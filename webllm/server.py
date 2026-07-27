"""FastAPI server: onboarding UI + login handoff + task runner.

Flow
----
1. GET  /                       -> the home page: the model-selection grid.
2. POST /api/login/start        -> launches visible browser windows for each
                                   selected provider, on the provider login page.
3. (user logs in + solves captchas by hand in those windows — the "waiting room")
4. POST /api/login/confirm      -> confirm ONE provider (per-card button); checks
                                   that its window is authenticated and persists it.
5. POST /api/orchestrator/ready -> once all selected providers are "Ready", build
                                   the orchestrator from the logged-in sessions.
   (POST /api/login/complete    -> legacy one-shot: re-check all + build at once.)
6. POST /api/run                -> runs the full dozen orchestrator (planner ->
                                   execute DAG -> synthesize) over the live
                                   browser sessions, returns the final answer.

Run:  python run_web.py          (http://127.0.0.1:8000)

Do NOT run this with uvicorn's ``--reload``: the autoreloader restarts the
process on any file change, which drops in-flight requests (the browser reports
them as a bare "Failed to fetch") and orphans the visible Chromium windows the
BrowserManager owns — the orphans keep holding their ``.profiles/<provider>``
locks, so the next login window silently fails to open.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dozen.cancellation import CancelToken
from dozen.presentation import sanitize_diagnostic

from .browser_manager import BrowserManager
from .conversations import compose_task_context, finish_run, get_history, prepare_run
from .pool import build_orchestrator
from .progress import RUNS, RunChannel
from .providers import PROVIDERS, available_providers
from .run_slot import RUN_SLOT

app = FastAPI(title="Dozen Web Orchestrator")

_STATIC = Path(__file__).parent / "static"


class _State:
    browser: Optional[BrowserManager] = None
    orchestrator = None
    providers: list[str] = []
    # Active-run ownership (which run is live, its cancel token, when to
    # release the slot) is owned solely by RUN_SLOT — see webllm/run_slot.py.


state = _State()


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class LoginStartRequest(BaseModel):
    providers: list[str]


class ConfirmRequest(BaseModel):
    provider: str


class RunRequest(BaseModel):
    prompt: str
    context: str = ""
    desired_output: str = ""
    # Optional: continue an existing conversation. Empty/stale ids are fine —
    # the conversation layer falls back to creating a fresh one.
    conversation_id: str = ""


def _terminal_result_event(result, *, cancelled: bool, conversation_id: str) -> dict:
    """Build one terminal SSE event whose status agrees with its payload.

    ``final_answer`` is ALWAYS a plain rendered string (the orchestrator's
    final rendering boundary produced it); structured artifact metadata travels
    separately inside ``summary`` as counts only — never file bodies. The
    additive ``answer_kind`` names the typed presentation for clients that want
    it; every pre-existing field keeps its exact shape.
    """
    answer_is_text = isinstance(result.final_answer, str)
    safe_error = sanitize_diagnostic(result.error) if result.error else ""
    if not answer_is_text and not safe_error:
        safe_error = "The run returned a non-text final result."
    if cancelled:
        status, icon, message = "cancelled", "🛑", "Orchestration cancelled."
    elif safe_error:
        status, icon, message = "error", "❌", "Orchestration failed."
    else:
        status, icon, message = "done", "🎉", "Orchestration complete."
    presentation = getattr(result, "presentation", None)
    summary = result.summary()
    summary["error"] = safe_error
    summary["warnings"] = [sanitize_diagnostic(w) for w in result.warnings]
    for subtask in summary.get("subtasks", []):
        if isinstance(subtask, dict):
            for key in ("title", "agent", "reasoning", "status"):
                value = subtask.get(key, "")
                subtask[key] = (
                    sanitize_diagnostic(value) if isinstance(value, str) else ""
                )
    payload = {
        "final_answer": result.final_answer if answer_is_text else "",
        "error": safe_error,
        "warnings": [sanitize_diagnostic(w) for w in result.warnings],
        "cancelled": cancelled,
        "summary": summary,
        "conversation_id": conversation_id,
    }
    if presentation is not None:
        payload["answer_kind"] = presentation.kind.value
    return {
        "phase": "result",
        "status": status,
        "icon": icon,
        "message": message,
        "result": payload,
    }


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _startup() -> None:
    # Runtime build identity FIRST, before anything else can fail: a stale
    # workspace must be visible in the logs of every boot, so a fix can never be
    # "verified" against a tree that never ran. Content-free — no request data.
    from dozen.build_info import startup_banner

    print(startup_banner(), file=sys.stderr, flush=True)
    # Visible windows are required for the manual-login + captcha handoff.
    state.browser = BrowserManager(headless=False)


@app.on_event("shutdown")
def _shutdown() -> None:
    if state.browser is not None:
        state.browser.shutdown()
    RUNS.shutdown()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/providers")
def list_providers() -> dict:
    return {"providers": available_providers()}


# --------------------------------------------------------------------------- #
# Model lifecycle — the REST endpoints the Model Library / Active drawer use.
# These drive the real Playwright handoff:
#   add  -> start-login  (opens a visible window at the provider login page)
#   user logs in by hand
#   ok   -> confirm-login (verifies + persists the session, marks Ready)
#   rm   -> DELETE        (closes the window, drops it from the pool)
# --------------------------------------------------------------------------- #
@app.post("/api/models/start-login")
def models_start_login(req: ConfirmRequest) -> dict:
    """Launch a visible Playwright window for ONE provider, at its login URL.

    The actual Chromium launch + navigation is SLOW (can take many seconds the
    first time), so we kick it off on the browser thread and return immediately.
    The window opens asynchronously while the UI already shows "connecting"; the
    client polls ``/api/active`` (or uses Confirm login) to learn when it's ready.
    This is what turns the old ~30s blocking add into a sub-second response.
    """
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    if req.provider not in PROVIDERS:
        raise HTTPException(404, f"Unknown provider {req.provider!r}.")
    # Track it in the selected set, then open (or reuse) its window in the bg.
    if req.provider not in state.providers:
        state.providers.append(req.provider)
    state.browser.start_login_async([req.provider])
    return {"provider": req.provider, "status": "connecting", "opened": True}


@app.post("/api/models/confirm-login")
def models_confirm_login(req: ConfirmRequest) -> dict:
    """Verify the window is authenticated, persist the session, mark Ready.

    The persistent context auto-saves cookies/localStorage to the on-disk
    profile (``webllm/.profiles/<provider>``), which is the *same* profile the
    WebAutomationLLMClient reuses later — so confirming is all it takes for the
    orchestrator to drive it in the background.
    """
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    if req.provider not in state.providers:
        raise HTTPException(400, f"{req.provider!r} has no open login window.")
    result = state.browser.confirm_provider(req.provider)  # {ready, name}
    # Keep the live orchestrator in sync with the ready pool.
    ready = state.browser.active_providers()
    state.orchestrator = build_orchestrator(state.browser, ready) if ready else None
    return {"provider": req.provider, "status": "ready" if result["ready"] else "error", **result}


@app.delete("/api/models/{provider_id}")
def models_delete(provider_id: str) -> dict:
    """Close the provider's window and remove it from the active pool."""
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    state.browser.remove_provider(provider_id)
    if provider_id in state.providers:
        state.providers.remove(provider_id)
    ready = state.browser.active_providers()
    state.orchestrator = build_orchestrator(state.browser, ready) if ready else None
    return {"removed": provider_id, "ready": ready}


@app.post("/api/login/start")
def login_start(req: LoginStartRequest) -> dict:
    if not req.providers:
        raise HTTPException(400, "Select at least one provider.")
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    # Accumulate across calls so models can be added one-by-one from the
    # Model Library without disturbing already-open windows.
    state.providers = sorted(set(state.providers) | set(req.providers))
    results = state.browser.start_login(req.providers)
    return {"started": results}


@app.get("/api/login/status")
def login_status() -> dict:
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    return {"status": state.browser.login_status()}


@app.post("/api/login/confirm")
def login_confirm(req: ConfirmRequest) -> dict:
    """Confirm a SINGLE provider (the waiting-room per-card button)."""
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    if req.provider not in state.providers:
        raise HTTPException(400, f"{req.provider!r} was not part of this login batch.")
    result = state.browser.confirm_provider(req.provider)
    return {"provider": req.provider, **result}


@app.post("/api/login/complete")
def login_complete() -> dict:
    """Re-check every window and build the orchestrator from logged-in ones."""
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    confirmed = state.browser.confirm_login()
    ready = [k for k, ok in confirmed.items() if ok]
    if not ready:
        raise HTTPException(
            409,
            "None of the opened sessions appear logged in yet. Finish logging "
            "in (and solving any captcha) in the browser windows, then retry.",
        )
    state.orchestrator = build_orchestrator(state.browser, ready)
    return {"logged_in": confirmed, "ready_providers": ready}


@app.post("/api/orchestrator/ready")
def orchestrator_ready() -> dict:
    """Finalize the handoff: build the orchestrator from all logged-in sessions.

    Called once every selected provider shows "Ready" in the waiting room.
    """
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    ready = state.browser.active_providers()
    if not ready:
        raise HTTPException(409, "No providers are logged in yet.")
    state.orchestrator = build_orchestrator(state.browser, ready)
    return {"ready_providers": ready}


@app.get("/api/active")
def active() -> dict:
    """Snapshot of every opened session and which ones are logged in/ready."""
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    return {
        "selected": state.providers,
        "ready": state.browser.active_providers(),
        "status": state.browser.login_status(),
    }


@app.post("/api/provider/remove")
def provider_remove(req: ConfirmRequest) -> dict:
    """Remove a model from the active pool (closes its window, keeps profile)."""
    if state.browser is None:
        raise HTTPException(500, "Browser manager not initialised.")
    state.browser.remove_provider(req.provider)
    if req.provider in state.providers:
        state.providers.remove(req.provider)
    # Rebuild (or tear down) the orchestrator to reflect the new pool.
    ready = state.browser.active_providers()
    state.orchestrator = build_orchestrator(state.browser, ready) if ready else None
    return {"removed": req.provider, "ready": ready}


@app.post("/api/run")
def run_task(req: RunRequest) -> dict:
    """Start an orchestration run in the background; return its ``run_id``.

    The run streams structured progress over ``GET /api/run/{run_id}/events``
    (Server-Sent Events) and finishes by emitting a terminal ``result`` event —
    so the UI never blocks waiting for the whole pipeline to complete.
    """
    if state.orchestrator is None:
        raise HTTPException(409, "Complete login before running a task.")
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty.")

    from dozen import Task

    # Fresh cancellation token + event channel for this run.
    cancel = CancelToken()
    channel = RUNS.create()

    # Admission control — the single-active-run invariant. Claim the one slot
    # BEFORE creating any conversation or execution state, so a rejected run
    # leaves no side effects behind. A second concurrent run is refused
    # deterministically with HTTP 409, naming the run that holds the slot.
    acquired, active_run_id = RUN_SLOT.try_acquire(channel.run_id, cancel)
    if not acquired:
        RUNS.discard(channel.run_id)
        raise HTTPException(
            409,
            f"Another run is currently active ({active_run_id}); DOZEN executes "
            "one workflow at a time. Retry after the active run finishes.",
        )

    orchestrator = state.orchestrator
    started = False
    try:
        # Conversation layer (Phase 1.3): resolve/create the conversation, store
        # the user prompt, and (Phase 1.4) build the history context — all BEFORE
        # the workflow starts.
        conversation = prepare_run(
            prompt=req.prompt,
            conversation_id=req.conversation_id,
            run_id=channel.run_id,
            context=req.context,
            desired_output=req.desired_output,
        )

        # Phase 1.4: the orchestrator now receives conversation context + the
        # current prompt. History is only PREPENDED into Task.context — planner,
        # executor and providers are untouched; empty history == stateless run.
        task = Task(
            prompt=req.prompt,
            context=compose_task_context(conversation.formatted_history, req.context),
            desired_output=req.desired_output,
        )
        if conversation.formatted_history:
            note = f"Context: {conversation.context_messages} past messages injected (~{conversation.context_tokens} tokens)"
            if conversation.context_truncated:
                note += " — oldest messages trimmed to fit the budget"
            channel.emit({"phase": "context", "status": "log", "icon": "🧠", "message": note})

        def _worker() -> None:
            try:
                result = orchestrator.run(task, cancel, channel.emit)
                # Store the assistant response before announcing the result.
                safe_error = sanitize_diagnostic(result.error) if result.error else ""
                safe_answer = (
                    result.final_answer if isinstance(result.final_answer, str) else ""
                )
                finish_run(
                    conversation,
                    final_answer=safe_answer,
                    error=safe_error,
                    cancelled=cancel.cancelled,
                )
                channel.emit(_terminal_result_event(
                    result,
                    cancelled=cancel.cancelled,
                    conversation_id=conversation.conversation_id,
                ))
            except Exception as exc:  # noqa: BLE001 - report instead of crashing thread
                # Bounded + sanitized: an exception message must never carry a
                # raw provider payload into persistence or the SSE stream.
                reason = sanitize_diagnostic(exc)
                finish_run(conversation, final_answer="", error=reason)
                channel.emit({
                    "phase": "error", "status": "error", "icon": "❌",
                    "message": f"Run failed: {reason}",
                })
            finally:
                # Release the active slot on EVERY terminal path — success,
                # failure, cancellation or unexpected exception. Release is
                # owner-checked, so a finishing run never clears a successor's
                # slot. Then keep the completed channel briefly so EventSource
                # retries don't turn a normal stream close into a spurious 404.
                RUN_SLOT.release(channel.run_id)
                RUNS.close(channel.run_id)

        threading.Thread(target=_worker, name=f"run-{channel.run_id}", daemon=True).start()
        started = True
        return {"run_id": channel.run_id, "conversation_id": conversation.conversation_id}
    finally:
        # Startup failed AFTER the slot was acquired (context build raised, or
        # the worker thread could not be launched): the worker's finally will
        # never run, so release ownership here. No-op on the success path.
        if not started:
            RUN_SLOT.release(channel.run_id)
            RUNS.discard(channel.run_id)


@app.get("/api/run/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    """Server-Sent Events stream of live progress for one run."""
    channel: Optional[RunChannel] = RUNS.get(run_id)
    if channel is None:
        raise HTTPException(404, "Unknown or already-finished run.")

    def gen():
        # Initial comment flushes headers and confirms the connection.
        yield ": connected\n\n"
        idx = 0
        last_beat = time.monotonic()
        while True:
            batch, done = channel.wait_batch(idx, timeout=1.0)
            for ev in batch:
                idx += 1
                safe_ev = dict(ev) if isinstance(ev, dict) else {
                    "phase": "log", "status": "log", "message": "Progress update."
                }
                event_message = safe_ev.get("message", "")
                safe_ev["message"] = (
                    sanitize_diagnostic(event_message)
                    if isinstance(event_message, str) else "Progress update."
                )
                yield "data: " + json.dumps(safe_ev, ensure_ascii=False) + "\n\n"
            if done:
                break
            if not batch and (time.monotonic() - last_beat) > 10:
                last_beat = time.monotonic()
                yield ": keepalive\n\n"  # comment line; keeps the socket warm

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )


@app.get("/api/conversation/{conversation_id}")
def conversation_history(conversation_id: str, limit: int = 500) -> dict:
    """The stored history of one conversation (Phase 1.3).

    Used to verify persistence across refreshes/restarts and as the seam for
    the future history UI. Read-only; never touches the orchestrator.
    """
    payload, error = get_history(conversation_id, limit=limit)
    if payload is None:
        raise HTTPException(404, error or "Conversation not found.")
    return payload


# --------------------------------------------------------------------------- #
# Reliability debug API (Phase 2.1.3) — developer-only, read-only, and OFF by
# default. Enable with DOZEN_RELIABILITY_DEBUG=1. When disabled, every
# endpoint returns 404, indistinguishable from not existing.
# --------------------------------------------------------------------------- #
def _debug_api():
    from .reliability_debug import get_debug_api
    from dozen.reliability.debug import DebugApiDisabled

    api = get_debug_api()
    if not api.enabled:
        raise HTTPException(404, "Not found.")
    return api, DebugApiDisabled


@app.get("/api/debug/build")
def debug_build() -> dict:
    """The runtime build identity (developer-only, gated by the debug mode).

    The remote response remains path-free even when debug mode is enabled;
    absolute paths are confined to the local startup banner. When debug is off
    this endpoint returns 404, exactly like every other debug route.
    """
    from dozen.build_info import build_identity

    _debug_api()  # raises 404 unless DOZEN_RELIABILITY_DEBUG is enabled
    return build_identity(reveal_paths=False)


@app.get("/api/debug/reliability/attempts")
def debug_reliability_attempts(
    limit: int = 50,
    provider: str = "",
    stage: str = "",
    run_id: str = "",
    conversation_id: str = "",
) -> dict:
    api, _ = _debug_api()
    return api.attempts(
        limit=limit,
        provider=provider or None,
        stage=stage or None,
        run_id=run_id or None,
        conversation_id=conversation_id or None,
    )


@app.get("/api/debug/reliability/attempt/{attempt_id}")
def debug_reliability_attempt(attempt_id: str) -> dict:
    api, _ = _debug_api()
    found = api.attempt(attempt_id)
    if found is None:
        raise HTTPException(404, "Unknown attempt id.")
    return found


@app.get("/api/debug/reliability/stats")
def debug_reliability_stats() -> dict:
    api, _ = _debug_api()
    return api.stats()


@app.get("/api/debug/reliability/providers")
def debug_reliability_providers() -> dict:
    api, _ = _debug_api()
    return api.providers()


@app.get("/api/debug/reliability/stages")
def debug_reliability_stages() -> dict:
    api, _ = _debug_api()
    return api.stages()


@app.get("/api/debug/reliability/health")
def debug_reliability_health() -> dict:
    api, _ = _debug_api()
    return api.health()


@app.get("/api/debug/reliability/health/history")
def debug_reliability_health_history(provider: str = "", limit: int = 50) -> dict:
    api, _ = _debug_api()
    return api.health_history(provider or None, limit=limit)


@app.get("/api/debug/reliability/health/{provider}")
def debug_reliability_health_provider(provider: str) -> dict:
    api, _ = _debug_api()
    found = api.health_provider(provider)
    if found is None:
        raise HTTPException(404, "Unknown provider (no health data recorded).")
    return found


@app.post("/api/stop")
def stop_task() -> dict:
    """Abort the in-flight orchestration (the Stop button).

    Trips ONLY the active run's cancellation token, which every stage checks:
    planning, the DAG executor, worker calls, the MutationObserver waits and
    the prompt-injection loop all unwind at their next checkpoint instead of
    blocking to completion. Safe to call when no run is active, more than once,
    and concurrently with a run completing — RUN_SLOT owns the transition, so a
    completed or replaced run is never cancelled by a stale Stop.
    """
    return RUN_SLOT.stop_active()


# Mount static assets last so /api/* routes take precedence.
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
