"""Server-side wiring for the Conversation Management layer (Phases 1.3+1.4).

This module is the ONLY bridge between the web server and ``dozen.context``:
the endpoints call ``prepare_run`` / ``finish_run`` and never see a manager,
repository, or model class. Everything here is deliberately fail-soft — a
conversation-layer problem must never break an orchestration run, so every
helper catches, reports through its return value, and lets the run proceed.

Phase 1.4: ``prepare_run`` now also builds the conversation context (the
history BEFORE the current prompt) via the ContextBuilder, and
``compose_task_context`` assembles the final ``Task.context`` string. Any
context failure degrades to stateless execution.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dozen.context.context_builder import (
    DEFAULT_TOKEN_BUDGET,
    BuiltContext,
    ContextBuilder,
)
from dozen.context.manager import ConversationService
from dozen.context.manager.session import ConversationSession

# Storage root: env override first (tests use it), else `<repo>/.conversations`
# anchored absolutely so it does not depend on the working directory.
_ENV_ROOT = "DOZEN_CONVERSATIONS_ROOT"
_ENV_BUDGET = "DOZEN_CONTEXT_BUDGET_TOKENS"
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / ".conversations"

_service: Optional[ConversationService] = None
_builder: Optional[ContextBuilder] = None
_service_lock = threading.Lock()


def get_service() -> ConversationService:
    """Process-wide ConversationService singleton (lazily built)."""
    global _service
    with _service_lock:
        if _service is None:
            root = os.environ.get(_ENV_ROOT, str(_DEFAULT_ROOT))
            _service = ConversationService(root_path=root)
        return _service


def get_context_builder() -> ContextBuilder:
    """Process-wide ContextBuilder over the same ConversationManager."""
    global _builder
    service = get_service()  # takes the lock; must be resolved first
    with _service_lock:
        if _builder is None:
            try:
                budget = int(os.environ.get(_ENV_BUDGET, DEFAULT_TOKEN_BUDGET))
            except ValueError:
                budget = DEFAULT_TOKEN_BUDGET
            _builder = ContextBuilder(service.manager, budget_tokens=budget)
        return _builder


def reset_service() -> None:
    """Drop the singletons (tests; simulating a server restart)."""
    global _service, _builder
    with _service_lock:
        if _service is not None:
            try:
                _service.close()
            except Exception:
                pass
        _service = None
        _builder = None


@dataclass
class RunConversation:
    """Everything the run worker needs to finish recording later, plus the
    context block (Phase 1.4) built from history BEFORE the current prompt."""

    conversation_id: str
    created: bool
    session: Optional[ConversationSession]
    recording: bool  # False when the conversation layer is degraded
    # Context injection (Phase 1.4). Empty string == stateless run.
    formatted_history: str = ""
    context_messages: int = 0
    context_tokens: int = 0
    context_truncated: bool = False
    context_fallback_reason: Optional[str] = None


def prepare_run(
    prompt: str,
    conversation_id: Optional[str],
    run_id: str,
    context: str = "",
    desired_output: str = "",
    *,
    inject_history: bool = False,
) -> RunConversation:
    """Resolve/create the conversation, optionally build the history context,
    store the user message, open a session — in that order: the context is
    built BEFORE the current prompt is recorded, so the prompt never duplicates
    into its own history.

    ``inject_history`` defaults to **False**: DOZEN is a one-shot orchestrator.
    Injecting prior turns made every planner prompt grow without bound — a long
    tail of earlier failures ended up dwarfing the actual request, and models
    answered the transcript instead of the task. Messages are still RECORDED
    (that is what ``/api/conversation/{id}`` reads); they are simply never fed
    back into a prompt. Pass ``inject_history=True`` only to exercise the
    context builder directly.

    Runs BEFORE the workflow starts. Never raises: if persistence is broken,
    the run continues stateless with ``recording=False``, exactly as pre-1.3.
    """
    try:
        service = get_service()
        cid, created = service.ensure_conversation(conversation_id, title_hint=prompt)
        built = (
            _build_context_safe(str(cid), prompt)
            if inject_history
            else _empty_context(str(cid))
        )
        stored = service.record_user_message(
            cid, prompt, run_id=run_id, context=context, desired_output=desired_output
        )
        session = service.begin_run(cid, run_id=run_id)
        return RunConversation(
            conversation_id=str(cid),
            created=created,
            session=session,
            recording=stored.ok,
            formatted_history=built.formatted_history,
            context_messages=built.message_count,
            context_tokens=built.estimated_tokens,
            context_truncated=built.truncated,
            context_fallback_reason=built.fallback_reason,
        )
    except Exception:
        # Conversation layer down entirely: degrade to pre-1.3 behavior.
        return RunConversation(
            conversation_id=(conversation_id or ""), created=False,
            session=None, recording=False,
        )


def _empty_context(conversation_id: str) -> BuiltContext:
    """A stateless context — the one-shot default. No history is even read."""
    return BuiltContext(
        conversation_id=conversation_id, formatted_history="",
        estimated_tokens=0, message_count=0, truncated=False,
        fallback=False, fallback_reason=None,
    )


def _build_context_safe(conversation_id: str, prompt: str) -> BuiltContext:
    """Context building can never take the run down with it."""
    try:
        return get_context_builder().build_context(conversation_id, prompt)
    except Exception as exc:  # noqa: BLE001
        return BuiltContext(
            conversation_id=conversation_id, formatted_history="",
            estimated_tokens=0, message_count=0, truncated=False,
            fallback=True, fallback_reason=f"builder crashed: {exc}",
        )


def compose_task_context(formatted_history: str, user_context: str = "") -> str:
    """Assemble the final ``Task.context``: conversation history first (with a
    one-line frame so the planner reads it as background, not as the task),
    then any caller-supplied context. Either part may be empty."""
    parts: list[str] = []
    if formatted_history.strip():
        parts.append(
            "PREVIOUS CONVERSATION (earlier turns between the user and the "
            "assistant, oldest first — treat as background memory):\n"
            + formatted_history
        )
    if user_context.strip():
        parts.append(user_context.strip())
    return "\n\n".join(parts)


def finish_run(
    handle: RunConversation,
    final_answer: str,
    error: Optional[str] = None,
    cancelled: bool = False,
) -> None:
    """Store the assistant response and close the session. Never raises."""
    try:
        service = get_service()
        if handle.recording and handle.conversation_id:
            from dozen.context.domain.types import ConversationId

            service.record_assistant_message(
                ConversationId(handle.conversation_id),
                final_answer,
                run_id=str(handle.session.run_id) if handle.session else None,
                error=error,
                cancelled=cancelled,
            )
        service.finish_run(handle.session)
    except Exception:
        pass  # recording is best-effort by contract


def get_history(conversation_id: str, limit: int = 500) -> tuple[Optional[dict], Optional[str]]:
    """(payload, error_message) for the debug/history endpoint."""
    try:
        result = get_service().get_history(conversation_id, limit=limit)
    except Exception as exc:
        return None, f"conversation layer unavailable: {exc}"
    if result.ok:
        return result.unwrap(), None
    return None, f"{result.error.code.value}: {result.error.message}"
