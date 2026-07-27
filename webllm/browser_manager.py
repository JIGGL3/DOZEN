"""Thread-safe Playwright session manager — one worker thread PER PROVIDER.

WHY A THREAD PER PROVIDER
-------------------------
Playwright's *sync* API is bound to the thread that created it; you cannot
touch a Page from another thread. The ``dozen`` executor runs subtasks on a
ThreadPoolExecutor, so browser work must be marshalled onto Playwright-owning
threads.

An earlier design used ONE thread (and one job queue) for every provider —
which silently serialized the whole pool: a 250-second Copilot generation
blocked the ChatGPT and Gemini prompts queued behind it, so "parallel"
subtasks executed one after another. Now each provider gets its own dedicated
worker thread with its own job queue and its OWN Playwright instance (created
lazily on that thread; multiple sync Playwright instances in one process are
fine as long as each stays on its thread). Calls to *different* providers run
truly in parallel, while calls to the *same* provider queue up naturally —
exactly what you want when scripting a single logged-in chat window.

SESSION PERSISTENCE
-------------------
Each provider gets its own persistent browser context (a real on-disk profile
under ``.profiles/<key>``). Cookies/localStorage survive process restarts, so
you typically log in by hand once and can re-run the orchestrator for days
without logging in again.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from dozen.cancellation import CancelledError

from .browser_cancellation import (
    DEFAULT_INTERRUPTION_POLICY,
    CancellationObservation,
    InterruptionActionResult,
    InterruptionOutcome,
    InterruptionPolicy,
    InterruptionReason,
    InterruptionRequest,
    InterruptionSnapshot,
    StopActionStatus,
    outcome_for_status,
)
from .browser_jobs import (
    BrowserJobEvent,
    BrowserJobId,
    BrowserJobRegistry,
    BrowserJobSnapshot,
    EventSink,
    JobState,
    TerminalCause,
)
from .browser_queue import (
    DEFAULT_QUEUE_POLICY,
    AdmissionOutcome,
    AdmissionSnapshot,
    BrowserQueueRejectedError,
    ProviderJobQueue,
    QueueEntry,
    QueueEvent,
    QueueEventKind,
    QueueEventSink,
    QueuePolicy,
    QueueSnapshot,
    make_admission_snapshot,
)
from .providers import ProviderAdapter, ProviderError, get_adapter

# Kept out of the geometry policy on purpose: this flag is anti-bot hygiene
# (hides navigator.webdriver), not window sizing.
_ANTIBOT_ARG = "--disable-blink-features=AutomationControlled"


def provider_window_launch_kwargs(headless: bool) -> dict[str, Any]:
    """The ONE authoritative geometry policy for provider Chromium windows.

    Every provider context — first login, worker restart, and closed-tab
    recovery — is launched through ``_ensure_session``, which takes its
    geometry exclusively from here.

    Headed (the production mode; visible windows are required for the manual
    login/captcha handoff):

    * ``no_viewport=True`` — disables Playwright's viewport EMULATION so the
      page always tracks the native window size. With a fixed ``viewport``
      (the previous ``1280x900``), Playwright does NOT resize the native
      window; it letterboxes page content into a fixed region anchored at the
      window's top-left. That produced exactly the reported symptoms: an
      unused strip on the right (window wider than 1280) and a chat composer
      pushed below the visible bottom edge (900px of content in a window
      whose bottom already sat off the work area).
    * ``--start-maximized`` — asks Chromium for a NATIVE maximized window.
      Windows sizes a maximized window to the monitor's *work area*, which
      inherently respects the taskbar (any size/edge), display scaling, the
      work-area origin, and whichever monitor the window opens on — with no
      hard-coded coordinates. The window stays a normal, resizable window
      (maximized is not kiosk and not fullscreen), and the flag overrides any
      stale window placement remembered inside the persistent profile.

    Headless (test/CI convenience; no native window exists to size): keep the
    previous fixed viewport so scraping geometry stays deterministic.
    """
    if headless:
        return {
            "viewport": {"width": 1280, "height": 900},
            "args": [_ANTIBOT_ARG],
        }
    return {
        "no_viewport": True,
        "args": [_ANTIBOT_ARG, "--start-maximized"],
    }


@dataclass
class _Job:
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    physical_done: threading.Event = field(default_factory=threading.Event)
    result_lock: threading.Lock = field(default_factory=threading.Lock)
    result: Any = None
    error: Optional[BaseException] = None
    caller_cause: Optional[TerminalCause] = None
    # --- Phase 5A browser-job lifecycle (None for legacy jobs: login/status/…) #
    job_id: Optional[str] = None
    provider: Optional[str] = None
    should_cancel: Optional[Callable[[], bool]] = None
    # --- Phase 5B active interruption (None for legacy jobs) ------------------ #
    # The authoritative interruption-request record and the composite
    # cancellation observation the running provider operation polls.
    interruption: Optional[InterruptionRequest] = None
    cancel_observation: Optional[CancellationObservation] = None
    # True once a produced result/exception was discarded because the caller no
    # longer owned the job (timed out / cancelled). Diagnostics only.
    late_discarded: bool = False
    # True only when the adapter returned or raised normally (not cooperative
    # cancellation), proving its physical call has already ended.
    provider_call_settled: bool = False


@dataclass
class _Session:
    adapter: ProviderAdapter
    context: Any = None   # playwright BrowserContext
    page: Any = None      # playwright Page
    logged_in: bool = False


class _Worker:
    """One provider's dedicated browser thread: bounded queue + lazy Playwright."""

    def __init__(
        self,
        key: str,
        run_loop: Callable[["_Worker"], None],
        *,
        policy: QueuePolicy = DEFAULT_QUEUE_POLICY,
    ) -> None:
        self.key = key
        # Phase 5C: a bounded FIFO provider queue replaces the unbounded
        # ``queue.Queue``. Prompt/global bounds are enforced atomically by the
        # manager's admission lock; the queue itself enforces the control bound
        # and physically removes cancelled/timed-out entries (no tombstones).
        self.jobs: ProviderJobQueue = ProviderJobQueue(key, policy)
        self.playwright: Any = None  # created lazily ON this worker's thread
        # Set during shutdown so the run loop abandons still-queued jobs instead
        # of starting fresh browser work.
        self.stopping = threading.Event()
        self.current_lock = threading.Lock()
        self.current_job: Optional[_Job] = None
        self.thread = threading.Thread(
            target=run_loop, args=(self,), name=f"playwright-{key}", daemon=True
        )
        self.thread.start()


class BrowserManager:
    """Owns one persistent context per provider, each on its own thread."""

    def __init__(
        self,
        profiles_dir: Optional[str] = None,
        headless: bool = False,
        *,
        job_event_sink: Optional["EventSink"] = None,
        interruption_policy: Optional[InterruptionPolicy] = None,
        queue_policy: Optional[QueuePolicy] = None,
        queue_event_sink: Optional["QueueEventSink"] = None,
    ):
        self.headless = headless
        self.profiles_dir = Path(profiles_dir or Path(__file__).parent / ".profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._workers: dict[str, _Worker] = {}
        self._workers_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        # Last window-launch failure per provider. ``start_login_async`` is
        # fire-and-forget, so without this the reason a login window never
        # appeared (locked profile, missing Chromium, crashed launch) was
        # discarded and the UI sat on "connecting" forever. Recorded on the
        # worker thread, read by ``_status_one``; guarded by its own lock so it
        # never participates in the session/worker lock ordering.
        self._launch_errors: dict[str, str] = {}
        self._launch_errors_lock = threading.Lock()
        self._shutdown = False
        # Phase 5A: this manager's OWN browser-job registry (never a global
        # singleton). Owns lifecycle state, provider→current-job ownership and
        # a bounded terminal history.
        self._registry = BrowserJobRegistry(event_sink=job_event_sink)
        # Phase 5B: one authoritative interruption policy, and a direct handle on
        # the event sink so interruption events can be emitted OUTSIDE any lock
        # with sink failures isolated (the registry owns lifecycle events; the
        # manager owns interruption events).
        self._interruption_policy = interruption_policy or DEFAULT_INTERRUPTION_POLICY
        self._job_event_sink = job_event_sink
        # Phase 5C: bounded provider queues + deterministic admission
        # backpressure. ``_admission_lock`` is a strictly OUTER lock that makes
        # every prompt admission decision atomic against provider/global capacity,
        # provider quarantine, manager shutdown and concurrent submitters — it is
        # the sole owner of the shutdown flag for admission purposes. Quarantine
        # is a set of providers left unsafe to reuse by a Phase 5B interruption;
        # new prompt work is rejected deterministically until explicit teardown.
        self._queue_policy = queue_policy or DEFAULT_QUEUE_POLICY
        self._queue_event_sink = queue_event_sink
        self._admission_lock = threading.Lock()
        self._admission_cond = threading.Condition(self._admission_lock)
        self._pending_admission_publications = 0
        self._admission_publication_state = threading.local()
        self._quarantined: set[str] = set()

    # ------------------------------------------------------------------ #
    # Job plumbing (runs callables on the owning provider's thread)
    # ------------------------------------------------------------------ #
    def _ensure_worker_locked(self, key: str) -> _Worker:
        """Return (creating if needed) ``key``'s worker. Caller holds admission
        lock; this takes the strictly-inner ``_workers_lock``.

        Order is always ``_admission_lock`` → ``_workers_lock`` → queue lock, so
        no admission/shutdown/worker path can deadlock.
        """
        with self._workers_lock:
            if self._shutdown:
                raise RuntimeError("BrowserManager is shut down.")
            worker = self._workers.get(key)
            if worker is None or not worker.thread.is_alive():
                worker = _Worker(key, self._run_loop, policy=self._queue_policy)
                self._workers[key] = worker
            return worker

    def _enqueue(self, key: str, fn: Callable[[], Any]) -> _Job:
        """Queue a CONTROL job (login/status/confirm/remove) and return it.

        Control jobs have their own small reserved capacity so they can never
        grow without bound and can never consume the prompt-queue bound. The
        atomic admission lock keeps enqueue race-free against shutdown, so no
        control job can be inserted after the queue's close boundary. Raises
        :class:`BrowserQueueRejectedError` when the control bound is reached (the
        existing control callers already treat any exception as a failed op).
        """
        job = _Job(fn=fn)
        with self._admission_lock:
            worker = self._ensure_worker_locked(key)
            admitted = worker.jobs.enqueue_control(job)
        if not admitted:
            snap = make_admission_snapshot(
                AdmissionOutcome.REJECTED_PROVIDER_FULL,
                key,
                provider_queued_depth=worker.jobs.control_depth(),
                provider_capacity=self._queue_policy.max_queued_control_jobs_per_provider,
                global_queued_depth=0,
                global_capacity=self._queue_policy.max_total_queued_prompts,
                reason="control queue is full",
                policy=self._queue_policy,
            )
            self._emit_queue_event_from_admission(snap)
            raise BrowserQueueRejectedError(snap)
        return job

    @staticmethod
    def _await(job: _Job, timeout: float) -> Any:
        if not job.done.wait(timeout=timeout):
            raise TimeoutError("Browser job timed out.")
        if job.error is not None:
            raise job.error
        return job.result

    def _submit_to(self, key: str, fn: Callable[[], Any], timeout: float = 600.0) -> Any:
        """Run ``fn`` on ``key``'s thread and block for its result."""
        return self._await(self._enqueue(key, fn), timeout)

    def _run_loop(self, worker: _Worker) -> None:
        while True:
            entry = worker.jobs.get()
            if entry is None:  # closed and drained — deterministic shutdown
                break
            job = entry.job
            if worker.stopping.is_set():
                # Manager is shutting down: never start fresh browser work.
                self._abandon_on_shutdown(job)
                continue
            if entry.is_prompt:
                # Queue-wait expiry: a prompt that waited past its bounded queue
                # deadline is never submitted to the provider — it settles with
                # the existing compatible timeout behavior.
                if entry.is_expired(time.monotonic()):
                    self._expire_queued_prompt(entry, reason="queue-wait-expired")
                    continue
                # The prompt left the queue to run; report the new depth.
                self._emit_queue_event(
                    QueueEventKind.QUEUE_DEPTH_CHANGED,
                    worker.key,
                    job_id=job.job_id,
                    prompt_depth=worker.jobs.prompt_depth(),
                    control_depth=worker.jobs.control_depth(),
                    global_depth=self._global_prompt_depth(),
                    reason="dequeued-for-run",
                )
                with worker.current_lock:
                    worker.current_job = job
                try:
                    self._run_lifecycle_job(job)
                finally:
                    with worker.current_lock:
                        worker.current_job = None
            else:
                self._run_legacy_job(job)
        self._teardown_worker(worker)

    @staticmethod
    def _run_legacy_job(job: _Job) -> None:
        """Run a non-lifecycle job (login/status/confirm/remove) as always."""
        try:
            result = job.fn()
            with job.result_lock:
                job.result = result
        except BaseException as exc:  # noqa: BLE001 - marshalled to caller
            with job.result_lock:
                job.error = exc
            # Keep a server-side trace; the caller gets the exception too.
            traceback.print_exc()
        finally:
            job.done.set()
            job.physical_done.set()

    def _run_lifecycle_job(self, job: _Job) -> None:
        """Execute a prompt job under the browser-job lifecycle state machine.

        Atomically claims ``QUEUED → RUNNING`` before any browser work. A job the
        caller already timed out or cancelled is never submitted to the provider;
        a late result (arriving after the caller stopped owning the job) is
        discarded, never delivered. Provider ownership is released owner-checked
        in ``finally``. This method NEVER holds a registry lock across the browser
        operation.
        """
        reg = self._registry
        jid = job.job_id
        assert jid is not None

        # Pre-start guard: cancellation observed before browser work begins means
        # the provider prompt must never be sent.
        if job.should_cancel is not None:
            try:
                if job.should_cancel():
                    if reg.mark_cancelled(jid, reason="cancelled-before-start").transitioned:
                        with job.result_lock:
                            job.caller_cause = TerminalCause.CANCELLED
                        job.done.set()
            except Exception:  # noqa: BLE001 - a bad predicate must not crash us
                pass

        # Atomic claim. Fails if the job was TIMED_OUT / CANCELLED / ABANDONED
        # while queued — in which case we settle it and skip submission.
        if not reg.mark_running(jid).transitioned:
            # The prompt was never submitted, so no browser action is needed even
            # if cancellation/timeout requested one while the job sat queued.
            if job.interruption is not None:
                job.interruption.complete(
                    InterruptionOutcome.NOT_NEEDED, "not-runnable-at-claim"
                )
            reg.mark_physical_settled(jid, reason="not-runnable-at-claim")
            job.done.set()
            job.physical_done.set()
            return

        try:
            result = job.fn()
        except CancelledError as exc:
            # User pressed Stop: record cancellation, never reclassify as failure.
            with job.result_lock:
                job.error = exc
            if reg.mark_cancelled(jid, reason="cancelled-during-send").transitioned:
                with job.result_lock:
                    job.caller_cause = TerminalCause.CANCELLED
                job.done.set()
            else:
                # Caller already timed out/cancelled: this is a late unwind.
                with job.result_lock:
                    job.error = None
                job.late_discarded = True
                reg.record_late_result_discarded(jid, reason="late-cancel")
        except BaseException as exc:  # noqa: BLE001 - marshalled to caller
            job.provider_call_settled = True
            with job.result_lock:
                job.error = exc
            failed = reg.mark_failed(
                jid,
                failure_category=type(exc).__name__,
                failure_message=str(exc),
                reason="send-failed",
            )
            if not failed.transitioned:
                # Late exception after the caller stopped owning the job: record
                # diagnostically without changing successor state.
                with job.result_lock:
                    job.error = None
                job.late_discarded = True
                reg.record_late_result_discarded(jid, reason="late-exception")
            else:
                with job.result_lock:
                    job.caller_cause = TerminalCause.FAILED
                job.done.set()
            traceback.print_exc()
        else:
            job.provider_call_settled = True
            with job.result_lock:
                job.result = result
            if not reg.mark_completed(jid).transitioned:
                # Caller no longer owns the result (timed out/cancelled): discard
                # it so it can never be delivered to this or a later job.
                with job.result_lock:
                    job.result = None
                job.late_discarded = True
                reg.record_late_result_discarded(jid, reason="late-result")
            else:
                with job.result_lock:
                    job.caller_cause = TerminalCause.COMPLETED
                job.done.set()
        finally:
            # Phase 5B: if a caller cancelled/timed out this physically-running
            # job, actively stop generation NOW — on this worker thread, while the
            # page is still generating and BEFORE ownership is released. Owner-
            # checked and exactly-once; a no-op when no interruption was requested.
            safe_to_reuse = self._maybe_interrupt(job)
            if safe_to_reuse:
                reg.release_provider(jid)  # owner-checked; no-op if not owner
                # Completion/failure may already have settled via the registry
                # wrapper; safe cancellation settles here after confirmed idle.
                reg.mark_physical_settled(jid, reason="worker-finally")
                job.physical_done.set()
            elif job.provider is not None:
                # Phase 5C: reuse could NOT be proven safe (unsupported/failed
                # stop, or quiescence unconfirmed). The physical op is left
                # un-settled exactly as Phase 5B requires (no tab restart, no
                # forced settlement); the provider is quarantined so new prompt
                # admissions are rejected deterministically instead of piling up
                # behind an occupied tab. No recovery is attempted here.
                self._quarantine_provider(
                    job.provider, job_id=jid, reason="interruption-unsafe"
                )
            job.done.set()

    # ------------------------------------------------------------------ #
    # Active interruption (production-hardening 5B)
    # ------------------------------------------------------------------ #
    # These run ONLY on the provider's own worker thread (called from
    # _run_lifecycle_job's finally). The caller thread never performs a browser
    # action; it only records an interruption request on the job's
    # InterruptionRequest. The single stop click is gated by begin_action() so it
    # happens at most once per job, owner-checked immediately before and after any
    # bounded wait, so a stale Job A can never stop a successor Job B.
    def _maybe_interrupt(self, job: _Job) -> bool:
        """Stop once and report whether the provider tab is proven reusable."""
        req = job.interruption
        if req is None or not req.is_requested():
            return True  # normal completion path: no interruption quarantine.
        if job.provider_call_settled:
            # Natural result/failure won before the worker reached the stop seam.
            # No browser click is needed and the completed provider call is safe
            # to release; any late body/error has already been discarded.
            req.complete(
                InterruptionOutcome.SETTLED_BEFORE_ACTION,
                "provider-call-settled",
            )
            self._emit_interruption_event(job, "interruption-settled-before-action")
            return True

        policy = self._interruption_policy
        reason = req.reason
        # Policy gate: interruption globally disabled, or disabled for this reason.
        if (
            not policy.active_interruption_enabled
            or (reason is InterruptionReason.CANCELLED and not policy.interrupt_on_cancel)
            or (
                reason is InterruptionReason.CALLER_TIMEOUT
                and not policy.interrupt_on_caller_timeout
            )
        ):
            req.complete(InterruptionOutcome.NOT_NEEDED, "policy-disabled")
            return False

        # Owner check BEFORE any action: only the current physical owner of a job
        # that actually started may be stopped.
        if not self._interruption_owner_ok(job):
            req.complete(InterruptionOutcome.SETTLED_BEFORE_ACTION, "not-owner-pre")
            return True
        req.mark_required()

        # Exactly-once gate. A second cancellation signal can never click again.
        if not req.begin_action():
            snap = req.snapshot()
            return snap.outcome in (
                InterruptionOutcome.NOT_NEEDED.value,
                InterruptionOutcome.SETTLED_BEFORE_ACTION.value,
            ) or (
                snap.outcome == InterruptionOutcome.STOPPED.value
                and snap.diagnostic is None
            )
        self._emit_interruption_event(job, "interruption-started")

        try:
            action = self._perform_stop_action(job)
        except Exception as exc:  # noqa: BLE001 - a stop failure is a diagnostic
            req.complete(InterruptionOutcome.FAILED, f"stop-error:{type(exc).__name__}")
            self._emit_interruption_event(job, "interruption-failed")
            return False

        outcome = outcome_for_status(action.status)
        diagnostic = action.diagnostic
        if outcome is InterruptionOutcome.STOPPED and not action.quiescent:
            outcome = InterruptionOutcome.FAILED
            diagnostic = action.diagnostic or "quiescence-not-confirmed"
        req.complete(outcome, diagnostic)
        self._emit_interruption_event(job, f"interruption-{outcome.value}")
        return (
            action.status in (
                StopActionStatus.ALREADY_IDLE,
                StopActionStatus.SETTLED_BEFORE_ACTION,
            )
            or (action.status is StopActionStatus.STOPPED and action.quiescent)
        )

    def _interruption_owner_ok(self, job: _Job) -> bool:
        """True only if ``job`` is still the provider's running physical owner."""
        reg = self._registry
        if job.provider is None or job.job_id is None:
            return False
        if reg.active_job(job.provider) != job.job_id:
            return False
        snap = reg.snapshot(job.job_id)
        if snap is None or snap.physical_settled or snap.started_at is None:
            return False
        return True

    def _perform_stop_action(self, job: _Job) -> InterruptionActionResult:
        """Resolve the provider page and delegate to the adapter's stop seam.

        Worker-thread only (Playwright is thread-affine). Never opens a second
        Playwright connection, never reloads/closes the page. Re-checks ownership
        AFTER resolving the session and immediately BEFORE clicking, so a
        successor that won the provider slot in the interim is never clicked.
        """
        assert job.provider is not None
        adapter = get_adapter(job.provider)
        with self._sessions_lock:
            sess = self._sessions.get(job.provider)
        if not self._session_alive(sess):
            # No live page to act on; caller cancellation/timeout stays
            # authoritative and the physical op unwinds naturally.
            return InterruptionActionResult.unsupported("no-live-session")
        if not self._interruption_owner_ok(job):
            return InterruptionActionResult.already_idle("owner-changed-pre-click")
        return adapter.interrupt_generation(
            sess.page,
            job.cancel_observation,
            self._interruption_policy,
            owner_check=lambda: self._interruption_owner_ok(job),
        )

    def _emit_interruption_event(self, job: _Job, phase: str) -> None:
        """Emit a content-free interruption lifecycle event OUTSIDE any lock.

        Reuses the existing job-event sink so observers get one ordered stream.
        Sink failures are isolated; a bad observer can never break interruption.
        """
        sink = self._job_event_sink
        req = job.interruption
        if sink is None or req is None or job.job_id is None or job.provider is None:
            return
        snap = self._registry.snapshot(job.job_id)
        state = snap.state if snap is not None else JobState.RUNNING.value
        isnap = req.snapshot()
        try:
            event = BrowserJobEvent(
                job_id=job.job_id,
                provider=job.provider,
                old_state=state,
                new_state=state,
                timestamp=time.time(),
                reason=phase[: self._interruption_policy.max_event_reason_length],
                late_result_discarded=job.late_discarded,
                physical_settled=snap.physical_settled if snap is not None else False,
                interruption_reason=isnap.reason,
                interruption_outcome=isnap.outcome,
                interruption_attempted=isnap.action_attempted,
            )
        except Exception:  # noqa: BLE001 - never let event construction break us
            return
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - an observer must never break the job
            pass

    def _abandon_on_shutdown(self, job: _Job) -> None:
        """Release a still-queued job's waiter during manager shutdown."""
        if job.job_id is not None:
            state = self._registry.state(job.job_id)
            if state in (JobState.TIMED_OUT, JobState.CANCELLED):
                self._registry.mark_physical_settled(
                    job.job_id, reason="manager-shutdown-before-start"
                )
            else:
                self._registry.mark_abandoned(
                    job.job_id,
                    reason="manager-shutdown",
                    physical_settled=True,
                )
        with job.result_lock:
            if job.caller_cause is None:
                job.caller_cause = TerminalCause.SHUTDOWN
            if job.error is None:
                job.error = RuntimeError(
                    "BrowserManager was shut down before the job ran."
                )
        job.done.set()
        job.physical_done.set()

    def _ensure_playwright(self, worker: _Worker) -> Any:
        """Start this worker's Playwright lazily, on its own thread."""
        if worker.playwright is None:
            from playwright.sync_api import sync_playwright

            worker.playwright = sync_playwright().start()
        return worker.playwright

    def _current_worker(self, key: str) -> _Worker:
        with self._workers_lock:
            w = self._workers.get(key)
        if w is None:
            raise RuntimeError(f"No worker thread for provider {key!r}.")
        return w

    # ------------------------------------------------------------------ #
    # Public API (same names, signatures and return shapes as always)
    # ------------------------------------------------------------------ #
    def start_login(self, provider_keys: list[str]) -> dict[str, Any]:
        """Launch a visible window per provider, pointed at its login page.

        Fans out to each provider's thread so several windows boot in
        parallel, then aggregates the per-provider results.
        """
        jobs: list[tuple[str, Optional[_Job], Optional[str]]] = []
        for key in provider_keys:
            try:
                jobs.append((key, self._enqueue(key, lambda k=key: self._start_login_one(k)), None))
            except Exception as exc:  # noqa: BLE001
                jobs.append((key, None, str(exc)))
        results: dict[str, Any] = {}
        for key, job, err in jobs:
            if job is None:
                results[key] = {"opened": False, "error": err}
                continue
            try:
                results[key] = self._await(job, timeout=600.0)
            except Exception as exc:  # noqa: BLE001
                results[key] = {"opened": False, "error": str(exc)}
        return results

    def start_login_async(self, provider_keys: list[str]) -> None:
        """Non-blocking variant: kick off the (slow) window launches in the
        background so the API can return ``connecting`` immediately.

        Nothing awaits these jobs, so the launch outcome is recorded on the
        manager (see ``_set_launch_error``) instead of being dropped — that is
        what lets the UI say *why* a window never opened rather than spinning on
        "connecting" indefinitely.
        """
        for key in provider_keys:
            # Clear the previous failure so a retry starts from a clean slate
            # and a stale reason can never be mistaken for a fresh one.
            self._set_launch_error(key, None)
            try:
                self._enqueue(key, lambda k=key: self._start_login_one(k))
            except Exception as exc:  # noqa: BLE001 - report, don't 500
                self._set_launch_error(key, str(exc) or exc.__class__.__name__)

    def login_status(self) -> dict[str, Any]:
        with self._sessions_lock:
            keys = list(self._sessions.keys())
        jobs = [(k, self._enqueue(k, lambda k=k: self._status_one(k))) for k in keys]
        status: dict[str, Any] = {}
        for key, job in jobs:
            try:
                status[key] = self._await(job, timeout=60.0)
            except Exception:  # noqa: BLE001 - report a dead session, not a 500
                with self._sessions_lock:
                    sess = self._sessions.get(key)
                status[key] = {
                    "name": sess.adapter.name if sess else key,
                    "open": False,
                    "logged_in": False,
                    "error": self.launch_error(key),
                }
        # A launch that failed before its context existed leaves NO session, so
        # it would otherwise vanish from this snapshot entirely — the UI would
        # keep showing "connecting" for a window that will never appear. Report
        # those directly (no job: the worker may be exactly what's wedged).
        with self._launch_errors_lock:
            failed = [k for k in self._launch_errors if k not in status]
        for key in failed:
            try:
                name = get_adapter(key).name
            except Exception:  # noqa: BLE001 - unknown key is still reportable
                name = key
            status[key] = {
                "name": name,
                "open": False,
                "logged_in": False,
                "error": self.launch_error(key),
            }
        return status

    def confirm_login(self) -> dict[str, Any]:
        """Re-check every open session and mark the ones that look logged in."""
        with self._sessions_lock:
            keys = list(self._sessions.keys())
        jobs = [(k, self._enqueue(k, lambda k=k: self._confirm_one(k))) for k in keys]
        out: dict[str, Any] = {}
        for key, job in jobs:
            try:
                out[key] = self._await(job, timeout=120.0)
            except Exception:  # noqa: BLE001
                out[key] = False
        return out

    def confirm_provider(self, provider: str) -> dict[str, Any]:
        """Re-check ONE provider's window and persist its session state."""
        with self._sessions_lock:
            if provider not in self._sessions:
                raise ProviderError(
                    f"No open browser window for {provider!r}. Start login first."
                )
        return self._submit_to(provider, lambda: self._confirm_provider(provider),
                               timeout=120.0)

    def remove_provider(self, provider: str) -> dict[str, Any]:
        """Close ONE provider's window and drop it from the active pool."""
        # Removing is an explicit reset: a stale launch failure must not follow
        # the provider back into the pool when it is added again.
        self._set_launch_error(provider, None)
        with self._sessions_lock:
            has_session = provider in self._sessions
        # Do not clear quarantine until teardown has actually run.  A provider
        # may have a live worker/queued jobs even when a test seam or failed
        # session creation left no session record; clearing early would admit new
        # work onto the stale worker if control admission or teardown then failed.
        if not has_session and self._peek_worker(provider) is None:
            self._clear_quarantine(provider)
            return {"removed": True}
        return self._submit_to(provider, lambda: self._remove_provider(provider),
                               timeout=60.0)

    @staticmethod
    def _default_prompt_timeout(provider: str) -> float:
        return max(120.0, get_adapter(provider).tuning.generation_timeout_s + 60)

    def submit_prompt(
        self,
        provider: str,
        prompt: str,
        should_cancel: Optional[Callable[[], bool]] = None,
        *,
        correlation: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> "BrowserJobHandle":
        """Register a browser job, enqueue it, and return a handle immediately.

        Additive API (Phase 5A). The job runs on the provider's own thread, so
        prompts to different providers execute concurrently while prompts to the
        same provider queue up — exactly as before. The returned handle lets a
        caller wait for the result, observe lifecycle state, and record a caller
        timeout through one authoritative path. It exposes NO active browser
        cancellation.

        Phase 5C: submission is now admitted through one atomic, bounded-capacity
        decision. When the provider queue is full, the global queue is full, or the
        provider is quarantined, this raises :class:`BrowserQueueRejectedError`
        (a ``ProviderError`` subclass) *before* returning a handle and without ever
        registering a ``QUEUED`` record. Manager shutdown continues to return a
        deterministically-settled handle (its ``result()`` raises ``RuntimeError``),
        preserving the corrected Phase 5A/5B shutdown contract.
        """
        if timeout is None:
            timeout = self._default_prompt_timeout(provider)
        return self._admit_prompt(provider, prompt, should_cancel, correlation, timeout)

    def _admit_prompt(
        self,
        provider: str,
        prompt: str,
        should_cancel: Optional[Callable[[], bool]],
        correlation: Optional[str],
        timeout: float,
    ) -> "BrowserJobHandle":
        """Atomically decide admission and enqueue, or reject with a typed reason.

        The whole capacity decision (shutdown → quarantine → availability →
        per-provider capacity → global capacity → register → enqueue) runs under
        ``_admission_lock`` so it can never over-admit or leave an orphaned record.
        Because shutdown flips ``_shutdown`` under this SAME lock before it ever
        closes a queue, the queue is provably open here — enqueue cannot lose to a
        concurrent close. All lifecycle/queue event emission and any expired-job
        settlement happen strictly OUTSIDE the lock so no observer runs under it.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        requested_timeout = float(timeout)
        if requested_timeout <= 0 or not math.isfinite(requested_timeout):
            raise ValueError("timeout must be finite and > 0")

        now_mono = time.monotonic()
        deadline_mono = now_mono + min(
            requested_timeout, self._queue_policy.max_queue_wait_s
        )
        policy = self._queue_policy
        expired: list[QueueEntry] = []
        queued_event: Optional[BrowserJobEvent] = None
        job: Optional[_Job] = None
        job_id: Optional[BrowserJobId] = None
        entry: Optional[QueueEntry] = None
        q: Optional[ProviderJobQueue] = None
        provider_label = provider if isinstance(provider, str) and provider else "unknown"

        with self._admission_lock:
            worker: Optional[_Worker] = None
            if self._shutdown:
                outcome = AdmissionOutcome.REJECTED_MANAGER_SHUTTING_DOWN
            elif provider_label == "unknown":
                outcome = AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE
            elif policy.reject_quarantined_provider and provider in self._quarantined:
                outcome = AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED
            else:
                try:
                    get_adapter(provider)
                except ProviderError:
                    outcome = AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE
                else:
                    try:
                        worker = self._ensure_worker_locked(provider)
                    except Exception:  # noqa: BLE001 - typed admission failure
                        outcome = (
                            AdmissionOutcome.REJECTED_MANAGER_SHUTTING_DOWN
                            if self._shutdown
                            else AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE
                        )
                if worker is not None:
                    q = worker.jobs
                    # Global capacity is physical capacity. Sweep every provider,
                    # not only the target, before deciding whether it is full.
                    expired = self._collect_expired_prompts_locked(now_mono)
                    provider_depth = q.prompt_depth()
                    global_depth = self._global_prompt_depth_locked()
                    if provider_depth >= policy.max_queued_prompts_per_provider:
                        outcome = AdmissionOutcome.REJECTED_PROVIDER_FULL
                    elif global_depth >= policy.max_total_queued_prompts:
                        outcome = AdmissionOutcome.REJECTED_GLOBAL_FULL
                    else:
                        try:
                            job_id, queued_event = self._registry.register_deferred(
                                provider, correlation=correlation
                            )
                            interruption = InterruptionRequest(
                                job_id=job_id.value,
                                provider=provider,
                                policy=self._interruption_policy,
                            )
                            observation = CancellationObservation(
                                job_id=job_id.value,
                                provider=provider,
                                request=interruption,
                                external=should_cancel,
                            )
                            job = _Job(
                                fn=lambda: self._send_prompt(provider, prompt, observation),
                                job_id=job_id.value,
                                provider=provider,
                                should_cancel=observation,
                                interruption=interruption,
                                cancel_observation=observation,
                            )
                            entry = QueueEntry(
                                job=job,
                                is_prompt=True,
                                job_id=job_id.value,
                                enqueued_mono=now_mono,
                                deadline_mono=deadline_mono,
                                claimable=False,
                            )
                            q.enqueue_prompt(entry)
                        except Exception:  # noqa: BLE001 - admission must roll back fully
                            if q is not None and job_id is not None:
                                q.remove_prompt(job_id.value)
                            if job_id is not None:
                                self._registry.rollback_deferred(job_id)
                            queued_event = None
                            job = None
                            job_id = None
                            entry = None
                            outcome = AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE
                        else:
                            outcome = AdmissionOutcome.ACCEPTED
            try:
                snapshot = self._admission_snapshot_locked(
                    outcome, provider_label, worker
                )
            except Exception:
                if q is not None and job_id is not None:
                    q.remove_prompt(job_id.value)
                    self._registry.rollback_deferred(job_id)
                raise
            if outcome is AdmissionOutcome.ACCEPTED:
                self._pending_admission_publications += 1

        # ---- OUTSIDE the admission lock: settle, emit, return/raise --------- #
        for entry in expired:
            self._expire_queued_prompt(entry, reason="queue-wait-expired-admission")

        if outcome is AdmissionOutcome.ACCEPTED:
            assert q is not None and job_id is not None and job is not None
            publish_error = False
            publication_state = self._admission_publication_state
            publication_state.active = True
            publication_state.shutdown_requested = False
            try:
                self._emit_queue_event_from_admission(snapshot, job_id=job_id.value)
                self._registry.emit(queued_event)  # QUEUED lifecycle event, no lock held
            finally:
                try:
                    deferred_shutdown = bool(publication_state.shutdown_requested)
                    if not deferred_shutdown and not q.publish_prompt(job_id.value):
                        publish_error = True
                except Exception:  # noqa: BLE001 - never orphan an accepted handle
                    publish_error = True
                finally:
                    publication_state.active = False
                    with self._admission_cond:
                        self._pending_admission_publications -= 1
                        self._admission_cond.notify_all()
            if deferred_shutdown:
                # A synchronous observer requested shutdown while admission
                # events were being published on this same thread. Defer the
                # drain until both events exist, then settle the still-
                # unclaimable entry before returning its handle.
                self.shutdown()
            if publish_error:
                q.remove_prompt(job_id.value)
                self._registry.mark_abandoned(
                    job_id,
                    reason="queue-publication-failed",
                    physical_settled=True,
                )
                with job.result_lock:
                    job.caller_cause = TerminalCause.FAILED
                    job.error = ProviderError("Provider queue publication failed.")
                self._registry.mark_physical_settled(
                    job_id, reason="queue-publication-failed"
                )
                job.physical_done.set()
                job.done.set()
            return BrowserJobHandle(
                self, job_id=job_id, provider=provider, job=job, default_timeout=timeout
            )

        self._emit_queue_event_from_admission(snapshot)
        if outcome is AdmissionOutcome.REJECTED_MANAGER_SHUTTING_DOWN:
            # Preserve the Phase 5A/5B shutdown contract: a settled handle whose
            # result() deterministically raises RuntimeError, never a raised
            # rejection (existing callers and regression tests depend on this).
            return self._settled_shutdown_handle(provider, correlation, timeout)

        raise BrowserQueueRejectedError(snapshot)

    # ------------------------------------------------------------------ #
    # Admission / queue accounting helpers (Phase 5C)
    # ------------------------------------------------------------------ #
    def _global_prompt_depth_locked(self) -> int:
        """Sum queued-prompt depth across every provider (derived, never a
        drifting counter). Caller holds ``_admission_lock``."""
        with self._workers_lock:
            workers = list(self._workers.values())
        return sum(w.jobs.prompt_depth() for w in workers)

    def _global_prompt_depth(self) -> int:
        with self._admission_lock:
            return self._global_prompt_depth_locked()

    def _collect_expired_prompts_locked(self, now_mono: float) -> list[QueueEntry]:
        """Bounded global sweep used only on prompt admission."""
        with self._workers_lock:
            workers = list(self._workers.values())
        expired: list[QueueEntry] = []
        for worker in workers:
            expired.extend(worker.jobs.collect_expired(now_mono))
        return expired

    def _peek_worker(self, key: str) -> Optional[_Worker]:
        with self._workers_lock:
            return self._workers.get(key)

    def _admission_snapshot_locked(
        self,
        outcome: AdmissionOutcome,
        provider: str,
        worker: Optional[_Worker],
    ) -> AdmissionSnapshot:
        """Build a content-free admission snapshot. Caller holds ``_admission_lock``."""
        w = worker if worker is not None else self._peek_worker(provider)
        provider_depth = w.jobs.prompt_depth() if w is not None else 0
        global_depth = self._global_prompt_depth_locked()
        return make_admission_snapshot(
            outcome,
            provider,
            provider_queued_depth=provider_depth,
            provider_capacity=self._queue_policy.max_queued_prompts_per_provider,
            global_queued_depth=global_depth,
            global_capacity=self._queue_policy.max_total_queued_prompts,
            policy=self._queue_policy,
        )

    def _settled_shutdown_handle(
        self, provider: str, correlation: Optional[str], timeout: float
    ) -> "BrowserJobHandle":
        """Return a pre-settled handle for a submission that lost to shutdown.

        Matches the corrected Phase 5A/5B behavior: a ``QUEUED`` record is created
        and immediately abandoned+settled (all outside the admission lock), so the
        handle never hangs and ``result()`` raises ``RuntimeError``.
        """
        job_id = self._registry.register(provider, correlation=correlation)
        job = _Job(fn=lambda: None, job_id=job_id.value, provider=provider)
        self._registry.mark_abandoned(job_id, reason="manager-shutdown")
        self._registry.mark_physical_settled(job_id, reason="manager-shutdown")
        with job.result_lock:
            job.caller_cause = TerminalCause.SHUTDOWN
            job.error = RuntimeError("BrowserManager is shut down.")
        job.done.set()
        job.physical_done.set()
        return BrowserJobHandle(
            self, job_id=job_id, provider=provider, job=job, default_timeout=timeout
        )

    def _remove_queued_entry(self, provider: str, job_id: str) -> bool:
        """Physically remove a still-queued prompt (queued cancel/timeout).

        Frees its logical slot immediately and prevents tombstone buildup behind a
        blocked running job. Returns True if an entry was removed here (i.e. the
        worker had not already claimed it). Emits a content-free removal event.
        """
        worker = self._peek_worker(provider)
        if worker is None:
            return False
        entry = worker.jobs.remove_prompt(job_id)
        if entry is None:
            return False
        self._emit_queue_event(
            QueueEventKind.QUEUED_JOB_REMOVED,
            provider,
            job_id=job_id,
            prompt_depth=worker.jobs.prompt_depth(),
            control_depth=worker.jobs.control_depth(),
            global_depth=self._global_prompt_depth(),
            reason="queued-removed",
        )
        return True

    def _expire_queued_prompt(self, entry: QueueEntry, *, reason: str) -> None:
        """Settle a queued prompt whose bounded queue wait expired.

        Never submits the prompt to the provider. Preserves the existing compatible
        timeout behavior and the original ``TIMED_OUT`` terminal cause, releases the
        logical slot (the entry is already removed from physical storage) and emits
        a safe expiration event. Idempotent against a caller who already settled the
        same job.
        """
        job = entry.job
        jid = job.job_id
        if jid is None:
            job.done.set()
            job.physical_done.set()
            return
        result = self._registry.mark_timed_out(jid, reason=reason)
        if result.transitioned:
            with job.result_lock:
                if job.caller_cause is None:
                    job.caller_cause = TerminalCause.TIMED_OUT
        self._registry.mark_physical_settled(jid, reason=reason)
        job.physical_done.set()
        job.done.set()
        if not result.transitioned:
            return
        worker = self._peek_worker(job.provider) if job.provider else None
        self._emit_queue_event(
            QueueEventKind.QUEUE_WAIT_EXPIRED,
            job.provider or "unknown",
            job_id=jid,
            outcome=AdmissionOutcome.EXPIRED_BEFORE_START,
            prompt_depth=worker.jobs.prompt_depth() if worker else 0,
            global_depth=self._global_prompt_depth(),
            reason=reason,
        )

    def _quarantine_provider(
        self, provider: str, *, job_id: Optional[str] = None, reason: str = "quarantine"
    ) -> None:
        """Mark ``provider`` unsafe for new prompt admission (Phase 5C).

        Additive to Phase 5B: the physical operation is NOT force-settled and the
        tab is NOT restarted (no recovery). New prompt admissions are rejected
        deterministically until explicit teardown clears the quarantine.
        """
        with self._admission_lock:
            newly = provider not in self._quarantined
            self._quarantined.add(provider)
        if newly:
            worker = self._peek_worker(provider)
            self._emit_queue_event(
                QueueEventKind.PROVIDER_QUARANTINED,
                provider,
                job_id=job_id,
                prompt_depth=worker.jobs.prompt_depth() if worker else 0,
                global_depth=self._global_prompt_depth(),
                reason=reason,
            )

    def _clear_quarantine(self, provider: str) -> None:
        """Clear a provider's quarantine on explicit teardown/removal."""
        with self._admission_lock:
            self._quarantined.discard(provider)

    def is_quarantined(self, provider: str) -> bool:
        with self._admission_lock:
            return provider in self._quarantined

    # ------------------------------------------------------------------ #
    # Content-free queue observability events (Phase 5C)
    # ------------------------------------------------------------------ #
    def _emit_queue_event(
        self,
        kind: QueueEventKind,
        provider: str,
        *,
        job_id: Optional[str] = None,
        outcome: Optional[AdmissionOutcome] = None,
        prompt_depth: int = 0,
        control_depth: int = 0,
        global_depth: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        """Emit ONE content-free queue event OUTSIDE every lock; isolate failures."""
        sink = self._queue_event_sink
        if sink is None:
            return
        limit = self._queue_policy.max_admission_diagnostic_length
        safe_reason = None
        if reason is not None:
            text = str(reason).replace("\n", " ").replace("\r", " ").strip()
            safe_reason = (text[: limit - 1].rstrip() + "…") if len(text) > limit else (text or None)
        try:
            event = QueueEvent(
                kind=kind.value,
                provider=provider,
                timestamp=time.time(),
                job_id=job_id,
                outcome=outcome.value if outcome is not None else None,
                prompt_depth=max(0, prompt_depth),
                control_depth=max(0, control_depth),
                global_depth=max(0, global_depth),
                reason=safe_reason,
            )
        except Exception:  # noqa: BLE001 - event construction must never break us
            return
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - an observer must never break admission
            pass

    def _emit_queue_event_from_admission(
        self, snapshot: AdmissionSnapshot, *, job_id: Optional[str] = None
    ) -> None:
        kind = (
            QueueEventKind.ADMISSION_ACCEPTED
            if snapshot.accepted
            else QueueEventKind.ADMISSION_REJECTED
        )
        self._emit_queue_event(
            kind,
            snapshot.provider,
            job_id=job_id,
            outcome=AdmissionOutcome(snapshot.outcome),
            prompt_depth=snapshot.provider_queued_depth,
            global_depth=snapshot.global_queued_depth,
            reason=snapshot.reason,
        )

    # ------------------------------------------------------------------ #
    # Queue snapshots (read-only, content-free)
    # ------------------------------------------------------------------ #
    def queue_snapshot(self, provider: str) -> Optional[QueueSnapshot]:
        worker = self._peek_worker(provider)
        if worker is None:
            return None
        with self._admission_lock:
            global_depth = self._global_prompt_depth_locked()
            quarantined = provider in self._quarantined
        return worker.jobs.snapshot(
            global_queued_depth=global_depth,
            quarantined=quarantined,
            running_job_id=self._registry.active_job(provider),
        )

    def queue_snapshots(self) -> list[QueueSnapshot]:
        with self._workers_lock:
            providers = list(self._workers.keys())[
                : self._queue_policy.max_queue_snapshot_entries
            ]
        return [s for p in providers if (s := self.queue_snapshot(p)) is not None]

    def send_prompt(
        self,
        provider: str,
        prompt: str,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Drive one provider's chat UI with ``prompt`` and scrape the reply.

        Backward-compatible wrapper over :meth:`submit_prompt`: submit a job,
        block for its result, and return/raise exactly what callers relied on
        before Phase 5A (``str`` on success; ``TimeoutError`` on deadline;
        ``CancelledError`` on Stop; ``ProviderError`` for provider failures).
        """
        return self.submit_prompt(provider, prompt, should_cancel).result()

    # ------------------------------------------------------------------ #
    # Browser-job observability (read-only views into this manager's registry)
    # ------------------------------------------------------------------ #
    def browser_job(self, job_id: str) -> Optional[BrowserJobSnapshot]:
        return self._registry.snapshot(job_id)

    def browser_job_snapshots(self) -> list[BrowserJobSnapshot]:
        return self._registry.snapshots()

    def active_browser_job(self, provider: str) -> Optional[str]:
        return self._registry.active_job(provider)

    def active_providers(self) -> list[str]:
        with self._sessions_lock:
            return [k for k, s in self._sessions.items() if s.logged_in]

    def shutdown(self) -> None:
        publication_state = self._admission_publication_state
        if bool(getattr(publication_state, "active", False)):
            # Waiting here would self-deadlock: this thread owns the publication
            # whose completion shutdown normally waits for. Close admission now;
            # the publication finalizer performs the physical drain immediately
            # after both events have been emitted.
            with self._admission_cond:
                self._shutdown = True
            publication_state.shutdown_requested = True
            return
        # Flip the shutdown flag under the admission lock FIRST, so no in-flight
        # prompt admission can enqueue after this point — its whole decision runs
        # under this same lock, which is why a queue is provably open (never yet
        # closed) inside _admit_prompt. New admissions after this observe shutdown.
        with self._admission_cond:
            self._shutdown = True
            while self._pending_admission_publications:
                self._admission_cond.wait()
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            # Flag the loop to abandon (not execute) any still-queued jobs, then
            # close the queue so a blocked worker wakes with the None sentinel.
            w.stopping.set()
            with w.current_lock:
                current = w.current_job
            if current is not None:
                self._abandon_running_on_shutdown(current)
            for entry in w.jobs.drain():
                self._abandon_on_shutdown(entry.job)
            w.jobs.close()
            self._emit_queue_event(
                QueueEventKind.PROVIDER_QUEUE_CLOSED, w.key, reason="manager-shutdown"
            )
        for w in workers:
            w.thread.join(timeout=15)

    def _abandon_running_on_shutdown(self, job: _Job) -> None:
        """Release a running prompt's callers and request worker-thread unwind."""
        if job.job_id is None:
            return
        state = self._registry.state(job.job_id)
        if state is JobState.RUNNING:
            transitioned = self._registry.mark_abandoned(
                job.job_id, reason="manager-shutdown-running"
            ).transitioned
            if transitioned:
                with job.result_lock:
                    job.caller_cause = TerminalCause.SHUTDOWN
                    job.error = RuntimeError(
                        "BrowserManager shut down while the job was running."
                    )
                if job.interruption is not None:
                    job.interruption.request(
                        InterruptionReason.CANCELLED,
                        "manager-shutdown",
                        require_physical=True,
                    )
                job.done.set()

    # ------------------------------------------------------------------ #
    # Resilience helpers — these ONLY run on the owning provider's thread
    # ------------------------------------------------------------------ #
    @staticmethod
    def _session_alive(sess: Optional["_Session"]) -> bool:
        """True only if the session's window is still open and usable."""
        if sess is None or sess.context is None or sess.page is None:
            return False
        try:
            return not sess.page.is_closed()
        except Exception:
            # Touching a torn-down page/context raises — treat as dead.
            return False

    @staticmethod
    def _is_closed_error(exc: BaseException) -> bool:
        """Heuristically detect Playwright 'target/page/context closed' errors.

        Done by name/message (not isinstance) so we don't have to import
        Playwright's private error classes, and it survives version changes.
        """
        name = type(exc).__name__
        msg = str(exc).lower()
        return (
            name == "TargetClosedError"
            or "has been closed" in msg
            or "target page, context or browser has been closed" in msg
            or "target closed" in msg
            or "browser has been closed" in msg
        )

    def _ensure_logged_in(
        self, adapter: ProviderAdapter, sess: _Session, provider: str
    ) -> None:
        try:
            sess.logged_in = adapter.is_logged_in(sess.page)
        except Exception:
            sess.logged_in = False
        if not sess.logged_in:
            raise ProviderError(
                f"Session for {provider!r} is not logged in (the window may "
                "have been closed). Re-add it in 'My Active Models' and sign in."
            )

    # ------------------------------------------------------------------ #
    # Implementations — these ONLY run on the owning provider's thread
    # ------------------------------------------------------------------ #
    def _set_launch_error(self, key: str, error: Optional[str]) -> None:
        """Record (or clear) why ``key``'s login window failed to open."""
        with self._launch_errors_lock:
            if error:
                # Bounded: this string is surfaced to the UI, never a payload.
                self._launch_errors[key] = error[:300]
            else:
                self._launch_errors.pop(key, None)

    def launch_error(self, key: str) -> str:
        """The last window-launch failure for ``key`` (empty when healthy)."""
        with self._launch_errors_lock:
            return self._launch_errors.get(key, "")

    def _start_login_one(self, key: str) -> dict[str, Any]:
        adapter = get_adapter(key)
        try:
            sess = self._ensure_session(adapter)
            # Navigate to the login page; relaunch once if the window died.
            try:
                adapter.open_login(sess.page)
            except Exception as exc:  # noqa: BLE001
                if self._is_closed_error(exc):
                    sess = self._ensure_session(adapter, force_new=True)
                    adapter.open_login(sess.page)
                else:
                    raise
            # If the persisted profile is still authenticated, detect it now.
            try:
                sess.logged_in = adapter.is_logged_in(sess.page)
            except Exception:
                sess.logged_in = False
            self._set_launch_error(key, None)
            return {"opened": True, "already_logged_in": sess.logged_in}
        except Exception as exc:  # noqa: BLE001
            # Keep the reason: the async path discards this return value, and a
            # lost launch failure is exactly what makes a window "never pop out"
            # with nothing to show for it.
            reason = str(exc) or exc.__class__.__name__
            self._set_launch_error(key, reason)
            return {"opened": False, "error": reason}

    def _ensure_session(
        self, adapter: ProviderAdapter, force_new: bool = False
    ) -> _Session:
        """Return a live session for ``adapter``, (re)launching if necessary.

        If the existing window was manually closed or crashed, its stale
        context is discarded and a fresh persistent context is launched from
        the SAME on-disk profile — so saved cookies/logins carry over and the
        recovery is usually invisible to the user.
        """
        with self._sessions_lock:
            sess = self._sessions.get(adapter.key)

        if sess is not None and not force_new and self._session_alive(sess):
            return sess

        # Stale, closed, or forced: tear down any old context before relaunch.
        if sess is not None and sess.context is not None:
            try:
                sess.context.close()
            except Exception:
                pass

        profile = self.profiles_dir / adapter.key
        profile.mkdir(parents=True, exist_ok=True)

        pw = self._ensure_playwright(self._current_worker(adapter.key))
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            headless=self.headless,
            # Window geometry comes from ONE policy (see the helper's
            # docstring); never add sizing/positioning arguments here.
            **provider_window_launch_kwargs(self.headless),
        )
        page = context.pages[0] if context.pages else context.new_page()
        new_sess = _Session(adapter=adapter, context=context, page=page)
        with self._sessions_lock:
            self._sessions[adapter.key] = new_sess
        return new_sess

    def _status_one(self, key: str) -> dict[str, Any]:
        with self._sessions_lock:
            sess = self._sessions.get(key)
        error = self.launch_error(key)
        if sess is None:
            return {"name": key, "open": False, "logged_in": False, "error": error}
        return {
            "name": sess.adapter.name,
            "open": self._session_alive(sess),
            "logged_in": sess.logged_in,
            "error": error,
        }

    def _confirm_one(self, key: str) -> bool:
        with self._sessions_lock:
            sess = self._sessions.get(key)
        if sess is None:
            return False
        try:
            sess.logged_in = sess.adapter.is_logged_in(sess.page)
        except Exception:
            sess.logged_in = False
        return sess.logged_in

    def _confirm_provider(self, provider: str) -> dict[str, Any]:
        with self._sessions_lock:
            sess = self._sessions.get(provider)
        if sess is None:
            raise ProviderError(
                f"No open browser window for {provider!r}. Start login first."
            )
        # If the user closed the window, relaunch from the saved profile so we
        # can still verify (and persist) the login.
        if not self._session_alive(sess):
            sess = self._ensure_session(get_adapter(provider), force_new=True)
        try:
            sess.logged_in = sess.adapter.is_logged_in(sess.page)
        except Exception:
            sess.logged_in = False
        # The persistent context auto-saves cookies/storage to the on-disk
        # profile, so confirming is enough to keep the session for reuse.
        return {"ready": sess.logged_in, "name": sess.adapter.name}

    def _remove_provider(self, provider: str) -> dict[str, Any]:
        with self._sessions_lock:
            sess = self._sessions.pop(provider, None)
        if sess is not None and sess.context is not None:
            try:
                sess.context.close()  # cookies remain on disk in the profile
            except Exception:
                pass
        # Explicit teardown clears any Phase 5C quarantine: the user removed the
        # provider (and will re-add/log in to get a fresh tab), so the unsafe-
        # reuse condition no longer applies to a future session.
        self._clear_quarantine(provider)
        return {"removed": True}

    def _send_prompt(
        self,
        provider: str,
        prompt: str,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        adapter = get_adapter(provider)
        with self._sessions_lock:
            sess = self._sessions.get(provider)

        # Never added (or already removed) -> nothing to recover.
        if sess is None:
            raise ProviderError(
                f"No active browser session for {provider!r}. "
                "Add it in 'My Active Models' and log in first."
            )

        # Window manually closed / crashed -> relaunch from the saved profile.
        if not self._session_alive(sess):
            sess = self._ensure_session(adapter, force_new=True)

        self._ensure_logged_in(adapter, sess, provider)

        try:
            return adapter.send(sess.page, prompt, should_cancel)
        except CancelledError:
            # User pressed Stop: surface it cleanly, never retry/relaunch.
            raise
        except Exception as exc:  # noqa: BLE001
            # If the window died mid-interaction, relaunch once and retry.
            if self._is_closed_error(exc):
                sess = self._ensure_session(adapter, force_new=True)
                self._ensure_logged_in(adapter, sess, provider)
                try:
                    return adapter.send(sess.page, prompt, should_cancel)
                except CancelledError:
                    raise
                except Exception as exc2:  # noqa: BLE001
                    raise ProviderError(
                        f"[{adapter.name}] Send failed even after re-launching "
                        f"the window: {exc2}"
                    ) from exc2
            # Already-clear adapter errors pass through; wrap anything else so
            # the WebAutomationLLMClient's retry logic gets a clean ProviderError
            # instead of a raw Playwright exception crashing the worker thread.
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"[{adapter.name}] Send failed: {exc}") from exc

    def _teardown_worker(self, worker: _Worker) -> None:
        """Close this provider's context and stop its Playwright — on its own
        thread, as required by the sync API."""
        retained_owner = self._registry.active_job(worker.key)
        with self._sessions_lock:
            sess = self._sessions.pop(worker.key, None)
        if sess is not None and sess.context is not None:
            try:
                sess.context.close()
            except Exception:
                pass
        if worker.playwright is not None:
            try:
                worker.playwright.stop()
            except Exception:
                pass
            worker.playwright = None
        # Existing shutdown teardown makes a quarantined tab physically unable
        # to continue generation. Release retained ownership only afterward.
        if retained_owner is not None:
            self._registry.mark_physical_settled(
                retained_owner, reason="worker-shutdown-settled"
            )
        # The tab is gone; a future session starts fresh, so drop any quarantine.
        self._clear_quarantine(worker.key)


class BrowserJobHandle:
    """Additive result handle for one submitted browser job (Phase 5A/5B).

    A handle is the caller's view of a single physical browser operation. It
    exposes the job's identity and lifecycle snapshot, waits for the result via
    the existing per-job result mechanism, and records caller cancellation /
    timeout through ONE authoritative path (a compare-and-transition on the
    registry). The caller thread NEVER touches Playwright: for a physically
    running job it records an interruption request that the owning provider
    worker observes and acts on (Phase 5B), so generation is actively stopped
    without the caller ever driving the browser. It exposes a read-only
    :meth:`interruption_snapshot` view of that request.
    """

    def __init__(
        self,
        manager: "BrowserManager",
        *,
        job_id: BrowserJobId,
        provider: str,
        job: _Job,
        default_timeout: float,
    ) -> None:
        self._mgr = manager
        self._job_id = job_id
        self._provider = provider
        self._job = job
        self._default_timeout = default_timeout

    # --- identity / observation ------------------------------------------- #
    @property
    def job_id(self) -> str:
        return self._job_id.value

    @property
    def provider(self) -> str:
        return self._provider

    def snapshot(self) -> Optional[BrowserJobSnapshot]:
        return self._mgr._registry.snapshot(self._job_id)

    def interruption_snapshot(self) -> Optional[InterruptionSnapshot]:
        """A read-only, content-free view of this job's interruption state.

        Returns ``None`` for legacy jobs that carry no interruption record. The
        mutable request internals are never exposed.
        """
        req = self._job.interruption
        return req.snapshot() if req is not None else None

    def state(self) -> Optional[JobState]:
        return self._mgr._registry.state(self._job_id)

    def timed_out(self) -> bool:
        snap = self.snapshot()
        if snap is not None:
            return snap.terminal_cause == TerminalCause.TIMED_OUT.value
        with self._job.result_lock:
            return self._job.caller_cause is TerminalCause.TIMED_OUT

    def cancelled(self) -> bool:
        snap = self.snapshot()
        if snap is not None:
            return snap.terminal_cause == TerminalCause.CANCELLED.value
        with self._job.result_lock:
            return self._job.caller_cause is TerminalCause.CANCELLED

    # --- caller-side lifecycle intent ------------------------------------- #
    def _record_interruption(self, reason: "InterruptionReason") -> None:
        """Record a physical-interruption request for a RUNNING job.

        Caller-thread only; performs no Playwright work. Records the request on
        the job's authoritative :class:`InterruptionRequest` so the owning worker
        actively stops generation. Idempotent; the first reason wins.
        """
        req = self._job.interruption
        if req is not None:
            req.request(reason, "handle", require_physical=True)

    def mark_caller_timeout(self, *, reason: str = "caller-deadline") -> bool:
        """Record that the caller's deadline expired — one authoritative path.

        Returns True only if this call won the completion-vs-timeout race (i.e.
        the job was not already terminal). For a QUEUED job it settles immediately
        (no prompt was ever submitted); for a RUNNING job it records a
        ``CALLER_TIMEOUT`` interruption request so the owning worker actively
        stops generation. The caller thread never touches Playwright, and the
        ``TIMED_OUT`` terminal cause is preserved.
        """
        transition = self._mgr._registry.mark_timed_out(
            self._job_id, reason=reason
        )
        if transition.transitioned:
            with self._job.result_lock:
                self._job.caller_cause = TerminalCause.TIMED_OUT
            if transition.snapshot is not None and transition.snapshot.started_at is None:
                # Queued (never started): free its logical slot immediately by
                # physically removing the entry, then settle. This is what makes a
                # queued timeout release capacity without waiting for the worker to
                # reach the dead entry (Phase 5C), and prevents tombstone buildup.
                self._mgr._remove_queued_entry(self._provider, self._job_id.value)
                self._mgr._registry.mark_physical_settled(
                    self._job_id, reason="queued-timeout"
                )
                self._job.physical_done.set()
            else:
                # Physically running: ask the owning worker to stop generation.
                self._record_interruption(InterruptionReason.CALLER_TIMEOUT)
            self._job.done.set()
        return transition.transitioned

    def request_cancel(self, *, reason: str = "caller-cancel") -> bool:
        """Record cancellation intent and, for a running job, actively stop it.

        Moves the lifecycle record to ``CANCELLED`` if the job is not yet terminal
        (so a still-queued job is skipped by the worker and a running job's late
        result is discarded), wakes any caller waiters immediately, and — for a
        physically RUNNING job — records a ``CANCELLED`` interruption request that
        the owning provider worker observes and acts on. It never performs a
        Playwright operation itself, and it is idempotent. Returns True if the
        intent was recorded.
        """
        transition = self._mgr._registry.mark_cancelled(
            self._job_id, reason=reason
        )
        if transition.transitioned:
            with self._job.result_lock:
                self._job.caller_cause = TerminalCause.CANCELLED
            if transition.snapshot is not None and transition.snapshot.started_at is None:
                # Queued (never started): free its logical slot immediately by
                # physically removing the entry, then settle (Phase 5C). A new
                # submission can reuse the released slot without waiting for the
                # worker to dequeue the cancelled job, and no tombstone lingers.
                self._mgr._remove_queued_entry(self._provider, self._job_id.value)
                self._mgr._registry.mark_physical_settled(
                    self._job_id, reason="queued-cancel"
                )
                self._job.physical_done.set()
            else:
                # Physically running: ask the owning worker to stop generation.
                self._record_interruption(InterruptionReason.CANCELLED)
            self._job.done.set()
        return transition.transitioned

    # --- result delivery -------------------------------------------------- #
    def result(self, timeout: Optional[float] = None) -> str:
        """Block for the job's result, or raise a compatible exception.

        Semantics match the pre-5A ``send_prompt`` contract:

        * returns the scraped ``str`` on success,
        * raises ``TimeoutError`` if the caller's deadline expires first (or the
          job otherwise settled as timed out / abandoned),
        * raises ``CancelledError`` on a Stop,
        * raises the provider exception (e.g. ``ProviderError``) on failure.

        On a deadline expiry the caller stops waiting immediately; the underlying
        browser operation may still be running and its late result is discarded.
        """
        if timeout is None:
            timeout = self._default_timeout

        if not self._job.done.wait(timeout):
            # Deadline expired before the worker finished. Try to win the race.
            if self.mark_caller_timeout():
                raise TimeoutError("Browser job timed out.")
            # Lost the race: the worker settled the job concurrently. Its data is
            # published right after its transition; wait (briefly) to read it.
            self._job.done.wait()

        return self._resolve_delivered()

    def _resolve_delivered(self) -> str:
        snap = self.snapshot()
        state = JobState(snap.state) if snap is not None else None
        with self._job.result_lock:
            internal_cause = self._job.caller_cause
            result = self._job.result
            error = self._job.error
        cause = (
            snap.terminal_cause
            if snap is not None and snap.terminal_cause is not None
            else (internal_cause.value if internal_cause is not None else None)
        )
        # Trusted caller causes always outrank any late provider settlement.
        if cause == TerminalCause.CANCELLED.value:
            if isinstance(error, CancelledError):
                raise error
            raise CancelledError("Browser job was cancelled.")
        if cause == TerminalCause.TIMED_OUT.value:
            raise TimeoutError("Browser job timed out.")
        if state is JobState.COMPLETED or cause == TerminalCause.COMPLETED.value:
            self._mgr._registry.mark_result_delivered(self._job_id)
            return result
        if error is not None:
            raise error
        if cause == TerminalCause.SHUTDOWN.value:
            raise RuntimeError("BrowserManager was shut down before the job ran.")
        # ABANDONED / anything else with no delivered result.
        raise TimeoutError("Browser job timed out.")
