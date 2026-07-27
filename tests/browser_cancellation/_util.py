"""Browser-free scaffolding for the Phase 5B active-interruption tests.

Named with a leading underscore so ``unittest discover`` never collects it.

``InterruptibleBrowserManager`` subclasses the real :class:`BrowserManager` and
overrides exactly two seams:

* ``_send_prompt`` — the single seam that would otherwise drive Playwright, and
* ``_perform_stop_action`` — the single seam that would otherwise resolve a page
  and click a real stop control.

Everything else (per-provider worker threads, the job queue, the run loop, the
registry, handles, the interruption request/observation, owner checks, the
exactly-once ``begin_action`` gate, event emission and shutdown) runs for real,
so these tests exercise the genuine concurrency machinery without a browser.

The stop-action seam records the calling thread id and call count so tests can
prove the stop action runs exactly once and on the provider worker thread.
"""

from __future__ import annotations

import tempfile
import threading
import time
from typing import Callable, Optional

from dozen.cancellation import CancelledError
from webllm.browser_cancellation import InterruptionActionResult
from webllm.browser_manager import BrowserManager, _Job


class Gate:
    """A releasable stand-in for one blocking physical ``send``.

    Blocks until ``release`` is set, then returns ``value`` or raises ``exc``. Its
    completion can be interleaved deterministically with a caller cancel/timeout.
    """

    def __init__(self, value: Optional[str] = None, exc: Optional[BaseException] = None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.value = value
        self.exc = exc
        self.worker_ident: Optional[int] = None

    def __call__(self, provider: str, prompt: str, should_cancel) -> str:
        self.worker_ident = threading.get_ident()
        self.started.set()
        self.release.wait(timeout=10)
        if self.exc is not None:
            raise self.exc
        return self.value if self.value is not None else f"reply:{prompt}"


class CooperativeGate:
    """A cooperative stand-in that polls ``should_cancel`` like a real adapter.

    Records the worker thread id, then loops checking the composite cancellation
    observation; the instant it trips, it raises :class:`CancelledError` — exactly
    as the real provider observation loop unwinds mid-generation.
    """

    def __init__(self, value: str = "reply", poll_s: float = 0.005):
        self.started = threading.Event()
        self.value = value
        self.poll_s = poll_s
        self.worker_ident: Optional[int] = None
        self.saw_cancel = False

    def __call__(self, provider: str, prompt: str, should_cancel) -> str:
        self.worker_ident = threading.get_ident()
        self.started.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            if should_cancel is not None and should_cancel():
                self.saw_cancel = True
                raise CancelledError("cooperative stop")
            time.sleep(self.poll_s)
        return self.value


class InterruptibleBrowserManager(BrowserManager):
    """A BrowserManager with test-controlled send and stop-action seams."""

    def __init__(self, *, stop_result: Optional[InterruptionActionResult] = None,
                 stop_exc: Optional[BaseException] = None, **kwargs):
        tmp = tempfile.mkdtemp(prefix="bm_cancel_")
        super().__init__(profiles_dir=tmp, headless=True, **kwargs)
        self._tmp = tmp
        self.gates: dict[str, Gate | CooperativeGate] = {}
        self.handlers: dict[str, Callable[[str, str, object], str]] = {}
        self.sent: list[tuple[str, str]] = []
        self._sent_lock = threading.Lock()
        # Stop-action instrumentation.
        self._stop_result = (
            stop_result if stop_result is not None
            else InterruptionActionResult.stopped(quiescent=True)
        )
        self._stop_exc = stop_exc
        self.stop_lock = threading.Lock()
        self.stop_calls = 0
        self.stop_idents: list[int] = []
        self.stop_jobs: list[str] = []
        # Optional callback invoked (on the worker thread) inside the stop seam.
        self.on_stop: Optional[Callable[[_Job], None]] = None

    # --- send seam -------------------------------------------------------- #
    def _send_prompt(self, provider: str, prompt: str, should_cancel=None) -> str:
        with self._sent_lock:
            self.sent.append((provider, prompt))
        gate = self.gates.get(prompt)
        if gate is not None:
            return gate(provider, prompt, should_cancel)
        handler = self.handlers.get(prompt)
        if handler is not None:
            return handler(provider, prompt, should_cancel)
        return f"reply:{prompt}"

    # --- stop-action seam (runs ONLY on the provider worker thread) ------- #
    def _perform_stop_action(self, job: _Job) -> InterruptionActionResult:
        with self.stop_lock:
            self.stop_calls += 1
            self.stop_idents.append(threading.get_ident())
            if job.job_id is not None:
                self.stop_jobs.append(job.job_id)
        if self.on_stop is not None:
            self.on_stop(job)
        if self._stop_exc is not None:
            raise self._stop_exc
        return self._stop_result

    def prompts_sent(self) -> list[str]:
        with self._sent_lock:
            return [p for _, p in self.sent]


def poll_state(mgr, job_id, want, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mgr._registry.state(job_id) is want:
            return True
        time.sleep(0.005)
    return False


def poll(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False
