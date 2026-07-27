"""Shared, browser-free scaffolding for the Phase 5C bounded-queue tests.

Named with a leading underscore so ``unittest discover`` never collects it.

Reuses the real per-provider worker threads, run loop, registry, handles and
shutdown from :class:`BrowserManager`; only the single Playwright seam
(``_send_prompt``) is replaced by a controllable, browser-free stand-in — so
these tests exercise the genuine bounded-queue and admission machinery without
ever launching a browser.
"""

from __future__ import annotations

import tempfile
import threading
from typing import Callable, Optional

from webllm.browser_manager import BrowserManager
from webllm.browser_queue import QueuePolicy


class Gate:
    """A controllable stand-in for one physical browser ``send``.

    ``started`` fires when the worker enters the send; the call then blocks until
    ``release`` is set (bounded so a broken test can never hang the suite), after
    which it returns ``value`` or raises ``exc``.
    """

    def __init__(self, value: Optional[str] = None, exc: Optional[BaseException] = None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.value = value
        self.exc = exc

    def __call__(self, provider: str, prompt: str, should_cancel) -> str:
        self.started.set()
        self.release.wait(timeout=10)
        if self.exc is not None:
            raise self.exc
        return self.value if self.value is not None else f"reply:{prompt}"


class QueueTestManager(BrowserManager):
    """A BrowserManager with a fully test-controlled ``_send_prompt`` seam and an
    explicit queue policy, plus small helpers for filling and draining queues."""

    def __init__(self, *, queue_policy: Optional[QueuePolicy] = None, **kwargs):
        tmp = tempfile.mkdtemp(prefix="bm_queue_")
        super().__init__(
            profiles_dir=tmp, headless=True, queue_policy=queue_policy, **kwargs
        )
        self._tmp = tmp
        self.gates: dict[str, Gate] = {}
        self.handlers: dict[str, Callable[[str, str, object], str]] = {}
        self.sent: list[tuple[str, str]] = []
        self._sent_lock = threading.Lock()

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

    # --- helpers -------------------------------------------------------- #
    def prompts_sent(self) -> list[str]:
        with self._sent_lock:
            return [p for _, p in self.sent]

    def install_blocker(self, provider: str, name: str = "blocker") -> Gate:
        """Submit a job that occupies the provider's running slot until released.

        Returns the :class:`Gate`; the returned handle is intentionally dropped —
        the caller only needs the running slot held. ``started`` is awaited so the
        worker is provably busy (queue empty) before queued jobs are added.
        """
        gate = Gate(value=f"reply:{name}")
        self.gates[name] = gate
        self.submit_prompt(provider, name)
        assert gate.started.wait(2), "blocker never started"
        return gate

    def release_all(self) -> None:
        for gate in self.gates.values():
            gate.release.set()

    def shutdown(self) -> None:  # ensure blocked gates never stall teardown
        self.release_all()
        super().shutdown()
