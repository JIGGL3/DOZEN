"""Shared, browser-free test scaffolding for the Phase 5A lifecycle tests.

Named with a leading underscore so ``unittest discover`` (which only collects
``test*.py``) never treats it as a test module.

``ControllableBrowserManager`` subclasses the real :class:`BrowserManager` and
overrides ONLY ``_send_prompt`` — the single seam that would otherwise drive
Playwright. Everything else (per-provider worker threads, the job queue, the run
loop, the registry, handles, shutdown) runs for real, so these tests exercise the
genuine concurrency machinery without ever launching a browser.
"""

from __future__ import annotations

import tempfile
import threading
from typing import Callable, Optional

from webllm.browser_manager import BrowserManager


class Gate:
    """A controllable stand-in for one physical browser ``send``.

    ``started`` fires when the worker enters the send; the call then blocks until
    ``release`` is set, after which it returns ``value`` or raises ``exc``. This
    lets a test deterministically interleave a caller timeout/cancel with the
    browser operation's completion.
    """

    def __init__(self, value: Optional[str] = None, exc: Optional[BaseException] = None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.value = value
        self.exc = exc
        self.cancel_seen = False

    def __call__(self, provider: str, prompt: str, should_cancel) -> str:
        self.started.set()
        # Wait to be released (bounded so a broken test cannot hang the suite).
        self.release.wait(timeout=10)
        if should_cancel is not None:
            try:
                self.cancel_seen = bool(should_cancel())
            except Exception:
                self.cancel_seen = False
        if self.exc is not None:
            raise self.exc
        return self.value if self.value is not None else f"reply:{prompt}"


class ControllableBrowserManager(BrowserManager):
    """A BrowserManager whose ``_send_prompt`` is fully test-controlled."""

    def __init__(self, **kwargs):
        tmp = tempfile.mkdtemp(prefix="bm_jobs_")
        super().__init__(profiles_dir=tmp, headless=True, **kwargs)
        self._tmp = tmp
        # Route by prompt text: a per-prompt Gate, or a per-prompt callable.
        self.gates: dict[str, Gate] = {}
        self.handlers: dict[str, Callable[[str, str, object], str]] = {}
        # Records every prompt actually sent to the (fake) provider.
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

    def prompts_sent(self) -> list[str]:
        with self._sent_lock:
            return [p for _, p in self.sent]
