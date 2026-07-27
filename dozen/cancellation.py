"""Cooperative cancellation for the orchestration pipeline.

This is the Python equivalent of a JS ``AbortController``: a single thread-safe
token that every long-running stage (planner, executor, workers, the browser
``send`` loop, MutationObserver waits) checks at safe points. Setting it from any
thread (e.g. the HTTP handler for the "Stop" button) makes the in-flight run
unwind quickly instead of blocking until the next slow web call finishes.

Usage
-----
    token = CancelToken()
    ...
    token.check()            # raises CancelledError if cancelled
    if token.cancelled: ...  # non-raising probe
    token.cancel()           # trip it (from any thread)
    token.reset()            # reuse for the next run
"""

from __future__ import annotations

import threading


class CancelledError(Exception):
    """Raised at a checkpoint when a run has been cancelled by the user."""


class CancelToken:
    """A thread-safe, resettable cancellation flag (an AbortController analogue)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Trip the token. Safe to call from any thread."""
        self._event.set()

    def reset(self) -> None:
        """Clear the token so it can be reused for the next run."""
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`CancelledError` if the token has been tripped."""
        if self._event.is_set():
            raise CancelledError("Orchestration was cancelled by the user.")


# A shared no-op token so callers can always assume a token exists and never
# have to branch on ``None``. It is never tripped.
class _NullToken(CancelToken):
    def cancel(self) -> None:  # pragma: no cover - intentionally inert
        pass


NULL_TOKEN: CancelToken = _NullToken()
