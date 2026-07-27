"""Live progress plumbing for the SSE "Live Execution Log".

A run is started in a background thread and emits structured progress events
(dicts) via :meth:`RunChannel.emit`. The SSE endpoint consumes them with
:meth:`RunChannel.wait_batch`, which buffers everything so NO event is missed
even if the browser's EventSource connects a moment after the run starts.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Optional


DEFAULT_COMPLETED_RUN_RETENTION_S = 30.0


class RunChannel:
    """A buffered, thread-safe event stream for one orchestration run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        self._done = False

    def emit(self, event: dict[str, Any]) -> None:
        """Append an event (called from any worker thread)."""
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def close(self) -> None:
        """Mark the run finished; wakes any waiting SSE consumer."""
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def wait_batch(self, idx: int, timeout: float) -> tuple[list[dict[str, Any]], bool]:
        """Return ``(events_since_idx, done)``, blocking up to ``timeout`` sec.

        ``idx`` is the number of events the caller has already consumed. The
        returned ``done`` is True only once the run is closed AND the caller has
        been handed every buffered event.
        """
        with self._cond:
            if idx >= len(self._events) and not self._done:
                self._cond.wait(timeout)
            batch = self._events[idx:]
            done = self._done and (idx + len(batch)) >= len(self._events)
        return batch, done


class RunRegistry:
    """Tracks run channels and briefly retains completed SSE streams."""

    def __init__(
        self, completed_run_retention_s: float = DEFAULT_COMPLETED_RUN_RETENTION_S
    ) -> None:
        if completed_run_retention_s < 0:
            raise ValueError("completed_run_retention_s must be non-negative.")
        self._runs: dict[str, RunChannel] = {}
        self._lock = threading.Lock()
        self._completed_run_retention_s = completed_run_retention_s
        self._cleanup_timers: dict[str, threading.Timer] = {}

    def create(self) -> RunChannel:
        run_id = uuid.uuid4().hex[:12]
        channel = RunChannel(run_id)
        with self._lock:
            self._runs[run_id] = channel
        return channel

    def get(self, run_id: str) -> Optional[RunChannel]:
        with self._lock:
            return self._runs.get(run_id)

    def close(self, run_id: str) -> bool:
        """Close a run and retain it long enough for an SSE reconnect."""
        with self._lock:
            channel = self._runs.get(run_id)
            if channel is None:
                return False
            if run_id in self._cleanup_timers:
                return True

            channel.close()
            timer = threading.Timer(
                self._completed_run_retention_s,
                self.discard,
                args=(run_id,),
            )
            timer.daemon = True
            self._cleanup_timers[run_id] = timer

        timer.start()
        return True

    def discard(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
            timer = self._cleanup_timers.pop(run_id, None)
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def shutdown(self) -> None:
        """Release retained channels when the application is shutting down."""
        with self._lock:
            timers = list(self._cleanup_timers.values())
            self._cleanup_timers.clear()
            self._runs.clear()
        for timer in timers:
            timer.cancel()


# Module-level singleton used by the server.
RUNS = RunRegistry()
