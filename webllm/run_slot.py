"""Single-active-run admission control (production-hardening Phase 1).

DOZEN's production composition shares mutable execution state across runs
(one orchestrator, one browser pool, one client cancellation token). Until
run-scoped execution exists, AT MOST ONE workflow may be active per server
process — this module makes that invariant explicit and safe instead of
accidental.

``RunSlot`` is the single authoritative source of truth for:

* whether a run is active,
* which run is active,
* which cancellation token belongs to it,
* when the slot is released.

The internal lock guards only STATE TRANSITIONS (idle→active,
active→cancelling, active→idle); the workflow itself never executes under
the lock. Release is owner-checked: a stale or foreign run id can never
clear another run's slot, so a finishing run cannot disturb its successor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from dozen.cancellation import CancelToken


@dataclass(frozen=True)
class ActiveRun:
    """Immutable record of the run currently owning the slot."""

    run_id: str
    cancel: CancelToken
    started_at: float  # monotonic seconds — diagnostics only


class RunSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[ActiveRun] = None

    # ------------------------------------------------------------------ #
    # Transitions
    # ------------------------------------------------------------------ #
    def try_acquire(self, run_id: str, cancel: CancelToken) -> tuple[bool, Optional[str]]:
        """Idle→Active, atomically.

        Returns ``(True, run_id)`` when this run now owns the slot, or
        ``(False, active_run_id)`` when another run is already active.
        """
        with self._lock:
            if self._active is not None:
                return False, self._active.run_id
            self._active = ActiveRun(
                run_id=run_id, cancel=cancel, started_at=time.monotonic()
            )
            return True, run_id

    def release(self, run_id: str) -> bool:
        """Active→Idle, owner-checked.

        Only the run that owns the slot may release it; releasing with a
        stale or foreign id is a safe no-op (returns False). Callers invoke
        this from ``finally`` blocks so the slot can never leak on exceptions.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active = None
                return True
            return False

    def stop_active(self) -> dict:
        """Active→cancelling. Trips ONLY the current owner's token.

        Idempotent and safe against concurrent completion: the token is
        cancelled while the transition lock is held, so a run that finishes
        (and releases) concurrently can never have a *successor's* token
        tripped by this call — the token cancelled is always the one that
        belonged to the slot owner observed here.
        """
        with self._lock:
            active = self._active
            if active is None:
                return {"stopped": False, "detail": "No task is currently running."}
            active.cancel.cancel()  # CancelToken.cancel() is idempotent
            return {"stopped": True, "run_id": active.run_id}

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #
    def active_run_id(self) -> Optional[str]:
        with self._lock:
            return self._active.run_id if self._active else None

    def is_active(self) -> bool:
        return self.active_run_id() is not None


# Module-level singleton owned by the server composition layer. One slot per
# server process — the explicit, current-stage single-active-run invariant.
RUN_SLOT = RunSlot()
