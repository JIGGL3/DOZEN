"""Reliability observability events (Phase 2.1.3).

Typed publisher over the Phase 1.3 ``InProcessEventBus``. Observation only:
payloads carry ids, counters and enum values — never message content, never
prompts. Publishing is fail-open by the bus's own contract (a raising
subscriber cannot break the publisher), and every emit here is additionally
guarded so a broken bus cannot break recording.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..context.domain.enums import EventType
from ..context.manager.events import ConversationEvents, InProcessEventBus
from .models import ExecutionAttempt
from .types import AttemptStatus

_STATUS_EVENT = {
    AttemptStatus.FAILED: EventType.ATTEMPT_FAILED,
    AttemptStatus.CANCELLED: EventType.ATTEMPT_CANCELLED,
}


class ReliabilityEvents:
    """Emits attempt lifecycle events. One instance per recorder."""

    def __init__(self, bus: Optional[InProcessEventBus] = None) -> None:
        self.bus = bus or InProcessEventBus()
        # Reuse the context layer's publisher plumbing (ids + timestamps).
        self._publisher = ConversationEvents(self.bus)

    # ------------------------------------------------------------------ #
    def attempt_started(self, attempt: ExecutionAttempt) -> None:
        self._emit(EventType.ATTEMPT_STARTED, attempt)

    def attempt_finished(self, attempt: ExecutionAttempt) -> None:
        # The generic terminal event, always emitted…
        self._emit(EventType.ATTEMPT_FINISHED, attempt)
        # …plus the specific one for failed/cancelled outcomes.
        specific = _STATUS_EVENT.get(attempt.status)
        if specific is not None:
            self._emit(specific, attempt)

    def attempt_recorded(self, attempt: ExecutionAttempt) -> None:
        self._emit(EventType.ATTEMPT_RECORDED, attempt)

    def attempt_evicted(self, attempt: ExecutionAttempt) -> None:
        self._emit(EventType.ATTEMPT_EVICTED, attempt)

    # ---------------- provider health (Phase 2.2.2, observational) ------- #
    def provider_health_changed(self, provider: str, from_state: str,
                                to_state: str, score: float, reason: str) -> None:
        self._emit_health(EventType.PROVIDER_HEALTH_CHANGED, provider,
                          from_state, to_state, score, reason)

    def provider_recovered(self, provider: str, from_state: str,
                           to_state: str, score: float, reason: str) -> None:
        self._emit_health(EventType.PROVIDER_RECOVERED, provider,
                          from_state, to_state, score, reason)

    def provider_degraded(self, provider: str, from_state: str,
                          to_state: str, score: float, reason: str) -> None:
        self._emit_health(EventType.PROVIDER_DEGRADED, provider,
                          from_state, to_state, score, reason)

    def provider_quarantined(self, provider: str, from_state: str,
                             to_state: str, score: float, reason: str) -> None:
        self._emit_health(EventType.PROVIDER_QUARANTINED, provider,
                          from_state, to_state, score, reason)

    def health_snapshot(self, provider: str, from_state: str,
                        to_state: str, score: float, reason: str) -> None:
        self._emit_health(EventType.HEALTH_SNAPSHOT, provider,
                          from_state, to_state, score, reason)

    def _emit_health(self, event_type: EventType, provider: str,
                     from_state: str, to_state: str,
                     score: float, reason: str) -> None:
        try:
            self._publisher._emit(  # noqa: SLF001 - stable internal plumbing
                event_type, None,
                provider=provider, from_state=from_state, to_state=to_state,
                score=score, reason=reason,
            )
        except Exception:
            pass  # observation must never break health computation

    # ------------------------------------------------------------------ #
    def _emit(self, event_type: EventType, attempt: ExecutionAttempt) -> None:
        try:
            self._publisher._emit(  # noqa: SLF001 - same package family, stable plumbing
                event_type,
                attempt.conversation_id,
                attempt_id=attempt.attempt_id,
                provider=attempt.provider,
                stage=attempt.execution_stage.value,
                status=attempt.status.value,
                run_id=attempt.run_id,
                latency_ms=attempt.latency_ms,
            )
        except Exception:
            pass  # observation must never break recording


# Process-wide default bus/events (mirrors default_recorder()).
_default: Optional[ReliabilityEvents] = None
_default_lock = threading.Lock()


def default_reliability_events() -> ReliabilityEvents:
    global _default
    with _default_lock:
        if _default is None:
            _default = ReliabilityEvents()
        return _default
