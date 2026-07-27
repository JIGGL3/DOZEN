"""ReliabilityDebugApi — the read-only developer inspection facade
(Phase 2.1.3).

Framework-agnostic: returns plain JSON-safe dicts; the web layer maps them to
HTTP. Disabled by default (``ObservabilityConfig.enable_debug_api``); when
disabled, every call raises ``DebugApiDisabled`` so transports can translate
that to 404 uniformly. Nothing here can mutate the recorder — every read goes
through immutable snapshots.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..context.manager.events import InProcessEventBus
from ..context.domain.enums import EventType
from .config import ObservabilityConfig
from .query import AttemptQuery
from .recorder import InMemoryExecutionRecorder
from .types import ExecutionStage, parse_enum

logger = logging.getLogger("dozen.reliability")


class DebugApiDisabled(RuntimeError):
    """Raised when the debug API is called while disabled by configuration."""


class ReliabilityDebugApi:
    def __init__(
        self,
        recorder: InMemoryExecutionRecorder,
        config: Optional[ObservabilityConfig] = None,
    ) -> None:
        self.config = config or ObservabilityConfig()
        self._recorder = recorder
        self._query = AttemptQuery(
            recorder,
            statistics_cache_seconds=self.config.statistics_cache_seconds,
        )
        if self.config.enable_attempt_logging:
            attach_attempt_logging(recorder.events.bus)

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self.config.enable_debug_api

    def _guard(self) -> None:
        if not self.enabled:
            raise DebugApiDisabled("Reliability debug API is disabled by configuration.")

    def _clamp(self, limit: Optional[int]) -> int:
        ceiling = max(1, self.config.max_attempts_returned)
        if limit is None or limit <= 0:
            return ceiling
        return min(limit, ceiling)

    # ------------------------------------------------------------------ #
    # Endpoint payloads (all JSON-safe dicts, all read-only)
    # ------------------------------------------------------------------ #
    def attempts(
        self,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> dict[str, object]:
        self._guard()
        if run_id:
            found = self._query.attempts_for_run(run_id)
        elif conversation_id:
            found = self._query.attempts_for_conversation(conversation_id)
        elif provider:
            found = self._query.attempts_for_provider(provider)
        elif stage:
            found = self._query.attempts_for_stage(parse_enum(ExecutionStage, stage))
        else:
            found = list(reversed(self._query.recent_attempts(self._clamp(limit))))
        found = found[-self._clamp(limit):]
        newest_first = list(reversed(found))
        return {
            "count": len(newest_first),
            "ring_size": self._recorder.count(),
            "ring_capacity": self._recorder.capacity,
            "attempts": [s.to_dict() for s in self._query.summaries(newest_first)],
        }

    def attempt(self, attempt_id: str) -> Optional[dict[str, object]]:
        self._guard()
        found = self._query.attempt_by_id(attempt_id)
        return self._query.details(found).to_dict() if found else None

    def stats(self) -> dict[str, object]:
        self._guard()
        return self._query.attempt_statistics().to_dict()

    def providers(self) -> dict[str, object]:
        self._guard()
        stats = self._query.attempt_statistics()
        return {"providers": [p.to_dict() for p in stats.provider_breakdown]}

    def stages(self) -> dict[str, object]:
        self._guard()
        stats = self._query.attempt_statistics()
        return {"stages": [s.to_dict() for s in stats.stage_breakdown]}

    # ---------------- provider health (Phase 2.2.2, read-only) ----------- #
    def health(self) -> dict[str, object]:
        self._guard()
        manager = self._recorder.health
        return {
            "statistics": manager.health_statistics(),
            "providers": [r.to_dict() for r in manager.all_health()],
        }

    def health_provider(self, provider: str) -> Optional[dict[str, object]]:
        self._guard()
        manager = self._recorder.health
        if provider not in manager._known():  # noqa: SLF001 - read-only check
            return None
        return manager.emit_snapshot(provider).to_dict()

    def health_history(
        self, provider: Optional[str] = None, limit: int = 50
    ) -> dict[str, object]:
        self._guard()
        history = self._recorder.health.health_history(provider, limit=limit)
        return {"count": len(history),
                "transitions": [t.to_dict() for t in history]}


def attach_attempt_logging(
    bus: InProcessEventBus,
    log: Optional[logging.Logger] = None,
) -> Callable[[], None]:
    """Subscribe a one-line-per-recorded-attempt logger to the bus.
    Returns the unsubscribe callable. Observation only."""
    target = log or logger

    def on_recorded(event) -> None:
        payload = event.payload
        target.debug(
            "attempt %s provider=%s stage=%s status=%s latency_ms=%s run=%s",
            payload.get("attempt_id"), payload.get("provider"),
            payload.get("stage"), payload.get("status"),
            payload.get("latency_ms"), payload.get("run_id"),
        )

    return bus.subscribe(on_recorded, EventType.ATTEMPT_RECORDED)
