"""Read-only query API over the in-memory recorder (Phase 2.1.3).

Every method works on an immutable ``snapshot()`` of the ring — queries can
never observe a half-updated state and can never mutate anything. Statistics
are pure computation with an optional TTL cache (``statistics_cache_seconds``)
so a polling debug UI doesn't recompute on every request.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .models import ExecutionAttempt
from .recorder import InMemoryExecutionRecorder
from .types import AttemptStatus, ExecutionStage
from .viewmodels import (
    AttemptDetails,
    AttemptSummary,
    ProviderSummary,
    StageSummary,
    StatisticsSummary,
)


def _average(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _breakdown(attempts: list[ExecutionAttempt], key: Callable[[ExecutionAttempt], str]):
    groups: dict[str, list[ExecutionAttempt]] = {}
    for attempt in attempts:
        groups.setdefault(key(attempt), []).append(attempt)
    return groups


class AttemptQuery:
    def __init__(
        self,
        recorder: InMemoryExecutionRecorder,
        statistics_cache_seconds: float = 0.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recorder = recorder
        self._cache_ttl = max(0.0, statistics_cache_seconds)
        self._time = time_source
        self._cache_lock = threading.Lock()
        self._cached_stats: Optional[StatisticsSummary] = None
        self._cached_at: float = -1.0
        self._cached_count: int = -1

    # ------------------------------------------------------------------ #
    # Attempt lookups (read-only; all work on an immutable snapshot)
    # ------------------------------------------------------------------ #
    def recent_attempts(self, limit: int = 50) -> list[ExecutionAttempt]:
        """Newest first."""
        snapshot = self._recorder.snapshot()
        if limit <= 0:
            return []
        return list(reversed(snapshot[-limit:]))

    def attempt_by_id(self, attempt_id: str) -> Optional[ExecutionAttempt]:
        for attempt in self._recorder.snapshot():
            if attempt.attempt_id == attempt_id:
                return attempt
        return None

    def attempts_for_run(self, run_id: str) -> list[ExecutionAttempt]:
        return [a for a in self._recorder.snapshot() if a.run_id == run_id]

    def attempts_for_provider(self, provider: str) -> list[ExecutionAttempt]:
        return [a for a in self._recorder.snapshot() if a.provider == provider]

    def attempts_for_stage(self, stage: ExecutionStage) -> list[ExecutionAttempt]:
        return [a for a in self._recorder.snapshot() if a.execution_stage is stage]

    def attempts_for_conversation(self, conversation_id: str) -> list[ExecutionAttempt]:
        return [a for a in self._recorder.snapshot()
                if a.conversation_id == conversation_id]

    # ------------------------------------------------------------------ #
    # Statistics (pure computation; NO health scoring)
    # ------------------------------------------------------------------ #
    def attempt_statistics(self) -> StatisticsSummary:
        with self._cache_lock:
            if self._cache_valid():
                return self._cached_stats  # type: ignore[return-value]
        stats = self._compute_statistics()
        with self._cache_lock:
            self._cached_stats = stats
            self._cached_at = self._time()
            self._cached_count = stats.total_attempts
        return stats

    def _cache_valid(self) -> bool:
        if self._cached_stats is None or self._cache_ttl <= 0:
            return False
        if self._time() - self._cached_at > self._cache_ttl:
            return False
        # New attempts invalidate immediately, TTL or not — cheap count check.
        return self._recorder.count() == self._cached_count

    def _compute_statistics(self) -> StatisticsSummary:
        attempts = list(self._recorder.snapshot())
        succeeded = [a for a in attempts if a.status is AttemptStatus.SUCCEEDED]
        failed = [a for a in attempts if a.status is AttemptStatus.FAILED]
        cancelled = [a for a in attempts if a.status is AttemptStatus.CANCELLED]
        latencies = [a.latency_ms for a in attempts if a.latency_ms is not None]

        def group_summary(cls, name: str, group: list[ExecutionAttempt]):
            return cls(
                **{("provider" if cls is ProviderSummary else "stage"): name},
                attempts=len(group),
                succeeded=sum(1 for a in group if a.status is AttemptStatus.SUCCEEDED),
                failed=sum(1 for a in group if a.status is AttemptStatus.FAILED),
                cancelled=sum(1 for a in group if a.status is AttemptStatus.CANCELLED),
                average_latency_ms=_average(
                    [a.latency_ms for a in group if a.latency_ms is not None]
                ),
            )

        providers = _breakdown(attempts, lambda a: str(a.provider))
        stages = _breakdown(attempts, lambda a: a.execution_stage.value)
        with_latency = [a for a in attempts if a.latency_ms is not None]
        longest = max(with_latency, key=lambda a: a.latency_ms) if with_latency else None

        return StatisticsSummary(
            total_attempts=len(attempts),
            successful_attempts=len(succeeded),
            failed_attempts=len(failed),
            cancelled_attempts=len(cancelled),
            average_latency_ms=_average(latencies),
            provider_breakdown=tuple(
                group_summary(ProviderSummary, name, group)
                for name, group in sorted(providers.items())
            ),
            stage_breakdown=tuple(
                group_summary(StageSummary, name, group)
                for name, group in sorted(stages.items())
            ),
            longest_attempt=AttemptSummary.from_attempt(longest) if longest else None,
            newest_attempt=AttemptSummary.from_attempt(attempts[-1]) if attempts else None,
            oldest_attempt=AttemptSummary.from_attempt(attempts[0]) if attempts else None,
            ring_capacity=self._recorder.capacity,
            computed_at=self._recorder._clock.now(),  # noqa: SLF001 - same package
        )

    # ------------------------------------------------------------------ #
    # DTO conveniences used by the debug API
    # ------------------------------------------------------------------ #
    def summaries(self, attempts: list[ExecutionAttempt]) -> list[AttemptSummary]:
        return [AttemptSummary.from_attempt(a) for a in attempts]

    def details(self, attempt: ExecutionAttempt) -> AttemptDetails:
        return AttemptDetails.from_attempt(attempt)
