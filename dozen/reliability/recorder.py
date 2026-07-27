"""InMemoryExecutionRecorder — passive attempt bookkeeping (Phase 2.1.2).

Implements the ``ExecutionRecorder`` port from Phase 2.1.1 over a thread-safe
in-memory ring buffer (default 1000 finished attempts; oldest evicted first).
Nothing is persisted and nothing is ever *done* with the records here —
observability only. Persistence and reaction arrive in later phases.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from typing import Optional

from ..context.domain.types import ProviderId, RunId
from .attempt_factory import AttemptFactory
from .clock import ReliabilityClock, SystemReliabilityClock
from .metadata import ProviderMetadata, WorkflowMetadata
from .models import (
    ExecutionAttempt,
    FailoverDecision,
    FailureEvent,
    RecoveryOutcome,
)
from .types import AttemptStatus, ExecutionStage

DEFAULT_CAPACITY = 1000


class InMemoryExecutionRecorder:
    """Ring-buffered, thread-safe, purely observational."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        factory: Optional[AttemptFactory] = None,
        clock: Optional[ReliabilityClock] = None,
        events: Optional["ReliabilityEvents"] = None,
        health_manager: Optional["PassiveHealthManager"] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._clock = clock or SystemReliabilityClock()
        self._factory = factory or AttemptFactory(clock=self._clock)
        self._lock = threading.Lock()
        self._ring: deque[ExecutionAttempt] = deque(maxlen=capacity)
        # Lazy import keeps recorder importable without the events module.
        if events is None:
            from .events import ReliabilityEvents as _RE
            events = _RE()
        self.events = events
        # Built lazily on the first FAILED attempt (Phase 2.2.1).
        self._classifier = None
        # Passive health feed (Phase 2.2.2): every finished attempt flows to
        # the health manager. Computation only — health is never read here.
        if health_manager is None:
            from .health import PassiveHealthManager
            health_manager = PassiveHealthManager(events=events)
        self.health = health_manager

    # ------------------------------------------------------------------ #
    # ExecutionRecorder port (Phase 2.1.1 interface, verbatim shapes)
    # ------------------------------------------------------------------ #
    def begin_attempt(
        self,
        provider: ProviderId,
        run_id: Optional[RunId] = None,
        subtask_id: Optional[str] = None,
        attempt_number: int = 1,
    ) -> ExecutionAttempt:
        attempt = self.begin(
            ProviderMetadata(provider=provider),
            WorkflowMetadata(run_id=run_id, subtask_id=subtask_id),
            attempt_number=attempt_number,
        )
        return attempt

    def record_failure(
        self, attempt: ExecutionAttempt, event: FailureEvent
    ) -> ExecutionAttempt:
        return attempt.with_failure(event)

    def record_recovery(
        self, attempt: ExecutionAttempt, outcome: RecoveryOutcome
    ) -> ExecutionAttempt:
        return attempt.with_recovery(outcome)

    def record_failover(
        self, attempt: ExecutionAttempt, decision: FailoverDecision
    ) -> ExecutionAttempt:
        return attempt.with_failover(decision)

    def finish(
        self,
        attempt: ExecutionAttempt,
        status: AttemptStatus,
        latency_ms: Optional[float] = None,
        result_metadata: Optional[dict[str, object]] = None,
    ) -> ExecutionAttempt:
        """Terminalize and store. The stored attempt is the returned copy;
        the input (like all models) is untouched."""
        done = attempt.with_status(
            status, finished_at=self._clock.now(), latency_ms=latency_ms
        )
        if result_metadata:
            merged = dict(done.result_metadata)
            merged.update(result_metadata)
            done = dataclasses.replace(done, result_metadata=merged)
        if status is AttemptStatus.FAILED:
            done = self._classify_failure(done)
        evicted: Optional[ExecutionAttempt] = None
        with self._lock:
            if len(self._ring) == self._ring.maxlen:
                evicted = self._ring[0]  # about to fall off the ring
            self._ring.append(done)
        if evicted is not None:
            self.events.attempt_evicted(evicted)
        self.events.attempt_recorded(done)
        self.events.attempt_finished(done)
        try:
            self.health.record_attempt(done)   # passive feed; fail-open
        except Exception:
            pass
        return done

    # ------------------------------------------------------------------ #
    # Failure classification (Phase 2.2.1) — classify only, never react.
    # ------------------------------------------------------------------ #
    def _classify_failure(self, done: ExecutionAttempt) -> ExecutionAttempt:
        """Attach a FailureClassification (and matching FailureEvent) to a
        FAILED attempt. Pure bookkeeping: any error here is swallowed and the
        attempt is stored unclassified — classification can never alter or
        block execution."""
        try:
            from .classifier import FailureClassifier
            from .evidence import FailureEvidence
            from .models import FailureEvent
            from .types import FailureEventId

            if self._classifier is None:
                self._classifier = FailureClassifier()
            evidence = FailureEvidence.from_attempt(done)
            verdict = self._classifier.classify(evidence)
            event = FailureEvent(
                id=FailureEventId(self._factory.ids.new_id()),
                provider=done.provider,
                failure_type=verdict.failure_type,
                detected_at=self._clock.now(),
                confidence=verdict.confidence,
                message=verdict.explanation,
                raw_signal=evidence.exception_message,
                evidence=tuple(
                    f"{m.name}:{m.confidence:.2f}" for m in verdict.matched_rules
                ),
                run_id=done.run_id,
                task_id=done.task_id,
                subtask_id=done.subtask_id,
                attempt_id=done.attempt_id,
            )
            merged = dict(done.result_metadata)
            merged["failure_classification"] = {
                "failure_type": verdict.failure_type.value,
                "confidence": verdict.confidence,
                "matched_rules": [m.to_dict() for m in verdict.matched_rules],
                "explanation": verdict.explanation,
            }
            return dataclasses.replace(
                done.with_failure(event), result_metadata=merged
            )
        except Exception:
            return done  # classification is observational; never fatal

    # ------------------------------------------------------------------ #
    # Richer construction used by the decorator
    # ------------------------------------------------------------------ #
    def begin(
        self,
        provider_meta: ProviderMetadata,
        workflow: Optional[WorkflowMetadata] = None,
        attempt_number: int = 1,
        stage: ExecutionStage = ExecutionStage.UNKNOWN,
        prompt_character_count: int = 0,
    ) -> ExecutionAttempt:
        attempt = self._factory.create(
            provider_meta, workflow, attempt_number,
            stage=stage, prompt_character_count=prompt_character_count,
        )
        self.events.attempt_started(attempt)
        return attempt

    # ------------------------------------------------------------------ #
    # Introspection (read-only copies)
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        return self._ring.maxlen or 0

    def count(self) -> int:
        with self._lock:
            return len(self._ring)

    def attempts(self) -> list[ExecutionAttempt]:
        """Oldest -> newest."""
        with self._lock:
            return list(self._ring)

    def recent(self, n: int = 50) -> list[ExecutionAttempt]:
        """Newest -> oldest, at most ``n``."""
        with self._lock:
            items = list(self._ring)
        return list(reversed(items[-max(0, n):])) if n > 0 else []

    def for_run(self, run_id: RunId) -> list[ExecutionAttempt]:
        with self._lock:
            return [a for a in self._ring if a.run_id == run_id]

    def for_provider(self, provider: ProviderId) -> list[ExecutionAttempt]:
        with self._lock:
            return [a for a in self._ring if a.provider == provider]

    # -- Phase 2.1.3: non-mutating peeks + immutable snapshot -------------- #
    def peek(self) -> Optional[ExecutionAttempt]:
        """The newest recorded attempt (alias of peek_latest)."""
        return self.peek_latest()

    def peek_latest(self) -> Optional[ExecutionAttempt]:
        with self._lock:
            return self._ring[-1] if self._ring else None

    def peek_oldest(self) -> Optional[ExecutionAttempt]:
        with self._lock:
            return self._ring[0] if self._ring else None

    def snapshot(self) -> tuple[ExecutionAttempt, ...]:
        """Immutable point-in-time view, oldest -> newest. A tuple of frozen
        models: nothing the caller does to it can touch the ring."""
        with self._lock:
            return tuple(self._ring)

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()


# Process-wide default recorder (mirrors default_registry()). The wiring in
# webllm/pool.py records into this instance; nothing reads it yet.
_default: Optional[InMemoryExecutionRecorder] = None
_default_lock = threading.Lock()


def default_recorder() -> InMemoryExecutionRecorder:
    global _default
    with _default_lock:
        if _default is None:
            _default = InMemoryExecutionRecorder()
        return _default
