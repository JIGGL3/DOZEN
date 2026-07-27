"""Debug view models (Phase 2.1.3) — lightweight DTOs for the debug API.

Internal dataclasses (ExecutionAttempt et al.) are never exposed directly:
these DTOs are the stable wire shape, deliberately flat, JSON-safe, and free
to diverge from the domain models later without breaking the debug API.
Read-only projections — construction and ``to_dict`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import ExecutionAttempt


@dataclass(frozen=True)
class AttemptSummary:
    """One row in an attempts listing."""

    attempt_id: str
    provider: str
    stage: str
    status: str
    started_at: str
    latency_ms: Optional[float]
    run_id: Optional[str]
    conversation_id: Optional[str]
    model_name: str
    response_present: bool
    failure_present: bool

    @classmethod
    def from_attempt(cls, attempt: ExecutionAttempt) -> "AttemptSummary":
        return cls(
            attempt_id=attempt.attempt_id,
            provider=attempt.provider,
            stage=attempt.execution_stage.value,
            status=attempt.status.value,
            started_at=attempt.started_at,
            latency_ms=attempt.latency_ms,
            run_id=attempt.run_id,
            conversation_id=attempt.conversation_id,
            model_name=attempt.model_name,
            response_present=attempt.response_present,
            failure_present=attempt.failure_present,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "latency_ms": self.latency_ms,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "model_name": self.model_name,
            "response_present": self.response_present,
            "failure_present": self.failure_present,
        }


@dataclass(frozen=True)
class AttemptDetails:
    """Everything a developer needs about one attempt."""

    summary: AttemptSummary
    agent_name: Optional[str]
    task_id: Optional[str]
    subtask_id: Optional[str]
    attempt_number: int
    retry_number: int
    parent_attempt_id: Optional[str]
    finished_at: Optional[str]
    prompt_character_count: int
    response_character_count: int
    debug_notes: tuple[str, ...]
    result_metadata: dict[str, object]
    failure_count: int
    recovery_count: int
    failover_count: int
    metadata_version: int

    @classmethod
    def from_attempt(cls, attempt: ExecutionAttempt) -> "AttemptDetails":
        return cls(
            summary=AttemptSummary.from_attempt(attempt),
            agent_name=attempt.agent_name,
            task_id=attempt.task_id,
            subtask_id=attempt.subtask_id,
            attempt_number=attempt.attempt_number,
            retry_number=attempt.retry_number,
            parent_attempt_id=attempt.parent_attempt_id,
            finished_at=attempt.finished_at,
            prompt_character_count=attempt.prompt_character_count,
            response_character_count=attempt.response_character_count,
            debug_notes=attempt.debug_notes,
            result_metadata=dict(attempt.result_metadata),
            failure_count=len(attempt.failures),
            recovery_count=len(attempt.recoveries),
            failover_count=len(attempt.failovers),
            metadata_version=attempt.metadata_version,
        )

    def to_dict(self) -> dict[str, object]:
        out = self.summary.to_dict()
        out.update({
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "attempt_number": self.attempt_number,
            "retry_number": self.retry_number,
            "parent_attempt_id": self.parent_attempt_id,
            "finished_at": self.finished_at,
            "prompt_character_count": self.prompt_character_count,
            "response_character_count": self.response_character_count,
            "debug_notes": list(self.debug_notes),
            "result_metadata": dict(self.result_metadata),
            "failure_count": self.failure_count,
            "recovery_count": self.recovery_count,
            "failover_count": self.failover_count,
            "metadata_version": self.metadata_version,
        })
        return out


@dataclass(frozen=True)
class ProviderSummary:
    provider: str
    attempts: int
    succeeded: int
    failed: int
    cancelled: int
    average_latency_ms: Optional[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "attempts": self.attempts,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "average_latency_ms": self.average_latency_ms,
        }


@dataclass(frozen=True)
class StageSummary:
    stage: str
    attempts: int
    succeeded: int
    failed: int
    cancelled: int
    average_latency_ms: Optional[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "attempts": self.attempts,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "average_latency_ms": self.average_latency_ms,
        }


@dataclass(frozen=True)
class StatisticsSummary:
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    cancelled_attempts: int
    average_latency_ms: Optional[float]
    provider_breakdown: tuple[ProviderSummary, ...] = ()
    stage_breakdown: tuple[StageSummary, ...] = ()
    longest_attempt: Optional[AttemptSummary] = None
    newest_attempt: Optional[AttemptSummary] = None
    oldest_attempt: Optional[AttemptSummary] = None
    ring_capacity: int = 0
    computed_at: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "cancelled_attempts": self.cancelled_attempts,
            "average_latency_ms": self.average_latency_ms,
            "provider_breakdown": [p.to_dict() for p in self.provider_breakdown],
            "stage_breakdown": [s.to_dict() for s in self.stage_breakdown],
            "longest_attempt": self.longest_attempt.to_dict() if self.longest_attempt else None,
            "newest_attempt": self.newest_attempt.to_dict() if self.newest_attempt else None,
            "oldest_attempt": self.oldest_attempt.to_dict() if self.oldest_attempt else None,
            "ring_capacity": self.ring_capacity,
            "computed_at": self.computed_at,
            **self.extra,
        }
