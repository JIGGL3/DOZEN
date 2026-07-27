"""Reliability Layer ports (SADD-002, Phase 2.1.1) — interfaces ONLY.

``runtime_checkable`` Protocols, exactly like the Phase 1.1 foundation ports:
contract tests can assert structural conformance of any future adapter without
inheriting from anything. NOTHING here executes — implementations arrive in
Phases 2.2+ and must satisfy these shapes unchanged.

``ReliabilityClient`` mirrors ``LLMClient.complete`` (dozen/llm_client.py)
keyword-for-keyword: the decorator that later wraps a real client must be a
drop-in at the one seam every Planner/Executor/Synthesizer call flows through.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from ..context.domain.types import ProviderId, RunId
from .models import (
    CheckpointReference,
    ExecutionAttempt,
    FailoverDecision,
    FailureEvent,
    HealthRecord,
    ProbeResult,
    ProviderSnapshot,
    RecoveryOutcome,
    RecoveryPlan,
)
from .types import AttemptStatus, FailureType, HealthState, RecoveryAction


# --------------------------------------------------------------------------- #
# Detection & classification
# --------------------------------------------------------------------------- #
@runtime_checkable
class FailureDetector(Protocol):
    """One evidence source (exception / DOM-state / liveness family)."""

    @property
    def name(self) -> str: ...

    def detect(
        self, provider: ProviderId, raw_signal: object, context: dict[str, object]
    ) -> Optional[FailureEvent]: ...


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
@runtime_checkable
class RecoveryStrategy(Protocol):
    """One executable recovery action (RETRY, RECOVER_TAB, ...)."""

    @property
    def action(self) -> RecoveryAction: ...

    def applies_to(self, failure_type: FailureType) -> bool: ...

    def execute(self, plan_step_context: dict[str, object]) -> bool: ...


@runtime_checkable
class RecoveryEngine(Protocol):
    def plan(self, event: FailureEvent) -> RecoveryPlan: ...

    def execute(self, plan: RecoveryPlan) -> RecoveryOutcome: ...


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #
@runtime_checkable
class FailoverEngine(Protocol):
    def select(
        self, subtask_context: dict[str, object], exclude: frozenset[str]
    ) -> Optional[FailoverDecision]: ...

    def transfer(
        self, call_context: dict[str, object], decision: FailoverDecision
    ) -> object: ...


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
@runtime_checkable
class CheckpointStore(Protocol):
    def append(self, run_id: RunId, record: dict[str, object]) -> CheckpointReference: ...

    def read(self, run_id: RunId) -> list[dict[str, object]]: ...

    def latest(self, run_id: RunId) -> Optional[CheckpointReference]: ...

    def list_runs(self) -> list[str]: ...


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@runtime_checkable
class HealthManager(Protocol):
    def record_attempt(self, attempt: ExecutionAttempt) -> Optional[HealthState]: ...

    def record_probe(self, probe: ProbeResult) -> Optional[HealthState]: ...

    def state(self, provider: ProviderId) -> HealthState: ...

    def snapshot(self, provider: ProviderId) -> HealthRecord: ...

    def routable_providers(self) -> list[str]: ...

    def begin_recovery(self, provider: ProviderId) -> None: ...

    def end_recovery(self, provider: ProviderId, success: bool) -> None: ...

    def mark_needs_human(self, provider: ProviderId, reason: str) -> None: ...

    def human_resolved(self, provider: ProviderId) -> None: ...


@runtime_checkable
class HealthMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def probe_now(self, provider: ProviderId) -> ProbeResult: ...

    def set_cadence(self, state: HealthState, seconds: float) -> None: ...


# --------------------------------------------------------------------------- #
# Provider directory (composed read model; SADD §5.6)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ProviderRegistry(Protocol):
    def get(self, provider: ProviderId) -> Optional[ProviderSnapshot]: ...

    def all(self) -> list[ProviderSnapshot]: ...

    def routable(self) -> list[ProviderSnapshot]: ...


# --------------------------------------------------------------------------- #
# The client seam (mirror of LLMClient.complete — keyword-for-keyword)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ReliabilityClient(Protocol):
    """Drop-in shape for the future decorator around WebAutomationLLMClient."""

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[object],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> object: ...


# --------------------------------------------------------------------------- #
# Attempt audit trail
# --------------------------------------------------------------------------- #
@runtime_checkable
class ExecutionRecorder(Protocol):
    """Builds the immutable ExecutionAttempt audit trail. Every method
    returns a NEW attempt instance (models are frozen)."""

    def begin_attempt(
        self,
        provider: ProviderId,
        run_id: Optional[RunId] = None,
        subtask_id: Optional[str] = None,
        attempt_number: int = 1,
    ) -> ExecutionAttempt: ...

    def record_failure(
        self, attempt: ExecutionAttempt, event: FailureEvent
    ) -> ExecutionAttempt: ...

    def record_recovery(
        self, attempt: ExecutionAttempt, outcome: RecoveryOutcome
    ) -> ExecutionAttempt: ...

    def record_failover(
        self, attempt: ExecutionAttempt, decision: FailoverDecision
    ) -> ExecutionAttempt: ...

    def finish(
        self,
        attempt: ExecutionAttempt,
        status: AttemptStatus,
        latency_ms: Optional[float] = None,
        result_metadata: Optional[dict[str, object]] = None,
    ) -> ExecutionAttempt: ...
