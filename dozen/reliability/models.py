"""Reliability Layer — immutable shared models (SADD-002, Phase 2.1.1).

Design rules (deliberately identical in spirit to the Phase 1.1 foundation):

* Every model is a **frozen dataclass** — attempts, failures, decisions and
  outcomes are facts; facts don't mutate. Progressive construction in later
  phases uses ``dataclasses.replace`` / the ``with_*`` helpers on
  ``ExecutionAttempt`` (pure copies, no in-place state).
* Sequences are tuples, so immutability is real, not rebinding-only.
* Serialization contract: ``to_dict()`` (JSON-safe, includes
  ``schema_version``), tolerant ``from_dict()`` (unknown keys preserved in
  ``extra`` and re-emitted; unknown enum values map to UNKNOWN with the raw
  string kept), ``to_json``/``from_json``, and structural ``validate()``.
* NO behavior: no engine logic, no I/O, no browser knowledge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

from ..context.domain.types import ConversationId, ProviderId, RunId, Timestamp
from .types import (
    AttemptId,
    AttemptStatus,
    CheckpointKind,
    CheckpointRecordId,
    ExecutionStage,
    FailureEventId,
    FailureType,
    HealthState,
    RecoveryAction,
    RecoveryPlanId,
    RecoveryStatus,
    parse_enum,
)

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def _is_timestamp(value: object) -> bool:
    return isinstance(value, str) and bool(_ISO_UTC_RE.match(value))


def _extra_of(data: Mapping[str, object], known: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in known and k != "schema_version"}


def _opt_str(data: Mapping[str, object], key: str) -> Optional[str]:
    value = data.get(key)
    return str(value) if isinstance(value, str) else None


def _opt_float(data: Mapping[str, object], key: str) -> Optional[float]:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _str_tuple(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key)
    return tuple(str(x) for x in value) if isinstance(value, (list, tuple)) else ()


def _validate_id(problems: list[str], name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{name} must be a non-empty string")


class _JsonMixin:
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)  # type: ignore[attr-defined]

    @classmethod
    def from_json(cls, payload: str) -> "object":
        return cls.from_dict(json.loads(payload))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# FailureEvent — one classified failure observation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FailureEvent(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "id", "provider", "failure_type", "confidence", "message", "raw_signal",
        "evidence", "detected_at", "run_id", "task_id", "subtask_id",
        "attempt_id", "screenshot_ref", "dom_snapshot_ref", "details",
    )

    id: FailureEventId
    provider: ProviderId
    failure_type: FailureType
    detected_at: Timestamp
    confidence: float = 0.0
    message: str = ""
    raw_signal: str = ""                       # repr of the triggering exception/signal
    evidence: tuple[str, ...] = ()             # detector evidence lines
    run_id: Optional[RunId] = None
    task_id: Optional[str] = None
    subtask_id: Optional[str] = None
    attempt_id: Optional[AttemptId] = None
    screenshot_ref: Optional[str] = None       # opaque path/ref; capture is later-phase
    dom_snapshot_ref: Optional[str] = None
    details: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "provider": self.provider,
            "failure_type": self.failure_type.value,
            "detected_at": self.detected_at,
            "confidence": self.confidence,
            "message": self.message,
            "raw_signal": self.raw_signal,
            "evidence": list(self.evidence),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "attempt_id": self.attempt_id,
            "screenshot_ref": self.screenshot_ref,
            "dom_snapshot_ref": self.dom_snapshot_ref,
            "details": dict(self.details),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FailureEvent":
        extra = _extra_of(data, cls._KNOWN)
        raw_type = data.get("failure_type", "unknown")
        failure_type = parse_enum(FailureType, raw_type)
        if failure_type is FailureType.UNKNOWN and str(raw_type) != "unknown":
            extra["failure_type__raw"] = raw_type
        details = data.get("details")
        run = _opt_str(data, "run_id")
        attempt = _opt_str(data, "attempt_id")
        return cls(
            id=FailureEventId(str(data.get("id", ""))),
            provider=ProviderId(str(data.get("provider", ""))),
            failure_type=failure_type,
            detected_at=Timestamp(str(data.get("detected_at", ""))),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            message=str(data.get("message", "")),
            raw_signal=str(data.get("raw_signal", "")),
            evidence=_str_tuple(data, "evidence"),
            run_id=RunId(run) if run else None,
            task_id=_opt_str(data, "task_id"),
            subtask_id=_opt_str(data, "subtask_id"),
            attempt_id=AttemptId(attempt) if attempt else None,
            screenshot_ref=_opt_str(data, "screenshot_ref"),
            dom_snapshot_ref=_opt_str(data, "dom_snapshot_ref"),
            details=dict(details) if isinstance(details, dict) else {},
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        _validate_id(problems, "provider", self.provider)
        if not _is_timestamp(self.detected_at):
            problems.append("detected_at must be ISO-8601 UTC (…Z)")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be within [0, 1]")
        return problems


# --------------------------------------------------------------------------- #
# ProviderStatistics + HealthRecord — the health manager's read model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderStatistics(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "attempts_1h", "successes_1h", "attempts_24h", "successes_24h",
        "latency_ewma_ms", "latency_p95_ms", "consecutive_failures",
        "quarantine_count", "failure_histogram", "last_success_at",
        "last_failure_at",
    )

    attempts_1h: int = 0
    successes_1h: int = 0
    attempts_24h: int = 0
    successes_24h: int = 0
    latency_ewma_ms: float = 0.0
    latency_p95_ms: float = 0.0
    consecutive_failures: int = 0
    quarantine_count: int = 0
    failure_histogram: dict[str, int] = field(default_factory=dict)  # FailureType.value -> count
    last_success_at: Optional[Timestamp] = None
    last_failure_at: Optional[Timestamp] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def success_rate_1h(self) -> float:
        return (self.successes_1h / self.attempts_1h) if self.attempts_1h else 0.0

    @property
    def success_rate_24h(self) -> float:
        return (self.successes_24h / self.attempts_24h) if self.attempts_24h else 0.0

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "attempts_1h": self.attempts_1h,
            "successes_1h": self.successes_1h,
            "attempts_24h": self.attempts_24h,
            "successes_24h": self.successes_24h,
            "latency_ewma_ms": self.latency_ewma_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "consecutive_failures": self.consecutive_failures,
            "quarantine_count": self.quarantine_count,
            "failure_histogram": dict(self.failure_histogram),
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProviderStatistics":
        histogram = data.get("failure_histogram")
        last_ok = _opt_str(data, "last_success_at")
        last_bad = _opt_str(data, "last_failure_at")
        return cls(
            attempts_1h=int(data.get("attempts_1h", 0) or 0),
            successes_1h=int(data.get("successes_1h", 0) or 0),
            attempts_24h=int(data.get("attempts_24h", 0) or 0),
            successes_24h=int(data.get("successes_24h", 0) or 0),
            latency_ewma_ms=float(data.get("latency_ewma_ms", 0.0) or 0.0),
            latency_p95_ms=float(data.get("latency_p95_ms", 0.0) or 0.0),
            consecutive_failures=int(data.get("consecutive_failures", 0) or 0),
            quarantine_count=int(data.get("quarantine_count", 0) or 0),
            failure_histogram={str(k): int(v) for k, v in histogram.items()}
            if isinstance(histogram, dict) else {},
            last_success_at=Timestamp(last_ok) if last_ok else None,
            last_failure_at=Timestamp(last_bad) if last_bad else None,
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for name in (
            "attempts_1h", "successes_1h", "attempts_24h", "successes_24h",
            "consecutive_failures", "quarantine_count",
        ):
            if int(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        if self.successes_1h > self.attempts_1h:
            problems.append("successes_1h cannot exceed attempts_1h")
        if self.successes_24h > self.attempts_24h:
            problems.append("successes_24h cannot exceed attempts_24h")
        for name in ("latency_ewma_ms", "latency_p95_ms"):
            if float(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        for name in ("last_success_at", "last_failure_at"):
            value = getattr(self, name)
            if value is not None and not _is_timestamp(value):
                problems.append(f"{name} must be ISO-8601 UTC (…Z)")
        return problems


@dataclass(frozen=True)
class HealthRecord(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("provider", "state", "confidence", "statistics", "updated_at", "state_reason")

    provider: ProviderId
    state: HealthState
    updated_at: Timestamp
    confidence: float = 0.0
    statistics: ProviderStatistics = field(default_factory=ProviderStatistics)
    state_reason: str = ""                     # human-readable cause of the current state
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "provider": self.provider,
            "state": self.state.value,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "statistics": self.statistics.to_dict(),
            "state_reason": self.state_reason,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HealthRecord":
        extra = _extra_of(data, cls._KNOWN)
        raw_state = data.get("state", "unknown")
        state = parse_enum(HealthState, raw_state)
        if state is HealthState.UNKNOWN and str(raw_state) != "unknown":
            extra["state__raw"] = raw_state
        stats = data.get("statistics")
        return cls(
            provider=ProviderId(str(data.get("provider", ""))),
            state=state,
            updated_at=Timestamp(str(data.get("updated_at", ""))),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            statistics=ProviderStatistics.from_dict(stats) if isinstance(stats, dict)
            else ProviderStatistics(),
            state_reason=str(data.get("state_reason", "")),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "provider", self.provider)
        if not _is_timestamp(self.updated_at):
            problems.append("updated_at must be ISO-8601 UTC (…Z)")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be within [0, 1]")
        problems.extend(self.statistics.validate())
        return problems


# --------------------------------------------------------------------------- #
# RecoveryStep / RecoveryPlan / RecoveryOutcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecoveryStep(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("action", "order", "timeout_s", "note")

    action: RecoveryAction
    order: int = 0
    timeout_s: float = 60.0
    note: str = ""
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "action": self.action.value,
            "order": self.order,
            "timeout_s": self.timeout_s,
            "note": self.note,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RecoveryStep":
        extra = _extra_of(data, cls._KNOWN)
        raw = data.get("action", "unknown")
        action = parse_enum(RecoveryAction, raw)
        if action is RecoveryAction.UNKNOWN and str(raw) != "unknown":
            extra["action__raw"] = raw
        return cls(
            action=action,
            order=int(data.get("order", 0) or 0),
            timeout_s=float(data.get("timeout_s", 60.0) or 0.0),
            note=str(data.get("note", "")),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.order < 0:
            problems.append("order must be >= 0")
        if self.timeout_s <= 0:
            problems.append("timeout_s must be > 0")
        if self.action is RecoveryAction.UNKNOWN:
            problems.append("action is unknown")
        return problems


@dataclass(frozen=True)
class RecoveryPlan(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("id", "failure", "steps", "budget", "created_at")

    id: RecoveryPlanId
    failure: FailureEvent
    created_at: Timestamp
    steps: tuple[RecoveryStep, ...] = ()
    budget: int = 2                            # max in-place actions (SADD §8)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "failure": self.failure.to_dict(),
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
            "budget": self.budget,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RecoveryPlan":
        failure = data.get("failure")
        steps = data.get("steps")
        return cls(
            id=RecoveryPlanId(str(data.get("id", ""))),
            failure=FailureEvent.from_dict(failure) if isinstance(failure, dict)
            else FailureEvent(
                id=FailureEventId(""), provider=ProviderId(""),
                failure_type=FailureType.UNKNOWN, detected_at=Timestamp(""),
            ),
            created_at=Timestamp(str(data.get("created_at", ""))),
            steps=tuple(RecoveryStep.from_dict(s) for s in steps if isinstance(s, dict))
            if isinstance(steps, (list, tuple)) else (),
            budget=int(data.get("budget", 2) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        if self.budget < 0:
            problems.append("budget must be >= 0")
        orders = [s.order for s in self.steps]
        if orders != sorted(orders):
            problems.append("steps must be ordered by their order field")
        problems.extend(self.failure.validate())
        for s in self.steps:
            problems.extend(s.validate())
        return problems


@dataclass(frozen=True)
class RecoveryOutcome(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "plan_id", "status", "executed_steps", "succeeded_action",
        "duration_ms", "finished_at", "notes",
    )

    plan_id: RecoveryPlanId
    status: RecoveryStatus
    finished_at: Timestamp
    executed_steps: tuple[RecoveryStep, ...] = ()
    succeeded_action: Optional[RecoveryAction] = None
    duration_ms: float = 0.0
    notes: tuple[str, ...] = ()
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "finished_at": self.finished_at,
            "executed_steps": [s.to_dict() for s in self.executed_steps],
            "succeeded_action": self.succeeded_action.value if self.succeeded_action else None,
            "duration_ms": self.duration_ms,
            "notes": list(self.notes),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RecoveryOutcome":
        extra = _extra_of(data, cls._KNOWN)
        raw_status = data.get("status", "unknown")
        status = parse_enum(RecoveryStatus, raw_status)
        if status is RecoveryStatus.UNKNOWN and str(raw_status) != "unknown":
            extra["status__raw"] = raw_status
        steps = data.get("executed_steps")
        succeeded = _opt_str(data, "succeeded_action")
        return cls(
            plan_id=RecoveryPlanId(str(data.get("plan_id", ""))),
            status=status,
            finished_at=Timestamp(str(data.get("finished_at", ""))),
            executed_steps=tuple(RecoveryStep.from_dict(s) for s in steps if isinstance(s, dict))
            if isinstance(steps, (list, tuple)) else (),
            succeeded_action=parse_enum(RecoveryAction, succeeded) if succeeded else None,
            duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
            notes=_str_tuple(data, "notes"),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "plan_id", self.plan_id)
        if not _is_timestamp(self.finished_at):
            problems.append("finished_at must be ISO-8601 UTC (…Z)")
        if self.duration_ms < 0:
            problems.append("duration_ms must be >= 0")
        if self.status is RecoveryStatus.RECOVERED and self.succeeded_action is None:
            problems.append("RECOVERED outcome must name the succeeded_action")
        for s in self.executed_steps:
            problems.extend(s.validate())
        return problems


# --------------------------------------------------------------------------- #
# ProbeResult — one Health Monitor observation (capture is a later phase)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeResult(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "provider", "probed_at", "alive", "logged_in", "url_ok",
        "input_resolvable", "captcha_present", "rate_limited",
        "latency_ms", "evidence", "error",
    )

    provider: ProviderId
    probed_at: Timestamp
    alive: bool = False
    logged_in: Optional[bool] = None           # None == probe did not check
    url_ok: Optional[bool] = None
    input_resolvable: Optional[bool] = None
    captcha_present: Optional[bool] = None
    rate_limited: Optional[bool] = None
    latency_ms: float = 0.0
    evidence: tuple[str, ...] = ()
    error: Optional[str] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "provider": self.provider,
            "probed_at": self.probed_at,
            "alive": self.alive,
            "logged_in": self.logged_in,
            "url_ok": self.url_ok,
            "input_resolvable": self.input_resolvable,
            "captcha_present": self.captcha_present,
            "rate_limited": self.rate_limited,
            "latency_ms": self.latency_ms,
            "evidence": list(self.evidence),
            "error": self.error,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProbeResult":
        def opt_bool(key: str) -> Optional[bool]:
            value = data.get(key)
            return bool(value) if isinstance(value, bool) else None

        return cls(
            provider=ProviderId(str(data.get("provider", ""))),
            probed_at=Timestamp(str(data.get("probed_at", ""))),
            alive=bool(data.get("alive", False)),
            logged_in=opt_bool("logged_in"),
            url_ok=opt_bool("url_ok"),
            input_resolvable=opt_bool("input_resolvable"),
            captcha_present=opt_bool("captcha_present"),
            rate_limited=opt_bool("rate_limited"),
            latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
            evidence=_str_tuple(data, "evidence"),
            error=_opt_str(data, "error"),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "provider", self.provider)
        if not _is_timestamp(self.probed_at):
            problems.append("probed_at must be ISO-8601 UTC (…Z)")
        if self.latency_ms < 0:
            problems.append("latency_ms must be >= 0")
        return problems


# --------------------------------------------------------------------------- #
# FailoverDecision — one explainable provider-selection verdict
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FailoverDecision(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "from_provider", "to_provider", "subtask_id", "run_id", "scores",
        "factors", "excluded", "decided_at", "reason",
    )

    from_provider: ProviderId
    decided_at: Timestamp
    to_provider: Optional[ProviderId] = None   # None == no routable provider left
    subtask_id: Optional[str] = None
    run_id: Optional[RunId] = None
    scores: dict[str, float] = field(default_factory=dict)          # candidate -> final score
    factors: dict[str, dict[str, float]] = field(default_factory=dict)  # candidate -> factor map
    excluded: tuple[str, ...] = ()
    reason: str = ""
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "from_provider": self.from_provider,
            "decided_at": self.decided_at,
            "to_provider": self.to_provider,
            "subtask_id": self.subtask_id,
            "run_id": self.run_id,
            "scores": dict(self.scores),
            "factors": {k: dict(v) for k, v in self.factors.items()},
            "excluded": list(self.excluded),
            "reason": self.reason,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FailoverDecision":
        scores = data.get("scores")
        factors = data.get("factors")
        to_provider = _opt_str(data, "to_provider")
        run = _opt_str(data, "run_id")
        return cls(
            from_provider=ProviderId(str(data.get("from_provider", ""))),
            decided_at=Timestamp(str(data.get("decided_at", ""))),
            to_provider=ProviderId(to_provider) if to_provider else None,
            subtask_id=_opt_str(data, "subtask_id"),
            run_id=RunId(run) if run else None,
            scores={str(k): float(v) for k, v in scores.items()}
            if isinstance(scores, dict) else {},
            factors={
                str(k): {str(fk): float(fv) for fk, fv in v.items()}
                for k, v in factors.items() if isinstance(v, dict)
            } if isinstance(factors, dict) else {},
            excluded=_str_tuple(data, "excluded"),
            reason=str(data.get("reason", "")),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "from_provider", self.from_provider)
        if not _is_timestamp(self.decided_at):
            problems.append("decided_at must be ISO-8601 UTC (…Z)")
        if self.to_provider is not None and self.to_provider == self.from_provider:
            problems.append("failover target must differ from the failed provider")
        if self.to_provider is not None and self.to_provider not in self.scores:
            problems.append("chosen provider must appear in scores")
        return problems


# --------------------------------------------------------------------------- #
# ExecutionAttempt — the audit unit: one provider call and everything
# that happened to it (SADD-002; new in Phase 2.1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionAttempt(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "attempt_id", "run_id", "conversation_id", "provider", "agent_name",
        "task_id", "subtask_id", "attempt_number", "status", "started_at",
        "finished_at", "latency_ms", "failures", "recoveries", "failovers",
        "screenshots", "dom_snapshots", "result_metadata",
        # Phase 2.1.3 observability extension (all optional — old payloads
        # without these keys parse to the defaults; backward compatible):
        "execution_stage", "provider_name", "model_name", "workflow_name",
        "retry_number", "parent_attempt_id", "prompt_character_count",
        "response_character_count", "response_present", "failure_present",
        "debug_notes", "metadata_version", "future_screenshot_path",
        "future_dom_snapshot_path", "future_failure_evidence",
    )

    attempt_id: AttemptId
    provider: ProviderId
    started_at: Timestamp
    run_id: Optional[RunId] = None
    conversation_id: Optional[ConversationId] = None
    agent_name: Optional[str] = None
    task_id: Optional[str] = None
    subtask_id: Optional[str] = None
    attempt_number: int = 1
    status: AttemptStatus = AttemptStatus.PENDING
    finished_at: Optional[Timestamp] = None
    latency_ms: Optional[float] = None
    failures: tuple[FailureEvent, ...] = ()          # failure history
    recoveries: tuple[RecoveryOutcome, ...] = ()     # retry/recovery history
    failovers: tuple[FailoverDecision, ...] = ()     # failover history
    screenshots: tuple[str, ...] = ()                # opaque refs
    dom_snapshots: tuple[str, ...] = ()              # opaque refs
    result_metadata: dict[str, object] = field(default_factory=dict)
    # ---- Phase 2.1.3: observability fields (additive) ------------------- #
    execution_stage: ExecutionStage = ExecutionStage.UNKNOWN
    provider_name: str = ""                          # display name; key is `provider`
    model_name: str = ""
    workflow_name: Optional[str] = None
    retry_number: int = 0
    parent_attempt_id: Optional[AttemptId] = None
    prompt_character_count: int = 0
    response_character_count: int = 0
    response_present: bool = False
    failure_present: bool = False
    debug_notes: tuple[str, ...] = ()
    metadata_version: int = 1
    # Reserved placeholders (SADD-002 later phases; never populated in 2.1.3):
    future_screenshot_path: Optional[str] = None
    future_dom_snapshot_path: Optional[str] = None
    future_failure_evidence: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    # -- pure-copy helpers (immutability-preserving; no logic, no I/O) ------ #
    def with_status(
        self,
        status: AttemptStatus,
        finished_at: Optional[Timestamp] = None,
        latency_ms: Optional[float] = None,
    ) -> "ExecutionAttempt":
        return replace(
            self, status=status,
            finished_at=finished_at if finished_at is not None else self.finished_at,
            latency_ms=latency_ms if latency_ms is not None else self.latency_ms,
        )

    def with_failure(self, event: FailureEvent) -> "ExecutionAttempt":
        return replace(self, failures=self.failures + (event,), failure_present=True)

    def with_stage(self, stage: ExecutionStage) -> "ExecutionAttempt":
        return replace(self, execution_stage=stage)

    def with_response(self, character_count: int) -> "ExecutionAttempt":
        return replace(
            self,
            response_character_count=max(0, character_count),
            response_present=character_count > 0,
        )

    def with_note(self, note: str) -> "ExecutionAttempt":
        return replace(self, debug_notes=self.debug_notes + (note,))

    def with_recovery(self, outcome: RecoveryOutcome) -> "ExecutionAttempt":
        return replace(self, recoveries=self.recoveries + (outcome,))

    def with_failover(self, decision: FailoverDecision) -> "ExecutionAttempt":
        return replace(self, failovers=self.failovers + (decision,))

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "attempt_number": self.attempt_number,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "failures": [f.to_dict() for f in self.failures],
            "recoveries": [r.to_dict() for r in self.recoveries],
            "failovers": [f.to_dict() for f in self.failovers],
            "screenshots": list(self.screenshots),
            "dom_snapshots": list(self.dom_snapshots),
            "result_metadata": dict(self.result_metadata),
            "execution_stage": self.execution_stage.value,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "workflow_name": self.workflow_name,
            "retry_number": self.retry_number,
            "parent_attempt_id": self.parent_attempt_id,
            "prompt_character_count": self.prompt_character_count,
            "response_character_count": self.response_character_count,
            "response_present": self.response_present,
            "failure_present": self.failure_present,
            "debug_notes": list(self.debug_notes),
            "metadata_version": self.metadata_version,
            "future_screenshot_path": self.future_screenshot_path,
            "future_dom_snapshot_path": self.future_dom_snapshot_path,
            "future_failure_evidence": dict(self.future_failure_evidence),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExecutionAttempt":
        extra = _extra_of(data, cls._KNOWN)
        raw_status = data.get("status", "pending")
        status = parse_enum(AttemptStatus, raw_status)
        if status is AttemptStatus.UNKNOWN and str(raw_status) != "unknown":
            extra["status__raw"] = raw_status
        raw_stage = data.get("execution_stage", "unknown")
        stage = parse_enum(ExecutionStage, raw_stage)
        if stage is ExecutionStage.UNKNOWN and str(raw_stage) != "unknown":
            extra["execution_stage__raw"] = raw_stage
        run = _opt_str(data, "run_id")
        conv = _opt_str(data, "conversation_id")
        finished = _opt_str(data, "finished_at")
        parent = _opt_str(data, "parent_attempt_id")
        failures = data.get("failures")
        recoveries = data.get("recoveries")
        failovers = data.get("failovers")
        meta = data.get("result_metadata")
        evidence = data.get("future_failure_evidence")
        return cls(
            attempt_id=AttemptId(str(data.get("attempt_id", ""))),
            provider=ProviderId(str(data.get("provider", ""))),
            started_at=Timestamp(str(data.get("started_at", ""))),
            run_id=RunId(run) if run else None,
            conversation_id=ConversationId(conv) if conv else None,
            agent_name=_opt_str(data, "agent_name"),
            task_id=_opt_str(data, "task_id"),
            subtask_id=_opt_str(data, "subtask_id"),
            attempt_number=int(data.get("attempt_number", 1) or 1),
            status=status,
            finished_at=Timestamp(finished) if finished else None,
            latency_ms=_opt_float(data, "latency_ms"),
            failures=tuple(FailureEvent.from_dict(f) for f in failures if isinstance(f, dict))
            if isinstance(failures, (list, tuple)) else (),
            recoveries=tuple(RecoveryOutcome.from_dict(r) for r in recoveries if isinstance(r, dict))
            if isinstance(recoveries, (list, tuple)) else (),
            failovers=tuple(FailoverDecision.from_dict(f) for f in failovers if isinstance(f, dict))
            if isinstance(failovers, (list, tuple)) else (),
            screenshots=_str_tuple(data, "screenshots"),
            dom_snapshots=_str_tuple(data, "dom_snapshots"),
            result_metadata=dict(meta) if isinstance(meta, dict) else {},
            execution_stage=stage,
            provider_name=str(data.get("provider_name", "")),
            model_name=str(data.get("model_name", "")),
            workflow_name=_opt_str(data, "workflow_name"),
            retry_number=int(data.get("retry_number", 0) or 0),
            parent_attempt_id=AttemptId(parent) if parent else None,
            prompt_character_count=int(data.get("prompt_character_count", 0) or 0),
            response_character_count=int(data.get("response_character_count", 0) or 0),
            response_present=bool(data.get("response_present", False)),
            failure_present=bool(data.get("failure_present", False)),
            debug_notes=_str_tuple(data, "debug_notes"),
            metadata_version=int(data.get("metadata_version", 1) or 1),
            future_screenshot_path=_opt_str(data, "future_screenshot_path"),
            future_dom_snapshot_path=_opt_str(data, "future_dom_snapshot_path"),
            future_failure_evidence=dict(evidence) if isinstance(evidence, dict) else {},
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "attempt_id", self.attempt_id)
        _validate_id(problems, "provider", self.provider)
        if not _is_timestamp(self.started_at):
            problems.append("started_at must be ISO-8601 UTC (…Z)")
        if self.finished_at is not None and not _is_timestamp(self.finished_at):
            problems.append("finished_at must be ISO-8601 UTC (…Z)")
        if self.attempt_number < 1:
            problems.append("attempt_number must be >= 1")
        if self.latency_ms is not None and self.latency_ms < 0:
            problems.append("latency_ms must be >= 0")
        if self.retry_number < 0:
            problems.append("retry_number must be >= 0")
        for name in ("prompt_character_count", "response_character_count",
                     "metadata_version"):
            if int(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        terminal = (
            AttemptStatus.SUCCEEDED, AttemptStatus.FAILED,
            AttemptStatus.DELEGATED, AttemptStatus.CANCELLED,
        )
        if self.status in terminal and self.finished_at is None:
            problems.append("terminal attempts must have finished_at")
        for f in self.failures:
            problems.extend(f.validate())
        for r in self.recoveries:
            problems.extend(r.validate())
        for d in self.failovers:
            problems.extend(d.validate())
        return problems


# --------------------------------------------------------------------------- #
# ProviderSnapshot — the registry's composed read model (SADD-002 §5.6)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderSnapshot(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "provider", "display_name", "tier", "capabilities", "max_prompt_chars",
        "health", "confidence", "statistics", "logged_in", "tab_open",
        "worker_alive", "recent_failures", "snapshot_at",
    )

    provider: ProviderId
    snapshot_at: Timestamp
    display_name: str = ""
    tier: str = ""
    capabilities: dict[str, float] = field(default_factory=dict)
    max_prompt_chars: Optional[int] = None
    health: HealthState = HealthState.UNKNOWN
    confidence: float = 0.0
    statistics: ProviderStatistics = field(default_factory=ProviderStatistics)
    logged_in: Optional[bool] = None
    tab_open: Optional[bool] = None
    worker_alive: Optional[bool] = None
    recent_failures: tuple[FailureEvent, ...] = ()
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "provider": self.provider,
            "snapshot_at": self.snapshot_at,
            "display_name": self.display_name,
            "tier": self.tier,
            "capabilities": dict(self.capabilities),
            "max_prompt_chars": self.max_prompt_chars,
            "health": self.health.value,
            "confidence": self.confidence,
            "statistics": self.statistics.to_dict(),
            "logged_in": self.logged_in,
            "tab_open": self.tab_open,
            "worker_alive": self.worker_alive,
            "recent_failures": [f.to_dict() for f in self.recent_failures],
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProviderSnapshot":
        extra = _extra_of(data, cls._KNOWN)
        raw_health = data.get("health", "unknown")
        health = parse_enum(HealthState, raw_health)
        if health is HealthState.UNKNOWN and str(raw_health) != "unknown":
            extra["health__raw"] = raw_health
        caps = data.get("capabilities")
        stats = data.get("statistics")
        failures = data.get("recent_failures")
        mpc = data.get("max_prompt_chars")

        def opt_bool(key: str) -> Optional[bool]:
            value = data.get(key)
            return bool(value) if isinstance(value, bool) else None

        return cls(
            provider=ProviderId(str(data.get("provider", ""))),
            snapshot_at=Timestamp(str(data.get("snapshot_at", ""))),
            display_name=str(data.get("display_name", "")),
            tier=str(data.get("tier", "")),
            capabilities={str(k): float(v) for k, v in caps.items()}
            if isinstance(caps, dict) else {},
            max_prompt_chars=int(mpc) if isinstance(mpc, (int, float)) else None,
            health=health,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            statistics=ProviderStatistics.from_dict(stats) if isinstance(stats, dict)
            else ProviderStatistics(),
            logged_in=opt_bool("logged_in"),
            tab_open=opt_bool("tab_open"),
            worker_alive=opt_bool("worker_alive"),
            recent_failures=tuple(
                FailureEvent.from_dict(f) for f in failures if isinstance(f, dict)
            ) if isinstance(failures, (list, tuple)) else (),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "provider", self.provider)
        if not _is_timestamp(self.snapshot_at):
            problems.append("snapshot_at must be ISO-8601 UTC (…Z)")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be within [0, 1]")
        if self.max_prompt_chars is not None and self.max_prompt_chars <= 0:
            problems.append("max_prompt_chars must be > 0 when set")
        problems.extend(self.statistics.validate())
        return problems


# --------------------------------------------------------------------------- #
# CheckpointReference — pointer to one record in a run's checkpoint log
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckpointReference(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("run_id", "record_id", "kind", "sequence", "created_at", "path")

    run_id: RunId
    record_id: CheckpointRecordId
    kind: CheckpointKind
    created_at: Timestamp
    sequence: int = 0
    path: Optional[str] = None                 # storage location hint (later phase)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "record_id": self.record_id,
            "kind": self.kind.value,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "path": self.path,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CheckpointReference":
        extra = _extra_of(data, cls._KNOWN)
        raw_kind = data.get("kind", "unknown")
        kind = parse_enum(CheckpointKind, raw_kind)
        if kind is CheckpointKind.UNKNOWN and str(raw_kind) != "unknown":
            extra["kind__raw"] = raw_kind
        return cls(
            run_id=RunId(str(data.get("run_id", ""))),
            record_id=CheckpointRecordId(str(data.get("record_id", ""))),
            kind=kind,
            created_at=Timestamp(str(data.get("created_at", ""))),
            sequence=int(data.get("sequence", 0) or 0),
            path=_opt_str(data, "path"),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "run_id", self.run_id)
        _validate_id(problems, "record_id", self.record_id)
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        if self.sequence < 0:
            problems.append("sequence must be >= 0")
        return problems
