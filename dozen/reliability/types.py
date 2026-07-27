"""Reliability Layer — enums and branded id types (SADD-002, Phase 2.1.1).

Every enum is a ``str`` enum with an ``UNKNOWN`` member, mirroring the
foundation-layer convention: values written by a *future* schema must never
crash this one. ``parse_enum`` maps unrecognized raw values to ``UNKNOWN``
while callers preserve the raw string (forward compatibility).
"""

from __future__ import annotations

from enum import Enum
from typing import NewType, Type, TypeVar

# --------------------------------------------------------------------------- #
# Branded ids (shared vocabulary with dozen.context re-used where it exists:
# ConversationId / RunId / ProviderId / Timestamp come from the foundation).
# --------------------------------------------------------------------------- #
AttemptId = NewType("AttemptId", str)
FailureEventId = NewType("FailureEventId", str)
RecoveryPlanId = NewType("RecoveryPlanId", str)
CheckpointRecordId = NewType("CheckpointRecordId", str)

E = TypeVar("E", bound=Enum)


def parse_enum(enum_cls: Type[E], value: object) -> E:
    """Parse a member; unknown raw values become the enum's UNKNOWN member."""
    try:
        return enum_cls(str(value))
    except ValueError:
        return enum_cls("unknown")  # every reliability enum defines UNKNOWN


# --------------------------------------------------------------------------- #
# 1. Failure taxonomy (SADD-002 §7)
# --------------------------------------------------------------------------- #
class FailureType(str, Enum):
    TIMEOUT = "timeout"
    GENERATION_STALLED = "generation_stalled"
    DOM_CHANGED = "dom_changed"
    RATE_LIMIT = "rate_limit"
    MODEL_BUSY = "model_busy"
    LOGIN_REQUIRED = "login_required"
    CAPTCHA = "captcha"
    NETWORK_ERROR = "network_error"
    BROWSER_CRASH = "browser_crash"
    TAB_CLOSED = "tab_closed"
    PROMPT_REJECTED = "prompt_rejected"
    OUTPUT_CORRUPTED = "output_corrupted"
    UNEXPECTED_UI = "unexpected_ui"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# 2. Provider health states (SADD-002 §5.1)
# --------------------------------------------------------------------------- #
class HealthState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SUSPECT = "suspect"
    RECOVERING = "recovering"
    QUARANTINED = "quarantined"
    NEEDS_HUMAN = "needs_human"
    OFFLINE = "offline"


# --------------------------------------------------------------------------- #
# 3. Recovery actions (SADD-002 §5.4)
# --------------------------------------------------------------------------- #
class RecoveryAction(str, Enum):
    RETRY = "retry"
    REFRESH = "refresh"
    NEW_CHAT = "new_chat"
    RECOVER_TAB = "recover_tab"
    RESTART_WORKER = "restart_worker"
    REAUTH = "reauth"
    WAIT = "wait"
    SHRINK_PROMPT = "shrink_prompt"
    DELEGATE = "delegate"
    ESCALATE = "escalate"
    ABORT = "abort"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Supporting enums required by the models (same forward-compat rules)
# --------------------------------------------------------------------------- #
class RecoveryStatus(str, Enum):
    """Terminal verdict of one executed RecoveryPlan (SADD-002 §5.4)."""

    RECOVERED = "recovered"
    DELEGATED = "delegated"
    ESCALATED = "escalated"
    ABORTED = "aborted"
    EXHAUSTED = "exhausted"      # budget spent, nothing worked
    UNKNOWN = "unknown"


class AttemptStatus(str, Enum):
    """Lifecycle status of one ExecutionAttempt."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DELEGATED = "delegated"      # completed by another provider via failover
    CANCELLED = "cancelled"      # user pressed Stop — never a failure
    UNKNOWN = "unknown"


class ExecutionStage(str, Enum):
    """Which pipeline component produced an attempt (Phase 2.1.3)."""

    PLANNER = "planner"
    WORKER = "worker"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"
    REPAIR = "repair"
    UNKNOWN = "unknown"


class CheckpointKind(str, Enum):
    """Record kinds of the event-sourced run log (SADD-002 §5.7)."""

    RUN_STARTED = "run_started"
    PLAN_READY = "plan_ready"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    FAILURE = "failure"
    RECOVERY = "recovery"
    FAILOVER = "failover"
    SYNTHESIS_STARTED = "synthesis_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    UNKNOWN = "unknown"
