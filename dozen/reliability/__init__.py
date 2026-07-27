"""Reliability Layer — foundation (SADD-002, Phase 2.1.1).

Contracts and shared models ONLY. Nothing in this package detects, recovers,
probes, retries, delegates, checkpoints, or touches a browser — those arrive
in Phases 2.2+ and must build on these types without modifying them.

Import rule: this package may import ``dozen.context`` (shared id types and,
in later phases, its EventBus and storage engine). It must NEVER import
Playwright, ``webllm``, or orchestrator pipeline modules — browser interaction
always goes through interfaces defined here and implemented elsewhere.

Phase 2.1.2 adds PASSIVE execution recording: build_orchestrator wraps the
web client in the transparent ReliabilityClientDecorator, which records one
ExecutionAttempt per complete() call into an in-memory ring buffer. Nothing
reads the ring yet; results, exceptions and attributes pass through
untouched — behavior remains bit-for-bit identical.
"""

from __future__ import annotations

from .config import (
    CheckpointConfig,
    DetectionConfig,
    FailoverConfig,
    HealthConfig,
    RecoveryConfig,
    ReliabilityConfig,
)
from .interfaces import (
    CheckpointStore,
    ExecutionRecorder,
    FailoverEngine,
    FailureDetector,
    HealthManager,
    HealthMonitor,
    ProviderRegistry,
    RecoveryEngine,
    RecoveryStrategy,
    ReliabilityClient,
)
from .models import (
    CheckpointReference,
    ExecutionAttempt,
    FailoverDecision,
    FailureEvent,
    HealthRecord,
    ProbeResult,
    ProviderSnapshot,
    ProviderStatistics,
    RecoveryOutcome,
    RecoveryPlan,
    RecoveryStep,
)
from .attempt_factory import AttemptFactory
from .classifier import FailureClassification, FailureClassifier, MatchedRule
from .client import ReliabilityClientDecorator, infer_execution_stage, wrap_client
from .evidence import FailureEvidence
from .health import (
    HealthTransition,
    PassiveHealthManager,
    ProviderHealthRecord,
    default_health_manager,
)
from .metrics import FAILURE_SEVERITY, WindowMetrics, severity_weight
from .rules import DEFAULT_RULES, ClassificationRule
from .scoring import HealthScorer, ScoreBreakdown
from .windows import Observation, RollingWindow, WindowSet, WindowTotals
from .clock import ReliabilityClock, SystemReliabilityClock
from .config import ObservabilityConfig
from .debug import DebugApiDisabled, ReliabilityDebugApi, attach_attempt_logging
from .events import ReliabilityEvents, default_reliability_events
from .metadata import ProviderMetadata, TimingMetadata, WorkflowMetadata
from .query import AttemptQuery
from .recorder import InMemoryExecutionRecorder, default_recorder
from .viewmodels import (
    AttemptDetails,
    AttemptSummary,
    ProviderSummary,
    StageSummary,
    StatisticsSummary,
)
from .registry import (
    DuplicateRegistrationError,
    FailureDescriptor,
    ReliabilityRegistry,
    default_registry,
)
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

__all__ = [
    # types
    "AttemptId", "AttemptStatus", "CheckpointKind", "CheckpointRecordId",
    "FailureEventId", "FailureType", "HealthState", "RecoveryAction",
    "RecoveryPlanId", "RecoveryStatus", "parse_enum",
    # models
    "CheckpointReference", "ExecutionAttempt", "FailoverDecision",
    "FailureEvent", "HealthRecord", "ProbeResult", "ProviderSnapshot",
    "ProviderStatistics", "RecoveryOutcome", "RecoveryPlan", "RecoveryStep",
    # config
    "CheckpointConfig", "DetectionConfig", "FailoverConfig", "HealthConfig",
    "RecoveryConfig", "ReliabilityConfig",
    # interfaces
    "CheckpointStore", "ExecutionRecorder", "FailoverEngine", "FailureDetector",
    "HealthManager", "HealthMonitor", "ProviderRegistry", "RecoveryEngine",
    "RecoveryStrategy", "ReliabilityClient",
    # registry
    "DuplicateRegistrationError", "FailureDescriptor", "ReliabilityRegistry",
    "default_registry",
    # recording (Phase 2.1.2)
    "AttemptFactory", "InMemoryExecutionRecorder", "ProviderMetadata",
    "ReliabilityClientDecorator", "ReliabilityClock", "SystemReliabilityClock",
    "TimingMetadata", "WorkflowMetadata", "default_recorder", "wrap_client",
    # observability (Phase 2.1.3)
    "AttemptDetails", "AttemptQuery", "AttemptSummary", "DebugApiDisabled",
    "ExecutionStage", "ObservabilityConfig", "ProviderSummary",
    "ReliabilityDebugApi", "ReliabilityEvents", "StageSummary",
    "StatisticsSummary", "attach_attempt_logging", "default_reliability_events",
    "infer_execution_stage",
    # classification (Phase 2.2.1)
    "ClassificationRule", "DEFAULT_RULES", "FailureClassification",
    "FailureClassifier", "FailureEvidence", "MatchedRule",
    # passive health (Phase 2.2.2)
    "FAILURE_SEVERITY", "HealthScorer", "HealthTransition", "Observation",
    "PassiveHealthManager", "ProviderHealthRecord", "RollingWindow",
    "ScoreBreakdown", "WindowMetrics", "WindowSet", "WindowTotals",
    "default_health_manager", "severity_weight",
]
