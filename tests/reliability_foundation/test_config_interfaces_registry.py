"""Configuration defaults/round-trips, Protocol conformance (typing), and
the static extension registry."""

from __future__ import annotations

import threading
import unittest
from typing import Any, Optional, Sequence

from dozen.context.domain.types import ProviderId, RunId, Timestamp
from dozen.context.utils import generate_ulid, utc_now_iso
from dozen.reliability import (
    AttemptId,
    AttemptStatus,
    CheckpointKind,
    CheckpointRecordId,
    CheckpointReference,
    CheckpointStore,
    DuplicateRegistrationError,
    ExecutionAttempt,
    ExecutionRecorder,
    FailoverDecision,
    FailoverEngine,
    FailureDescriptor,
    FailureDetector,
    FailureEvent,
    FailureEventId,
    FailureType,
    HealthManager,
    HealthMonitor,
    HealthRecord,
    HealthState,
    ProbeResult,
    ProviderRegistry,
    ProviderSnapshot,
    RecoveryAction,
    RecoveryEngine,
    RecoveryOutcome,
    RecoveryPlan,
    RecoveryPlanId,
    RecoveryStatus,
    RecoveryStep,
    RecoveryStrategy,
    ReliabilityClient,
    ReliabilityConfig,
    ReliabilityRegistry,
    default_registry,
)
from dozen.reliability.config import HealthConfig, RecoveryConfig

NOW = Timestamp(utc_now_iso())


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(ReliabilityConfig.defaults().validate(), [])

    def test_default_values_match_sadd(self) -> None:
        cfg = ReliabilityConfig.defaults()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.detection.stall_window_s, 45.0)
        self.assertEqual(cfg.detection.min_confidence, 0.5)
        self.assertEqual(cfg.recovery.per_attempt_budget, 2)
        self.assertEqual(cfg.recovery.per_run_budget, 12)
        self.assertEqual(cfg.health.quarantine_ladder_s, [30.0, 60.0, 120.0, 300.0, 600.0])
        self.assertEqual(cfg.failover.health_multipliers["healthy"], 1.0)
        self.assertEqual(cfg.failover.health_multipliers["degraded"], 0.7)
        self.assertEqual(cfg.failover.affinity_penalty, 0.5)
        self.assertEqual(cfg.checkpoint.root_path, ".runs")
        self.assertEqual(cfg.checkpoint.retention_runs, 20)

    def test_round_trip(self) -> None:
        cfg = ReliabilityConfig.defaults()
        cfg.enabled = False
        cfg.recovery.per_run_budget = 5
        cfg.health.monitor_cadence_s["healthy"] = 60.0
        again = ReliabilityConfig.from_json(cfg.to_json())
        self.assertEqual(again.to_dict(), cfg.to_dict())
        self.assertFalse(again.enabled)
        self.assertEqual(again.recovery.per_run_budget, 5)

    def test_validation_catches_bad_values(self) -> None:
        bad_health = HealthConfig(ewma_alpha=0.0, degraded_after=0,
                                  quarantine_ladder_s=[60.0, 30.0])
        problems = bad_health.validate()
        self.assertTrue(any("ewma_alpha" in p for p in problems))
        self.assertTrue(any("degraded_after" in p for p in problems))
        self.assertTrue(any("non-decreasing" in p for p in problems))
        bad_recovery = RecoveryConfig(per_attempt_budget=-1,
                                      action_timeouts_s={"retry": 0.0})
        problems = bad_recovery.validate()
        self.assertTrue(any("per_attempt_budget" in p for p in problems))
        self.assertTrue(any("action_timeouts_s[retry]" in p for p in problems))

    def test_unknown_config_fields_survive(self) -> None:
        data = ReliabilityConfig.defaults().to_dict()
        data["future_subsystem"] = {"enabled": True}
        parsed = ReliabilityConfig.from_dict(data)
        self.assertEqual(parsed.to_dict()["future_subsystem"], {"enabled": True})


# --------------------------------------------------------------------------- #
# Interfaces — structural typing (runtime_checkable)
# --------------------------------------------------------------------------- #
class _FakeDetector:
    @property
    def name(self) -> str:
        return "fake"

    def detect(self, provider, raw_signal, context) -> Optional[FailureEvent]:
        return None


class _FakeStrategy:
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.RETRY

    def applies_to(self, failure_type: FailureType) -> bool:
        return True

    def execute(self, plan_step_context) -> bool:
        return True


class _FakeRecoveryEngine:
    def plan(self, event: FailureEvent) -> RecoveryPlan:
        return RecoveryPlan(id=RecoveryPlanId("p"), failure=event, created_at=NOW)

    def execute(self, plan: RecoveryPlan) -> RecoveryOutcome:
        return RecoveryOutcome(plan_id=plan.id, status=RecoveryStatus.ABORTED, finished_at=NOW)


class _FakeFailoverEngine:
    def select(self, subtask_context, exclude) -> Optional[FailoverDecision]:
        return None

    def transfer(self, call_context, decision) -> object:
        return object()


class _FakeCheckpointStore:
    def append(self, run_id, record) -> CheckpointReference:
        return CheckpointReference(
            run_id=run_id, record_id=CheckpointRecordId(generate_ulid()),
            kind=CheckpointKind.RUN_STARTED, created_at=NOW,
        )

    def read(self, run_id) -> list:
        return []

    def latest(self, run_id) -> Optional[CheckpointReference]:
        return None

    def list_runs(self) -> list:
        return []


class _FakeHealthManager:
    def record_attempt(self, attempt) -> Optional[HealthState]:
        return None

    def record_probe(self, probe) -> Optional[HealthState]:
        return None

    def state(self, provider) -> HealthState:
        return HealthState.UNKNOWN

    def snapshot(self, provider) -> HealthRecord:
        return HealthRecord(provider=provider, state=HealthState.UNKNOWN, updated_at=NOW)

    def routable_providers(self) -> list:
        return []

    def begin_recovery(self, provider) -> None: ...
    def end_recovery(self, provider, success) -> None: ...
    def mark_needs_human(self, provider, reason) -> None: ...
    def human_resolved(self, provider) -> None: ...


class _FakeHealthMonitor:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def probe_now(self, provider) -> ProbeResult:
        return ProbeResult(provider=provider, probed_at=NOW)

    def set_cadence(self, state, seconds) -> None: ...


class _FakeProviderRegistry:
    def get(self, provider) -> Optional[ProviderSnapshot]:
        return None

    def all(self) -> list:
        return []

    def routable(self) -> list:
        return []


class _FakeClient:
    """Mirrors LLMClient.complete keyword-for-keyword."""

    def complete(self, *, provider: str, model: str, messages: Sequence[object],
                 temperature: float = 0.2, max_tokens: int = 4096, **kwargs: Any) -> object:
        return {"text": "ok"}


class _FakeRecorder:
    def begin_attempt(self, provider, run_id=None, subtask_id=None,
                      attempt_number=1) -> ExecutionAttempt:
        return ExecutionAttempt(
            attempt_id=AttemptId(generate_ulid()), provider=provider, started_at=NOW,
        )

    def record_failure(self, attempt, event) -> ExecutionAttempt:
        return attempt.with_failure(event)

    def record_recovery(self, attempt, outcome) -> ExecutionAttempt:
        return attempt.with_recovery(outcome)

    def record_failover(self, attempt, decision) -> ExecutionAttempt:
        return attempt.with_failover(decision)

    def finish(self, attempt, status, latency_ms=None,
               result_metadata=None) -> ExecutionAttempt:
        return attempt.with_status(status, finished_at=NOW, latency_ms=latency_ms)


class TestProtocolConformance(unittest.TestCase):
    def test_all_interfaces_satisfied_structurally(self) -> None:
        checks: list[tuple[object, type]] = [
            (_FakeDetector(), FailureDetector),
            (_FakeStrategy(), RecoveryStrategy),
            (_FakeRecoveryEngine(), RecoveryEngine),
            (_FakeFailoverEngine(), FailoverEngine),
            (_FakeCheckpointStore(), CheckpointStore),
            (_FakeHealthManager(), HealthManager),
            (_FakeHealthMonitor(), HealthMonitor),
            (_FakeProviderRegistry(), ProviderRegistry),
            (_FakeClient(), ReliabilityClient),
            (_FakeRecorder(), ExecutionRecorder),
        ]
        for instance, port in checks:
            with self.subTest(port=port.__name__):
                self.assertIsInstance(instance, port)

    def test_real_llm_client_already_satisfies_the_client_port(self) -> None:
        """The decorator seam is real: today's LLMClient shape conforms."""
        from dozen.llm_client import LLMClient
        self.assertIsInstance(LLMClient(mock=True), ReliabilityClient)

    def test_recorder_contract_is_pure(self) -> None:
        recorder = _FakeRecorder()
        first = recorder.begin_attempt(ProviderId("anthropic"), run_id=RunId("r1"))
        second = recorder.finish(first, AttemptStatus.SUCCEEDED, latency_ms=10.0)
        self.assertIsNot(first, second)
        self.assertEqual(first.status, AttemptStatus.PENDING)  # original untouched
        self.assertEqual(second.status, AttemptStatus.SUCCEEDED)


# --------------------------------------------------------------------------- #
# Registry — static registration only
# --------------------------------------------------------------------------- #
class TestReliabilityRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ReliabilityRegistry()

    def test_register_and_read_back_all_kinds(self) -> None:
        detector, strategy = _FakeDetector(), _FakeStrategy()
        self.registry.register_detector("timeout-exceptions", detector)
        self.registry.register_strategy("retry-fresh-chat", strategy)
        self.registry.register_recovery_action("retry", RecoveryAction.RETRY)
        descriptor = FailureDescriptor(
            name="timeout", failure_type=FailureType.TIMEOUT,
            description="no response within deadline",
            default_actions=(RecoveryAction.RETRY, RecoveryAction.NEW_CHAT,
                             RecoveryAction.DELEGATE),
        )
        self.registry.register_failure(descriptor)

        self.assertIs(self.registry.detector("timeout-exceptions"), detector)
        self.assertIs(self.registry.strategy("retry-fresh-chat"), strategy)
        self.assertIs(self.registry.recovery_action("retry"), RecoveryAction.RETRY)
        self.assertEqual(self.registry.failure("timeout"), descriptor)
        self.assertEqual(self.registry.snapshot(), {
            "detectors": ["timeout-exceptions"],
            "strategies": ["retry-fresh-chat"],
            "recovery_actions": ["retry"],
            "failures": ["timeout"],
        })

    def test_duplicate_registration_rejected_unless_replace(self) -> None:
        self.registry.register_detector("d", _FakeDetector())
        with self.assertRaises(DuplicateRegistrationError):
            self.registry.register_detector("d", _FakeDetector())
        replacement = _FakeDetector()
        self.registry.register_detector("d", replacement, replace=True)
        self.assertIs(self.registry.detector("d"), replacement)

    def test_invalid_names_and_types_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.register_strategy("   ", _FakeStrategy())
        with self.assertRaises(TypeError):
            self.registry.register_recovery_action("bad", "not-an-action")  # type: ignore[arg-type]

    def test_future_failure_kind_without_enum_member(self) -> None:
        custom = FailureDescriptor(name="quota_exceeded",
                                   description="daily quota exhausted")
        self.registry.register_failure(custom)
        stored = self.registry.failure("quota_exceeded")
        self.assertIs(stored.failure_type, FailureType.UNKNOWN)  # until the enum catches up

    def test_reads_return_copies(self) -> None:
        self.registry.register_recovery_action("retry", RecoveryAction.RETRY)
        table = self.registry.recovery_actions()
        table["injected"] = RecoveryAction.ABORT
        self.assertIsNone(self.registry.recovery_action("injected"))

    def test_thread_safety_smoke(self) -> None:
        errors: list[BaseException] = []

        def hammer(seed: int) -> None:
            try:
                for i in range(100):
                    name = f"d-{seed}-{i}"
                    self.registry.register_detector(name, _FakeDetector())
                    assert self.registry.detector(name) is not None
                    self.registry.snapshot()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.registry.detectors()), 800)

    def test_default_registry_is_a_singleton_and_empty(self) -> None:
        self.assertIs(default_registry(), default_registry())
        # Phase 2.1.1 registers nothing at import time — no runtime behavior.
        snapshot = default_registry().snapshot()
        self.assertEqual(sum(len(v) for v in snapshot.values()), 0)


if __name__ == "__main__":
    unittest.main()
