"""Reliability models: creation, validation, immutability, serialization
round-trips, forward compatibility, and the ExecutionAttempt audit trail."""

from __future__ import annotations

import dataclasses
import unittest

from dozen.context.domain.types import ConversationId, ProviderId, RunId, Timestamp
from dozen.context.utils import generate_ulid, utc_now_iso
from dozen.reliability import (
    AttemptId,
    AttemptStatus,
    CheckpointKind,
    CheckpointRecordId,
    CheckpointReference,
    ExecutionAttempt,
    FailoverDecision,
    FailureEvent,
    FailureEventId,
    FailureType,
    HealthRecord,
    HealthState,
    ProbeResult,
    ProviderSnapshot,
    ProviderStatistics,
    RecoveryAction,
    RecoveryOutcome,
    RecoveryPlan,
    RecoveryPlanId,
    RecoveryStatus,
    RecoveryStep,
)

NOW = Timestamp(utc_now_iso())
CLAUDE = ProviderId("anthropic")


def make_failure(failure_type: FailureType = FailureType.TIMEOUT) -> FailureEvent:
    return FailureEvent(
        id=FailureEventId(generate_ulid()),
        provider=CLAUDE,
        failure_type=failure_type,
        detected_at=NOW,
        confidence=0.7,
        message="Timed out waiting for a response",
        raw_signal="ProviderError('Timed out waiting for a fresh response.')",
        evidence=("observer fired: no", "dom probe: logged-in ok"),
        run_id=RunId("run-1"),
        subtask_id="s1",
    )


def make_plan(failure: FailureEvent | None = None) -> RecoveryPlan:
    return RecoveryPlan(
        id=RecoveryPlanId(generate_ulid()),
        failure=failure or make_failure(),
        created_at=NOW,
        steps=(
            RecoveryStep(action=RecoveryAction.RETRY, order=0, note="fresh chat"),
            RecoveryStep(action=RecoveryAction.NEW_CHAT, order=1),
            RecoveryStep(action=RecoveryAction.DELEGATE, order=2),
        ),
        budget=2,
    )


class TestCreationAndValidation(unittest.TestCase):
    def test_valid_failure_event(self) -> None:
        self.assertEqual(make_failure().validate(), [])

    def test_invalid_failure_event_reports_problems(self) -> None:
        bad = FailureEvent(
            id=FailureEventId(""), provider=ProviderId(""),
            failure_type=FailureType.CAPTCHA, detected_at=Timestamp("yesterday"),
            confidence=1.5,
        )
        problems = bad.validate()
        self.assertTrue(any("id" in p for p in problems))
        self.assertTrue(any("provider" in p for p in problems))
        self.assertTrue(any("detected_at" in p for p in problems))
        self.assertTrue(any("confidence" in p for p in problems))

    def test_valid_plan_and_step_ordering(self) -> None:
        self.assertEqual(make_plan().validate(), [])
        shuffled = RecoveryPlan(
            id=RecoveryPlanId(generate_ulid()), failure=make_failure(), created_at=NOW,
            steps=(
                RecoveryStep(action=RecoveryAction.DELEGATE, order=2),
                RecoveryStep(action=RecoveryAction.RETRY, order=0),
            ),
        )
        self.assertTrue(any("ordered" in p for p in shuffled.validate()))

    def test_recovered_outcome_must_name_action(self) -> None:
        nameless = RecoveryOutcome(
            plan_id=RecoveryPlanId(generate_ulid()),
            status=RecoveryStatus.RECOVERED, finished_at=NOW,
        )
        self.assertTrue(any("succeeded_action" in p for p in nameless.validate()))
        named = dataclasses.replace(nameless, succeeded_action=RecoveryAction.RETRY)
        self.assertEqual(named.validate(), [])

    def test_statistics_sanity_rules(self) -> None:
        bad = ProviderStatistics(attempts_1h=3, successes_1h=5, latency_ewma_ms=-1)
        problems = bad.validate()
        self.assertTrue(any("successes_1h" in p for p in problems))
        self.assertTrue(any("latency_ewma_ms" in p for p in problems))
        good = ProviderStatistics(attempts_1h=10, successes_1h=9)
        self.assertEqual(good.validate(), [])
        self.assertAlmostEqual(good.success_rate_1h, 0.9)
        self.assertEqual(ProviderStatistics().success_rate_1h, 0.0)  # no div-by-zero

    def test_failover_decision_rules(self) -> None:
        self_target = FailoverDecision(
            from_provider=CLAUDE, decided_at=NOW, to_provider=CLAUDE,
            scores={"anthropic": 1.0},
        )
        self.assertTrue(any("differ" in p for p in self_target.validate()))
        unscored = FailoverDecision(
            from_provider=CLAUDE, decided_at=NOW, to_provider=ProviderId("openai"),
            scores={"deepseek": 0.5},
        )
        self.assertTrue(any("scores" in p for p in unscored.validate()))
        nobody_left = FailoverDecision(from_provider=CLAUDE, decided_at=NOW)
        self.assertEqual(nobody_left.validate(), [])  # None target is legal

    def test_terminal_attempt_needs_finished_at(self) -> None:
        attempt = ExecutionAttempt(
            attempt_id=AttemptId(generate_ulid()), provider=CLAUDE,
            started_at=NOW, status=AttemptStatus.SUCCEEDED,
        )
        self.assertTrue(any("finished_at" in p for p in attempt.validate()))


class TestImmutability(unittest.TestCase):
    def test_models_are_frozen(self) -> None:
        for model in (
            make_failure(), make_plan(),
            ProbeResult(provider=CLAUDE, probed_at=NOW),
            HealthRecord(provider=CLAUDE, state=HealthState.HEALTHY, updated_at=NOW),
            ProviderStatistics(),
        ):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    model.provider = "mutated"  # type: ignore[misc, attr-defined]

    def test_sequences_are_tuples(self) -> None:
        failure = make_failure()
        self.assertIsInstance(failure.evidence, tuple)
        self.assertIsInstance(make_plan().steps, tuple)


class TestSerializationRoundTrip(unittest.TestCase):
    def assert_round_trip(self, model) -> None:
        again = type(model).from_dict(model.to_dict())
        self.assertEqual(again.to_dict(), model.to_dict())
        via_json = type(model).from_json(model.to_json())
        self.assertEqual(via_json.to_dict(), model.to_dict())

    def test_all_models_round_trip(self) -> None:
        failure = make_failure()
        plan = make_plan(failure)
        outcome = RecoveryOutcome(
            plan_id=plan.id, status=RecoveryStatus.RECOVERED, finished_at=NOW,
            executed_steps=plan.steps[:1], succeeded_action=RecoveryAction.RETRY,
            duration_ms=1234.5, notes=("first retry landed",),
        )
        decision = FailoverDecision(
            from_provider=CLAUDE, decided_at=NOW, to_provider=ProviderId("openai"),
            subtask_id="s1", scores={"openai": 0.86, "deepseek": 0.72},
            factors={"openai": {"base": 0.92, "health": 1.0}},
            excluded=("anthropic",), reason="LOGIN_REQUIRED on anthropic",
        )
        stats = ProviderStatistics(
            attempts_1h=20, successes_1h=19, attempts_24h=200, successes_24h=180,
            latency_ewma_ms=8000.0, latency_p95_ms=21000.0,
            failure_histogram={"timeout": 3, "captcha": 1}, last_success_at=NOW,
        )
        models = [
            failure, plan, outcome, decision, stats,
            HealthRecord(provider=CLAUDE, state=HealthState.DEGRADED, updated_at=NOW,
                         confidence=0.66, statistics=stats, state_reason="2 timeouts"),
            ProbeResult(provider=CLAUDE, probed_at=NOW, alive=True, logged_in=True,
                        captcha_present=False, latency_ms=42.0, evidence=("url ok",)),
            ProviderSnapshot(provider=CLAUDE, snapshot_at=NOW, display_name="Claude",
                             tier="frontier", capabilities={"coding": 0.95},
                             max_prompt_chars=60000, health=HealthState.HEALTHY,
                             confidence=0.9, statistics=stats, logged_in=True,
                             tab_open=True, worker_alive=True,
                             recent_failures=(failure,)),
            CheckpointReference(run_id=RunId("run-1"),
                                record_id=CheckpointRecordId(generate_ulid()),
                                kind=CheckpointKind.SUBTASK_COMPLETED,
                                created_at=NOW, sequence=4, path=".runs/run-1"),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.validate(), [])
                self.assert_round_trip(model)

    def test_unknown_fields_survive(self) -> None:
        data = make_failure().to_dict()
        data["field_from_the_future"] = {"nested": True}
        parsed = FailureEvent.from_dict(data)
        self.assertEqual(parsed.to_dict()["field_from_the_future"], {"nested": True})

    def test_unknown_enum_value_maps_to_unknown_and_keeps_raw(self) -> None:
        data = make_failure().to_dict()
        data["failure_type"] = "quota_exceeded"  # future taxonomy member
        parsed = FailureEvent.from_dict(data)
        self.assertIs(parsed.failure_type, FailureType.UNKNOWN)
        self.assertEqual(parsed.extra["failure_type__raw"], "quota_exceeded")
        self.assertEqual(parsed.to_dict()["failure_type__raw"], "quota_exceeded")


class TestExecutionAttempt(unittest.TestCase):
    """The audit unit: full-lifecycle construction stays immutable."""

    def make_attempt(self) -> ExecutionAttempt:
        return ExecutionAttempt(
            attempt_id=AttemptId(generate_ulid()),
            provider=CLAUDE,
            started_at=NOW,
            run_id=RunId("run-9"),
            conversation_id=ConversationId(generate_ulid()),
            agent_name="claude (frontier)",
            task_id="task_abc",
            subtask_id="s3",
            attempt_number=2,
            screenshots=("shots/a.png",),
            dom_snapshots=("dom/a.html",),
        )

    def test_creation_with_all_supported_fields(self) -> None:
        attempt = self.make_attempt()
        self.assertEqual(attempt.validate(), [])
        self.assertEqual(attempt.status, AttemptStatus.PENDING)
        self.assertEqual(attempt.attempt_number, 2)
        self.assertEqual(attempt.screenshots, ("shots/a.png",))

    def test_lifecycle_via_pure_copies(self) -> None:
        attempt = self.make_attempt()
        failure = make_failure(FailureType.LOGIN_REQUIRED)
        outcome = RecoveryOutcome(
            plan_id=RecoveryPlanId(generate_ulid()),
            status=RecoveryStatus.DELEGATED, finished_at=NOW,
        )
        decision = FailoverDecision(
            from_provider=CLAUDE, decided_at=NOW, to_provider=ProviderId("openai"),
            scores={"openai": 0.86},
        )
        done = (
            attempt
            .with_failure(failure)
            .with_recovery(outcome)
            .with_failover(decision)
            .with_status(AttemptStatus.DELEGATED, finished_at=NOW, latency_ms=15000.0)
        )
        # History accumulated…
        self.assertEqual(len(done.failures), 1)
        self.assertEqual(len(done.recoveries), 1)
        self.assertEqual(len(done.failovers), 1)
        self.assertEqual(done.status, AttemptStatus.DELEGATED)
        self.assertEqual(done.validate(), [])
        # …while the original is untouched (pure copies, frozen models):
        self.assertEqual(attempt.failures, ())
        self.assertEqual(attempt.status, AttemptStatus.PENDING)

    def test_full_attempt_round_trip(self) -> None:
        done = (
            self.make_attempt()
            .with_failure(make_failure())
            .with_status(AttemptStatus.FAILED, finished_at=NOW, latency_ms=90000.0)
        )
        again = ExecutionAttempt.from_dict(done.to_dict())
        self.assertEqual(again.to_dict(), done.to_dict())
        self.assertEqual(again.failures[0].failure_type, FailureType.TIMEOUT)

    def test_cancelled_status_exists_and_is_terminal(self) -> None:
        cancelled = self.make_attempt().with_status(
            AttemptStatus.CANCELLED, finished_at=NOW
        )
        self.assertEqual(cancelled.validate(), [])


if __name__ == "__main__":
    unittest.main()
