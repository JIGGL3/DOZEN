"""Phase 2.2.2 — passive provider health.

A scripted FakeClock drives every time-sensitive behavior (window expiry,
decay, idle recovery) deterministically. All health data flows in through
record_attempt with real ExecutionAttempt objects — exactly the production
path — and nothing anywhere reads health to make decisions.
"""

from __future__ import annotations

import json
import threading
import unittest

from dozen.context.domain.types import ProviderId, Timestamp
from dozen.context.utils import generate_ulid
from dozen.reliability import (
    AttemptId,
    AttemptStatus,
    ExecutionStage,
    FailureEvent,
    FailureEventId,
    FailureType,
    HealthScorer,
    HealthState,
    InMemoryExecutionRecorder,
    Observation,
    PassiveHealthManager,
    RollingWindow,
    severity_weight,
)
from dozen.reliability.config import HealthConfig
from dozen.reliability.models import ExecutionAttempt


class FakeClock:
    """Controllable time: advance() moves both monotonic and wall time."""

    def __init__(self) -> None:
        self.mono = 1000.0

    def advance(self, seconds: float) -> None:
        self.mono += seconds

    def monotonic(self) -> float:
        return self.mono

    def now(self) -> Timestamp:
        return Timestamp("2026-07-11T12:00:00.000Z")


def make_attempt(
    provider: str = "anthropic",
    status: AttemptStatus = AttemptStatus.SUCCEEDED,
    latency_ms: float = 10_000.0,
    failure_type: FailureType | None = None,
    confidence: float = 0.9,
) -> ExecutionAttempt:
    attempt = ExecutionAttempt(
        attempt_id=AttemptId(generate_ulid()),
        provider=ProviderId(provider),
        started_at=Timestamp("2026-07-11T12:00:00.000Z"),
        status=status,
        finished_at=Timestamp("2026-07-11T12:00:01.000Z"),
        latency_ms=latency_ms,
        execution_stage=ExecutionStage.WORKER,
    )
    if failure_type is not None:
        attempt = attempt.with_failure(FailureEvent(
            id=FailureEventId(generate_ulid()),
            provider=ProviderId(provider),
            failure_type=failure_type,
            detected_at=Timestamp("2026-07-11T12:00:01.000Z"),
            confidence=confidence,
        ))
    return attempt


def make_manager(clock: FakeClock | None = None, **config_overrides):
    clock = clock or FakeClock()
    config = HealthConfig(**config_overrides) if config_overrides else HealthConfig()
    return PassiveHealthManager(config=config, clock=clock), clock


def feed(mgr, clock, provider, count, status=AttemptStatus.SUCCEEDED,
         failure_type=None, latency_ms=10_000.0, gap_s=1.0, confidence=0.9):
    for _ in range(count):
        mgr.record_attempt(make_attempt(provider, status, latency_ms,
                                        failure_type, confidence))
        clock.advance(gap_s)


class TestRollingWindows(unittest.TestCase):
    def test_incremental_add_and_totals(self) -> None:
        window = RollingWindow(60.0)
        for i in range(10):
            window.add(Observation(success=i % 2 == 0, latency_ms=100.0 * (i + 1),
                                   severity_weight=0.5 if i % 2 else 0.0), now=100.0 + i)
        totals = window.totals(now=110.0)
        self.assertEqual(totals.count, 10)
        self.assertEqual(totals.successes, 5)
        self.assertEqual(totals.failures, 5)
        self.assertEqual(totals.success_rate, 0.5)
        self.assertAlmostEqual(totals.average_latency_ms, 550.0)
        self.assertIsNotNone(totals.p95_latency_ms)

    def test_window_expiration(self) -> None:
        window = RollingWindow(60.0)
        window.add(Observation(success=True), now=100.0)
        self.assertEqual(window.totals(now=100.0).count, 1)
        self.assertEqual(window.totals(now=159.0).count, 1)   # still inside
        self.assertEqual(window.totals(now=200.0).count, 0)   # naturally expired

    def test_lifetime_never_expires(self) -> None:
        window = RollingWindow(None)
        window.add(Observation(success=True), now=0.0)
        self.assertEqual(window.totals(now=10_000_000.0).count, 1)

    def test_windows_compute_independently(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 5)
        clock.advance(120.0)            # past the 1m window, inside 5m
        record = mgr.health_record("openai")
        self.assertEqual(record.rolling_windows["1m"]["observations"], 0)
        self.assertEqual(record.rolling_windows["5m"]["observations"], 5)
        self.assertEqual(record.rolling_windows["30m"]["observations"], 5)
        self.assertEqual(record.rolling_windows["lifetime"]["observations"], 5)


class TestStateTransitions(unittest.TestCase):
    def test_first_success_unknown_to_healthy(self) -> None:
        mgr, clock = make_manager()
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.UNKNOWN)
        changed = mgr.record_attempt(make_attempt("openai"))
        self.assertIs(changed, HealthState.HEALTHY)

    def test_100_successes_stay_healthy(self) -> None:
        """Manual acceptance 1."""
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 100)
        record = mgr.health_record("openai")
        self.assertIs(record.current_state, HealthState.HEALTHY)
        self.assertGreaterEqual(record.overall_score, 0.9)
        self.assertEqual(record.consecutive_successes, 100)
        self.assertEqual(record.success_rate, 1.0)

    def test_repeated_timeouts_degrade_gradually(self) -> None:
        """Manual acceptance 2: healthy -> degraded -> suspect."""
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 10)  # solid baseline
        feed(mgr, clock, "openai", 2, AttemptStatus.FAILED, FailureType.TIMEOUT)
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.DEGRADED)
        feed(mgr, clock, "openai", 2, AttemptStatus.FAILED, FailureType.TIMEOUT)
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.SUSPECT)
        # scores fell along the way, and every transition is explained:
        history = mgr.health_history("openai")
        self.assertTrue(all(t.reason for t in history))
        self.assertEqual([t.to_state for t in reversed(history)],
                         [HealthState.HEALTHY, HealthState.DEGRADED,
                          HealthState.SUSPECT])

    def test_repeated_dom_failures_reach_suspect(self) -> None:
        """Manual acceptance 3."""
        mgr, clock = make_manager()
        feed(mgr, clock, "gemini", 6)
        feed(mgr, clock, "gemini", 4, AttemptStatus.FAILED, FailureType.DOM_CHANGED)
        self.assertIs(mgr.state(ProviderId("gemini")), HealthState.SUSPECT)

    def test_repeated_login_failures_need_human(self) -> None:
        """Manual acceptance 4."""
        mgr, clock = make_manager()
        feed(mgr, clock, "copilot", 5)
        feed(mgr, clock, "copilot", 2, AttemptStatus.FAILED, FailureType.LOGIN_REQUIRED)
        self.assertIs(mgr.state(ProviderId("copilot")), HealthState.NEEDS_HUMAN)
        record = mgr.health_record("copilot")
        self.assertIn("login/captcha", record.metadata["state_reason"])
        # sticky against other failures:
        feed(mgr, clock, "copilot", 1, AttemptStatus.FAILED, FailureType.TIMEOUT)
        self.assertIs(mgr.state(ProviderId("copilot")), HealthState.NEEDS_HUMAN)

    def test_repeated_crashes_quarantine(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "grok", 5)
        feed(mgr, clock, "grok", 3, AttemptStatus.FAILED, FailureType.BROWSER_CRASH)
        self.assertIs(mgr.state(ProviderId("grok")), HealthState.QUARANTINED)
        self.assertIsNotNone(
            mgr.health_record("grok").current_quarantine_reason
        )

    def test_success_streak_recovers(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 5)
        feed(mgr, clock, "openai", 3, AttemptStatus.FAILED, FailureType.BROWSER_CRASH)
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.QUARANTINED)
        feed(mgr, clock, "openai", 1)   # first success -> RECOVERING
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.RECOVERING)
        clock.advance(1200.0)           # let old crashes decay
        feed(mgr, clock, "openai", 6)   # streak + score -> HEALTHY
        self.assertIs(mgr.state(ProviderId("openai")), HealthState.HEALTHY)

    def test_early_blip_is_capped_at_degraded(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "fresh", 1, AttemptStatus.FAILED, FailureType.TIMEOUT)
        self.assertIn(mgr.state(ProviderId("fresh")),
                      (HealthState.DEGRADED,))  # never straight to quarantine

    def test_cancellations_are_neutral(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 5)
        feed(mgr, clock, "openai", 10, AttemptStatus.CANCELLED)
        record = mgr.health_record("openai")
        self.assertIs(record.current_state, HealthState.HEALTHY)
        self.assertEqual(record.consecutive_failures, 0)
        self.assertEqual(record.success_rate, 1.0)  # cancellations don't count
        self.assertEqual(
            record.rolling_windows["5m"]["cancellation_rate"], 10 / 15
        )

    def test_explicit_human_flow(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "claude", 3)
        mgr.mark_needs_human(ProviderId("claude"), "captcha wall observed")
        self.assertIs(mgr.state(ProviderId("claude")), HealthState.NEEDS_HUMAN)
        mgr.human_resolved(ProviderId("claude"))
        self.assertIs(mgr.state(ProviderId("claude")), HealthState.RECOVERING)


class TestDecay(unittest.TestCase):
    def test_idle_score_recovers(self) -> None:
        """Manual acceptance 5: decay lifts health with zero new traffic."""
        mgr, clock = make_manager()
        feed(mgr, clock, "openai", 6)
        feed(mgr, clock, "openai", 3, AttemptStatus.FAILED, FailureType.TIMEOUT)
        hurt = mgr.health_record("openai")
        self.assertIn(hurt.current_state, (HealthState.DEGRADED, HealthState.SUSPECT))
        hurt_score = hurt.overall_score
        clock.advance(3600.0)  # one idle hour == 6 half-lives
        recovered = mgr.health_record("openai")
        self.assertGreater(recovered.overall_score, hurt_score)
        self.assertIn(recovered.current_state,
                      (HealthState.DEGRADED, HealthState.HEALTHY))
        history = mgr.health_history("openai", limit=1)
        if recovered.current_state is HealthState.HEALTHY:
            self.assertIn("decay recovery", history[0].reason)

    def test_needs_human_never_lifts_by_decay(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "copilot", 3)
        feed(mgr, clock, "copilot", 2, AttemptStatus.FAILED, FailureType.CAPTCHA)
        self.assertIs(mgr.state(ProviderId("copilot")), HealthState.NEEDS_HUMAN)
        clock.advance(100_000.0)
        self.assertIs(mgr.state(ProviderId("copilot")), HealthState.NEEDS_HUMAN)


class TestScoring(unittest.TestCase):
    def test_latency_affects_score(self) -> None:
        mgr_fast, clock_fast = make_manager()
        mgr_slow, clock_slow = make_manager()
        feed(mgr_fast, clock_fast, "p", 10, latency_ms=5_000.0)
        feed(mgr_slow, clock_slow, "p", 10, latency_ms=110_000.0)
        fast = mgr_fast.health_record("p").overall_score
        slow = mgr_slow.health_record("p").overall_score
        self.assertGreater(fast, slow)
        self.assertGreaterEqual(fast - slow, 0.10)  # latency weight is 0.15

    def test_confidence_weighting(self) -> None:
        """A low-confidence failure verdict hurts the score less."""
        mgr_hi, c1 = make_manager()
        mgr_lo, c2 = make_manager()
        feed(mgr_hi, c1, "p", 6)
        feed(mgr_lo, c2, "p", 6)
        feed(mgr_hi, c1, "p", 2, AttemptStatus.FAILED,
             FailureType.BROWSER_CRASH, confidence=0.95)
        feed(mgr_lo, c2, "p", 2, AttemptStatus.FAILED,
             FailureType.BROWSER_CRASH, confidence=0.2)
        self.assertLess(mgr_hi.health_record("p").overall_score,
                        mgr_lo.health_record("p").overall_score)
        self.assertGreater(severity_weight(FailureType.BROWSER_CRASH, 0.95),
                           severity_weight(FailureType.BROWSER_CRASH, 0.2))

    def test_neutral_prior_for_unseen_provider(self) -> None:
        mgr, _ = make_manager()
        record = mgr.health_record("never-used")
        self.assertEqual(record.overall_score, 0.5)
        self.assertIs(record.current_state, HealthState.UNKNOWN)

    def test_scorer_explanation_documents_components(self) -> None:
        mgr, clock = make_manager()
        feed(mgr, clock, "p", 5)
        breakdown = mgr.health_record("p").score_breakdown
        for key in ("success", "latency", "failure", "trend", "streak"):
            self.assertIn(key, breakdown["components"])
        self.assertIn("->", breakdown["explanation"])


class TestQueriesEventsSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr, self.clock = make_manager()
        feed(self.mgr, self.clock, "openai", 10)
        feed(self.mgr, self.clock, "gemini", 4)
        feed(self.mgr, self.clock, "gemini", 4, AttemptStatus.FAILED,
             FailureType.TIMEOUT)
        feed(self.mgr, self.clock, "grok", 3)
        feed(self.mgr, self.clock, "grok", 3, AttemptStatus.FAILED,
             FailureType.BROWSER_CRASH)

    def test_query_correctness(self) -> None:
        self.assertEqual(self.mgr.healthy_providers(), ["openai"])
        self.assertEqual(self.mgr.degraded_providers(), ["gemini"])
        self.assertEqual(self.mgr.quarantined_providers(), ["grok"])
        stats = self.mgr.health_statistics()
        self.assertEqual(stats["providers"], 3)
        self.assertEqual(stats["by_state"]["healthy"], 1)
        self.assertGreater(stats["transitions_recorded"], 3)
        self.assertEqual(len(self.mgr.all_health()), 3)

    def test_history_newest_first(self) -> None:
        history = self.mgr.health_history("gemini")
        self.assertGreaterEqual(len(history), 2)
        self.assertIs(history[0].to_state, HealthState.SUSPECT)

    def test_events_emitted(self) -> None:
        from dozen.context.domain.enums import EventType
        mgr, clock = make_manager()
        heard: list[EventType] = []
        mgr.events.bus.subscribe(lambda e: heard.append(e.type))
        feed(mgr, clock, "p", 3)
        feed(mgr, clock, "p", 3, AttemptStatus.FAILED, FailureType.BROWSER_CRASH)
        feed(mgr, clock, "p", 1)
        self.assertIn(EventType.PROVIDER_HEALTH_CHANGED, heard)
        self.assertIn(EventType.PROVIDER_DEGRADED, heard)
        self.assertIn(EventType.PROVIDER_QUARANTINED, heard)
        self.assertIn(EventType.PROVIDER_RECOVERED, heard)

    def test_record_serialization(self) -> None:
        record = self.mgr.health_record("gemini")
        payload = json.dumps(record.to_dict())
        self.assertIn("rolling_windows", payload)
        self.assertIn("score_breakdown", payload)
        for t in self.mgr.health_history("gemini"):
            json.dumps(t.to_dict())

    def test_port_snapshot_backward_compatible(self) -> None:
        snap = self.mgr.snapshot(ProviderId("openai"))
        self.assertEqual(snap.validate(), [])
        self.assertIs(snap.state, HealthState.HEALTHY)


class TestConcurrencyAndIntegration(unittest.TestCase):
    def test_concurrent_updates(self) -> None:
        mgr, _ = make_manager()
        errors: list[BaseException] = []

        def worker(seed: int) -> None:
            try:
                for i in range(50):
                    ok = (seed + i) % 4 != 0
                    mgr.record_attempt(make_attempt(
                        f"prov-{seed % 3}",
                        AttemptStatus.SUCCEEDED if ok else AttemptStatus.FAILED,
                        failure_type=None if ok else FailureType.TIMEOUT,
                    ))
                    mgr.health_record(f"prov-{seed % 3}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        total = sum(
            r.rolling_windows["lifetime"]["observations"] for r in mgr.all_health()
        )
        self.assertEqual(total, 450)

    def test_recorder_feeds_health_automatically(self) -> None:
        """The production path: decorator -> recorder -> health, no wiring."""
        from dozen.llm_client import LLMClient, LLMMessage
        from dozen.reliability import ReliabilityClientDecorator

        recorder = InMemoryExecutionRecorder()
        decorated = ReliabilityClientDecorator(LLMClient(mock=True), recorder=recorder)
        decorated.complete(provider="openai", model="gpt",
                           messages=[LLMMessage("user", "hi")])
        self.assertIs(recorder.health.state(ProviderId("openai")),
                      HealthState.HEALTHY)

    def test_debug_endpoints(self) -> None:
        import os
        from fastapi import HTTPException
        from webllm.reliability_debug import reset_debug_api
        from webllm.server import (
            debug_reliability_health,
            debug_reliability_health_history,
            debug_reliability_health_provider,
        )
        from dozen.reliability.recorder import default_recorder

        old = os.environ.pop("DOZEN_RELIABILITY_DEBUG", None)
        try:
            reset_debug_api()
            with self.assertRaises(HTTPException):
                debug_reliability_health()          # disabled -> 404

            os.environ["DOZEN_RELIABILITY_DEBUG"] = "1"
            reset_debug_api()
            default_recorder().health.record_attempt(make_attempt("openai"))
            payload = debug_reliability_health()
            self.assertIn("providers", payload)
            provider_payload = debug_reliability_health_provider("openai")
            self.assertEqual(provider_payload["provider"], "openai")
            history = debug_reliability_health_history()
            self.assertIn("transitions", history)
            with self.assertRaises(HTTPException):
                debug_reliability_health_provider("never-seen-provider")
        finally:
            if old is None:
                os.environ.pop("DOZEN_RELIABILITY_DEBUG", None)
            else:
                os.environ["DOZEN_RELIABILITY_DEBUG"] = old
            reset_debug_api()

    def test_orchestration_output_identical_with_health_active(self) -> None:
        from dozen import Task
        from dozen.agent_pool import AgentPool, AgentSpec
        from dozen.config import OrchestratorConfig
        from dozen.orchestrator import Orchestrator
        from dozen.llm_client import LLMClient
        from dozen.reliability import ReliabilityClientDecorator

        def run(client) -> str:
            pool = AgentPool([
                AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.8}, tier=4),
                AgentSpec(name="beta", provider="anthropic", model="claude",
                          strengths={"writing": 0.9, "reasoning": 0.8}, tier=4),
            ])
            # Phase 4G: model synthesis is eliminated, so the second provider is
            # exercised by a WORKER instead. A summarization prompt routes a
            # writing subtask to the anthropic-backed agent, keeping both
            # providers' health observable without any synthesis stage.
            cfg = OrchestratorConfig(max_parallelism=2, max_repair_attempts=1,
                                     verify_outputs=False, use_llm_router=False)
            return Orchestrator(client=client, pool=pool, config=cfg).run(
                Task(prompt="Summarize what a mutex does.")
            ).final_answer or ""

        recorder = InMemoryExecutionRecorder()
        observed = run(ReliabilityClientDecorator(LLMClient(mock=True),
                                                  recorder=recorder))
        bare = run(LLMClient(mock=True))
        self.assertEqual(observed, bare)  # byte-for-byte
        # …and health was computed on the side for both providers:
        self.assertEqual(
            set(recorder.health.healthy_providers()), {"openai", "anthropic"}
        )


if __name__ == "__main__":
    unittest.main()
