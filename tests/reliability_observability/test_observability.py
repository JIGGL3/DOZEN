"""Phase 2.1.3 — observability layer.

Read-only everywhere: queries, stats, DTOs, debug API, events. The last test
class re-proves byte-identical orchestration output with the full
observability stack active.
"""

from __future__ import annotations

import json
import threading
import unittest

from dozen.context.domain.enums import EventType
from dozen.context.domain.types import ProviderId, RunId, Timestamp
from dozen.llm_client import LLMClient, LLMMessage
from dozen.reliability import (
    AttemptQuery,
    AttemptStatus,
    ExecutionStage,
    InMemoryExecutionRecorder,
    ObservabilityConfig,
    ProviderMetadata,
    ReliabilityClientDecorator,
    ReliabilityDebugApi,
    DebugApiDisabled,
    WorkflowMetadata,
)
from dozen.reliability.models import ExecutionAttempt
from dozen.reliability.types import AttemptId


def record_attempt(
    recorder: InMemoryExecutionRecorder,
    provider: str = "anthropic",
    stage: ExecutionStage = ExecutionStage.WORKER,
    status: AttemptStatus = AttemptStatus.SUCCEEDED,
    latency_ms: float = 100.0,
    run_id: str | None = "run-1",
    conversation_id: str | None = None,
) -> ExecutionAttempt:
    attempt = recorder.begin(
        ProviderMetadata(provider=ProviderId(provider), model="m"),
        WorkflowMetadata(
            run_id=RunId(run_id) if run_id else None,
            conversation_id=conversation_id,  # type: ignore[arg-type]
        ),
        stage=stage,
        prompt_character_count=42,
    )
    return recorder.finish(attempt, status, latency_ms=latency_ms)


class TestExecutionAttemptCompat(unittest.TestCase):
    def test_old_shape_dict_parses_with_defaults(self) -> None:
        """A 2.1.2-era serialized attempt (no new keys) must load cleanly."""
        old = {
            "schema_version": 1,
            "attempt_id": "01OLD", "provider": "openai",
            "started_at": "2026-07-10T10:00:00.000Z",
            "status": "succeeded", "finished_at": "2026-07-10T10:00:01.000Z",
            "latency_ms": 1000.0, "attempt_number": 1,
            "failures": [], "recoveries": [], "failovers": [],
            "screenshots": [], "dom_snapshots": [], "result_metadata": {},
        }
        parsed = ExecutionAttempt.from_dict(old)
        self.assertIs(parsed.execution_stage, ExecutionStage.UNKNOWN)
        self.assertEqual(parsed.retry_number, 0)
        self.assertEqual(parsed.prompt_character_count, 0)
        self.assertFalse(parsed.response_present)
        self.assertEqual(parsed.debug_notes, ())
        self.assertEqual(parsed.metadata_version, 1)
        self.assertIsNone(parsed.future_screenshot_path)
        self.assertEqual(parsed.validate(), [])

    def test_new_fields_round_trip(self) -> None:
        recorder = InMemoryExecutionRecorder()
        done = record_attempt(recorder, stage=ExecutionStage.PLANNER)
        done = done.with_note("first note").with_response(1234)
        again = ExecutionAttempt.from_dict(done.to_dict())
        self.assertEqual(again.to_dict(), done.to_dict())
        self.assertIs(again.execution_stage, ExecutionStage.PLANNER)
        self.assertEqual(again.debug_notes, ("first note",))
        self.assertTrue(again.response_present)
        self.assertEqual(again.response_character_count, 1234)

    def test_helpers_are_pure(self) -> None:
        recorder = InMemoryExecutionRecorder()
        base = record_attempt(recorder)
        staged = base.with_stage(ExecutionStage.VERIFIER)
        self.assertIs(base.execution_stage, ExecutionStage.WORKER)
        self.assertIs(staged.execution_stage, ExecutionStage.VERIFIER)


class TestRingAdditions(unittest.TestCase):
    def test_peeks_and_snapshot(self) -> None:
        recorder = InMemoryExecutionRecorder(capacity=10)
        self.assertIsNone(recorder.peek())
        self.assertIsNone(recorder.peek_oldest())
        first = record_attempt(recorder, provider="openai")
        last = record_attempt(recorder, provider="google")
        self.assertEqual(recorder.peek().attempt_id, last.attempt_id)
        self.assertEqual(recorder.peek_latest().attempt_id, last.attempt_id)
        self.assertEqual(recorder.peek_oldest().attempt_id, first.attempt_id)
        snap = recorder.snapshot()
        self.assertIsInstance(snap, tuple)          # immutable view
        self.assertEqual(len(snap), 2)

    def test_snapshot_is_point_in_time(self) -> None:
        recorder = InMemoryExecutionRecorder()
        record_attempt(recorder)
        snap = recorder.snapshot()
        record_attempt(recorder)
        self.assertEqual(len(snap), 1)              # unaffected by later writes
        self.assertEqual(recorder.count(), 2)


class TestQueryAndStatistics(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = InMemoryExecutionRecorder()
        record_attempt(self.recorder, "anthropic", ExecutionStage.PLANNER,
                       AttemptStatus.SUCCEEDED, 200.0, "run-1", "conv-A")
        record_attempt(self.recorder, "openai", ExecutionStage.WORKER,
                       AttemptStatus.SUCCEEDED, 400.0, "run-1", "conv-A")
        record_attempt(self.recorder, "openai", ExecutionStage.WORKER,
                       AttemptStatus.FAILED, 900.0, "run-2", "conv-B")
        record_attempt(self.recorder, "google", ExecutionStage.SYNTHESIZER,
                       AttemptStatus.CANCELLED, 100.0, "run-2", "conv-B")
        self.query = AttemptQuery(self.recorder)

    def test_filters(self) -> None:
        self.assertEqual(len(self.query.attempts_for_run("run-1")), 2)
        self.assertEqual(len(self.query.attempts_for_provider("openai")), 2)
        self.assertEqual(len(self.query.attempts_for_stage(ExecutionStage.WORKER)), 2)
        self.assertEqual(len(self.query.attempts_for_conversation("conv-B")), 2)
        self.assertEqual(self.query.attempts_for_run("ghost"), [])

    def test_recent_and_by_id(self) -> None:
        newest_two = self.query.recent_attempts(2)
        self.assertEqual(len(newest_two), 2)
        self.assertGreater(newest_two[0].attempt_id, newest_two[1].attempt_id)
        target = newest_two[0]
        self.assertEqual(self.query.attempt_by_id(target.attempt_id), target)
        self.assertIsNone(self.query.attempt_by_id("missing"))

    def test_statistics_correctness(self) -> None:
        stats = self.query.attempt_statistics()
        self.assertEqual(stats.total_attempts, 4)
        self.assertEqual(stats.successful_attempts, 2)
        self.assertEqual(stats.failed_attempts, 1)
        self.assertEqual(stats.cancelled_attempts, 1)
        self.assertEqual(stats.average_latency_ms, 400.0)  # (200+400+900+100)/4
        self.assertEqual(stats.longest_attempt.latency_ms, 900.0)
        self.assertEqual(stats.newest_attempt.provider, "google")
        self.assertEqual(stats.oldest_attempt.provider, "anthropic")
        providers = {p.provider: p for p in stats.provider_breakdown}
        self.assertEqual(providers["openai"].attempts, 2)
        self.assertEqual(providers["openai"].failed, 1)
        self.assertEqual(providers["openai"].average_latency_ms, 650.0)
        stages = {s.stage: s for s in stats.stage_breakdown}
        self.assertEqual(stages["worker"].attempts, 2)
        self.assertEqual(stages["planner"].succeeded, 1)

    def test_statistics_cache_invalidates_on_new_attempts(self) -> None:
        cached_query = AttemptQuery(self.recorder, statistics_cache_seconds=3600)
        first = cached_query.attempt_statistics()
        self.assertIs(cached_query.attempt_statistics(), first)   # served from cache
        record_attempt(self.recorder)                             # new data
        second = cached_query.attempt_statistics()
        self.assertIsNot(second, first)
        self.assertEqual(second.total_attempts, 5)

    def test_queries_never_mutate(self) -> None:
        before = [a.attempt_id for a in self.recorder.snapshot()]
        self.query.attempt_statistics()
        self.query.recent_attempts(100)
        self.query.attempts_for_provider("openai")
        after = [a.attempt_id for a in self.recorder.snapshot()]
        self.assertEqual(before, after)


class TestViewModels(unittest.TestCase):
    def test_dto_serialization_is_json_safe(self) -> None:
        recorder = InMemoryExecutionRecorder()
        record_attempt(recorder)
        query = AttemptQuery(recorder)
        stats = query.attempt_statistics()
        payload = json.dumps(stats.to_dict())     # must not raise
        self.assertIn("provider_breakdown", payload)
        details = query.details(recorder.peek())
        json.dumps(details.to_dict())
        self.assertEqual(details.summary.provider, "anthropic")
        self.assertEqual(details.failure_count, 0)


class TestEvents(unittest.TestCase):
    def test_lifecycle_events_emitted(self) -> None:
        recorder = InMemoryExecutionRecorder()
        heard: list[EventType] = []
        recorder.events.bus.subscribe(lambda e: heard.append(e.type))
        record_attempt(recorder, status=AttemptStatus.SUCCEEDED)
        self.assertIn(EventType.ATTEMPT_STARTED, heard)
        self.assertIn(EventType.ATTEMPT_RECORDED, heard)
        self.assertIn(EventType.ATTEMPT_FINISHED, heard)
        self.assertNotIn(EventType.ATTEMPT_FAILED, heard)

    def test_failed_and_cancelled_specific_events(self) -> None:
        recorder = InMemoryExecutionRecorder()
        heard: list[EventType] = []
        recorder.events.bus.subscribe(lambda e: heard.append(e.type))
        record_attempt(recorder, status=AttemptStatus.FAILED)
        record_attempt(recorder, status=AttemptStatus.CANCELLED)
        self.assertIn(EventType.ATTEMPT_FAILED, heard)
        self.assertIn(EventType.ATTEMPT_CANCELLED, heard)

    def test_eviction_event(self) -> None:
        recorder = InMemoryExecutionRecorder(capacity=2)
        evicted: list[str] = []
        recorder.events.bus.subscribe(
            lambda e: evicted.append(e.payload["attempt_id"]),
            EventType.ATTEMPT_EVICTED,
        )
        first = record_attempt(recorder)
        record_attempt(recorder)
        record_attempt(recorder)   # pushes `first` off the ring
        self.assertEqual(evicted, [first.attempt_id])


class TestDebugApi(unittest.TestCase):
    def make_api(self, enabled: bool, **cfg) -> tuple[ReliabilityDebugApi, InMemoryExecutionRecorder]:
        recorder = InMemoryExecutionRecorder()
        api = ReliabilityDebugApi(
            recorder, ObservabilityConfig(enable_debug_api=enabled, **cfg)
        )
        return api, recorder

    def test_disabled_by_default(self) -> None:
        recorder = InMemoryExecutionRecorder()
        api = ReliabilityDebugApi(recorder)  # default config
        self.assertFalse(api.enabled)
        with self.assertRaises(DebugApiDisabled):
            api.attempts()
        with self.assertRaises(DebugApiDisabled):
            api.stats()

    def test_enabled_payloads(self) -> None:
        api, recorder = self.make_api(True)
        record_attempt(recorder, "anthropic", ExecutionStage.PLANNER)
        record_attempt(recorder, "openai", ExecutionStage.WORKER,
                       AttemptStatus.FAILED, run_id="run-9")
        listing = api.attempts()
        self.assertEqual(listing["count"], 2)
        self.assertEqual(listing["attempts"][0]["provider"], "openai")  # newest first
        by_run = api.attempts(run_id="run-9")
        self.assertEqual(by_run["count"], 1)
        by_stage = api.attempts(stage="planner")
        self.assertEqual(by_stage["count"], 1)
        target_id = listing["attempts"][0]["attempt_id"]
        detail = api.attempt(target_id)
        self.assertEqual(detail["attempt_id"], target_id)
        self.assertIsNone(api.attempt("missing"))
        stats = api.stats()
        self.assertEqual(stats["total_attempts"], 2)
        self.assertEqual(len(api.providers()["providers"]), 2)
        self.assertEqual(len(api.stages()["stages"]), 2)

    def test_max_attempts_returned_enforced(self) -> None:
        api, recorder = self.make_api(True, max_attempts_returned=3)
        for _ in range(10):
            record_attempt(recorder)
        self.assertEqual(api.attempts()["count"], 3)
        self.assertEqual(api.attempts(limit=9999)["count"], 3)

    def test_server_endpoints_gate_on_environment(self) -> None:
        import os
        from fastapi import HTTPException
        from webllm.reliability_debug import reset_debug_api
        from webllm.server import debug_reliability_attempts, debug_reliability_stats

        old = os.environ.pop("DOZEN_RELIABILITY_DEBUG", None)
        try:
            reset_debug_api()
            with self.assertRaises(HTTPException) as ctx:
                debug_reliability_attempts()
            self.assertEqual(ctx.exception.status_code, 404)   # disabled == invisible

            os.environ["DOZEN_RELIABILITY_DEBUG"] = "1"
            reset_debug_api()
            payload = debug_reliability_stats()
            self.assertIn("total_attempts", payload)           # enabled == JSON
        finally:
            if old is None:
                os.environ.pop("DOZEN_RELIABILITY_DEBUG", None)
            else:
                os.environ["DOZEN_RELIABILITY_DEBUG"] = old
            reset_debug_api()


class TestConcurrentReads(unittest.TestCase):
    def test_reads_race_writes_safely(self) -> None:
        recorder = InMemoryExecutionRecorder(capacity=64)
        query = AttemptQuery(recorder, statistics_cache_seconds=0.05)
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer() -> None:
            while not stop.is_set():
                record_attempt(recorder)

        def reader() -> None:
            try:
                while not stop.is_set():
                    query.attempt_statistics()
                    query.recent_attempts(10)
                    recorder.snapshot()
                    recorder.peek_latest()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        import time
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=20)
        self.assertEqual(errors, [])


class TestStageInferenceAndIdenticalBehavior(unittest.TestCase):
    def _run_orchestration(self, client) -> str:
        from dozen import Task
        from dozen.agent_pool import AgentPool, AgentSpec
        from dozen.config import OrchestratorConfig
        from dozen.orchestrator import Orchestrator

        pool = AgentPool([
            AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.8}, tier=4),
            AgentSpec(name="beta", provider="anthropic", model="claude",
                      strengths={"writing": 0.9, "reasoning": 0.8}, tier=4),
        ])
        # Phase 4F: model synthesis is optional polish now; this test asserts
        # the SYNTHESIZER stage is observable, so the trusted configuration
        # enables the polish pass explicitly.
        cfg = OrchestratorConfig(max_parallelism=2, max_repair_attempts=1,
                                 verify_outputs=False, use_llm_router=False,
                                 enable_model_polish=True)
        orch = Orchestrator(client=client, pool=pool, config=cfg)
        result = orch.run(Task(prompt="Summarize what a mutex does."))
        return result.final_answer or ""

    def test_stages_inferred_and_output_identical(self) -> None:
        recorder = InMemoryExecutionRecorder()
        decorated = ReliabilityClientDecorator(LLMClient(mock=True), recorder=recorder)
        answer_observed = self._run_orchestration(decorated)
        answer_bare = self._run_orchestration(LLMClient(mock=True))
        # Manual acceptance #5: byte-for-byte identical output.
        self.assertEqual(answer_observed, answer_bare)
        # Manual acceptance #1/#2: planner and workers visible, each attempt
        # carrying provider/stage/status/latency. Phase 4G: model synthesis is
        # eliminated from production, so no SYNTHESIZER stage is ever recorded.
        stages = {a.execution_stage for a in recorder.attempts()}
        self.assertIn(ExecutionStage.PLANNER, stages)
        self.assertIn(ExecutionStage.WORKER, stages)
        self.assertNotIn(ExecutionStage.SYNTHESIZER, stages)
        for attempt in recorder.attempts():
            self.assertTrue(attempt.provider)
            self.assertIsNotNone(attempt.latency_ms)
            self.assertIs(attempt.status, AttemptStatus.SUCCEEDED)
            self.assertGreater(attempt.prompt_character_count, 0)
            self.assertTrue(attempt.response_present)

    def test_direct_calls_are_unknown_stage(self) -> None:
        recorder = InMemoryExecutionRecorder()
        decorated = ReliabilityClientDecorator(LLMClient(mock=True), recorder=recorder)
        decorated.complete(provider="openai", model="gpt",
                           messages=[LLMMessage("user", "hi")])
        self.assertIs(recorder.peek().execution_stage, ExecutionStage.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
