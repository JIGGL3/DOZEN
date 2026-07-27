"""Phase 2.1.2 — passive execution recording.

The one non-negotiable: the decorator is TRANSPARENT. Same result object,
same exceptions, same attribute semantics, byte-for-byte identical behavior.
Recording happens on the side or not at all.
"""

from __future__ import annotations

import threading
import unittest

from dozen.cancellation import CancelledError
from dozen.context.domain.types import ProviderId, RunId, Timestamp
from dozen.llm_client import LLMClient, LLMMessage
from dozen.reliability import (
    AttemptFactory,
    AttemptStatus,
    ExecutionRecorder,
    InMemoryExecutionRecorder,
    ProviderMetadata,
    ReliabilityClient,
    ReliabilityClientDecorator,
    WorkflowMetadata,
    wrap_client,
)


class FakeClock:
    """Deterministic clock: scripted monotonic ticks, fixed wall time."""

    def __init__(self, ticks: list[float]) -> None:
        self.ticks = list(ticks)
        self.calls = 0

    def now(self) -> Timestamp:
        return Timestamp("2026-07-10T12:00:00.000Z")

    def monotonic(self) -> float:
        value = self.ticks[min(self.calls, len(self.ticks) - 1)]
        self.calls += 1
        return value


def make_decorated(capacity: int = 1000, clock=None):
    recorder = InMemoryExecutionRecorder(
        capacity=capacity,
        factory=AttemptFactory(clock=clock) if clock else None,
        clock=clock,
    )
    wrapped = LLMClient(mock=True)  # the real production base client, offline
    decorated = ReliabilityClientDecorator(wrapped, recorder=recorder, clock=clock)
    return decorated, wrapped, recorder


def ask(client, text: str = "hello"):
    return client.complete(
        provider="anthropic", model="claude", messages=[LLMMessage("user", text)]
    )


class TestTransparency(unittest.TestCase):
    def test_result_is_the_exact_wrapped_object(self) -> None:
        decorated, wrapped, _ = make_decorated()
        sentinel = object()
        wrapped.complete = lambda **kw: sentinel  # type: ignore[method-assign]
        self.assertIs(ask(decorated), sentinel)   # identity, not equality

    def test_response_matches_undecorated_byte_for_byte(self) -> None:
        decorated, _, _ = make_decorated()
        bare = LLMClient(mock=True)
        a = ask(decorated, "deterministic prompt")
        b = ask(bare, "deterministic prompt")
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.provider, b.provider)
        self.assertEqual(a.model, b.model)

    def test_exception_rethrown_is_the_same_object(self) -> None:
        decorated, wrapped, recorder = make_decorated()
        boom = ValueError("provider exploded")

        def explode(**kw):
            raise boom

        wrapped.complete = explode  # type: ignore[method-assign]
        with self.assertRaises(ValueError) as ctx:
            ask(decorated)
        self.assertIs(ctx.exception, boom)  # untouched, not wrapped

    def test_attribute_reads_and_writes_forward(self) -> None:
        decorated, wrapped, _ = make_decorated()
        # Read-through:
        self.assertIs(decorated.mock, wrapped.mock)
        # Write-through — the orchestrator's cancel_token handoff must land
        # on the REAL client (orchestrator.py:97 does exactly this):
        token = object()
        decorated.cancel_token = token
        self.assertIs(wrapped.cancel_token, token)
        self.assertIs(decorated.cancel_token, token)

    def test_complete_json_flows_through_and_is_recorded(self) -> None:
        """Planner/Verifier path: inherited complete_json must execute the
        original logic AND its inner complete() must hit the recorder."""
        decorated, _, recorder = make_decorated()
        data = decorated.complete_json(
            provider="anthropic", model="claude",
            messages=[LLMMessage("user", 'Return {"x": 1}')],
        )
        self.assertIsInstance(data, dict)
        self.assertGreaterEqual(recorder.count(), 1)  # inner complete recorded

    def test_kwargs_pass_through_unmodified(self) -> None:
        decorated, wrapped, _ = make_decorated()
        seen: dict = {}

        def spy(**kw):
            seen.update(kw)
            return LLMClient(mock=True).complete(
                provider=kw["provider"], model=kw["model"], messages=kw["messages"]
            )

        wrapped.complete = spy  # type: ignore[method-assign]
        marker = {"nested": ["untouched"]}
        decorated.complete(
            provider="openai", model="gpt", messages=[LLMMessage("user", "x")],
            temperature=0.9, max_tokens=123, custom_flag=marker,
        )
        self.assertEqual(seen["temperature"], 0.9)
        self.assertEqual(seen["max_tokens"], 123)
        self.assertIs(seen["custom_flag"], marker)  # same object, no copy

    def test_decorator_satisfies_the_client_port(self) -> None:
        decorated, _, recorder = make_decorated()
        self.assertIsInstance(decorated, ReliabilityClient)
        self.assertIsInstance(recorder, ExecutionRecorder)

    def test_recorder_crash_never_breaks_the_call(self) -> None:
        decorated, _, recorder = make_decorated()

        def sabotage(*a, **k):
            raise RuntimeError("recorder on fire")

        recorder.begin = sabotage       # type: ignore[method-assign]
        recorder.finish = sabotage      # type: ignore[method-assign]
        response = ask(decorated)       # must still succeed (fail-open)
        self.assertTrue(response.text)


class TestRecording(unittest.TestCase):
    def test_success_recording(self) -> None:
        decorated, _, recorder = make_decorated()
        response = ask(decorated)
        self.assertEqual(recorder.count(), 1)
        attempt = recorder.attempts()[0]
        self.assertEqual(attempt.provider, "anthropic")
        self.assertEqual(attempt.status, AttemptStatus.SUCCEEDED)
        self.assertEqual(attempt.result_metadata["model"], "claude")
        self.assertEqual(attempt.result_metadata["response_chars"], len(response.text))
        self.assertIsNotNone(attempt.finished_at)
        self.assertGreaterEqual(attempt.latency_ms, 0.0)
        self.assertEqual(attempt.validate(), [])

    def test_failure_recording_with_exception_details(self) -> None:
        decorated, wrapped, recorder = make_decorated()

        def explode(**kw):
            raise TimeoutError("the tab never answered")

        wrapped.complete = explode  # type: ignore[method-assign]
        with self.assertRaises(TimeoutError):
            ask(decorated)
        attempt = recorder.attempts()[0]
        self.assertEqual(attempt.status, AttemptStatus.FAILED)
        self.assertEqual(attempt.result_metadata["exception_type"], "TimeoutError")
        self.assertEqual(attempt.result_metadata["exception_message"],
                         "the tab never answered")
        self.assertIsNotNone(attempt.latency_ms)

    def test_cancellation_recorded_as_cancelled_not_failed(self) -> None:
        decorated, wrapped, recorder = make_decorated()

        def cancel(**kw):
            raise CancelledError("user pressed stop")

        wrapped.complete = cancel  # type: ignore[method-assign]
        with self.assertRaises(CancelledError):
            ask(decorated)
        self.assertEqual(recorder.attempts()[0].status, AttemptStatus.CANCELLED)

    def test_timing_accuracy_with_deterministic_clock(self) -> None:
        clock = FakeClock(ticks=[10.0, 10.5])  # start, finish
        decorated, _, recorder = make_decorated(clock=clock)
        ask(decorated)
        attempt = recorder.attempts()[0]
        self.assertEqual(attempt.latency_ms, 500.0)  # exactly (10.5-10.0)*1000
        self.assertEqual(attempt.started_at, "2026-07-10T12:00:00.000Z")

    def test_id_generation_unique_and_ordered(self) -> None:
        decorated, _, recorder = make_decorated()
        for _ in range(10):
            ask(decorated)
        ids = [a.attempt_id for a in recorder.attempts()]
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(ids, sorted(ids))  # monotonic ULIDs

    def test_workflow_context_provider_is_mined_when_present(self) -> None:
        recorder = InMemoryExecutionRecorder()
        decorated = ReliabilityClientDecorator(
            LLMClient(mock=True), recorder=recorder,
            context_provider=lambda: {"run_id": "run-7", "subtask_id": "s2",
                                      "ignored_key": "x"},
        )
        ask(decorated)
        attempt = recorder.attempts()[0]
        self.assertEqual(attempt.run_id, "run-7")
        self.assertEqual(attempt.subtask_id, "s2")
        self.assertEqual(recorder.for_run(RunId("run-7")), [attempt])


class TestRingBuffer(unittest.TestCase):
    def test_eviction_keeps_newest(self) -> None:
        decorated, _, recorder = make_decorated(capacity=3)
        for i in range(5):
            ask(decorated, f"call {i}")
        self.assertEqual(recorder.count(), 3)
        self.assertEqual(recorder.capacity, 3)
        kept = recorder.attempts()
        ids = [a.attempt_id for a in kept]
        self.assertEqual(ids, sorted(ids))  # oldest two evicted, order intact
        self.assertEqual([a.result_metadata["response_chars"] > 0 for a in kept],
                         [True, True, True])

    def test_recent_is_newest_first(self) -> None:
        decorated, _, recorder = make_decorated()
        for _ in range(4):
            ask(decorated)
        newest_first = recorder.recent(2)
        self.assertEqual(len(newest_first), 2)
        self.assertGreater(newest_first[0].attempt_id, newest_first[1].attempt_id)

    def test_default_capacity_is_1000(self) -> None:
        self.assertEqual(InMemoryExecutionRecorder().capacity, 1000)
        with self.assertRaises(ValueError):
            InMemoryExecutionRecorder(capacity=0)


class TestThreadSafety(unittest.TestCase):
    def test_parallel_calls_all_recorded_uniquely(self) -> None:
        decorated, _, recorder = make_decorated()
        errors: list[BaseException] = []

        def worker(seed: int) -> None:
            try:
                for i in range(25):
                    ask(decorated, f"w{seed}-{i}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(recorder.count(), 200)
        ids = {a.attempt_id for a in recorder.attempts()}
        self.assertEqual(len(ids), 200)
        self.assertTrue(all(a.status is AttemptStatus.SUCCEEDED
                            for a in recorder.attempts()))


class TestFactoryAndWiring(unittest.TestCase):
    def test_attempt_factory_consistency(self) -> None:
        factory = AttemptFactory()
        attempt = factory.create(
            ProviderMetadata(provider=ProviderId("openai"), model="gpt",
                             agent_name="gpt (frontier)"),
            WorkflowMetadata(task_id="t1", subtask_id="s1"),
            attempt_number=3,
        )
        self.assertEqual(attempt.validate(), [])
        self.assertEqual(attempt.status, AttemptStatus.PENDING)
        self.assertEqual(attempt.agent_name, "gpt (frontier)")
        self.assertEqual(attempt.attempt_number, 3)
        self.assertEqual(attempt.result_metadata, {"model": "gpt"})

    def test_wrap_client_helper(self) -> None:
        recorder = InMemoryExecutionRecorder()
        decorated = wrap_client(LLMClient(mock=True), recorder=recorder)
        self.assertIsInstance(decorated, ReliabilityClientDecorator)
        ask(decorated)
        self.assertEqual(recorder.count(), 1)

    def test_build_orchestrator_installs_the_decorator(self) -> None:
        """The ONLY wiring change of this phase: the orchestrator's client is
        now the decorator, and the real web client sits inside it."""
        from webllm.client import WebAutomationLLMClient
        from webllm.pool import build_orchestrator

        orchestrator = build_orchestrator(browser=object(), provider_keys=["openai"])
        self.assertIsInstance(orchestrator.client, ReliabilityClientDecorator)
        self.assertIsInstance(orchestrator.client._wrapped, WebAutomationLLMClient)
        # Transparency at the wiring level: attribute forwarding works on the
        # real stack too (this is what the orchestrator does at run start).
        sentinel = object()
        orchestrator.client.cancel_token = sentinel
        self.assertIs(orchestrator.client._wrapped.cancel_token, sentinel)


if __name__ == "__main__":
    unittest.main()
