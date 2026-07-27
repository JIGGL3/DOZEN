"""Parts G, I, J — oversized results, provider failures, deterministic fallback."""

from __future__ import annotations

import json
import unittest

from dozen.cancellation import CancelToken, CancelledError
from dozen.synthesis_scaling import FALLBACK_HEADLINE, SynthesisBudgetPolicy

from .harness import (
    RecordingClient,
    make_synthesizer,
    measure_final_frame,
    result,
    results_of,
    task_and_plan,
)

BROKEN_ENVELOPE = (
    '{"summary": "s", "key_decisions": ["k"], "artifacts": {"f.py": "def broken('
)


class TestOversizedSingleResult(unittest.TestCase):
    def test_large_output_is_chunk_reduced_in_order(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client)  # default 24000 policy
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([40000, 200]))
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        kinds = [k for k, _ in client.calls]
        self.assertEqual(kinds[:2], ["chunk", "chunk"])
        self.assertEqual(kinds[-1], "final")
        for size in client.sizes():
            self.assertLessEqual(size, 24000)
        # The tail of the original output reached the LAST chunk call — the
        # tail is never dropped.
        chunk_inputs = [text for kind, text in client.inputs if kind == "chunk"]
        self.assertIn("part 1 of 2", chunk_inputs[0])
        self.assertIn("part 2 of 2", chunk_inputs[1])
        self.assertIn("TAIL-0-END", chunk_inputs[1])
        # Ordered digests reach the final prompt.
        final_input = next(text for kind, text in client.inputs
                           if kind == "final")
        self.assertLess(final_input.index("(part 1 of 2)"),
                        final_input.index("(part 2 of 2)"))

    def test_chunk_failure_degrades_to_explicit_excerpt(self) -> None:
        client = RecordingClient(
            fail=lambda kind, i: RuntimeError("chunk provider down")
            if kind == "chunk" and i == 1 else None
        )
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([40000, 200]))
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        final_input = next(text for kind, text in client.inputs
                           if kind == "final")
        # Explicit, never silent: the marker AND the tail are both present.
        self.assertIn("characters omitted", final_input)
        self.assertIn("TAIL-0-END", final_input)


class TestProviderFailures(unittest.TestCase):
    def one_over_policy(self) -> tuple[SynthesisBudgetPolicy, list, object, object]:
        titles = ["S0", "S1"]
        contents = ["a" * 300, "b" * 300]
        base = measure_final_frame(contents, titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base - 1, reserved_frame_chars=100,
            max_capsule_chars=600, max_intermediate_chars=600,
        )
        results = [result("s0", "S0", contents[0]),
                   result("s1", "S1", contents[1])]
        task, plan = task_and_plan()
        return policy, results, task, plan

    def test_intermediate_failure_uses_deterministic_join(self) -> None:
        policy, results, task, plan = self.one_over_policy()
        client = RecordingClient(
            fail=lambda kind, i: RuntimeError("merge provider down")
            if kind == "merge" else None
        )
        synthesizer = make_synthesizer(client, policy=policy)
        answer = synthesizer.synthesize(task, plan, results)
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        kinds = [k for k, _ in client.calls]
        self.assertEqual(kinds, ["merge", "final"])
        final_input = next(text for kind, text in client.inputs
                           if kind == "final")
        self.assertIn("## S0", final_input)  # the deterministic join

    def test_malformed_intermediate_response_uses_deterministic_join(self) -> None:
        policy, results, task, plan = self.one_over_policy()
        client = RecordingClient(merge_reply=BROKEN_ENVELOPE)
        synthesizer = make_synthesizer(client, policy=policy)
        answer = synthesizer.synthesize(task, plan, results)
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        final_input = next(text for kind, text in client.inputs
                           if kind == "final")
        self.assertNotIn('"key_decisions"', final_input)
        self.assertIn("## S0", final_input)

    def test_final_failure_falls_back_deterministically(self) -> None:
        client = RecordingClient(
            fail=lambda kind, i: RuntimeError("final provider down")
            if kind == "final" else None
        )
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertIn(FALLBACK_HEADLINE, answer)
        self.assertIn("provider call failed", answer)
        self.assertIn("S0", answer)
        self.assertIn("TAIL-1-END", answer)

    def test_malformed_final_response_falls_back(self) -> None:
        client = RecordingClient(final_reply=BROKEN_ENVELOPE)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertIn(FALLBACK_HEADLINE, answer)
        self.assertNotIn('"key_decisions"', answer)
        self.assertNotIn("def broken(", answer)

    def test_valid_final_envelope_is_decoded(self) -> None:
        envelope = json.dumps({
            "summary": "done", "key_decisions": ["k"],
            "artifacts": {"answer.md": "Decoded final answer."},
            "confidence": 0.9,
        })
        client = RecordingClient(final_reply=envelope)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, "Decoded final answer.")

    def test_deeply_nested_final_envelope_falls_back_without_leak(self) -> None:
        value = "SAFE BODY"
        for _ in range(5):
            value = json.dumps({
                "summary": "done", "key_decisions": [],
                "artifacts": {"answer.md": value}, "confidence": 1,
            })
        client = RecordingClient(final_reply=value)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertIn(FALLBACK_HEADLINE, answer)
        self.assertNotIn('"artifacts"', answer)

    def test_requested_protocol_looking_json_is_not_decoded(self) -> None:
        data = '{"summary":"inventory","artifacts":{"count":3}}'
        client = RecordingClient(final_reply=data)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        task.prompt = "Return the inventory as raw JSON only."
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, data)

    def test_requested_fenced_protocol_looking_json_is_not_decoded(self) -> None:
        data = '```json\n{"summary":"inventory","artifacts":{"count":3}}\n```'
        client = RecordingClient(final_reply=data)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        task.prompt = "Return the inventory as JSON."
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, data)

    def test_requested_prefixed_json_fixture_is_not_decoded(self) -> None:
        data = 'Here is the example:\n{"summary":"inventory","artifacts":{"count":3}}'
        client = RecordingClient(final_reply=data)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        task.prompt = "Show an example JSON object."
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, data)

    def test_final_code_with_protocol_key_literals_is_not_decoded(self) -> None:
        code = 'const data = {"summary":"product","artifacts":{"x":1}};'
        client = RecordingClient(final_reply=code)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, code)

    def test_final_code_with_planner_fixture_is_not_rejected(self) -> None:
        code = (
            'const plan = {"analysis":"x","delegations":[],'
            '"synthesis_strategy":"y"};'
        )
        client = RecordingClient(final_reply=code)
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(answer, code)

    def test_cancellation_between_levels_propagates(self) -> None:
        client = RecordingClient(
            fail=lambda kind, i: CancelledError("stop pressed")
            if kind == "merge" and i == 1 else None
        )
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        with self.assertRaises(CancelledError):
            synthesizer.synthesize(task, plan, results_of([3000] * 10))
        # The second merge is the call that observes cancellation; nothing
        # starts after it unwinds.
        self.assertEqual(len(client.calls), 2)

    def test_cancelled_before_final_call_starts_no_call(self) -> None:
        client = RecordingClient()
        token = CancelToken()
        token.cancel()
        client.cancel_token = token
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        with self.assertRaises(CancelledError):
            synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual(client.calls, [])

    def test_cancellation_between_chunks_starts_no_later_chunk(self) -> None:
        token = CancelToken()

        class CancelAfterFirst(RecordingClient):
            def complete(self, **kwargs):
                response = super().complete(**kwargs)
                if len(self.calls) == 1:
                    token.cancel()
                return response

        client = CancelAfterFirst()
        client.cancel_token = token
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        with self.assertRaises(CancelledError):
            synthesizer.synthesize(task, plan, results_of([40000, 200]))
        self.assertEqual([kind for kind, _ in client.calls], ["chunk"])

    def test_cancellation_between_groups_starts_no_later_group(self) -> None:
        token = CancelToken()

        class CancelAfterFirst(RecordingClient):
            def complete(self, **kwargs):
                response = super().complete(**kwargs)
                if len(self.calls) == 1:
                    token.cancel()
                return response

        client = CancelAfterFirst()
        client.cancel_token = token
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        with self.assertRaises(CancelledError):
            synthesizer.synthesize(task, plan, results_of([3000] * 10))
        self.assertEqual([kind for kind, _ in client.calls], ["merge"])

    def test_cancellation_during_final_call_is_not_diagnostic(self) -> None:
        client = RecordingClient(
            fail=lambda kind, _i: CancelledError("during final")
            if kind == "final" else None
        )
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        with self.assertRaises(CancelledError):
            synthesizer.synthesize(task, plan, results_of([300, 300]))
        self.assertEqual([kind for kind, _ in client.calls], ["final"])


class TestFallbackShape(unittest.TestCase):
    def test_fallback_preserves_order_headings_and_contradictions(self) -> None:
        client = RecordingClient(
            fail=lambda kind, i: RuntimeError("down") if kind == "final" else None
        )
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        flagged = result(
            "s9", "Flagged", "Flagged body content.",
            error="contradicts the Alpha subtask about limits",
        )
        answer = synthesizer.synthesize(
            task, plan, results_of([300, 300]) + [flagged]
        )
        self.assertIn(FALLBACK_HEADLINE, answer)
        self.assertLess(answer.index("## S0"), answer.index("## S1"))
        self.assertLess(answer.index("## S1"), answer.index("## Flagged"))
        self.assertIn("Unresolved warnings / contradictions", answer)
        self.assertIn("contradicts the Alpha subtask about limits", answer)
        # It never claims full synthesis.
        self.assertIn("could not be completed", answer)


class TestLegacyUnboundedMode(unittest.TestCase):
    def test_budget_zero_keeps_the_single_unbounded_call(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client, max_input_chars=0)
        self.assertIsNone(synthesizer.policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([50000, 50000]))
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        self.assertEqual([k for k, _ in client.calls], ["final"])

    def test_budget_zero_provider_failure_propagates(self) -> None:
        client = RecordingClient(
            fail=lambda kind, i: RuntimeError("down") if kind == "final" else None
        )
        synthesizer = make_synthesizer(client, max_input_chars=0)
        task, plan = task_and_plan()
        with self.assertRaises(RuntimeError):
            synthesizer.synthesize(task, plan, results_of([300, 300]))


if __name__ == "__main__":
    unittest.main()
