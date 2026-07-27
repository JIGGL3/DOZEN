"""Part H — hierarchical/tree synthesis under the budget policy."""

from __future__ import annotations

import unittest

from dozen.synthesis_scaling import (
    FALLBACK_HEADLINE,
    SynthesisBudgetPolicy,
    build_capsule,
    DEFAULT_SYNTHESIS_POLICY,
)

from .harness import (
    RecordingClient,
    make_synthesizer,
    measure_final_frame,
    result,
    results_of,
    task_and_plan,
)


class TestSingleGroup(unittest.TestCase):
    def test_small_run_is_one_final_call(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([200, 200]))
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        self.assertEqual([k for k, _ in client.calls], ["final"])
        self.assertLessEqual(client.sizes()[0],
                             synthesizer.policy.max_call_input_chars)

    def test_exact_budget_boundary_is_allowed(self) -> None:
        titles = ["S0", "S1"]
        contents = ["a" * 300, "b" * 300]
        base = measure_final_frame(contents, titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base, reserved_frame_chars=100,
            max_capsule_chars=600, max_intermediate_chars=600,
        )
        client = RecordingClient()
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(
            task, plan,
            [result("s0", "S0", contents[0]), result("s1", "S1", contents[1])],
        )
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        self.assertEqual([k for k, _ in client.calls], ["final"])
        self.assertEqual(client.sizes("final")[0], base)  # exactly at budget

    def test_one_char_over_budget_triggers_reduction(self) -> None:
        titles = ["S0", "S1"]
        contents = ["a" * 300, "b" * 300]
        base = measure_final_frame(contents, titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base - 1, reserved_frame_chars=100,
            max_capsule_chars=600, max_intermediate_chars=600,
        )
        client = RecordingClient(merge_reply="condensed pair")
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(
            task, plan,
            [result("s0", "S0", contents[0]), result("s1", "S1", contents[1])],
        )
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        kinds = [k for k, _ in client.calls]
        self.assertEqual(kinds, ["merge", "final"])
        for size in client.sizes():
            self.assertLessEqual(size, policy.max_call_input_chars)


class TestMultiGroupAndTree(unittest.TestCase):
    def test_many_moderate_outputs_form_groups_then_one_final(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client)  # default 24000 policy
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([3000] * 10))
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        kinds = [k for k, _ in client.calls]
        self.assertGreaterEqual(kinds.count("merge"), 2)
        self.assertEqual(kinds[-1], "final")
        for size in client.sizes():
            self.assertLessEqual(size, 24000)

    def test_sections_keep_stable_order_in_prompts(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        synthesizer.synthesize(task, plan, results_of([3000] * 10))
        first_merge = next(text for kind, text in client.inputs if kind == "merge")
        self.assertLess(first_merge.index("S0"), first_merge.index("S1"))
        self.assertLess(first_merge.index("S1"), first_merge.index("S2"))

    def test_multi_level_tree_reaches_a_final_call(self) -> None:
        # Small budget forces at least two reduction levels before the final.
        titles = ["S0"]
        base = measure_final_frame([""], titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base + 2600,
            reserved_frame_chars=200,
            max_capsule_chars=800,
            max_intermediate_chars=800,
            max_group_size=2,
        )
        client = RecordingClient(merge_reply="m " * 400)  # 800 chars back
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([800] * 8))
        kinds = [k for k, _ in client.calls]
        self.assertGreaterEqual(kinds.count("merge"), 6)  # two levels: 4 + 2
        self.assertEqual(kinds[-1], "final")
        self.assertEqual(answer, "FINAL SYNTHESIS ANSWER.")
        for size in client.sizes():
            self.assertLessEqual(size, policy.max_call_input_chars)

    def test_frame_overhead_counts_against_the_budget(self) -> None:
        # Section contents alone fit the content budget, but the REAL prompt
        # (system + task brief + strategy) exceeds the cap — the synthesizer
        # must not issue that final call.
        titles = ["S0", "S1"]
        contents = ["a" * 100, "b" * 100]
        base = measure_final_frame(contents, titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base - 1, reserved_frame_chars=50,
            max_capsule_chars=300, max_intermediate_chars=300,
        )
        client = RecordingClient(merge_reply="c " * 200)  # stays big: 400 chars
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(
            task, plan,
            [result("s0", "S0", contents[0]), result("s1", "S1", contents[1])],
        )
        for kind, size in client.calls:
            self.assertLessEqual(size, policy.max_call_input_chars)
            self.assertNotEqual(kind, "final")
        self.assertTrue(answer)  # deterministic fallback, never an over-budget call

    def test_maximum_reduction_depth_falls_back_deterministically(self) -> None:
        titles = ["S0"]
        base = measure_final_frame([""], titles)
        # Merges return content as large as their inputs, so the tree can
        # never fit the final frame: depth must cap, then fall back.
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base + 1000,
            reserved_frame_chars=200,
            max_capsule_chars=800,
            max_intermediate_chars=800,
            max_group_size=2,
            max_reduction_depth=3,
        )
        client = RecordingClient(merge_reply="M" * 800)
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(task, plan, results_of([800] * 16))
        kinds = [k for k, _ in client.calls]
        self.assertNotIn("final", kinds)
        self.assertEqual(kinds.count("merge"), 8 + 4 + 2)  # exactly three levels
        self.assertIn(FALLBACK_HEADLINE, answer)
        self.assertIn("reduction depth limit reached", answer)
        for size in client.sizes():
            self.assertLessEqual(size, policy.max_call_input_chars)

    def test_intermediate_results_are_bounded(self) -> None:
        titles = ["S0", "S1", "S2"]
        contents = ["a" * 400, "b" * 400, "c" * 400]
        base = measure_final_frame(contents, titles)
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=base - 1, reserved_frame_chars=100,
            max_capsule_chars=500, max_intermediate_chars=200,
        )
        client = RecordingClient(merge_reply="R" * 5000)  # oversize reply
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        synthesizer.synthesize(
            task, plan,
            [result(f"s{i}", titles[i], contents[i]) for i in range(3)],
        )
        # Whatever happened next, no later prompt embedded the unbounded reply.
        for kind, text in client.inputs:
            if kind in ("merge", "final"):
                self.assertNotIn("R" * 1000, text)

    def test_strictly_bounded_call_count(self) -> None:
        policy = DEFAULT_SYNTHESIS_POLICY
        client = RecordingClient()
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        synthesizer.synthesize(task, plan, results_of([5000] * 40))
        ceiling = (policy.max_groups * policy.max_reduction_depth  # merges
                   + 1                                             # final
                   + policy.max_chunks_per_result * 40)            # chunking
        self.assertLessEqual(len(client.calls), ceiling)
        for size in client.sizes():
            self.assertLessEqual(size, policy.max_call_input_chars)

    def test_true_run_level_call_ceiling_is_independent_of_result_count(self) -> None:
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=5000, reserved_frame_chars=1000,
            max_capsule_chars=1000, max_intermediate_chars=1000,
            max_group_size=3, max_groups=4, max_reduction_depth=2,
            max_chunks_per_result=3, max_calls_per_run=10,
            max_fallback_chars=4000,
        )
        client = RecordingClient(chunk_reply="digest")
        synthesizer = make_synthesizer(client, policy=policy)
        task, plan = task_and_plan()
        answer = synthesizer.synthesize(
            task, plan,
            [result(f"s{i}", f"S{i}", "x" * 7000 + f"TAIL-{i}")
             for i in range(30)],
        )
        self.assertEqual(len(client.calls), policy.max_calls_per_run)
        self.assertLessEqual(len(answer), policy.max_fallback_chars)
        self.assertIn("omitted", answer)

    def test_web_wrapper_characters_are_included_in_client_measurement(self) -> None:
        from webllm.client import WebAutomationLLMClient, render_messages

        task, plan = task_and_plan()
        messages = __import__(
            "dozen.prompts", fromlist=["build_synthesizer_messages"]
        ).build_synthesizer_messages(task, plan.synthesis_strategy, [("A", "x" * 100)])
        client = WebAutomationLLMClient.__new__(WebAutomationLLMClient)
        measured = client.measure_input_chars(messages)
        self.assertEqual(measured, len(render_messages(messages)))
        self.assertGreater(measured, sum(len(m.content) for m in messages))


class TestDuplicatesAndProvenance(unittest.TestCase):
    def test_duplicate_content_is_suppressed_but_distinct_kept(self) -> None:
        client = RecordingClient()
        synthesizer = make_synthesizer(client)
        task, plan = task_and_plan()
        same = "Identical body produced twice by different subtasks."
        synthesizer.synthesize(task, plan, [
            result("s0", "First", same),
            result("s1", "Second", same),
            result("s2", "Third", "A genuinely distinct body of evidence."),
        ])
        final_input = next(text for kind, text in client.inputs
                           if kind == "final")
        self.assertEqual(final_input.count(same), 1)
        self.assertIn("identical to the output of 'First'", final_input)
        self.assertIn("A genuinely distinct body of evidence.", final_input)

    def test_merged_nodes_preserve_subtask_references(self) -> None:
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=24000, reserved_frame_chars=4000,
            max_capsule_chars=6000, max_intermediate_chars=100,
        )
        client = RecordingClient()
        synthesizer = make_synthesizer(client, policy=policy)
        capsules = [
            build_capsule(result("sa", "Alpha", "alpha body"), policy),
            build_capsule(result("sb", "Beta", "beta body"), policy),
        ]
        merged = synthesizer._merge_group(capsules, 1, 1, policy)
        self.assertEqual(merged.depends_on, ("sa", "sb"))
        self.assertEqual(merged.subtask_id, "sa+sb")
        self.assertIn("Alpha", merged.title)
        self.assertIn("Beta", merged.title)

    def test_contradictions_survive_merging(self) -> None:
        policy = DEFAULT_SYNTHESIS_POLICY
        client = RecordingClient()
        synthesizer = make_synthesizer(client, policy=policy)
        capsules = [
            build_capsule(result("sa", "Alpha", "alpha body",
                                 error="conflicts with Beta about limits"),
                          policy),
            build_capsule(result("sb", "Beta", "beta body"), policy),
        ]
        merged = synthesizer._merge_group(capsules, 1, 1, policy)
        self.assertIn("conflicts with Beta about limits", merged.warnings)


if __name__ == "__main__":
    unittest.main()
