"""Part F/G helpers — capsules, explicit bounding, deterministic chunking."""

from __future__ import annotations

import json
import unittest

from dozen.synthesis_scaling import (
    DEFAULT_SYNTHESIS_POLICY,
    SynthesisBudgetPolicy,
    SynthesisCapsule,
    bound_text,
    build_capsule,
    plan_groups,
    split_chunks,
)

from .harness import result


class TestBoundText(unittest.TestCase):
    def test_no_bounding_when_within_limit(self) -> None:
        self.assertEqual(bound_text("short", 100), "short")
        self.assertEqual(bound_text("anything", 0), "anything")  # disabled

    def test_bounding_is_explicit_and_keeps_the_tail(self) -> None:
        text = "HEAD " + ("x" * 5000) + " TAIL-SENTINEL"
        bounded = bound_text(text, 400)
        self.assertLessEqual(len(bounded), 400)
        self.assertIn("characters omitted", bounded)
        self.assertIn("TAIL-SENTINEL", bounded)
        self.assertTrue(bounded.startswith("HEAD"))

    def test_degenerate_limit_still_declares_omission(self) -> None:
        bounded = bound_text("y" * 500, 60)
        self.assertIn("omitted", bounded)


class TestSplitChunks(unittest.TestCase):
    def test_chunks_cover_the_whole_text_in_order(self) -> None:
        text = "".join(f"<{i}>" for i in range(1000))
        chunks = split_chunks(text, 500, 12)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c) <= 500 for c in chunks[:-1]))

    def test_chunk_size_grows_instead_of_dropping_the_tail(self) -> None:
        text = "z" * 10000
        chunks = split_chunks(text, 100, 4)  # would need 100 chunks
        self.assertLessEqual(len(chunks), 4)
        self.assertEqual("".join(chunks), text)

    def test_empty_text(self) -> None:
        self.assertEqual(split_chunks("", 100, 4), [])


class TestCapsules(unittest.TestCase):
    def test_deterministic_construction(self) -> None:
        r = result("s1", "Research", "Findings body.")
        one = build_capsule(r, DEFAULT_SYNTHESIS_POLICY)
        two = build_capsule(r, DEFAULT_SYNTHESIS_POLICY)
        self.assertEqual(one, two)

    def test_capsule_is_immutable_and_serializable(self) -> None:
        capsule = build_capsule(
            result("s1", "Research", "Findings."), DEFAULT_SYNTHESIS_POLICY,
            key_decisions=["chose approach A"],
        )
        with self.assertRaises(Exception):
            capsule.content = "changed"  # type: ignore[misc]
        payload = json.loads(json.dumps(capsule.to_dict()))
        self.assertEqual(payload["subtask_id"], "s1")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["key_decisions"], ["chose approach A"])

    def test_capsule_round_trip_none_handling_and_public_bounds(self) -> None:
        capsule = SynthesisCapsule(
            subtask_id=None, title=None, status=None,  # type: ignore[arg-type]
            content="x" * 9000,
            key_decisions=[None], depends_on=[None], warnings=[None],  # type: ignore[list-item]
            agent_name=None, original_chars=None,  # type: ignore[arg-type]
        )
        restored = SynthesisCapsule(**capsule.to_dict())
        self.assertEqual(restored, capsule)
        self.assertLessEqual(
            len(restored.content),
            DEFAULT_SYNTHESIS_POLICY.max_intermediate_chars,
        )
        self.assertNotIn("None", json.dumps(restored.to_dict()))

    def test_oversized_content_is_bounded_explicitly(self) -> None:
        big = "word " * 3000 + "TAIL-MARK"
        capsule = build_capsule(
            result("s1", "Big", big), DEFAULT_SYNTHESIS_POLICY
        )
        self.assertLessEqual(len(capsule.content),
                             DEFAULT_SYNTHESIS_POLICY.max_capsule_chars)
        self.assertIn("characters omitted", capsule.content)
        self.assertIn("TAIL-MARK", capsule.content)
        self.assertTrue(capsule.truncated)
        self.assertEqual(capsule.original_chars, len(big))

    def test_clean_result_produces_no_warnings(self) -> None:
        capsule = build_capsule(
            result("s1", "Clean", "Output.", feedback="looks fine"),
            DEFAULT_SYNTHESIS_POLICY,
        )
        self.assertEqual(capsule.warnings, ())
        # render() therefore equals the content — legacy prompts unchanged.
        self.assertEqual(capsule.render(), "Output.")

    def test_flagged_result_preserves_its_contradiction(self) -> None:
        capsule = build_capsule(
            result("s1", "Flagged", "Output.",
                   error="contradicts subtask Beta on the cache TTL",
                   feedback="verifier saw conflicting numbers"),
            DEFAULT_SYNTHESIS_POLICY,
        )
        self.assertIn("contradicts subtask Beta on the cache TTL",
                      capsule.warnings)
        self.assertIn("verifier saw conflicting numbers", capsule.warnings)
        self.assertIn("Warnings / open contradictions:", capsule.render())

    def test_provenance_fields_are_kept(self) -> None:
        r = result("s7", "Named", "Body.")
        r.agent_name = "gemini"
        capsule = build_capsule(r, DEFAULT_SYNTHESIS_POLICY)
        self.assertEqual(capsule.subtask_id, "s7")
        self.assertEqual(capsule.agent_name, "gemini")


class TestPlanGroups(unittest.TestCase):
    def test_grouping_is_stable_and_order_preserving(self) -> None:
        sizes = [500, 500, 500, 500, 500]
        one = plan_groups(sizes, DEFAULT_SYNTHESIS_POLICY)
        two = plan_groups(sizes, DEFAULT_SYNTHESIS_POLICY)
        self.assertEqual(one, two)
        flattened = [i for group in one for i in group]
        self.assertEqual(flattened, sorted(flattened))

    def test_groups_respect_content_budget_and_size(self) -> None:
        policy = DEFAULT_SYNTHESIS_POLICY
        sizes = [9000, 9000, 9000, 9000]  # content budget 20000
        groups = plan_groups(sizes, policy)
        for group in groups:
            self.assertLessEqual(sum(sizes[i] for i in group),
                                 policy.content_budget)
            self.assertLessEqual(len(group), policy.max_group_size)

    def test_impossible_max_groups_fails_closed(self) -> None:
        policy = DEFAULT_SYNTHESIS_POLICY
        sizes = [15000] * 100  # greedy would produce 100 singleton groups
        with self.assertRaises(ValueError):
            plan_groups(sizes, policy)

    def test_every_returned_group_respects_all_policy_limits(self) -> None:
        policy = SynthesisBudgetPolicy(
            max_call_input_chars=1000, reserved_frame_chars=100,
            max_capsule_chars=100, max_intermediate_chars=100,
            max_group_size=2, max_groups=3,
        )
        groups = plan_groups([100] * 6, policy)
        self.assertLessEqual(len(groups), policy.max_groups)
        self.assertTrue(all(len(group) <= policy.max_group_size for group in groups))

    def test_empty_input(self) -> None:
        self.assertEqual(plan_groups([], DEFAULT_SYNTHESIS_POLICY), [])


if __name__ == "__main__":
    unittest.main()
