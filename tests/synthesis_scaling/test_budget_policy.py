"""Part E — the immutable authoritative synthesis-input policy."""

from __future__ import annotations

import json
import unittest

from dozen.synthesis_scaling import (
    DEFAULT_SYNTHESIS_POLICY,
    SynthesisBudgetPolicy,
    policy_for_budget,
)


class TestPolicyShape(unittest.TestCase):
    def test_conservative_defaults(self) -> None:
        policy = DEFAULT_SYNTHESIS_POLICY
        self.assertEqual(policy.max_call_input_chars, 24000)
        self.assertEqual(policy.reserved_frame_chars, 4000)
        self.assertEqual(policy.max_capsule_chars, 6000)
        self.assertEqual(policy.max_group_size, 6)
        self.assertEqual(policy.max_groups, 12)
        self.assertEqual(policy.max_reduction_depth, 3)
        self.assertEqual(policy.max_intermediate_chars, 8000)
        self.assertEqual(policy.max_chunks_per_result, 12)
        self.assertEqual(policy.max_calls_per_run, 64)
        self.assertEqual(policy.max_fallback_chars, 24000)
        self.assertEqual(policy.max_diagnostic_chars, 500)
        self.assertEqual(policy.schema_version, 1)
        self.assertEqual(policy.content_budget, 20000)

    def test_policy_is_immutable_and_serializable(self) -> None:
        policy = SynthesisBudgetPolicy()
        with self.assertRaises(Exception):
            policy.max_call_input_chars = 1  # type: ignore[misc]
        payload = json.loads(json.dumps(policy.to_dict()))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["max_call_input_chars"], 24000)
        self.assertEqual(SynthesisBudgetPolicy(**payload), policy)

    def test_invalid_policies_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_call_input_chars=0)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(reserved_frame_chars=24000)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_capsule_chars=0)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_capsule_chars=21000)  # > content budget
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_reduction_depth=0)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_groups=0)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_calls_per_run=0)
        with self.assertRaises(ValueError):
            SynthesisBudgetPolicy(max_fallback_chars=0)

    def test_budget_boundary_exact_and_one_over(self) -> None:
        policy = SynthesisBudgetPolicy()
        self.assertTrue(policy.call_within_budget(policy.max_call_input_chars))
        self.assertFalse(policy.call_within_budget(policy.max_call_input_chars + 1))

    def test_policy_for_budget_scales_and_clamps(self) -> None:
        derived = policy_for_budget(24000)
        self.assertEqual(derived.max_call_input_chars, 24000)
        self.assertEqual(derived.reserved_frame_chars, 4000)
        small = policy_for_budget(6000)
        self.assertEqual(small.max_call_input_chars, 6000)
        self.assertEqual(small.reserved_frame_chars, 1500)
        self.assertLessEqual(small.max_capsule_chars, small.content_budget)
        self.assertLessEqual(small.max_intermediate_chars, small.content_budget)
        with self.assertRaises(ValueError):
            policy_for_budget(0)


if __name__ == "__main__":
    unittest.main()
