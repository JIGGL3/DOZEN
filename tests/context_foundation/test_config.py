"""Configuration objects: defaults are valid, round-trips are exact,
bad values are reported structurally."""

from __future__ import annotations

import unittest

from dozen.context.config.models import (
    ContextConfig,
    EstimatorConfig,
    PipelineConfig,
    SummaryPolicy,
    WindowPolicy,
)


class TestConfig(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(ContextConfig.defaults().validate(), [])

    def test_round_trip(self) -> None:
        cfg = ContextConfig.defaults()
        cfg.window.reserved_for_response_tokens = 4096
        cfg.estimator.chars_per_token["prose"] = 3.9
        cfg.feature_flags["memory"] = True
        again = ContextConfig.from_json(cfg.to_json())
        self.assertEqual(again.to_dict(), cfg.to_dict())
        self.assertEqual(again.window.reserved_for_response_tokens, 4096)
        self.assertTrue(again.feature_flags["memory"])

    def test_window_policy_validation(self) -> None:
        bad = WindowPolicy(name="", min_verbatim_tail_tokens=-5)
        problems = bad.validate()
        self.assertTrue(any("name" in p for p in problems))
        self.assertTrue(any("min_verbatim_tail_tokens" in p for p in problems))

    def test_summary_policy_validation(self) -> None:
        bad = SummaryPolicy(max_level=0, target_compression_ratio=2.0,
                            placeholder_template="no placeholders here")
        problems = bad.validate()
        self.assertEqual(len(problems), 3)

    def test_estimator_validation(self) -> None:
        bad = EstimatorConfig(chars_per_token={"prose": 0.0}, safety_margin=1.5)
        problems = bad.validate()
        self.assertTrue(any("chars_per_token" in p for p in problems))
        self.assertTrue(any("safety_margin" in p for p in problems))

    def test_pipeline_validation_rejects_duplicates(self) -> None:
        bad = PipelineConfig(stage_order=["load", "load"])
        self.assertTrue(any("duplicates" in p for p in bad.validate()))

    def test_unknown_config_fields_survive(self) -> None:
        data = ContextConfig.defaults().to_dict()
        data["future_subsystem"] = {"enabled": True}
        parsed = ContextConfig.from_dict(data)
        self.assertEqual(parsed.to_dict()["future_subsystem"], {"enabled": True})


if __name__ == "__main__":
    unittest.main()
