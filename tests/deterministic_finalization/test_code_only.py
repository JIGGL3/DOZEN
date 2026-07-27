"""Phase 4F Part G — code-only contract enforcement."""

from __future__ import annotations

import unittest

from dozen.intent import demands_code_only, resolve_contract
from dozen.models import Task
from dozen.planner import PlanError, Planner
from dozen.prompts import artifact_planning_enabled

from ..artifact_decomposition.harness import QueuedPlannerClient, make_planner
from .harness import (
    FAILED_PROMPT,
    FinalizationClient,
    SIZE_REFUSAL_TEXT,
    envelope_for,
    make_orchestrator,
    plan_json,
    typed_entry,
    typed_envelope,
)


class TestCodeOnlyContract(unittest.TestCase):
    def test_the_exact_failed_prompt_resolves_code_only(self) -> None:
        contract = resolve_contract(FAILED_PROMPT)
        self.assertTrue(contract.code_only)
        self.assertTrue(contract.code_required)
        self.assertFalse(contract.architecture_only_allowed)
        self.assertFalse(contract.explanation_only_allowed)

    def test_code_only_forces_artifact_planning(self) -> None:
        contract = resolve_contract(FAILED_PROMPT)
        task = Task(prompt=FAILED_PROMPT, contract=contract)
        self.assertTrue(artifact_planning_enabled(task, 0))

    def test_explicit_prohibition_outranks_code_only(self) -> None:
        contract = resolve_contract(
            "Design the payment system. Do not write code. Code only, please."
        )
        self.assertFalse(contract.code_only)

    def test_review_only_phrase_is_not_promoted(self) -> None:
        contract = resolve_contract(
            "Review the module and note the only code path that matters."
        )
        self.assertFalse(contract.code_only)

    def test_code_only_without_action_signal_still_becomes_implement(self) -> None:
        # The exact live failure: no action verb matched, but "code only" alone
        # must not leave the request at UNKNOWN.
        contract = resolve_contract(
            "make me a thing that does the stuff. (give code only)"
        )
        self.assertEqual(contract.intent.value, "implement")
        self.assertTrue(contract.code_only)

    def test_code_only_wording_variants_bind_the_contract(self) -> None:
        cases = (
            "Only code",
            "Implementation only",
            "Build a runnable project. Do not include commentary.",
            "give code onli",
        )
        for prompt in cases:
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertEqual(contract.intent.value, "implement")
                self.assertTrue(contract.code_required)
                self.assertTrue(contract.code_only)

    def test_code_deliverable_variants_require_implementation(self) -> None:
        for prompt in ("Return complete implementation", "Generate source files"):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertEqual(contract.intent.value, "implement")
                self.assertTrue(contract.code_required)

    def test_negative_near_misses_do_not_promote(self) -> None:
        for prompt in (
            "Only review code",
            "Explain the only code path that matters",
            "Do not write code",
            "Show JSON only",
            "Do not include commentary",
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(resolve_contract(prompt).code_only)

    def test_secondary_detector_and_resolver_share_the_strong_signal(self) -> None:
        positive = (
            "Give code only",
            "Only code",
            "Implementation only",
            "No explanation, just source files",
            "give code onli",
        )
        negative = (
            "Do not give code",
            "Do not write code",
            "Review code only",
            "Explain the code only",
            "JSON only",
            'Explain what the quoted phrase "give code only" means.',
            "Build it. Give code only, but do not write code.",
        )
        for prompt in positive:
            with self.subTest(prompt=prompt):
                self.assertTrue(demands_code_only(prompt))
                self.assertTrue(resolve_contract(prompt).code_only)
        for prompt in negative:
            with self.subTest(prompt=prompt):
                self.assertFalse(demands_code_only(prompt))
                self.assertFalse(resolve_contract(prompt).code_only)

    def test_debug_and_modify_are_intact_code_only_contracts(self) -> None:
        from dozen.finalization import code_only_contract_intact

        for prompt, expected in (
            ("Fix the crash. Give code only.", "debug"),
            ("Update this endpoint. Give code only.", "modify"),
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertEqual(contract.intent.value, expected)
                self.assertTrue(code_only_contract_intact(contract))


class TestPlannerMandate(unittest.TestCase):
    def test_missing_artifact_plan_triggers_one_corrective_replan(self) -> None:
        no_plan = plan_json(include_artifact_plan=False)
        client = QueuedPlannerClient(no_plan, plan_json())
        task = Task(prompt=FAILED_PROMPT, contract=resolve_contract(FAILED_PROMPT))
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIsNotNone(plan.artifact_plan)
        self.assertIn("code-only", client.prompts[1].lower())

    def test_repeated_missing_artifact_plan_fails_planning_cleanly(self) -> None:
        no_plan = plan_json(include_artifact_plan=False)
        client = QueuedPlannerClient(no_plan, no_plan)
        task = Task(prompt=FAILED_PROMPT, contract=resolve_contract(FAILED_PROMPT))
        with self.assertRaises(PlanError) as ctx:
            make_planner(client).plan(task, 0, 2)
        self.assertIn("code-only", str(ctx.exception).lower())


class TestSuccessfulCodeOnlyOutput(unittest.TestCase):
    def test_no_introduction_or_conclusion(self) -> None:
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        self.assertEqual(result.error, "")
        for phrase in ("here is your", "here's your", "i have created",
                       "let me know if"):
            self.assertNotIn(phrase, result.final_answer.lower())

    def test_no_summaries_confidence_or_key_decisions(self) -> None:
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        for token in ('"confidence"', '"key_decisions"', '"summary"',
                     "kept modules deterministic",
                     "Implemented the assigned package."):
            self.assertNotIn(token, result.final_answer)

    def test_code_fences_stay_intact(self) -> None:
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        self.assertIn("```python", result.final_answer)
        self.assertIn("class Orchestrator:", result.final_answer)

    def test_json_source_content_survives_byte_identical(self) -> None:
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        self.assertIn('"max_parallel": 2', result.final_answer)

    def test_failure_diagnostic_only_appears_when_incomplete(self) -> None:
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        self.assertNotIn("could not be completed", result.final_answer)

    def test_every_required_file_appears_exactly_once(self) -> None:
        from .harness import FILES
        client = FinalizationClient()
        result = make_orchestrator(client).run(FAILED_PROMPT)
        for path in FILES:
            self.assertEqual(result.final_answer.count(f"### {path}"), 1, path)


if __name__ == "__main__":
    unittest.main()
