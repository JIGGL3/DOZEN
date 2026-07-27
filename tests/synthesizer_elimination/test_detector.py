"""Phase 4G — unit coverage for the synthesis-role detector and guards.

These tests pin the boundary between a FORBIDDEN synthesis/merge/integration/
final-assembly delegation and legitimate engineering work, and confirm the
plan-level validator and the executor dispatch guard agree with the detector.
"""

from __future__ import annotations

import unittest

from dozen.models import Plan, SubTask
from dozen.synthesis_guard import (
    MODEL_BASED_SYNTHESIS_ENABLED,
    detect_synthesis_role,
    forbidden_synthesis_reason,
    validate_no_synthesis_tasks,
)


class TestInvariant(unittest.TestCase):
    def test_model_based_synthesis_is_disabled(self) -> None:
        self.assertFalse(MODEL_BASED_SYNTHESIS_ENABLED)


class TestDetector(unittest.TestCase):
    # The exact roles the live run produced.
    LIVE_FAILURES = [
        dict(title="Integrate runtime components", dependency_count=3),
        dict(title="Final integration review", dependency_count=3),
        dict(title="Final project assembly", dependency_count=3),
        dict(title="Merge multiple worker sections", dependency_count=3),
    ]

    FORBIDDEN = [
        dict(instruction="Combine all the subtask outputs into one final answer."),
        dict(instruction="Synthesise the results from the other agents into one document."),
        dict(instruction="Consolidate the complete codebase into a single reply."),
        dict(instruction="Trust each output and merge it into the final deliverable."),
        dict(instruction="Return part 1 of 2 of the merged project."),
        dict(instruction="Produce the complete merged part now."),
        dict(instruction="Assemble the final project from every worker response."),
        dict(instruction="Stitch together the worker sections into the final report."),
        dict(instruction="Perform the final synthesis of all delegated results."),
        dict(title="Final QA review", instruction="Review and combine the sections."),
        dict(instruction="Create one unified deliverable from previous tasks",
             dependency_count=3),
        dict(instruction="Reconcile all upstream work into the definitive answer",
             dependency_count=3),
        dict(instruction="Produce the release candidate using every dependency",
             dependency_count=3),
        dict(instruction="Build a consolidated implementation from earlier modules",
             dependency_count=3),
        dict(instruction="Turn the previous task outputs into the final response",
             dependency_count=3),
        dict(instruction="Resolve and combine all dependent sections",
             dependency_count=3),
    ]

    ALLOWED = [
        dict(title="Verification and synthesis",
             instruction="Write the verifier and synthesis modules.",
             dependency_count=1),
        dict(title="Recommendation",
             instruction="Write the final recommendation.", dependency_count=0),
        dict(instruction="Write main.py that imports and wires the modules together.",
             dependency_count=1),
        dict(instruction="Integrate the payment API with the checkout page.",
             dependency_count=1),
        dict(instruction="Write a Python script that merges two CSV files by key.",
             dependency_count=3),
        dict(instruction="Research and compare four message queues."),
        dict(title="Executive summary",
             instruction="Write a one-paragraph summary of the design.",
             dependency_count=1),
        # Soft integration wording WITHOUT fan-in is legitimate.
        dict(instruction="Combine the modules into a single package init file.",
             dependency_count=1),
    ]

    def test_live_failures_are_all_flagged(self) -> None:
        for case in self.LIVE_FAILURES:
            with self.subTest(case=case):
                self.assertIsNotNone(detect_synthesis_role(**case))

    def test_forbidden_cases_are_flagged(self) -> None:
        for case in self.FORBIDDEN:
            with self.subTest(case=case):
                self.assertIsNotNone(detect_synthesis_role(**case))

    def test_allowed_cases_pass_through(self) -> None:
        for case in self.ALLOWED:
            with self.subTest(case=case):
                self.assertIsNone(detect_synthesis_role(**case))

    def test_soft_integration_requires_fan_in(self) -> None:
        # "Integrate the components" alone is allowed; the same wording on a
        # fan-in node (>=2 producers) is a forbidden synthesis role.
        self.assertIsNone(
            detect_synthesis_role(
                instruction="Integrate the runtime components.", dependency_count=1
            )
        )
        self.assertIsNotNone(
            detect_synthesis_role(
                instruction="Integrate the runtime components.", dependency_count=2
            )
        )

    def test_user_saying_synthesise_the_results_is_flagged_as_a_task(self) -> None:
        # A literal "synthesise the results" WORKER INSTRUCTION is a merge role.
        # (The user typing it in their PROMPT is handled end-to-end elsewhere:
        # it must still finalize deterministically, never spawn this task.)
        self.assertIsNotNone(
            detect_synthesis_role(instruction="Synthesise the results and reply.")
        )


class TestPlanValidator(unittest.TestCase):
    def _plan(self, *subtasks: SubTask) -> Plan:
        return Plan(analysis="a", subtasks=list(subtasks), synthesis_strategy="s")

    def test_clean_plan_is_ok(self) -> None:
        plan = self._plan(
            SubTask(title="Auth", instruction="Write the auth module.", id="s1"),
            SubTask(title="DB", instruction="Write the db module.", id="s2"),
        )
        self.assertTrue(validate_no_synthesis_tasks(plan).ok)

    def test_merge_subtask_is_fatal(self) -> None:
        plan = self._plan(
            SubTask(title="Auth", instruction="Write the auth module.", id="s1"),
            SubTask(title="DB", instruction="Write the db module.", id="s2"),
            SubTask(title="Final assembly",
                    instruction="Merge the worker outputs into the final project.",
                    depends_on=["s1", "s2"], id="s3"),
        )
        check = validate_no_synthesis_tasks(plan)
        self.assertFalse(check.ok)
        self.assertIn("s3", check.feedback())

    def test_direct_plan_is_ok(self) -> None:
        plan = Plan(analysis="a", subtasks=[], synthesis_strategy="",
                    direct_answer="here is the answer")
        self.assertTrue(validate_no_synthesis_tasks(plan).ok)

    def test_fan_in_integration_subtask_is_fatal(self) -> None:
        plan = self._plan(
            SubTask(title="A", instruction="Write module A.", id="s1"),
            SubTask(title="B", instruction="Write module B.", id="s2"),
            SubTask(title="Integrate", instruction="Integrate the components.",
                    depends_on=["s1", "s2"], id="s3"),
        )
        self.assertFalse(validate_no_synthesis_tasks(plan).ok)


class TestExecutorGuardHelper(unittest.TestCase):
    def test_forbidden_reason_from_subtask(self) -> None:
        subtask = SubTask(
            title="Final integration review",
            instruction="Review and merge every worker output.",
            depends_on=["s1", "s2", "s3"], id="s4",
        )
        self.assertIsNotNone(forbidden_synthesis_reason(subtask))

    def test_ordinary_subtask_is_allowed(self) -> None:
        subtask = SubTask(
            title="Auth", instruction="Write the auth module.", id="s1"
        )
        self.assertIsNone(forbidden_synthesis_reason(subtask))


if __name__ == "__main__":
    unittest.main()
