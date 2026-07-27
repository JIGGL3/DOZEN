"""Phase 4F — trusted result ordering: plan order in, byte-identical text out."""

from __future__ import annotations

import random
import unittest

from dozen.artifact_results import assemble_deliverable
from dozen.finalization import (
    FinalizationError,
    build_ordered_sections,
    render_code_only_deliverable,
    render_ordered_sections,
)
from dozen.models import Plan, SubTask, SubTaskResult, TaskStatus

from ..artifact_results.harness import (
    collect_dashboard,
    react_work_plan,
)


def make_plan(count: int = 4, *, chained: bool = False) -> Plan:
    subtasks = []
    for index in range(1, count + 1):
        subtasks.append(SubTask(
            title=f"Section {index}",
            instruction=f"Produce section {index}.",
            id=f"s{index}",
            depends_on=[f"s{index - 1}"] if chained and index > 1 else [],
        ))
    return Plan(analysis="a", subtasks=subtasks, synthesis_strategy="s")


def result_for(sid: str, body: str,
               status: TaskStatus = TaskStatus.COMPLETED) -> SubTaskResult:
    return SubTaskResult(subtask_id=sid, title=f"Title {sid}", status=status,
                         output=body)


class TestPlanOrder(unittest.TestCase):
    def render(self, plan, results) -> str:
        return render_ordered_sections(build_ordered_sections(plan, results))

    def test_sections_follow_plan_order_not_result_order(self) -> None:
        plan = make_plan(3)
        results = [
            result_for("s3", "Third body."),
            result_for("s1", "First body."),
            result_for("s2", "Second body."),
        ]
        text = self.render(plan, results)
        self.assertLess(text.index("First body."), text.index("Second body."))
        self.assertLess(text.index("Second body."), text.index("Third body."))

    def test_reverse_and_random_completion_are_byte_identical(self) -> None:
        plan = make_plan(6)
        results = [result_for(f"s{i}", f"Body number {i}.") for i in range(1, 7)]
        baseline = self.render(plan, results)
        self.assertEqual(self.render(plan, list(reversed(results))), baseline)
        rng = random.Random(4)
        for _ in range(5):
            shuffled = list(results)
            rng.shuffle(shuffled)
            self.assertEqual(self.render(plan, shuffled), baseline)

    def test_dependency_chains_keep_allocation_order(self) -> None:
        plan = make_plan(4, chained=True)
        results = [result_for(f"s{i}", f"Chained body {i}.") for i in (4, 2, 3, 1)]
        text = self.render(plan, results)
        positions = [text.index(f"Chained body {i}.") for i in range(1, 5)]
        self.assertEqual(positions, sorted(positions))

    def test_duplicate_result_identities_fail_closed(self) -> None:
        plan = make_plan(2)
        results = [
            result_for("s1", "Original first body."),
            result_for("s1", "Impostor body."),
            result_for("s2", "Second body."),
        ]
        with self.assertRaises(FinalizationError) as ctx:
            build_ordered_sections(plan, results)
        self.assertIn("duplicate", str(ctx.exception).lower())
        self.assertIn("s1", str(ctx.exception))

    def test_results_outside_the_plan_fail_closed(self) -> None:
        plan = make_plan(2)
        results = [
            result_for("s1", "First body."),
            result_for("s2", "Second body."),
            result_for("sX", "Injected body from nowhere."),
        ]
        with self.assertRaises(FinalizationError) as ctx:
            self.render(plan, results)
        self.assertIn("trusted plan", str(ctx.exception).lower())

    def test_duplicate_plan_identities_fail_closed(self) -> None:
        plan = make_plan(2)
        plan.subtasks[1].id = "s1"
        with self.assertRaises(FinalizationError) as ctx:
            build_ordered_sections(plan, [result_for("s1", "Body.")])
        self.assertIn("duplicate subtask id", str(ctx.exception).lower())

    def test_recursive_result_renders_at_the_parent_position(self) -> None:
        # The recursive child's completed result occupies the parent subtask's
        # allocated slot — nested children are never flattened elsewhere.
        plan = make_plan(3)
        child_text = "## Child part one\n\nAlpha.\n\n## Child part two\n\nBeta."
        results = [
            result_for("s2", child_text),
            result_for("s1", "Leading body."),
            result_for("s3", "Trailing body."),
        ]
        text = self.render(plan, results)
        self.assertLess(text.index("Leading body."), text.index("Alpha."))
        self.assertLess(text.index("Beta."), text.index("Trailing body."))


class TestManifestOrder(unittest.TestCase):
    def test_artifact_rendering_follows_manifest_order(self) -> None:
        work_plan = react_work_plan()
        collections = collect_dashboard()
        baseline = render_code_only_deliverable(
            assemble_deliverable(work_plan, collections)
        )
        manifest_paths = [spec.path for spec in work_plan.manifest.artifacts]
        positions = [baseline.index(f"### {path}") for path in manifest_paths]
        self.assertEqual(positions, sorted(positions))

    def test_reverse_collection_order_is_byte_identical(self) -> None:
        work_plan = react_work_plan()
        baseline = render_code_only_deliverable(
            assemble_deliverable(work_plan, collect_dashboard())
        )
        reversed_render = render_code_only_deliverable(
            assemble_deliverable(work_plan, list(reversed(collect_dashboard())))
        )
        self.assertEqual(reversed_render, baseline)

    def test_random_collection_order_is_byte_identical(self) -> None:
        work_plan = react_work_plan()
        baseline = render_code_only_deliverable(
            assemble_deliverable(work_plan, collect_dashboard())
        )
        rng = random.Random(11)
        for _ in range(4):
            shuffled = collect_dashboard()
            rng.shuffle(shuffled)
            self.assertEqual(
                render_code_only_deliverable(
                    assemble_deliverable(work_plan, shuffled)
                ),
                baseline,
            )


if __name__ == "__main__":
    unittest.main()
