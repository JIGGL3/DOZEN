"""Phase 4C — one subtask's collection result, completion state and feedback."""

from __future__ import annotations

import unittest

from dozen.artifact_results import (
    DEFAULT_RESULT_POLICY,
    CollectionErrorCode,
    SubtaskCollectionResult,
    collect_subtask_artifacts,
)
from dozen.artifacts import ArtifactSpec, WorkPackage

from ..artifact_decomposition.harness import manifest_of, work_plan_of
from .harness import (
    collect_for,
    entry,
    envelope,
    legacy_envelope,
    scope_for,
    worker_envelope_for,
)


class TestSubtaskCollection(unittest.TestCase):
    def test_all_owned_artifacts_returned(self) -> None:
        collection = collect_for("shell", worker_envelope_for("shell"))
        self.assertTrue(collection.engaged)
        self.assertTrue(collection.satisfied)
        self.assertEqual(len(collection.accepted), 3)
        self.assertEqual(collection.missing_required_artifact_ids, ())
        self.assertEqual(collection.package_ids, ("application-shell",))
        self.assertEqual(collection.feedback(), "")

    def test_required_artifact_missing(self) -> None:
        collection = collect_for(
            "shell",
            envelope(
                entry("src/app/App.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            ),
        )
        self.assertFalse(collection.satisfied)
        self.assertEqual(
            collection.missing_required_artifact_ids, ("src/app/router.tsx",)
        )
        self.assertEqual(len(collection.accepted), 2)

    def test_optional_artifact_missing_does_not_break_completion(self) -> None:
        specs = [
            ArtifactSpec(id="a", path="a.py"),
            ArtifactSpec(id="b", path="b.py", required=False),
        ]
        plan = work_plan_of(
            manifest_of(specs),
            [WorkPackage(id="p1", title="impl", owns=("a", "b"))],
            (("p1", "s1"),),
        )
        collection = collect_subtask_artifacts(
            envelope(entry("a", path="a.py", content="print(1)\n")),
            scope_for("s1", plan), work_plan=plan,
        )
        self.assertTrue(collection.satisfied)
        self.assertEqual(collection.missing_required_artifact_ids, ())
        self.assertEqual(collection.missing_optional_artifact_ids, ("b",))

    def test_optional_only_scope_still_requires_the_typed_contract(self) -> None:
        plan = work_plan_of(
            manifest_of([ArtifactSpec(id="b", path="b.py", required=False)]),
            [WorkPackage(id="p1", title="impl", owns=("b",))],
            (("p1", "s1"),),
        )
        collection = collect_subtask_artifacts(
            legacy_envelope(**{"b.py": "print(1)\n"}),
            scope_for("s1", plan),
            work_plan=plan,
        )
        self.assertFalse(collection.engaged)
        self.assertFalse(collection.satisfied)
        self.assertTrue(collection.feedback())

    def test_unexpected_artifact_is_reported(self) -> None:
        collection = collect_for(
            "dashboard",
            envelope(
                entry("src/features/dashboard/DashboardPage.tsx"),
                entry("package.json"),
            ),
        )
        self.assertEqual(collection.unexpected_artifact_ids, ("package.json",))
        self.assertFalse(collection.satisfied)
        self.assertEqual(len(collection.accepted), 1)

    def test_duplicate_artifact_in_one_response_is_reported(self) -> None:
        collection = collect_for(
            "shared-ui",
            envelope(
                entry("src/components/ui/Card.tsx"),
                entry("src/components/ui/Card.tsx", content="a second Card"),
            ),
        )
        self.assertEqual(
            collection.duplicate_artifact_ids, ("src/components/ui/Card.tsx",)
        )
        self.assertEqual(len(collection.accepted), 1)
        self.assertFalse(collection.satisfied)

    def test_invalid_plus_valid_artifacts_keeps_the_valid_ones(self) -> None:
        collection = collect_for(
            "shell",
            envelope(
                entry("src/app/App.tsx"),
                entry("src/app/router.tsx", path="src/router.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            ),
        )
        self.assertEqual(len(collection.accepted), 2)
        self.assertEqual(
            [r.code for r in collection.rejected], [CollectionErrorCode.PATH_MISMATCH]
        )
        self.assertEqual(
            collection.missing_required_artifact_ids, ("src/app/router.tsx",)
        )
        self.assertFalse(collection.satisfied)

    def test_completion_state_is_independent_of_worker_semantics(self) -> None:
        # A perfectly good prose answer that produced no artifacts is NOT
        # artifact-complete, and says so without any semantic verdict involved.
        collection = collect_for("shell", legacy_envelope(notes="I described it."))
        self.assertFalse(collection.engaged)
        self.assertFalse(collection.satisfied)
        self.assertEqual(len(collection.missing_required_artifact_ids), 3)

    def test_a_scope_that_owes_nothing_may_return_nothing(self) -> None:
        plan = work_plan_of(
            manifest_of([ArtifactSpec(id="a", path="a.py")]),
            [
                WorkPackage(id="p1", title="impl", owns=("a",)),
                WorkPackage(id="p2", title="review", validation_ids=(),
                            depends_on=("p1",), input_artifact_ids=("a",)),
            ],
            (("p1", "s1"), ("p2", "s2")),
        )
        scope = scope_for("s2", plan)
        collection = collect_subtask_artifacts(
            legacy_envelope(notes="Reviewed."), scope, work_plan=plan
        )
        self.assertTrue(collection.satisfied)
        self.assertEqual(collection.missing_required_artifact_ids, ())

    def test_stable_ordering(self) -> None:
        first = collect_for("shell", worker_envelope_for("shell"))
        second = collect_for("shell", worker_envelope_for("shell"))
        self.assertEqual(first, second)
        self.assertEqual(
            [c.artifact_id for c in first.accepted],
            [c.artifact_id for c in second.accepted],
        )

    def test_serialization_round_trip(self) -> None:
        collection = collect_for(
            "shell",
            envelope(entry("src/app/App.tsx"), entry("package.json")),
        )
        restored = SubtaskCollectionResult.from_dict(collection.to_dict())
        self.assertEqual(restored, collection)
        self.assertEqual(restored.satisfied, collection.satisfied)

    def test_attempt_and_recursion_provenance_are_recorded(self) -> None:
        collection = collect_for(
            "shell", worker_envelope_for("shell"),
            attempt=2, producer_subtask_id="inner-impl", recursion_depth=1,
        )
        for candidate in collection.accepted:
            self.assertEqual(candidate.attempt, 2)
            self.assertEqual(candidate.producer_subtask_id, "inner-impl")
            self.assertEqual(candidate.recursion_depth, 1)
            self.assertEqual(candidate.subtask_id, "shell")


class TestRepairFeedback(unittest.TestCase):
    def test_feedback_names_the_missing_artifacts(self) -> None:
        collection = collect_for("shell", envelope(entry("src/app/App.tsx")))
        feedback = collection.feedback()
        self.assertIn("src/app/router.tsx", feedback)
        self.assertIn("src/components/layout/DashboardLayout.tsx", feedback)
        self.assertIn("artifacts", feedback)

    def test_feedback_names_rejected_submissions(self) -> None:
        collection = collect_for(
            "dashboard",
            envelope(
                entry("src/features/dashboard/DashboardPage.tsx"),
                entry("package.json"),
            ),
        )
        self.assertIn("owned by another package", collection.feedback())

    def test_feedback_is_bounded(self) -> None:
        collection = collect_for(
            "shell",
            envelope(*[
                entry(f"ghost-{i}", path=f"ghost/{i}.tsx") for i in range(20)
            ]),
        )
        feedback = collection.feedback()
        self.assertLessEqual(len(feedback), DEFAULT_RESULT_POLICY.max_feedback_chars)
        self.assertTrue(feedback)

    def test_a_satisfied_collection_produces_no_feedback(self) -> None:
        self.assertEqual(collect_for("tests", worker_envelope_for("tests")).feedback(), "")


if __name__ == "__main__":
    unittest.main()
