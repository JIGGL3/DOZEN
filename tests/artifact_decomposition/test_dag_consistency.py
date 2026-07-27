"""Phase 4B — the package graph and the subtask DAG must agree."""

from __future__ import annotations

import unittest

from dozen.artifacts import (
    ArtifactContractError,
    ArtifactValidation,
    ValidationKind,
    WorkPackage,
    WorkPackageKind,
)
from dozen.decomposition import validate_artifact_decomposition

from .harness import (
    REACT_PACKAGES,
    REACT_SUBTASK_EDGES,
    artifact,
    manifest_of,
    plan_of,
    react_manifest,
    react_packages,
    react_plan,
    react_work_plan,
    work_plan_of,
)

CRITERION = ("the artifact is complete",)


def package(pid: str, owns=(), **kwargs) -> WorkPackage:
    return WorkPackage(id=pid, title=pid, objective=f"Produce {pid}.",
                       owns=tuple(owns), completion_criteria=CRITERION,
                       **kwargs)


class TestPackageDagConsistency(unittest.TestCase):
    def test_reference_react_decomposition_passes(self) -> None:
        check = validate_artifact_decomposition(react_plan())
        self.assertEqual(check.fatal, ())

    def test_package_dependency_reflected_in_subtask_dependency_passes(self) -> None:
        a, b = artifact("src/a.py"), artifact("src/b.py")
        wp = work_plan_of(
            manifest_of([a, b]),
            [package("pa", [a.id]),
             package("pb", [b.id], depends_on=("pa",),
                     input_artifact_ids=(a.id,))],
            [("pa", "sa"), ("pb", "sb")],
        )
        check = validate_artifact_decomposition(
            plan_of([("sa", []), ("sb", ["sa"])], wp)
        )
        self.assertEqual(check.fatal, ())

    def test_missing_subtask_dependency_fails(self) -> None:
        a, b = artifact("src/a.py"), artifact("src/b.py")
        wp = work_plan_of(
            manifest_of([a, b]),
            [package("pa", [a.id]),
             package("pb", [b.id], depends_on=("pa",),
                     input_artifact_ids=(a.id,))],
            [("pa", "sa"), ("pb", "sb")],
        )
        check = validate_artifact_decomposition(
            plan_of([("sa", []), ("sb", [])], wp)  # sb does NOT depend on sa
        )
        self.assertTrue(
            any("does not depend" in msg for msg in check.fatal), check.fatal
        )

    def test_same_subtask_package_dependency_passes(self) -> None:
        a, b = artifact("src/a.py"), artifact("src/b.py")
        wp = work_plan_of(
            manifest_of([a, b]),
            [package("pa", [a.id]),
             package("pb", [b.id], depends_on=("pa",),
                     input_artifact_ids=(a.id,))],
            [("pa", "s1"), ("pb", "s1")],  # both in ONE subtask
        )
        check = validate_artifact_decomposition(plan_of([("s1", [])], wp))
        self.assertEqual(check.fatal, ())

    def test_transitive_subtask_dependency_satisfies_the_package_edge(self) -> None:
        a, b, c = artifact("src/a.py"), artifact("src/b.py"), artifact("src/c.py")
        wp = work_plan_of(
            manifest_of([a, b, c]),
            [package("pa", [a.id]),
             package("pb", [b.id], depends_on=("pa",)),
             package("pc", [c.id], depends_on=("pa", "pb"))],
            [("pa", "sa"), ("pb", "sb"), ("pc", "sc")],
        )
        # sc depends on sb only; sa is reachable transitively.
        check = validate_artifact_decomposition(
            plan_of([("sa", []), ("sb", ["sa"]), ("sc", ["sb"])], wp)
        )
        self.assertEqual(check.fatal, ())

    def test_validation_before_producer_fails(self) -> None:
        source = artifact("src/a.py")
        result = artifact("test-result", kind="test_result")
        validation = ArtifactValidation(
            id="v1", kind=ValidationKind.UNIT_TESTS,
            target_artifact_ids=(source.id,), produces_artifact_id=result.id,
            success_criterion="tests pass",
        )
        wp = work_plan_of(
            manifest_of([source, result], [validation]),
            [package("impl", [source.id]),
             package("verify", [result.id], kind=WorkPackageKind.VALIDATION,
                     depends_on=("impl",), validation_ids=("v1",))],
            [("impl", "build"), ("verify", "check")],
        )
        check = validate_artifact_decomposition(
            plan_of([("build", []), ("check", [])], wp)  # check runs in parallel
        )
        self.assertTrue(
            any("validation package 'verify' would execute before producer"
                in msg for msg in check.fatal),
            check.fatal,
        )

    def test_integration_before_input_fails(self) -> None:
        a, bundle = artifact("src/a.py"), artifact("dist/bundle.js")
        wp = work_plan_of(
            manifest_of([a, bundle]),
            [package("impl", [a.id]),
             package("integrate", [bundle.id],
                     kind=WorkPackageKind.INTEGRATION,
                     depends_on=("impl",), input_artifact_ids=(a.id,))],
            [("impl", "build"), ("integrate", "assemble")],
        )
        check = validate_artifact_decomposition(
            plan_of([("build", []), ("assemble", [])], wp)
        )
        self.assertTrue(
            any("integration package 'integrate' would execute before its input"
                in msg for msg in check.fatal),
            check.fatal,
        )

    def test_artifact_dependency_without_a_package_path_is_rejected(self) -> None:
        # Phase 4A already forbids consuming another package's artifact with no
        # declared package dependency; that invariant still holds under 4B.
        a = artifact("src/a.py")
        b = artifact("src/b.py", depends_on=(a.id,))
        with self.assertRaises(ArtifactContractError) as ctx:
            work_plan_of(
                manifest_of([a, b]),
                [package("pa", [a.id]),
                 package("pb", [b.id], input_artifact_ids=(a.id,))],
                [("pa", "sa"), ("pb", "sb")],
            )
        self.assertIn("not declared dependencies", str(ctx.exception))

    def test_correct_multi_level_dependency_chain_passes(self) -> None:
        check = validate_artifact_decomposition(react_plan())
        self.assertEqual(check.fatal, ())
        chain = {sid: deps for sid, deps in REACT_SUBTASK_EDGES}
        self.assertEqual(chain["dashboard"], ["shell", "shared-ui"])
        self.assertIn("dashboard", chain["tests"])

    def test_parallel_independent_packages_remain_parallel(self) -> None:
        # shell and shared-ui both depend only on foundation: neither subtask
        # depends on the other, and validation does not demand that they do.
        plan = react_plan()
        by_id = {s.id: s for s in plan.subtasks}
        self.assertNotIn("shared-ui", by_id["shell"].depends_on)
        self.assertNotIn("shell", by_id["shared-ui"].depends_on)
        self.assertEqual(validate_artifact_decomposition(plan).fatal, ())

    def test_upstream_package_mapped_to_downstream_subtask_fails(self) -> None:
        # 'project-foundation' is mapped to the LAST subtask, so every package
        # that depends on it now precedes its producer.
        subtask_map = tuple(
            (pid, "validation" if pid == "project-foundation" else sub)
            for pid, (sub, _o, _d) in REACT_PACKAGES.items()
        )
        wp = work_plan_of(react_manifest(), react_packages(), subtask_map)
        check = validate_artifact_decomposition(react_plan(work_plan=wp))
        self.assertTrue(
            any("does not depend" in msg for msg in check.fatal), check.fatal
        )

    def test_validation_package_after_all_producers_passes(self) -> None:
        plan = react_plan()
        by_id = {s.id: s for s in plan.subtasks}
        for producer in ("foundation", "shell", "shared-ui", "dashboard", "tests"):
            self.assertIn(producer, by_id["validation"].depends_on)
        self.assertEqual(validate_artifact_decomposition(plan).fatal, ())

    def test_work_plan_reference_example_is_structurally_valid(self) -> None:
        wp = react_work_plan()
        self.assertEqual(len(wp.manifest.artifacts), 11)
        self.assertEqual(len(wp.packages), 6)
        self.assertEqual(wp.owner_of("src/app/App.tsx"), "application-shell")
        self.assertEqual(wp.subtask_for("dashboard-tests"), "tests")


if __name__ == "__main__":
    unittest.main()
