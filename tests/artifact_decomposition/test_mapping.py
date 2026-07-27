"""Phase 4B — executable-package to subtask mapping."""

from __future__ import annotations

import unittest

from dozen.artifacts import (
    ArtifactValidation,
    ValidationKind,
    WorkPackage,
    WorkPackageKind,
)
from dozen.decomposition import validate_artifact_decomposition

from .harness import (
    REACT_PACKAGES,
    artifact,
    manifest_of,
    plan_of,
    react_manifest,
    react_packages,
    react_plan,
    work_plan_of,
)


def package(pid: str, owns=(), **kwargs) -> WorkPackage:
    return WorkPackage(id=pid, title=pid, objective=f"Produce {pid}.",
                       owns=tuple(owns), **kwargs)


CRITERION = ("the artifact is complete",)


class TestExecutablePackageMapping(unittest.TestCase):
    def test_every_executable_package_mapped_passes(self) -> None:
        check = validate_artifact_decomposition(react_plan())
        self.assertEqual(check.fatal, ())

    def test_unmapped_producing_package_fails(self) -> None:
        subtask_map = tuple(
            (pid, sub) for pid, (sub, _o, _d) in REACT_PACKAGES.items()
            if pid != "shared-ui"
        )
        wp = work_plan_of(react_manifest(), react_packages(), subtask_map)
        check = validate_artifact_decomposition(react_plan(work_plan=wp))
        self.assertTrue(
            any("executable package 'shared-ui'" in msg for msg in check.fatal),
            check.fatal,
        )

    def test_every_executable_kind_requires_a_subtask(self) -> None:
        for kind in (
            WorkPackageKind.IMPLEMENTATION,
            WorkPackageKind.MODIFICATION,
            WorkPackageKind.INTEGRATION,
            WorkPackageKind.REVIEW,
        ):
            with self.subTest(kind=kind):
                spec = artifact("src/a.py")
                wp = work_plan_of(
                    manifest_of([spec]),
                    [package("p", [spec.id], kind=kind,
                             completion_criteria=CRITERION)],
                    (),  # no mapping
                )
                check = validate_artifact_decomposition(plan_of([("s", [])], wp))
                self.assertTrue(
                    any("has no mapped subtask" in msg for msg in check.fatal),
                    check.fatal,
                )

    def test_validation_package_producing_evidence_requires_a_subtask(self) -> None:
        source = artifact("src/a.py")
        result = artifact("test-result", kind="test_result")
        validation = ArtifactValidation(
            id="v1", kind=ValidationKind.UNIT_TESTS,
            target_artifact_ids=(source.id,), produces_artifact_id=result.id,
            success_criterion="tests pass",
        )
        wp = work_plan_of(
            manifest_of([source, result], [validation]),
            [
                package("impl", [source.id], completion_criteria=CRITERION),
                package("verify", [result.id],
                        kind=WorkPackageKind.VALIDATION,
                        validation_ids=("v1",), depends_on=("impl",),
                        completion_criteria=CRITERION),
            ],
            [("impl", "a")],  # 'verify' left unmapped
        )
        check = validate_artifact_decomposition(plan_of([("a", []), ("b", ["a"])], wp))
        self.assertTrue(
            any("executable package 'verify'" in msg for msg in check.fatal),
            check.fatal,
        )

    def test_declarative_coordination_package_may_remain_unmapped(self) -> None:
        spec = artifact("src/a.py")
        wp = work_plan_of(
            manifest_of([spec]),
            [
                package("impl", [spec.id], completion_criteria=CRITERION),
                WorkPackage(id="coord", title="Coordination",
                            objective="Track sequencing only.",
                            kind=WorkPackageKind.COORDINATION,
                            depends_on=("impl",)),
            ],
            [("impl", "a")],
        )
        check = validate_artifact_decomposition(plan_of([("a", [])], wp))
        self.assertEqual(check.fatal, ())
        self.assertTrue(
            any("remains declarative" in msg for msg in check.advisory),
            check.advisory,
        )

    def test_unknown_subtask_mapping_fails(self) -> None:
        spec = artifact("src/a.py")
        wp = work_plan_of(
            manifest_of([spec]),
            [package("impl", [spec.id], completion_criteria=CRITERION)],
            [("impl", "ghost")],
        )
        check = validate_artifact_decomposition(plan_of([("a", [])], wp))
        self.assertTrue(
            any("unknown subtask 'ghost'" in msg for msg in check.fatal),
            check.fatal,
        )

    def test_conflicting_mapping_for_one_package_is_rejected_by_the_contract(self) -> None:
        from dozen.artifacts import ArtifactContractError
        spec = artifact("src/a.py")
        with self.assertRaises(ArtifactContractError) as ctx:
            work_plan_of(
                manifest_of([spec]),
                [package("impl", [spec.id], completion_criteria=CRITERION)],
                [("impl", "a"), ("impl", "b")],
            )
        self.assertIn("more than one subtask", str(ctx.exception))

    def test_multiple_bounded_packages_may_share_one_subtask(self) -> None:
        specs = [artifact(f"src/a{i}.py") for i in range(4)]
        wp = work_plan_of(
            manifest_of(specs),
            [
                package("p1", [specs[0].id, specs[1].id],
                        completion_criteria=CRITERION),
                package("p2", [specs[2].id, specs[3].id],
                        completion_criteria=CRITERION),
            ],
            [("p1", "a"), ("p2", "a")],
        )
        check = validate_artifact_decomposition(plan_of([("a", [])], wp))
        self.assertEqual(check.fatal, ())

    def test_unbounded_combined_package_scope_fails(self) -> None:
        # Two individually bounded packages whose COMBINED ownership exceeds the
        # per-subtask limit may not share one subtask.
        total = 22  # > max_owned_artifacts_per_package (20)
        specs = [artifact(f"src/a{i}.py") for i in range(total + 4)]
        half = total // 2
        wp = work_plan_of(
            manifest_of(specs),
            [
                package("p1", [s.id for s in specs[:half]],
                        completion_criteria=CRITERION),
                package("p2", [s.id for s in specs[half:total]],
                        completion_criteria=CRITERION),
                package("p3", [s.id for s in specs[total:]],
                        completion_criteria=CRITERION),
            ],
            [("p1", "a"), ("p2", "a"), ("p3", "b")],
        )
        check = validate_artifact_decomposition(plan_of([("a", []), ("b", [])], wp))
        self.assertTrue(
            any("combined" in msg and "'a'" in msg for msg in check.fatal),
            check.fatal,
        )

    def test_combined_packages_cannot_own_a_large_manifest_wholesale(self) -> None:
        # Each package is individually below 80%, but together the one worker
        # would silently receive every file in an eleven-artifact deliverable.
        specs = [artifact(f"src/dashboard_{i}.py") for i in range(11)]
        wp = work_plan_of(
            manifest_of(specs),
            [
                package("p1", [s.id for s in specs[:6]],
                        completion_criteria=CRITERION),
                package("p2", [s.id for s in specs[6:]],
                        completion_criteria=CRITERION),
            ],
            [("p1", "only"), ("p2", "only")],
        )
        check = validate_artifact_decomposition(plan_of([("only", [])], wp))
        self.assertTrue(
            any("collectively" in msg and "wholesale" in msg
                for msg in check.fatal),
            check.fatal,
        )

    def test_a_subtask_does_not_silently_inherit_every_package(self) -> None:
        # All six React packages mapped into ONE subtask exceeds the bounded
        # packages-per-subtask limit, so it cannot silently absorb the manifest.
        subtask_map = tuple((pid, "only") for pid in REACT_PACKAGES)
        wp = work_plan_of(react_manifest(), react_packages(), subtask_map)
        check = validate_artifact_decomposition(
            plan_of([("only", [])], wp)
        )
        self.assertTrue(
            any("share one subtask" in msg for msg in check.fatal), check.fatal
        )

    def test_mapping_remains_deterministic_after_internal_id_conversion(self) -> None:
        # The same package/subtask relation validated twice yields the same
        # result, and derivation keys off the INTERNAL subtask ids the plan holds.
        plan = react_plan()
        first = validate_artifact_decomposition(plan)
        second = validate_artifact_decomposition(plan)
        self.assertEqual(first, second)
        mapped = {sid for _pid, sid in plan.artifact_plan.subtask_map}
        self.assertEqual(mapped, {s.id for s in plan.subtasks})


if __name__ == "__main__":
    unittest.main()
