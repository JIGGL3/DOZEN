"""Phase 4B — boundedness policy: deterministic, immutable, declared once."""

from __future__ import annotations

import dataclasses
import unittest

from dozen.decomposition import (
    DEFAULT_DECOMPOSITION_POLICY,
    DecompositionCheck,
    DecompositionPolicy,
    package_requires_subtask,
    validate_artifact_decomposition,
)
from dozen.artifacts import (
    ArtifactKind,
    ArtifactValidation,
    ValidationKind,
    WorkPackage,
    WorkPackageKind,
)

from .harness import artifact, manifest_of, plan_of, work_plan_of

POLICY = DEFAULT_DECOMPOSITION_POLICY


def sources(count: int, start: int = 0) -> list:
    return [artifact(f"src/file_{i}.py") for i in range(start, start + count)]


def checked_manifest(specs, **kwargs):
    """A manifest with one plan-wide validation (an observable completion signal)."""
    validations = kwargs.pop("validations", (
        ArtifactValidation(id="v1", kind=ValidationKind.CUSTOM,
                           success_criterion="output looks correct"),
    ))
    return manifest_of(specs, validations, **kwargs)


def single_subtask_plan(work_plan):
    return plan_of([("work", [])], work_plan)


def package(pid: str, owns, **kwargs) -> WorkPackage:
    return WorkPackage(id=pid, title=pid, objective=f"Produce {pid}.",
                       owns=tuple(owns), **kwargs)


class TestBoundednessPolicy(unittest.TestCase):
    def test_small_one_package_deliverable_passes(self) -> None:
        specs = sources(3)
        wp = work_plan_of(
            checked_manifest(specs),
            [package("all", [s.id for s in specs])],
            [("all", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertEqual(check.fatal, ())

    def test_large_one_package_deliverable_fails(self) -> None:
        specs = sources(POLICY.single_package_manifest_threshold + 3)
        wp = work_plan_of(
            checked_manifest(specs),
            [package("all", [s.id for s in specs])],
            [("all", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertTrue(
            any("wholesale" in msg for msg in check.fatal), check.fatal
        )

    def test_single_package_manifest_threshold_allows_legitimate_seven_files(self) -> None:
        cases = (
            (3, ArtifactKind.SOURCE_FILE, "three-file application", True),
            (6, ArtifactKind.SOURCE_FILE, "six-file application", True),
            (7, ArtifactKind.CONFIG_FILE, "seven small configuration files", True),
            (7, ArtifactKind.SOURCE_FILE, "seven related source files", True),
            (11, ArtifactKind.SOURCE_FILE, "eleven-file dashboard", False),
            (20, ArtifactKind.SOURCE_FILE, "twenty-file package", False),
        )
        for count, kind, label, should_pass in cases:
            with self.subTest(label=label):
                specs = [artifact(f"src/{i}.txt", kind=kind) for i in range(count)]
                wp = work_plan_of(
                    checked_manifest(specs),
                    [package("all", [spec.id for spec in specs])],
                    [("all", "work")],
                )
                check = validate_artifact_decomposition(single_subtask_plan(wp))
                self.assertEqual(check.ok, should_pass, check.fatal)

    def test_wholesale_percentage_boundary(self) -> None:
        # Exact 79% would require 100 artifacts, beyond Phase 4A's manifest
        # cap. 15/19 (78.95%) is the closest default-policy case; 16/20 is 80%.
        for owned, total, wholesale in (
            (15, 19, False), (16, 20, True), (17, 20, True),
        ):
            with self.subTest(owned=owned, total=total):
                specs = [
                    artifact(f"src/f{i}.py", required=i < owned)
                    for i in range(total)
                ]
                wp = work_plan_of(
                    checked_manifest(specs),
                    [package("big", [spec.id for spec in specs[:owned]])],
                    [("big", "work")],
                )
                check = validate_artifact_decomposition(
                    single_subtask_plan(wp)
                )
                self.assertEqual(
                    any("wholesale" in msg for msg in check.fatal),
                    wholesale,
                    check.fatal,
                )

    def _two_package_plan(self, big: int, rest: int):
        specs = sources(big) + sources(rest, start=big)
        big_ids = [s.id for s in specs[:big]]
        rest_ids = [s.id for s in specs[big:]]
        wp = work_plan_of(
            checked_manifest(specs),
            [package("big", big_ids), package("rest", rest_ids)],
            [("big", "work"), ("rest", "other")],
        )
        return plan_of([("work", []), ("other", [])], wp)

    def test_maximum_owned_artifact_boundary_passes(self) -> None:
        # 20 of 26: below both the owns limit and the wholesale fraction.
        limit = POLICY.max_owned_artifacts_per_package
        check = validate_artifact_decomposition(self._two_package_plan(limit, 6))
        self.assertEqual(check.fatal, ())
        self.assertTrue(any("near the limit" in msg for msg in check.advisory))

    def test_excess_owned_artifacts_fail(self) -> None:
        limit = POLICY.max_owned_artifacts_per_package
        check = validate_artifact_decomposition(
            self._two_package_plan(limit + 1, 6)
        )
        self.assertTrue(
            any("split it into smaller packages" in msg for msg in check.fatal),
            check.fatal,
        )

    def _input_plan(self, input_count: int):
        externals = [
            artifact(f"vendor/dep_{i}.json", external=True)
            for i in range(input_count)
        ]
        owned = sources(1)
        wp = work_plan_of(
            checked_manifest(externals + owned),
            [package("impl", [owned[0].id],
                     input_artifact_ids=tuple(e.id for e in externals))],
            [("impl", "work")],
        )
        return single_subtask_plan(wp)

    def test_maximum_input_boundary_passes(self) -> None:
        check = validate_artifact_decomposition(
            self._input_plan(POLICY.max_input_artifacts_per_package)
        )
        self.assertEqual(check.fatal, ())

    def test_excess_inputs_fail(self) -> None:
        check = validate_artifact_decomposition(
            self._input_plan(POLICY.max_input_artifacts_per_package + 1)
        )
        self.assertTrue(
            any("input artifacts" in msg for msg in check.fatal), check.fatal
        )

    def _multi_package_plan(self, count: int):
        specs = sources(count)
        packages = [package(f"p{i}", [specs[i].id]) for i in range(count)]
        wp = work_plan_of(
            checked_manifest(specs),
            packages,
            [(f"p{i}", "work") for i in range(count)],
        )
        return single_subtask_plan(wp)

    def test_maximum_packages_per_subtask_boundary_passes(self) -> None:
        check = validate_artifact_decomposition(
            self._multi_package_plan(POLICY.max_packages_per_subtask)
        )
        self.assertEqual(check.fatal, ())

    def test_excess_mapped_packages_fail(self) -> None:
        check = validate_artifact_decomposition(
            self._multi_package_plan(POLICY.max_packages_per_subtask + 1)
        )
        self.assertTrue(
            any("share one subtask" in msg for msg in check.fatal), check.fatal
        )

    def test_excess_validations_per_package_fail(self) -> None:
        specs = sources(1)
        count = POLICY.max_validations_per_package + 1
        validations = tuple(
            ArtifactValidation(id=f"v{i}", kind=ValidationKind.CUSTOM,
                               target_artifact_ids=(specs[0].id,),
                               success_criterion="ok")
            for i in range(count)
        )
        wp = work_plan_of(
            manifest_of(specs, validations),
            [package("impl", [specs[0].id],
                     validation_ids=tuple(v.id for v in validations))],
            [("impl", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertTrue(
            any("validations" in msg for msg in check.fatal), check.fatal
        )

    def test_maximum_validations_per_package_boundary_passes(self) -> None:
        specs = sources(1)
        validations = tuple(
            ArtifactValidation(
                id=f"v{i}", kind=ValidationKind.CUSTOM,
                target_artifact_ids=(specs[0].id,), success_criterion="ok",
            )
            for i in range(POLICY.max_validations_per_package)
        )
        wp = work_plan_of(
            manifest_of(specs, validations),
            [package("impl", [specs[0].id],
                     validation_ids=tuple(v.id for v in validations))],
            [("impl", "work")],
        )
        self.assertEqual(
            validate_artifact_decomposition(single_subtask_plan(wp)).fatal, ()
        )

    def test_advisory_threshold_exact_boundary(self) -> None:
        for owns, warns in ((15, False), (16, True)):
            with self.subTest(owns=owns):
                check = validate_artifact_decomposition(
                    self._two_package_plan(owns, 6)
                )
                self.assertEqual(
                    any("near the limit" in msg for msg in check.advisory),
                    warns,
                )

    def test_missing_completion_criteria_for_producing_package_fails(self) -> None:
        specs = sources(2)
        wp = work_plan_of(
            manifest_of(specs),  # no validations, no completion criteria
            [package("impl", [s.id for s in specs])],
            [("impl", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertTrue(
            any("no observable completion criterion" in msg
                for msg in check.fatal),
            check.fatal,
        )

    def test_package_criterion_satisfies_completion_rule(self) -> None:
        specs = sources(2)
        wp = work_plan_of(
            manifest_of(specs),
            [package("impl", [s.id for s in specs],
                     completion_criteria=("all files compile",))],
            [("impl", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertEqual(check.fatal, ())

    def test_manifest_wide_signal_downgrades_to_advisory(self) -> None:
        specs = sources(2)
        wp = work_plan_of(
            checked_manifest(specs),
            [package("impl", [s.id for s in specs])],
            [("impl", "work")],
        )
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertEqual(check.fatal, ())
        self.assertTrue(
            any("no package-level completion" in msg for msg in check.advisory)
        )

    def test_missing_objective_for_producing_package_fails(self) -> None:
        specs = sources(1)
        bare = WorkPackage(id="impl", title="", owns=(specs[0].id,))
        wp = work_plan_of(checked_manifest(specs), [bare], [("impl", "work")])
        check = validate_artifact_decomposition(single_subtask_plan(wp))
        self.assertTrue(
            any("no objective" in msg.replace("\n", " ") for msg in check.fatal),
            check.fatal,
        )

    def test_policy_is_deterministic_and_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_DECOMPOSITION_POLICY.max_owned_artifacts_per_package = 1  # type: ignore[misc]
        plan = self._two_package_plan(POLICY.max_owned_artifacts_per_package + 1, 6)
        first = validate_artifact_decomposition(plan)
        second = validate_artifact_decomposition(plan)
        self.assertEqual(first, second)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.fatal = ()  # type: ignore[misc]

    def test_policy_requires_no_provider_and_no_filesystem(self) -> None:
        import dozen.decomposition as module
        for forbidden in ("os", "io", "pathlib", "subprocess", "socket",
                          "requests", "urllib"):
            self.assertFalse(
                hasattr(module, forbidden),
                f"decomposition must not import {forbidden}",
            )

    def test_limits_are_declared_once_and_reach_the_prompt(self) -> None:
        from dozen.prompts import PLANNER_ARTIFACT_PLAN_RULE
        policy = DEFAULT_DECOMPOSITION_POLICY
        self.assertEqual(DecompositionPolicy(), policy)
        self.assertIn(
            f"within {policy.max_owned_artifacts_per_package} owned artifacts",
            PLANNER_ARTIFACT_PLAN_RULE.replace("\\\n", ""),
        )
        self.assertIn(
            f"{policy.max_input_artifacts_per_package} inputs",
            PLANNER_ARTIFACT_PLAN_RULE,
        )
        self.assertIn(
            f"at most {policy.max_packages_per_subtask} packages to one",
            PLANNER_ARTIFACT_PLAN_RULE.replace("\n", " "),
        )
        self.assertIn(
            f"{policy.max_validations_per_package} validations",
            PLANNER_ARTIFACT_PLAN_RULE,
        )

    def test_defaults_are_conservative_fractions_of_plan_limits(self) -> None:
        policy = DEFAULT_DECOMPOSITION_POLICY
        self.assertLessEqual(policy.max_owned_artifacts_per_package, 80)
        self.assertLessEqual(policy.max_packages_per_subtask, 24)
        self.assertLessEqual(policy.max_validations_per_package, 40)
        self.assertGreater(policy.single_package_manifest_threshold, 2)

    def test_plan_without_artifact_plan_is_trivially_valid(self) -> None:
        check = validate_artifact_decomposition(plan_of([("work", [])], None))
        self.assertEqual(check, DecompositionCheck())

    def test_executable_predicate(self) -> None:
        # Producing / validating packages are executable, whatever their kind.
        self.assertTrue(package_requires_subtask(package("p", ["a"])))
        self.assertTrue(package_requires_subtask(
            WorkPackage(id="v", title="v", kind=WorkPackageKind.VALIDATION,
                        validation_ids=("v1",))
        ))
        self.assertTrue(package_requires_subtask(
            WorkPackage(id="r", title="r", kind=WorkPackageKind.REVIEW,
                        owns=("a1",))
        ))
        # Only a package owning nothing, producing nothing and validating
        # nothing may remain declarative.
        self.assertFalse(package_requires_subtask(
            WorkPackage(id="c", title="c", kind=WorkPackageKind.COORDINATION)
        ))


if __name__ == "__main__":
    unittest.main()
