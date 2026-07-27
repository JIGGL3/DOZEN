"""Phase 4A — WorkPackage and ArtifactWorkPlan invariants and serialization."""

from __future__ import annotations

import unittest

from dozen.artifacts import (
    ArtifactContractError,
    ArtifactKind,
    ArtifactManifest,
    ArtifactSpec,
    ArtifactValidation,
    ArtifactWorkPlan,
    ValidationKind,
    WorkPackage,
    WorkPackageKind,
)


def spec(aid: str, path: str, **kwargs) -> ArtifactSpec:
    return ArtifactSpec(id=aid, path=path, **kwargs)


def manifest(artifacts, validations=(), **kwargs) -> ArtifactManifest:
    return ArtifactManifest(
        id=kwargs.pop("id", "m1"),
        title=kwargs.pop("title", "deliverable"),
        artifacts=tuple(artifacts),
        validations=tuple(validations),
        **kwargs,
    )


def plan(manifest_obj, packages, **kwargs) -> ArtifactWorkPlan:
    return ArtifactWorkPlan(
        id=kwargs.pop("id", "wp1"),
        manifest=manifest_obj,
        packages=tuple(packages),
        **kwargs,
    )


class TestWorkPackage(unittest.TestCase):
    def test_defaults(self) -> None:
        package = WorkPackage(id="p1", title="Foundation")
        self.assertEqual(package.kind, WorkPackageKind.IMPLEMENTATION)
        self.assertEqual(package.owns, ())
        self.assertEqual(package.depends_on, ())

    def test_empty_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            WorkPackage(id="", title="t")

    def test_self_dependency_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            WorkPackage(id="p1", title="t", depends_on=("p1",))

    def test_duplicate_owned_artifact_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            WorkPackage(id="p1", title="t", owns=("a1", "a1"))

    def test_outputs_must_be_owned(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            WorkPackage(id="p1", title="t", owns=("a1",),
                        output_artifact_ids=("a2",))
        self.assertIn("does not own", str(ctx.exception))

    def test_unknown_kind_falls_back_to_coordination(self) -> None:
        package = WorkPackage(id="p1", title="t", kind="wizardry")
        self.assertEqual(package.kind, WorkPackageKind.COORDINATION)

    def test_missing_or_null_kind_keeps_constructor_default(self) -> None:
        for payload in ({"id": "p1", "title": "t"},
                        {"id": "p1", "title": "t", "kind": None}):
            with self.subTest(payload=payload):
                self.assertEqual(
                    WorkPackage.from_dict(payload).kind,
                    WorkPackageKind.IMPLEMENTATION,
                )

    def test_scalar_json_arrays_are_rejected_on_deserialization(self) -> None:
        base = WorkPackage(id="p1", title="t").to_dict()
        for field in (
            "owns", "depends_on", "input_artifact_ids", "output_artifact_ids",
            "validation_ids", "completion_criteria", "capability_hints",
        ):
            with self.subTest(field=field):
                data = dict(base)
                data[field] = 42
                with self.assertRaises(ArtifactContractError):
                    WorkPackage.from_dict(data)

    def test_collections_are_immutable(self) -> None:
        owns = ["a1"]
        package = WorkPackage(id="p1", title="t", owns=owns)
        owns.append("a2")
        self.assertEqual(package.owns, ("a1",))
        with self.assertRaises(Exception):
            package.title = "changed"  # type: ignore[misc]

    def test_capability_hints_are_bounded_and_flat(self) -> None:
        package = WorkPackage(
            id="p1", title="t",
            capability_hints=["python\n```code```" * 100] * 30,
        )
        self.assertEqual(len(package.capability_hints), 12)
        self.assertTrue(all(len(item) <= 60 for item in package.capability_hints))
        self.assertTrue(all("\n" not in item for item in package.capability_hints))

    def test_serialization_round_trip(self) -> None:
        package = WorkPackage(
            id="p1", title="Shell", objective="build the shell",
            kind=WorkPackageKind.INTEGRATION, owns=("a1",),
            depends_on=("p0",), input_artifact_ids=("a0",),
            output_artifact_ids=("a1",), validation_ids=("v1",),
            completion_criteria=("compiles",), estimated_size="small",
            contract_intent="implement", capability_hints=("coding",),
        )
        again = WorkPackage.from_dict(package.to_dict())
        self.assertEqual(again, package)

    def test_no_provider_names_field_exists(self) -> None:
        # Providers are routing concerns; the package carries capability hints
        # at most, never a provider/model identity.
        fields = set(WorkPackage("p1", "t").to_dict())
        self.assertNotIn("provider", fields)
        self.assertNotIn("model", fields)
        self.assertIn("capability_hints", fields)

    def test_brief_is_bounded_and_flat(self) -> None:
        package = WorkPackage(
            id="p1", title="t", objective="x\ny\nz" * 200,
            owns=tuple(f"a{i}" for i in range(30)),
            depends_on=("p0",),
        )
        brief = package.to_brief()
        self.assertIn("WORK PACKAGE: p1", brief)
        self.assertIn("(+24 more)", brief)              # ids line clipped
        self.assertLess(len(brief), 800)
        self.assertNotIn("\n\n", brief)


class TestArtifactWorkPlanInvariants(unittest.TestCase):
    def make_manifest(self) -> ArtifactManifest:
        return manifest(
            [
                spec("config", "package.json", kind=ArtifactKind.CONFIG_FILE),
                spec("app", "src/App.tsx", kind=ArtifactKind.SOURCE_FILE),
                spec("tests", "tests/app.test.tsx", kind=ArtifactKind.TEST_FILE),
                spec("design-input", "docs/design.md",
                     kind=ArtifactKind.DOCUMENT, external=True),
            ],
            [ArtifactValidation(id="v1", kind=ValidationKind.UNIT_TESTS,
                                target_artifact_ids=("tests",))],
        )

    def test_valid_plan_constructs(self) -> None:
        wp = plan(self.make_manifest(), [
            WorkPackage(id="foundation", title="Foundation", owns=("config",)),
            WorkPackage(id="shell", title="Shell", owns=("app",),
                        depends_on=("foundation",),
                        input_artifact_ids=("design-input",)),
            WorkPackage(id="tests", title="Tests", owns=("tests",),
                        kind=WorkPackageKind.VALIDATION,
                        depends_on=("shell",), validation_ids=("v1",)),
        ])
        self.assertEqual(wp.owner_of("app"), "shell")
        self.assertEqual(wp.owner_of("design-input"), "")

    def test_duplicate_package_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a", owns=("config",)),
                WorkPackage(id="p1", title="b", owns=("app", "tests")),
            ])
        self.assertIn("duplicate package id", str(ctx.exception))

    def test_duplicate_ownership_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a", owns=("config", "app", "tests")),
                WorkPackage(id="p2", title="b", owns=("app",)),
            ])
        self.assertIn("owned by more than one package", str(ctx.exception))

    def test_claiming_artifact_outside_manifest_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a",
                            owns=("config", "app", "tests", "ghost")),
            ])
        self.assertIn("not in the manifest", str(ctx.exception))

    def test_unknown_package_dependency_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a",
                            owns=("config", "app", "tests"),
                            depends_on=("ghost",)),
            ])
        self.assertIn("unknown package", str(ctx.exception))

    def test_package_cycle_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a", owns=("config",),
                            depends_on=("p2",)),
                WorkPackage(id="p2", title="b", owns=("app", "tests"),
                            depends_on=("p1",)),
            ])
        self.assertIn("cycle", str(ctx.exception))

    def test_required_artifact_without_producer_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a", owns=("config", "app")),
                # "tests" is required, non-external, and unowned.
            ])
        self.assertIn("no owning work package", str(ctx.exception))
        self.assertIn("tests", str(ctx.exception))

    def test_external_inputs_need_no_producer(self) -> None:
        # "design-input" is external: no package owns it, and that is fine.
        wp = plan(self.make_manifest(), [
            WorkPackage(id="p1", title="a", owns=("config", "app", "tests")),
        ])
        self.assertEqual(wp.owner_of("design-input"), "")

    def test_consumed_nonexternal_artifact_needs_an_owner(self) -> None:
        m = manifest([
            spec("optional-input", "optional.py", required=False),
            spec("output", "output.py"),
        ])
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(m, [
                WorkPackage(
                    id="p1", title="a", owns=("output",),
                    input_artifact_ids=("optional-input",),
                ),
            ])
        self.assertIn("non-external artifact", str(ctx.exception))
        self.assertIn("no owning work package", str(ctx.exception))

        dependency_manifest = manifest([
            spec("optional-input", "optional.py", required=False),
            spec("output", "output.py", depends_on=("optional-input",)),
        ])
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(dependency_manifest, [
                WorkPackage(id="p1", title="a", owns=("output",)),
            ])
        self.assertIn("depends on non-external artifact", str(ctx.exception))
        self.assertIn("no owning work package", str(ctx.exception))

    def test_self_owned_modification_may_also_be_an_input(self) -> None:
        m = manifest([
            spec("source", "source.py", operation="modify"),
        ])
        wp = plan(m, [
            WorkPackage(
                id="p1", title="a", owns=("source",),
                input_artifact_ids=("source",),
            ),
        ])
        self.assertEqual(wp.owner_of("source"), "p1")

    def test_external_artifacts_cannot_be_owned_or_emitted(self) -> None:
        external = spec(
            "design-input", "docs/design.md",
            kind=ArtifactKind.DOCUMENT, external=True,
        )
        for package in (
            WorkPackage(id="p1", title="a", owns=("design-input",)),
            WorkPackage(
                id="p1", title="a", owns=("design-input",),
                output_artifact_ids=("design-input",),
            ),
        ):
            with self.subTest(package=package):
                with self.assertRaises(ArtifactContractError) as ctx:
                    plan(manifest([external]), [package])
                self.assertIn("cannot own external artifact", str(ctx.exception))

        declared_owner = spec(
            "design-input", "docs/design.md",
            kind=ArtifactKind.DOCUMENT, external=True, package_id="p1",
        )
        with self.assertRaises(ArtifactContractError):
            plan(
                manifest([declared_owner]),
                [WorkPackage(id="p1", title="a", owns=("design-input",))],
            )

    def test_optional_artifacts_need_no_producer(self) -> None:
        m = manifest([
            spec("must", "a.py"),
            spec("maybe", "b.py", required=False),
        ])
        wp = plan(m, [WorkPackage(id="p1", title="a", owns=("must",))])
        self.assertEqual(wp.owner_of("maybe"), "")

    def test_validation_package_may_inspect_unowned_artifacts(self) -> None:
        wp = plan(self.make_manifest(), [
            WorkPackage(id="build", title="Build",
                        owns=("config", "app", "tests")),
            WorkPackage(id="check", title="Check",
                        kind=WorkPackageKind.VALIDATION,
                        depends_on=("build",),
                        input_artifact_ids=("config", "app", "tests"),
                        validation_ids=("v1",)),
        ])
        self.assertEqual(wp.packages[1].owns, ())

    def test_integration_package_consumes_other_packages_outputs(self) -> None:
        wp = plan(self.make_manifest(), [
            WorkPackage(id="a", title="a", owns=("config",)),
            WorkPackage(id="b", title="b", owns=("app", "tests"),
                        kind=WorkPackageKind.INTEGRATION,
                        depends_on=("a",), input_artifact_ids=("config",)),
        ])
        self.assertEqual(wp.packages[1].kind, WorkPackageKind.INTEGRATION)

    def test_cross_package_inputs_require_dependency_reachability(self) -> None:
        m = manifest([spec("a", "a.py"), spec("b", "b.py")])
        with self.assertRaises(ArtifactContractError):
            plan(m, [
                WorkPackage(id="p1", title="a", owns=("a",)),
                WorkPackage(id="p2", title="b", owns=("b",),
                            input_artifact_ids=("a",)),
            ])
        wp = plan(m, [
            WorkPackage(id="p1", title="a", owns=("a",)),
            WorkPackage(id="middle", title="middle", depends_on=("p1",)),
            WorkPackage(id="p2", title="b", owns=("b",),
                        input_artifact_ids=("a",), depends_on=("middle",)),
        ])
        self.assertEqual(wp.owner_of("b"), "p2")

    def test_artifact_dependencies_require_package_dependency(self) -> None:
        m = manifest([
            spec("a", "a.py"),
            spec("b", "b.py", depends_on=("a",)),
        ])
        with self.assertRaises(ArtifactContractError):
            plan(m, [
                WorkPackage(id="p1", title="a", owns=("a",)),
                WorkPackage(id="p2", title="b", owns=("b",)),
            ])

    def test_result_producer_must_be_exactly_its_owning_package(self) -> None:
        m = manifest(
            [
                spec("source", "src/app.py", kind=ArtifactKind.SOURCE_FILE),
                spec("build-result", "build-result",
                     kind=ArtifactKind.BUILD_RESULT),
            ],
            [ArtifactValidation(
                id="build", kind=ValidationKind.BUILD,
                target_artifact_ids=("source",),
                produces_artifact_id="build-result",
            )],
        )
        with self.assertRaises(ArtifactContractError):
            plan(m, [
                WorkPackage(id="impl", title="impl",
                            owns=("source", "build-result")),
            ])
        with self.assertRaises(ArtifactContractError):
            plan(m, [
                WorkPackage(id="impl", title="impl", owns=("source",),
                            validation_ids=("build",)),
                WorkPackage(id="integration", title="integration",
                            owns=("build-result",), depends_on=("impl",)),
            ])

        wp = plan(m, [
            WorkPackage(id="impl", title="impl", owns=("source",)),
            WorkPackage(id="integration", title="integration",
                        kind=WorkPackageKind.INTEGRATION,
                        owns=("build-result",), depends_on=("impl",),
                        input_artifact_ids=("source",),
                        validation_ids=("build",)),
        ])
        self.assertEqual(wp.owner_of("build-result"), "integration")

    def test_owned_optional_result_requires_a_producing_validation(self) -> None:
        optional_result = spec(
            "result", "build-result", kind=ArtifactKind.BUILD_RESULT,
            required=False,
        )
        m = manifest([optional_result])
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(m, [WorkPackage(id="build", title="build", owns=("result",))])
        self.assertIn("no producing validation", str(ctx.exception))
        self.assertEqual(plan(m, []).owner_of("result"), "")

    def test_validation_targets_require_dependency_reachability(self) -> None:
        m = manifest(
            [
                spec("source", "src/app.py"),
                spec("result", "build-result", kind=ArtifactKind.BUILD_RESULT),
            ],
            [ArtifactValidation(
                id="build", kind=ValidationKind.BUILD,
                target_artifact_ids=("source",),
                produces_artifact_id="result",
            )],
        )
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(m, [
                WorkPackage(id="impl", title="impl", owns=("source",)),
                WorkPackage(id="check", title="check", owns=("result",),
                            validation_ids=("build",)),
            ])
        self.assertIn("not declared dependencies", str(ctx.exception))

        wp = plan(m, [
            WorkPackage(id="impl", title="impl", owns=("source",)),
            WorkPackage(id="check", title="check", owns=("result",),
                        depends_on=("impl",), validation_ids=("build",)),
        ])
        self.assertEqual(wp.owner_of("result"), "check")

    def test_inputs_must_exist_in_manifest(self) -> None:
        with self.assertRaises(ArtifactContractError):
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a",
                            owns=("config", "app", "tests"),
                            input_artifact_ids=("ghost",)),
            ])

    def test_unknown_validation_reference_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            plan(self.make_manifest(), [
                WorkPackage(id="p1", title="a",
                            owns=("config", "app", "tests"),
                            validation_ids=("ghost",)),
            ])

    def test_validation_targeting_unknown_package_rejected(self) -> None:
        m = manifest(
            [spec("a1", "a.py")],
            [ArtifactValidation(id="v1", target_package_id="ghost")],
        )
        with self.assertRaises(ArtifactContractError):
            plan(m, [WorkPackage(id="p1", title="a", owns=("a1",))])

    def test_spec_declared_owner_must_agree_with_packages(self) -> None:
        m = manifest([spec("a1", "a.py", package_id="p2")])
        with self.assertRaises(ArtifactContractError) as ctx:
            plan(m, [WorkPackage(id="p1", title="a", owns=("a1",))])
        self.assertIn("declares owner", str(ctx.exception))

    def test_subtask_map_invariants(self) -> None:
        m = manifest([spec("a1", "a.py")])
        packages = [WorkPackage(id="p1", title="a", owns=("a1",))]
        wp = plan(m, packages, subtask_map=(("p1", "sub_abc"),))
        self.assertEqual(wp.subtask_for("p1"), "sub_abc")
        self.assertEqual(wp.subtask_for("ghost"), "")
        with self.assertRaises(ArtifactContractError):
            plan(m, packages, subtask_map=(("ghost", "sub_abc"),))
        with self.assertRaises(ArtifactContractError):
            plan(m, packages, subtask_map=(("p1", "s1"), ("p1", "s2")))
        with self.assertRaises(ArtifactContractError):
            plan(m, packages, subtask_map=(("p1", ""),))
        with self.assertRaises(ArtifactContractError):
            plan(m, packages, subtask_map=(("p1", None),))

    def test_ownership_is_deterministic(self) -> None:
        wp = plan(self.make_manifest(), [
            WorkPackage(id="p1", title="a", owns=("config",)),
            WorkPackage(id="p2", title="b", owns=("app", "tests")),
        ])
        self.assertEqual([wp.owner_of(a) for a in ("config", "app", "tests")],
                         ["p1", "p2", "p2"])

    def test_mutation_resistance(self) -> None:
        packages = [WorkPackage(id="p1", title="a",
                                owns=("config", "app", "tests"))]
        wp = plan(self.make_manifest(), packages)
        packages.append(WorkPackage(id="p2", title="b"))
        self.assertEqual(len(wp.packages), 1)
        with self.assertRaises(Exception):
            wp.id = "changed"  # type: ignore[misc]


class TestArtifactWorkPlanSerialization(unittest.TestCase):
    def make_plan(self) -> ArtifactWorkPlan:
        m = manifest(
            [spec("a1", "a.py"), spec("a2", "b.py", required=False)],
            [ArtifactValidation(id="v1", kind=ValidationKind.SYNTAX)],
        )
        return plan(
            m,
            [WorkPackage(id="p1", title="a", owns=("a1",),
                         validation_ids=("v1",))],
            completion_criteria=("everything builds",),
            subtask_map=(("p1", "sub_1"),),
        )

    def test_round_trip(self) -> None:
        wp = self.make_plan()
        again = ArtifactWorkPlan.from_dict(wp.to_dict())
        self.assertEqual(again, wp)
        self.assertEqual(again.subtask_map, (("p1", "sub_1"),))

    def test_missing_manifest_fails(self) -> None:
        data = self.make_plan().to_dict()
        data.pop("manifest")
        with self.assertRaises(ArtifactContractError):
            ArtifactWorkPlan.from_dict(data)

    def test_malformed_subtask_map_fails(self) -> None:
        data = self.make_plan().to_dict()
        data["subtask_map"] = ["not-a-pair"]
        with self.assertRaises(ArtifactContractError):
            ArtifactWorkPlan.from_dict(data)
        data = self.make_plan().to_dict()
        data["subtask_map"] = [["p1", {"bad": "id"}]]
        with self.assertRaises(ArtifactContractError):
            ArtifactWorkPlan.from_dict(data)

    def test_scalar_nested_collections_raise_contract_error(self) -> None:
        for field in ("packages", "subtask_map"):
            with self.subTest(field=field):
                data = self.make_plan().to_dict()
                data[field] = 1
                with self.assertRaises(ArtifactContractError):
                    ArtifactWorkPlan.from_dict(data)

    def test_plan_brief_is_bounded_and_deterministic(self) -> None:
        m = manifest([spec(f"a{i}", f"src/f{i:03d}.py") for i in range(50)])
        packages = [
            WorkPackage(id=f"p{i}", title=f"pkg {i}",
                        owns=tuple(f"a{j}" for j in range(i * 2, i * 2 + 2)))
            for i in range(25)
        ]
        wp = plan(m, packages)
        brief = wp.to_brief()
        self.assertEqual(brief, wp.to_brief())            # deterministic
        self.assertIn("WORK PACKAGES:", brief)
        self.assertIn("more packages retained in the plan", brief)
        self.assertLess(len(brief), 5000)
        self.assertEqual(len(wp.packages), 25)            # full retention


if __name__ == "__main__":
    unittest.main()
