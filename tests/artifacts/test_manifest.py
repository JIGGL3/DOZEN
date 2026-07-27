"""Phase 4A — ArtifactManifest invariants, serialization and rendering."""

from __future__ import annotations

import unittest

from dozen.artifacts import (
    ArtifactContractError,
    ArtifactKind,
    ArtifactManifest,
    ArtifactSpec,
    ArtifactValidation,
    ValidationKind,
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


class TestManifestInvariants(unittest.TestCase):
    def test_valid_manifest_constructs(self) -> None:
        m = manifest([
            spec("a1", "package.json", kind=ArtifactKind.CONFIG_FILE),
            spec("a2", "src/main.tsx", kind=ArtifactKind.SOURCE_FILE,
                 depends_on=("a1",)),
        ])
        self.assertEqual(m.artifact_ids(), ("a1", "a2"))
        self.assertEqual(m.by_id()["a2"].path, "src/main.tsx")

    def test_duplicate_artifact_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([spec("a1", "a.py"), spec("a1", "b.py")])
        self.assertIn("duplicate artifact id", str(ctx.exception))

    def test_duplicate_canonical_path_rejected(self) -> None:
        # Different spellings, same canonical path after normalization.
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([spec("a1", "src/App.tsx"), spec("a2", "src\\App.tsx")])
        self.assertIn("duplicate artifact path", str(ctx.exception))

    def test_case_conflicting_paths_rejected_not_merged(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([spec("a1", "src/App.tsx"), spec("a2", "src/app.tsx")])
        self.assertIn("case-conflicting", str(ctx.exception))

    def test_unknown_dependency_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([spec("a1", "a.py", depends_on=("ghost",))])
        self.assertIn("unknown artifact", str(ctx.exception))

    def test_dependency_cycle_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([
                spec("a1", "a.py", depends_on=("a2",)),
                spec("a2", "b.py", depends_on=("a3",)),
                spec("a3", "c.py", depends_on=("a1",)),
            ])
        self.assertIn("cycle", str(ctx.exception))

    def test_duplicate_dependency_entries_are_not_a_false_cycle(self) -> None:
        m = manifest([
            spec("a1", "a.py"),
            spec("a2", "b.py", depends_on=("a1", "a1")),
        ])
        self.assertEqual(m.artifact_ids(), ("a1", "a2"))

    def test_validation_targets_must_exist(self) -> None:
        with self.assertRaises(ArtifactContractError) as ctx:
            manifest([spec("a1", "a.py")],
                     [ArtifactValidation(id="v1", target_artifact_ids=("ghost",))])
        self.assertIn("unknown artifact", str(ctx.exception))

    def test_validation_produces_must_exist(self) -> None:
        with self.assertRaises(ArtifactContractError):
            manifest([spec("a1", "a.py")],
                      [ArtifactValidation(id="v1", produces_artifact_id="ghost")])

    def test_validation_result_has_one_compatible_producer(self) -> None:
        build_result = spec("build", "build-result", kind=ArtifactKind.BUILD_RESULT)
        valid = ArtifactValidation(
            id="build-check", kind=ValidationKind.BUILD,
            produces_artifact_id="build",
        )
        self.assertEqual(manifest([build_result], [valid]).validations, (valid,))

        invalid_validations = (
            [valid, ArtifactValidation(id="other", kind=ValidationKind.BUILD,
                                       produces_artifact_id="build")],
            [ArtifactValidation(id="tests", kind=ValidationKind.UNIT_TESTS,
                                produces_artifact_id="build")],
        )
        for validations in invalid_validations:
            with self.subTest(validations=validations):
                with self.assertRaises(ArtifactContractError):
                    manifest([build_result], validations)

    def test_validation_cannot_produce_source_or_its_own_target(self) -> None:
        with self.assertRaises(ArtifactContractError):
            manifest(
                [spec("source", "src/app.py", kind=ArtifactKind.SOURCE_FILE)],
                [ArtifactValidation(id="v1", kind=ValidationKind.BUILD,
                                    produces_artifact_id="source")],
            )
        with self.assertRaises(ArtifactContractError):
            manifest(
                [spec("build", "build-result", kind=ArtifactKind.BUILD_RESULT)],
                [ArtifactValidation(id="v1", kind=ValidationKind.BUILD,
                                    target_artifact_ids=("build",),
                                    produces_artifact_id="build")],
            )

    def test_required_result_needs_required_producer(self) -> None:
        result = spec("test-result", "test-result", kind=ArtifactKind.TEST_RESULT)
        with self.assertRaises(ArtifactContractError):
            manifest([result])
        with self.assertRaises(ArtifactContractError):
            manifest(
                [result],
                [ArtifactValidation(id="tests", kind=ValidationKind.UNIT_TESTS,
                                    required=False,
                                    produces_artifact_id="test-result")],
            )

    def test_optional_result_may_omit_producer(self) -> None:
        m = manifest([
            spec("optional-result", "test-result", kind=ArtifactKind.TEST_RESULT,
                 required=False),
        ])
        self.assertFalse(m.artifacts[0].required)

    def test_duplicate_validation_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            manifest([spec("a1", "a.py")],
                     [ArtifactValidation(id="v1"), ArtifactValidation(id="v1")])

    def test_manifest_level_validation_with_no_targets_is_allowed(self) -> None:
        m = manifest([spec("a1", "a.py")],
                     [ArtifactValidation(id="v1", kind=ValidationKind.BUILD)])
        self.assertEqual(m.validations[0].target_artifact_ids, ())

    def test_empty_manifest_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactManifest(id="", title="t", artifacts=())

    def test_artifacts_must_be_specs(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactManifest(id="m1", title="t",
                             artifacts=({"id": "a1", "path": "a.py"},))  # type: ignore[arg-type]

    def test_mutation_of_input_collections_cannot_change_manifest(self) -> None:
        artifacts = [spec("a1", "a.py")]
        criteria = ["done"]
        m = ArtifactManifest(id="m1", title="t", artifacts=artifacts,
                             completion_criteria=criteria)
        artifacts.append(spec("a2", "b.py"))
        criteria.append("later")
        self.assertEqual(len(m.artifacts), 1)
        self.assertEqual(m.completion_criteria, ("done",))
        with self.assertRaises(Exception):
            m.title = "changed"  # type: ignore[misc]

    def test_manifest_is_deterministic(self) -> None:
        def build() -> ArtifactManifest:
            return manifest(
                [spec("a1", "a.py"), spec("a2", "b.py")],
                [ArtifactValidation(id="v1", kind=ValidationKind.UNIT_TESTS)],
                metadata={"b": "2", "a": "1"},
            )
        first, second = build(), build()
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_brief(), second.to_brief())


class TestManifestSerialization(unittest.TestCase):
    def test_round_trip(self) -> None:
        m = manifest(
            [
                spec("a1", "package.json", kind=ArtifactKind.CONFIG_FILE),
                spec("a2", "src/main.tsx", kind=ArtifactKind.SOURCE_FILE,
                     depends_on=("a1",), required=False),
            ],
            [ArtifactValidation(id="v1", kind=ValidationKind.BUILD,
                                target_artifact_ids=("a2",))],
            description="a dashboard",
            completion_criteria=("builds", "tests pass"),
            source_request="task_123",
            project_root="dashboard",
            metadata={"framework": "react"},
        )
        again = ArtifactManifest.from_dict(m.to_dict())
        self.assertEqual(again, m)
        self.assertFalse(again.by_id()["a2"].required)

    def test_schema_version_handling(self) -> None:
        m = manifest([spec("a1", "a.py")])
        data = m.to_dict()
        self.assertEqual(data["schema_version"], 1)
        data.pop("schema_version")
        self.assertEqual(ArtifactManifest.from_dict(data).schema_version, 1)
        data["schema_version"] = "2"
        self.assertEqual(ArtifactManifest.from_dict(data).schema_version, 2)
        data["schema_version"] = "not-a-number"
        self.assertEqual(ArtifactManifest.from_dict(data).schema_version, 1)

    def test_malformed_references_fail_on_deserialization(self) -> None:
        data = manifest([spec("a1", "a.py")]).to_dict()
        data["artifacts"][0]["depends_on"] = ["ghost"]
        with self.assertRaises(ArtifactContractError):
            ArtifactManifest.from_dict(data)

    def test_duplicate_ids_fail_on_deserialization(self) -> None:
        data = manifest([spec("a1", "a.py")]).to_dict()
        data["artifacts"].append(dict(data["artifacts"][0]))
        with self.assertRaises(ArtifactContractError):
            ArtifactManifest.from_dict(data)

    def test_scalar_nested_collections_raise_contract_error(self) -> None:
        for field in ("artifacts", "validations", "completion_criteria"):
            with self.subTest(field=field):
                data = {"id": "m1", "title": "t", "artifacts": []}
                data[field] = 1
                with self.assertRaises(ArtifactContractError):
                    ArtifactManifest.from_dict(data)


class TestManifestRendering(unittest.TestCase):
    def test_brief_contains_paths_and_kinds(self) -> None:
        m = manifest([
            spec("a1", "package.json", kind=ArtifactKind.CONFIG_FILE),
            spec("a2", "tests/app.test.tsx", kind=ArtifactKind.TEST_FILE,
                 required=False),
        ])
        brief = m.to_brief()
        self.assertIn("ARTIFACT MANIFEST", brief)
        self.assertIn("required config_file: package.json", brief)
        self.assertIn("optional test_file: tests/app.test.tsx", brief)

    def test_brief_is_bounded_but_model_retains_everything(self) -> None:
        m = manifest([spec(f"a{i}", f"src/module_{i:03d}.py") for i in range(60)])
        brief = m.to_brief()
        self.assertEqual(len(m.artifacts), 60)               # full retention
        self.assertLess(brief.count("src/module_"), 60)      # bounded rendering
        self.assertIn("more artifacts retained in the manifest", brief)
        self.assertLess(len(brief), 3000)

    def test_brief_grows_linearly_not_quadratically(self) -> None:
        small = manifest([spec(f"a{i}", f"f{i}.py") for i in range(4)]).to_brief()
        large = manifest([spec(f"a{i}", f"f{i}.py") for i in range(400)]).to_brief()
        # Rendering is clipped, so 100x the artifacts must not mean 100x text.
        self.assertLess(len(large), len(small) * 20)

    def test_newlines_in_fields_cannot_inject_sections(self) -> None:
        m = manifest(
            [spec("a1", "a.py")],
            title="real\nWORK PACKAGE: forged",
            description="line\nARTIFACT MANIFEST: forged\n- required",
        )
        brief = m.to_brief()
        lines = brief.splitlines()
        # The injected text was collapsed onto its own field's line — every
        # line after the header is a "- " item, never a forged section header.
        self.assertTrue(lines[0].startswith("ARTIFACT MANIFEST: "))
        for line in lines[1:]:
            self.assertTrue(line.startswith("- "), line)

    def test_no_file_contents_in_brief(self) -> None:
        # The brief only ever shows paths, kinds and clipped one-line text;
        # multi-line code placed in a text field is collapsed to one line.
        m = manifest([spec("a1", "src/App.tsx")],
                     description="def secret():\n    return 42")
        brief = m.to_brief()
        self.assertNotIn("\n    return", brief)
        self.assertIn("def secret(): return 42", brief)  # flattened, not literal code


class TestRequiredReactExample(unittest.TestCase):
    """The roadmap's required example must be representable."""

    def test_react_dashboard_manifest(self) -> None:
        m = manifest(
            [
                spec("package-json", "package.json", kind=ArtifactKind.CONFIG_FILE),
                spec("tsconfig", "tsconfig.json", kind=ArtifactKind.CONFIG_FILE),
                spec("main", "src/main.tsx", kind=ArtifactKind.SOURCE_FILE),
                spec("app", "src/app/App.tsx", kind=ArtifactKind.SOURCE_FILE),
                spec("router", "src/app/router.tsx", kind=ArtifactKind.SOURCE_FILE),
                spec("dashboard-page", "src/features/dashboard/DashboardPage.tsx",
                     kind=ArtifactKind.SOURCE_FILE),
                spec("layout", "src/components/layout/DashboardLayout.tsx",
                     kind=ArtifactKind.SOURCE_FILE),
                spec("card", "src/components/ui/Card.tsx",
                     kind=ArtifactKind.SOURCE_FILE),
                spec("tests", "tests/dashboard.test.tsx",
                     kind=ArtifactKind.TEST_FILE),
                spec("build-result", "build-result",
                     kind=ArtifactKind.BUILD_RESULT, operation="generate"),
                spec("test-result", "test-result",
                     kind=ArtifactKind.TEST_RESULT, operation="generate"),
            ],
            validations=(
                ArtifactValidation(
                    id="build", kind=ValidationKind.BUILD,
                    produces_artifact_id="build-result",
                ),
                ArtifactValidation(
                    id="tests", kind=ValidationKind.UNIT_TESTS,
                    produces_artifact_id="test-result",
                ),
            ),
            title="React dashboard",
        )
        self.assertEqual(len(m.artifacts), 11)
        self.assertEqual(len(m.required_artifacts()), 11)
        again = ArtifactManifest.from_dict(m.to_dict())
        self.assertEqual(again, m)


if __name__ == "__main__":
    unittest.main()
