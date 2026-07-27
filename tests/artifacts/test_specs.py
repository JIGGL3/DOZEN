"""Phase 4A — artifact path safety, ArtifactSpec and ArtifactValidation."""

from __future__ import annotations

import unittest

from dozen.artifacts import (
    ArtifactContractError,
    ArtifactKind,
    ArtifactOperation,
    ArtifactSpec,
    ArtifactValidation,
    ValidationKind,
    canonical_artifact_path,
)


class TestCanonicalArtifactPath(unittest.TestCase):
    def test_valid_relative_paths_pass_through(self) -> None:
        self.assertEqual(canonical_artifact_path("src/App.tsx"), "src/App.tsx")
        self.assertEqual(canonical_artifact_path("package.json"), "package.json")
        self.assertEqual(
            canonical_artifact_path("src/features/dashboard/DashboardPage.tsx"),
            "src/features/dashboard/DashboardPage.tsx",
        )

    def test_windows_separators_normalize(self) -> None:
        self.assertEqual(canonical_artifact_path("src\\app\\App.tsx"), "src/app/App.tsx")

    def test_redundant_segments_collapse(self) -> None:
        self.assertEqual(canonical_artifact_path("./src//app/./main.py"), "src/app/main.py")

    def test_absolute_paths_rejected(self) -> None:
        for path in ("/etc/passwd", "\\windows\\system32", "C:\\repo\\a.py",
                     "c:/repo/a.py", "~/secrets.txt"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_parent_traversal_rejected(self) -> None:
        for path in ("../a.py", "src/../../a.py", "src/..", "..\\a.py"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_empty_and_degenerate_paths_rejected(self) -> None:
        for path in ("", "   ", None, ".", "./.", "//"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_control_characters_rejected(self) -> None:
        for path in ("a\nb.py", "a\tb.py", "a\x00b.py", "a\x01b.py",
                     "a\x7fb.py", "a\x85b.py"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_whitespace_cannot_hide_traversal_or_dot_components(self) -> None:
        for path in ("src/.. /app.py", "src/ . /app.py", "src /app.py",
                     "src/app.py "):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_portable_windows_aliases_and_url_paths_rejected(self) -> None:
        for path in ("CON", "dir/aux.txt", "LPT9.log", "dir/file.",
                     "dir/a:b.py", "dir/a<b.py", "https://example.com/a.py"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    canonical_artifact_path(path)

    def test_unicode_paths_are_preserved(self) -> None:
        self.assertEqual(canonical_artifact_path("文档/设计.md"), "文档/设计.md")

    def test_logical_names_are_single_segment_paths(self) -> None:
        self.assertEqual(canonical_artifact_path("build-result"), "build-result")

    def test_case_is_preserved(self) -> None:
        self.assertEqual(canonical_artifact_path("src/App.TSX"), "src/App.TSX")

    def test_overlong_path_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            canonical_artifact_path("a/" * 200 + "x.py")


class TestArtifactSpec(unittest.TestCase):
    def test_minimal_spec_has_safe_defaults(self) -> None:
        spec = ArtifactSpec(id="a1", path="src/main.py")
        self.assertEqual(spec.kind, ArtifactKind.OTHER)
        self.assertEqual(spec.operation, ArtifactOperation.CREATE)
        self.assertTrue(spec.required)
        self.assertFalse(spec.external)
        self.assertEqual(spec.depends_on, ())
        self.assertEqual(spec.metadata, ())

    def test_path_is_canonicalized_at_construction(self) -> None:
        spec = ArtifactSpec(id="a1", path="src\\app\\App.tsx",
                            kind=ArtifactKind.SOURCE_FILE)
        self.assertEqual(spec.path, "src/app/App.tsx")

    def test_unsafe_paths_rejected_at_construction(self) -> None:
        for path in ("", "/abs.py", "../up.py"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactContractError):
                    ArtifactSpec(id="a1", path=path)

    def test_empty_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactSpec(id="", path="a.py")

    def test_logical_non_path_artifacts(self) -> None:
        spec = ArtifactSpec(id="build", path="build-result",
                            kind=ArtifactKind.BUILD_RESULT,
                            operation=ArtifactOperation.GENERATE)
        self.assertEqual(spec.path, "build-result")
        self.assertEqual(spec.kind, ArtifactKind.BUILD_RESULT)

    def test_required_optional_and_operation_preserved(self) -> None:
        spec = ArtifactSpec(id="a1", path="a.py", required=False,
                            operation=ArtifactOperation.MODIFY)
        self.assertFalse(spec.required)
        self.assertEqual(spec.operation, ArtifactOperation.MODIFY)
        # String forms coerce too (JSON round trips deliver strings).
        spec = ArtifactSpec(id="a1", path="a.py", required="false",
                            operation="delete")
        self.assertFalse(spec.required)
        self.assertEqual(spec.operation, ArtifactOperation.DELETE)

    def test_self_dependency_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactSpec(id="a1", path="a.py", depends_on=("a1",))

    def test_collections_are_immutable_and_input_mutation_is_harmless(self) -> None:
        deps = ["a0"]
        criteria = ["compiles"]
        spec = ArtifactSpec(id="a1", path="a.py", depends_on=deps,
                            completion_criteria=criteria,
                            metadata={"framework": "react"})
        deps.append("a9")
        criteria.append("later")
        self.assertEqual(spec.depends_on, ("a0",))
        self.assertEqual(spec.completion_criteria, ("compiles",))
        self.assertIsInstance(spec.depends_on, tuple)
        self.assertIsInstance(spec.metadata, tuple)
        with self.assertRaises(Exception):
            spec.required = False  # type: ignore[misc]

    def test_metadata_is_bounded_and_sorted(self) -> None:
        metadata = {f"key{i:02d}": "v" for i in range(40)}
        spec = ArtifactSpec(id="a1", path="a.py", metadata=metadata)
        self.assertLessEqual(len(spec.metadata), 16)
        self.assertEqual([k for k, _ in spec.metadata],
                         sorted(k for k, _ in spec.metadata))

    def test_metadata_key_collisions_after_clipping_are_rejected(self) -> None:
        prefix = "k" * 60
        with self.assertRaises(ArtifactContractError):
            ArtifactSpec(id="a1", path="a.py",
                         metadata={prefix + "x": "one", prefix + "y": "two"})

    def test_null_nested_references_do_not_become_string_none(self) -> None:
        spec = ArtifactSpec(id="a1", path="a.py", depends_on=[None])
        self.assertEqual(spec.depends_on, ())

    def test_mapping_shaped_reference_collection_is_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactSpec(id="a1", path="a.py", depends_on={"a0": True})

    def test_scalar_json_arrays_are_rejected_on_deserialization(self) -> None:
        base = ArtifactSpec(id="a1", path="a.py").to_dict()
        for field in ("depends_on", "completion_criteria"):
            with self.subTest(field=field):
                data = dict(base)
                data[field] = 42
                with self.assertRaises(ArtifactContractError):
                    ArtifactSpec.from_dict(data)

    def test_unknown_enum_values_fall_back_conservatively(self) -> None:
        spec = ArtifactSpec(id="a1", path="a.py", kind="quantum_blob",
                            operation="teleport")
        self.assertEqual(spec.kind, ArtifactKind.OTHER)
        self.assertEqual(spec.operation, ArtifactOperation.CREATE)

    def test_serialization_round_trip(self) -> None:
        spec = ArtifactSpec(
            id="a1", path="src/App.tsx", kind=ArtifactKind.SOURCE_FILE,
            required=False, operation=ArtifactOperation.MODIFY,
            description="root component", package_id="p1",
            depends_on=("a0",), language="tsx", media_type="text/tsx",
            external=True, completion_criteria=("renders",),
            metadata={"framework": "react"},
        )
        again = ArtifactSpec.from_dict(spec.to_dict())
        self.assertEqual(again, spec)
        self.assertFalse(again.required)          # booleans preserved
        self.assertTrue(again.external)

    def test_from_dict_handles_missing_and_null_fields(self) -> None:
        spec = ArtifactSpec.from_dict({"id": "a1", "path": "a.py",
                                       "kind": None, "depends_on": None,
                                       "required": None, "description": None})
        self.assertEqual(spec.kind, ArtifactKind.OTHER)
        self.assertTrue(spec.required)            # null -> safe default
        self.assertEqual(spec.depends_on, ())
        self.assertEqual(spec.description, "")

    def test_from_dict_rejects_non_objects(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactSpec.from_dict(["not", "an", "object"])  # type: ignore[arg-type]


class TestArtifactValidation(unittest.TestCase):
    def test_defaults(self) -> None:
        validation = ArtifactValidation(id="v1")
        self.assertEqual(validation.kind, ValidationKind.CUSTOM)
        self.assertTrue(validation.required)
        self.assertEqual(validation.target_artifact_ids, ())

    def test_empty_id_rejected(self) -> None:
        with self.assertRaises(ArtifactContractError):
            ArtifactValidation(id="  ")

    def test_command_is_a_declaration_not_executed(self) -> None:
        validation = ArtifactValidation(
            id="v1", kind=ValidationKind.UNIT_TESTS,
            command="npm test", success_criterion="all tests pass",
            produces_artifact_id="test-result",
        )
        self.assertEqual(validation.command, "npm test")
        # It stays a plain string field; there is no callable surface at all.
        self.assertIsInstance(validation.command, str)

    def test_unknown_kind_falls_back_to_custom(self) -> None:
        validation = ArtifactValidation(id="v1", kind="vibe_check")
        self.assertEqual(validation.kind, ValidationKind.CUSTOM)

    def test_serialization_round_trip(self) -> None:
        validation = ArtifactValidation(
            id="v1", kind=ValidationKind.BUILD, required=False,
            target_artifact_ids=("a1", "a2"), target_package_id="p9",
            command="npm run build", success_criterion="exit code 0",
            produces_artifact_id="build-result",
        )
        again = ArtifactValidation.from_dict(validation.to_dict())
        self.assertEqual(again, validation)
        self.assertFalse(again.required)

    def test_from_dict_missing_fields(self) -> None:
        validation = ArtifactValidation.from_dict({"id": "v1"})
        self.assertEqual(validation.kind, ValidationKind.CUSTOM)
        self.assertEqual(validation.target_package_id, "")

    def test_scalar_targets_rejected_on_deserialization(self) -> None:
        data = ArtifactValidation(id="v1").to_dict()
        data["target_artifact_ids"] = 42
        with self.assertRaises(ArtifactContractError):
            ArtifactValidation.from_dict(data)


if __name__ == "__main__":
    unittest.main()
