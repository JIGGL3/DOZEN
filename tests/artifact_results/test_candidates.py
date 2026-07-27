"""Phase 4C — the produced-artifact candidate model, its bounds and hashing."""

from __future__ import annotations

import unittest

from dozen.artifact_results import (
    DEFAULT_RESULT_POLICY,
    ArtifactCandidate,
    ArtifactResultError,
    CandidateRejection,
    CollectionErrorCode,
    ResultPolicy,
    content_hash,
    parse_artifact_candidates,
    scope_requires_artifacts,
)
from dozen.artifacts import ArtifactKind, ArtifactOperation

from .harness import CONTENT, entry, envelope, scope_for


def candidate(**kwargs) -> ArtifactCandidate:
    base = dict(
        artifact_id="src/app/App.tsx",
        path="src/app/App.tsx",
        subtask_id="shell",
        package_id="application-shell",
        content=CONTENT["src/app/App.tsx"],
        kind=ArtifactKind.SOURCE_FILE,
        language="tsx",
    )
    base.update(kwargs)
    return ArtifactCandidate(**base)


class TestCandidateModel(unittest.TestCase):
    def test_valid_source_candidate(self) -> None:
        item = candidate()
        self.assertEqual(item.artifact_id, "src/app/App.tsx")
        self.assertEqual(item.kind, ArtifactKind.SOURCE_FILE)
        self.assertEqual(item.operation, ArtifactOperation.CREATE)
        self.assertTrue(item.complete)
        self.assertEqual(item.producer_subtask_id, "shell")

    def test_valid_test_config_and_result_candidates(self) -> None:
        for artifact_id, kind in (
            ("tests/dashboard.test.tsx", ArtifactKind.TEST_FILE),
            ("package.json", ArtifactKind.CONFIG_FILE),
            ("build-result", ArtifactKind.BUILD_RESULT),
            ("test-result", ArtifactKind.TEST_RESULT),
        ):
            item = candidate(
                artifact_id=artifact_id, path=artifact_id,
                content=CONTENT[artifact_id], kind=kind,
            )
            self.assertEqual(item.kind, kind)
            self.assertTrue(item.content)

    def test_fields_are_immutable(self) -> None:
        item = candidate()
        with self.assertRaises(Exception):
            item.content = "rewritten"  # type: ignore[misc]
        with self.assertRaises(Exception):
            item.artifact_id = "other"  # type: ignore[misc]

    def test_content_is_preserved_exactly(self) -> None:
        raw = "  line one\n\n\tindented\n  "
        self.assertEqual(candidate(content=raw).content, raw)

    def test_hash_is_deterministic_and_content_addressed(self) -> None:
        first = candidate()
        second = candidate(subtask_id="other-subtask", package_id="other-package")
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.content_hash, content_hash(first.content))
        self.assertNotEqual(
            first.content_hash, candidate(content="different").content_hash
        )
        self.assertTrue(first.content_hash.startswith("sha256:"))

    def test_a_mismatched_supplied_hash_is_rejected(self) -> None:
        with self.assertRaises(ArtifactResultError):
            candidate(content_hash="sha256:deadbeef")

    def test_supplied_matching_hash_is_accepted(self) -> None:
        item = candidate(content_hash=content_hash(CONTENT["src/app/App.tsx"]))
        self.assertEqual(item.content_hash, content_hash(CONTENT["src/app/App.tsx"]))

    def test_serialization_round_trip(self) -> None:
        item = candidate(
            attempt=2, recursion_depth=1, producer_subtask_id="inner",
            metadata={"origin": "recursive-shell"}, summary="wrote the shell",
        )
        self.assertEqual(ArtifactCandidate.from_dict(item.to_dict()), item)

    def test_round_trip_with_missing_optional_fields(self) -> None:
        minimal = {
            "artifact_id": "src/app/App.tsx",
            "path": "src/app/App.tsx",
            "subtask_id": "shell",
            "package_id": "application-shell",
            "content": "x",
        }
        restored = ArtifactCandidate.from_dict(minimal)
        self.assertEqual(restored.attempt, 0)
        self.assertEqual(restored.recursion_depth, 0)
        self.assertEqual(restored.kind, ArtifactKind.OTHER)
        self.assertEqual(restored.metadata, ())

    def test_unknown_enum_values_fall_back_conservatively(self) -> None:
        restored = ArtifactCandidate.from_dict({
            "artifact_id": "a", "path": "a.txt", "subtask_id": "s",
            "package_id": "p", "content": "x",
            "kind": "quantum_file", "operation": "teleport",
        })
        self.assertEqual(restored.kind, ArtifactKind.OTHER)
        self.assertEqual(restored.operation, ArtifactOperation.CREATE)

    def test_structural_requirements(self) -> None:
        with self.assertRaises(ArtifactResultError):
            candidate(artifact_id="")
        with self.assertRaises(ArtifactResultError):
            candidate(subtask_id="")
        with self.assertRaises(ArtifactResultError):
            candidate(package_id="")
        with self.assertRaises(ArtifactResultError):
            candidate(content=123)
        with self.assertRaises(ArtifactResultError):
            candidate(path="../../etc/passwd")
        with self.assertRaises(ArtifactResultError):
            candidate(path="/abs/App.tsx")

    def test_metadata_is_bounded_and_sorted(self) -> None:
        item = candidate(metadata={
            f"key{i}": "v" * 400 for i in range(30)
        })
        self.assertLessEqual(len(item.metadata), DEFAULT_RESULT_POLICY.max_metadata_items)
        self.assertEqual(list(item.metadata), sorted(item.metadata))
        for _key, value in item.metadata:
            self.assertLessEqual(
                len(value), DEFAULT_RESULT_POLICY.max_metadata_value_chars
            )

    def test_summary_is_bounded(self) -> None:
        item = candidate(summary="x" * 5000)
        self.assertLessEqual(
            len(item.summary), DEFAULT_RESULT_POLICY.max_summary_chars
        )

    def test_input_mutation_cannot_affect_a_candidate(self) -> None:
        metadata = {"origin": "shell"}
        item = candidate(metadata=metadata)
        metadata["origin"] = "tampered"
        metadata["extra"] = "injected"
        self.assertEqual(item.metadata, (("origin", "shell"),))

    def test_provenance_is_complete(self) -> None:
        item = candidate(attempt=2, recursion_depth=1, producer_subtask_id="inner")
        self.assertEqual(
            item.provenance,
            ("application-shell", "shell", "inner", 2, 1),
        )
        described = item.describe_provenance()
        self.assertIn("application-shell", described)
        self.assertIn("inner", described)
        self.assertIn("attempt 2", described)


class TestCandidateScopeInvariants(unittest.TestCase):
    """Invariants that need the scope: manifest membership, ownership, kind."""

    def parse(self, subtask_id: str, *entries):
        return parse_artifact_candidates(envelope(*entries), scope_for(subtask_id))

    def codes(self, rejections) -> list[CollectionErrorCode]:
        return [rejection.code for rejection in rejections]

    def test_unknown_artifact_id_is_rejected(self) -> None:
        _ok, rejected, engaged = self.parse(
            "shell", entry("src/app/Ghost.tsx", path="src/app/Ghost.tsx")
        )
        self.assertTrue(engaged)
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.UNKNOWN_ARTIFACT_ID])

    def test_path_mismatch_is_rejected(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", entry("src/app/App.tsx", path="src/App.tsx")
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.PATH_MISMATCH])
        self.assertIn("src/app/App.tsx", rejected[0].message)

    def test_artifact_owned_by_another_package_is_out_of_scope(self) -> None:
        _ok, rejected, _ = self.parse("dashboard", entry("package.json"))
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.OUT_OF_SCOPE])

    def test_a_candidate_is_credited_to_its_owning_package_and_subtask(self) -> None:
        accepted, _rejected, _ = self.parse(
            "shell",
            entry(
                "src/app/App.tsx",
                package_id="attacker-package",
                subtask_id="attacker-subtask",
                producer_subtask_id="attacker-producer",
                attempt=999,
                recursion_depth=999,
                manifest_id="attacker-manifest",
                kind="test_result",
                operation="create",
            ),
        )
        self.assertEqual(accepted[0].package_id, "application-shell")
        self.assertEqual(accepted[0].subtask_id, "shell")
        self.assertEqual(accepted[0].producer_subtask_id, "shell")
        self.assertEqual(accepted[0].attempt, 0)
        self.assertEqual(accepted[0].recursion_depth, 0)
        self.assertEqual(accepted[0].kind, ArtifactKind.SOURCE_FILE)
        self.assertEqual(accepted[0].operation, ArtifactOperation.CREATE)

    def test_external_artifacts_cannot_be_produced(self) -> None:
        from dozen.artifacts import ArtifactSpec

        from ..artifact_decomposition.harness import (
            manifest_of, react_packages, work_plan_of,
        )
        from .harness import react_work_plan

        base = react_work_plan().manifest
        specs = list(base.artifacts) + [
            ArtifactSpec(id="vendor/logo.svg", path="vendor/logo.svg",
                         kind=ArtifactKind.DATA_FILE, external=True, required=False)
        ]
        plan = work_plan_of(
            manifest_of(specs, base.validations, title=base.title),
            react_packages(),
            react_work_plan().subtask_map,
        )
        scope = scope_for("shell", plan)
        _ok, rejected, _ = parse_artifact_candidates(
            envelope(entry("vendor/logo.svg", content="<svg/>")), scope
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.EXTERNAL_ARTIFACT])

    def test_directory_artifacts_cannot_carry_content(self) -> None:
        from dozen.artifacts import ArtifactSpec, WorkPackage

        from ..artifact_decomposition.harness import manifest_of, work_plan_of

        specs = [
            ArtifactSpec(id="src", path="src", kind=ArtifactKind.DIRECTORY),
            ArtifactSpec(id="src/main.ts", path="src/main.ts",
                         kind=ArtifactKind.SOURCE_FILE),
        ]
        plan = work_plan_of(
            manifest_of(specs),
            [WorkPackage(id="p1", title="impl", owns=("src", "src/main.ts"))],
            (("p1", "s1"),),
        )
        scope = scope_for("s1", plan)
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(
                entry("src", path="src", content="not allowed"),
                entry("src/main.ts", path="src/main.ts", content="export {};\n"),
            ),
            scope,
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.DIRECTORY_CONTENT])
        self.assertEqual([c.artifact_id for c in accepted], ["src/main.ts"])

    def test_directory_artifact_without_content_is_accepted(self) -> None:
        from dozen.artifacts import ArtifactSpec, WorkPackage

        from ..artifact_decomposition.harness import manifest_of, work_plan_of

        plan = work_plan_of(
            manifest_of([ArtifactSpec(id="src", path="src",
                                      kind=ArtifactKind.DIRECTORY)]),
            [WorkPackage(id="p1", title="impl", owns=("src",))],
            (("p1", "s1"),),
        )
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(entry("src", path="src", content="")), scope_for("s1", plan)
        )
        self.assertEqual(rejected, ())
        self.assertEqual(accepted[0].content, "")

    def test_empty_required_content_is_rejected(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", entry("src/app/App.tsx", content="   \n ")
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.MISSING_CONTENT])

    def test_invalid_content_type_is_rejected(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", entry("src/app/App.tsx", content={"base64": "AAAA"})
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.INVALID_CONTENT_TYPE])

    def test_result_artifacts_accept_structured_result_text(self) -> None:
        accepted, rejected, _ = self.parse(
            "validation", entry("build-result"), entry("test-result")
        )
        self.assertEqual(rejected, ())
        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0].kind, ArtifactKind.BUILD_RESULT)

    def test_invalid_operation_is_rejected(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", entry("src/app/App.tsx", operation="teleport")
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.INVALID_OPERATION])

    def test_worker_cannot_override_manifest_operation(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", entry("src/app/App.tsx", operation="delete")
        )
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.INVALID_OPERATION])
        self.assertIn("manifest-controlled", rejected[0].message)

    def test_manifest_delete_accepts_no_content_and_rejects_content(self) -> None:
        from dozen.artifacts import ArtifactSpec, WorkPackage

        from ..artifact_decomposition.harness import manifest_of, work_plan_of

        plan = work_plan_of(
            manifest_of([
                ArtifactSpec(
                    id="obsolete.py",
                    path="obsolete.py",
                    kind=ArtifactKind.SOURCE_FILE,
                    operation=ArtifactOperation.DELETE,
                )
            ]),
            [WorkPackage(id="cleanup", title="Cleanup", owns=("obsolete.py",))],
            (("cleanup", "s1"),),
        )
        scope = scope_for("s1", plan)
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(entry("obsolete.py", content="", operation="delete")), scope
        )
        self.assertEqual(rejected, ())
        self.assertEqual(accepted[0].operation, ArtifactOperation.DELETE)

        accepted, rejected, _ = parse_artifact_candidates(
            envelope(entry("obsolete.py", content="still here", operation="delete")),
            scope,
        )
        self.assertEqual(accepted, ())
        self.assertEqual(self.codes(rejected), [CollectionErrorCode.DELETE_CONTENT])


class TestResultPolicyBounds(unittest.TestCase):
    def test_content_size_boundary(self) -> None:
        policy = ResultPolicy(max_content_chars=100)
        scope = scope_for("shared-ui")
        at_limit = "x" * 100
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(entry("src/components/ui/Card.tsx", content=at_limit)),
            scope, policy=policy,
        )
        self.assertEqual(rejected, ())
        self.assertEqual(len(accepted), 1)

        _ok, rejected, _ = parse_artifact_candidates(
            envelope(entry("src/components/ui/Card.tsx", content="x" * 101)),
            scope, policy=policy,
        )
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.CONTENT_TOO_LARGE]
        )

    def test_aggregate_content_limit(self) -> None:
        policy = ResultPolicy(max_subtask_content_chars=120)
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(
                entry("src/app/App.tsx", content="a" * 100),
                entry("src/app/router.tsx", content="b" * 100),
            ),
            scope_for("shell"), policy=policy,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.AGGREGATE_TOO_LARGE]
        )

    def test_maximum_candidate_count(self) -> None:
        policy = ResultPolicy(max_candidates_per_response=2)
        accepted, rejected, _ = parse_artifact_candidates(
            envelope(
                entry("src/app/App.tsx"),
                entry("src/app/router.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            ),
            scope_for("shell"), policy=policy,
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.TOO_MANY_CANDIDATES]
        )

    def test_limits_live_in_one_policy_object(self) -> None:
        self.assertIsInstance(DEFAULT_RESULT_POLICY, ResultPolicy)
        self.assertGreater(DEFAULT_RESULT_POLICY.max_content_chars, 100_000)
        self.assertGreater(
            DEFAULT_RESULT_POLICY.max_total_content_chars,
            DEFAULT_RESULT_POLICY.max_subtask_content_chars,
        )
        with self.assertRaises(Exception):
            DEFAULT_RESULT_POLICY.max_content_chars = 1  # type: ignore[misc]

    def test_scope_requires_artifacts(self) -> None:
        self.assertTrue(scope_requires_artifacts(scope_for("shell")))
        self.assertFalse(scope_requires_artifacts(None))


class TestRejectionModel(unittest.TestCase):
    def test_round_trip(self) -> None:
        rejection = CandidateRejection(
            code=CollectionErrorCode.PATH_MISMATCH, message="wrong path",
            artifact_id="a", path="p", subtask_id="s", index=2,
        )
        self.assertEqual(
            CandidateRejection.from_dict(rejection.to_dict()), rejection
        )

    def test_unknown_code_falls_back(self) -> None:
        restored = CandidateRejection.from_dict(
            {"code": "meteor_strike", "message": "boom"}
        )
        self.assertEqual(restored.code, CollectionErrorCode.MALFORMED_ENTRY)


if __name__ == "__main__":
    unittest.main()
