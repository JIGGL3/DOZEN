"""Phase 4D Part A/B — the integrity models: immutability, status semantics,
serialization, bounds and determinism.
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from dozen.artifact_integrity import (
    DEFAULT_INTEGRITY_POLICY,
    ArtifactIntegrityError,
    ArtifactIntegrityResult,
    InspectionMode,
    IntegrityIssue,
    IntegrityIssueCode,
    IntegrityPolicy,
    IntegritySeverity,
    IntegrityStatus,
    detect_inspection_mode,
    evaluate_artifact_integrity,
    integrity_feedback,
)
from dozen.artifacts import ArtifactKind, ArtifactOperation

from .harness import APP_CUT_IN_ATTRIBUTE, CARD_BRACES_IN_STRINGS, check


def issue(**kwargs) -> IntegrityIssue:
    base = dict(
        code=IntegrityIssueCode.TRUNCATION_MARKER,
        severity=IntegritySeverity.FATAL,
        message="the content ends with an explicit truncation marker",
    )
    base.update(kwargs)
    return IntegrityIssue(**base)


class TestIntegrityIssue(unittest.TestCase):
    def test_the_issue_is_immutable(self) -> None:
        item = issue()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.message = "changed"  # type: ignore[misc]

    def test_evidence_is_bounded_and_newline_free(self) -> None:
        item = issue(evidence="x\ny" + "z" * 500)
        self.assertLessEqual(
            len(item.evidence), DEFAULT_INTEGRITY_POLICY.max_evidence_chars
        )
        self.assertNotIn("\n", item.evidence)

    def test_message_is_bounded(self) -> None:
        item = issue(message="m" * 5000)
        self.assertLessEqual(
            len(item.message), DEFAULT_INTEGRITY_POLICY.max_message_chars
        )

    def test_unknown_code_and_severity_fall_back(self) -> None:
        item = IntegrityIssue(code="not_a_code", severity="not_a_severity",
                              message="x")
        self.assertIs(item.code, IntegrityIssueCode.NOT_INSPECTED)
        self.assertIs(item.severity, IntegritySeverity.ADVISORY)
        self.assertFalse(item.fatal)

    def test_round_trip(self) -> None:
        item = issue(evidence="tail", offset=12, line=3, truncation=True)
        self.assertEqual(IntegrityIssue.from_dict(item.to_dict()), item)

    def test_missing_fields_round_trip(self) -> None:
        restored = IntegrityIssue.from_dict({"code": "unclosed_fence"})
        self.assertIs(restored.code, IntegrityIssueCode.UNCLOSED_FENCE)
        self.assertEqual(restored.message, "")
        self.assertEqual(restored.offset, -1)

    def test_none_never_leaks_as_a_string(self) -> None:
        item = IntegrityIssue(code=IntegrityIssueCode.EMPTY_CONTENT,
                              severity=IntegritySeverity.FATAL,
                              message=None, evidence=None)
        self.assertEqual(item.message, "")
        self.assertEqual(item.evidence, "")
        self.assertNotIn("None", json.dumps(item.to_dict()))

    def test_from_dict_rejects_non_objects(self) -> None:
        with self.assertRaises(ArtifactIntegrityError):
            IntegrityIssue.from_dict(["nope"])  # type: ignore[arg-type]


class TestIntegrityResult(unittest.TestCase):
    def test_the_result_is_immutable(self) -> None:
        result = check(CARD_BRACES_IN_STRINGS, "src/components/ui/Card.tsx")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = IntegrityStatus.INVALID  # type: ignore[misc]

    def test_no_mutable_collection_leaks(self) -> None:
        result = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        self.assertIsInstance(result.issues, tuple)
        self.assertIsInstance(result.fatal_issues, tuple)
        self.assertIsInstance(result.advisories, tuple)

    def test_status_semantics(self) -> None:
        valid = check(CARD_BRACES_IN_STRINGS, "src/components/ui/Card.tsx")
        invalid = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        not_applicable = check("", "assets", kind=ArtifactKind.DIRECTORY)
        self.assertIs(valid.status, IntegrityStatus.VALID)
        self.assertTrue(valid.structurally_complete)
        self.assertFalse(valid.truncation_suspected)
        self.assertIs(invalid.status, IntegrityStatus.INVALID)
        self.assertFalse(invalid.structurally_complete)
        self.assertTrue(invalid.truncation_suspected)
        self.assertIs(not_applicable.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertTrue(not_applicable.structurally_complete)

    def test_valid_does_not_claim_compiler_correctness(self) -> None:
        # Balanced, quoted, closed — and complete nonsense to a type checker.
        result = check("const x: number = 'not a number';\n", "src/x.ts")
        self.assertIs(result.status, IntegrityStatus.VALID)

    def test_issues_are_bounded(self) -> None:
        policy = IntegrityPolicy(max_issues=2)
        content = "function f() {\n  const a = `x\n"
        result = check(content, "src/f.ts", policy=policy)
        self.assertLessEqual(len(result.issues), 2)

    def test_issues_are_ordered_fatal_first(self) -> None:
        result = check("```tsx\nconst a = {\n", "src/a.tsx")
        severities = [i.severity for i in result.issues]
        self.assertEqual(
            severities, sorted(severities, key=lambda s: {
                IntegritySeverity.FATAL: 0,
                IntegritySeverity.SUSPECT: 1,
                IntegritySeverity.ADVISORY: 2,
            }[s])
        )

    def test_detection_is_deterministic(self) -> None:
        first = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        second = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_round_trip(self) -> None:
        result = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        self.assertEqual(ArtifactIntegrityResult.from_dict(result.to_dict()), result)

    def test_json_serializable(self) -> None:
        result = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        text = json.dumps(result.to_dict())
        self.assertIn("invalid", text)

    def test_asdict_compatible(self) -> None:
        result = check(CARD_BRACES_IN_STRINGS, "src/components/ui/Card.tsx")
        data = dataclasses.asdict(result)
        self.assertEqual(data["status"], IntegrityStatus.VALID)

    def test_unknown_status_falls_back_conservatively(self) -> None:
        restored = ArtifactIntegrityResult.from_dict({"status": "banana"})
        self.assertIs(restored.status, IntegrityStatus.SUSPICIOUS)
        self.assertFalse(restored.structurally_complete)

    def test_status_is_canonicalized_from_issue_severity(self) -> None:
        restored = ArtifactIntegrityResult.from_dict({
            "status": "valid",
            "issues": [issue(truncation=True).to_dict()],
        })
        self.assertIs(restored.status, IntegrityStatus.INVALID)
        self.assertFalse(restored.structurally_complete)

    def test_deserialized_issues_are_ordered_deterministically(self) -> None:
        advisory = issue(severity=IntegritySeverity.ADVISORY,
                         code=IntegrityIssueCode.NOT_INSPECTED)
        fatal = issue(severity=IntegritySeverity.FATAL)
        restored = ArtifactIntegrityResult.from_dict({
            "status": "valid",
            "issues": [advisory.to_dict(), fatal.to_dict()],
        })
        self.assertEqual(restored.issues, (fatal, advisory))

    def test_nonconclusive_serialized_valid_result_is_not_valid(self) -> None:
        restored = ArtifactIntegrityResult.from_dict({
            "status": "valid", "conclusive": False,
        })
        self.assertIs(restored.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertFalse(restored.conclusive)

    def test_unknown_mode_falls_back(self) -> None:
        restored = ArtifactIntegrityResult.from_dict(
            {"status": "valid", "mode": "brainfuck"}
        )
        self.assertIs(restored.mode, InspectionMode.TEXT)

    def test_issues_must_be_typed(self) -> None:
        with self.assertRaises(ArtifactIntegrityError):
            ArtifactIntegrityResult(status=IntegrityStatus.VALID, issues=("oops",))

    def test_content_is_never_embedded_in_the_result(self) -> None:
        body = "const secret = 'x" + "y" * 5000 + "\n"
        result = check(body, "src/big.ts")
        serialized = json.dumps(result.to_dict())
        self.assertLess(len(serialized), 4000)
        self.assertEqual(result.content_chars, len(body))


class TestPolicy(unittest.TestCase):
    def test_the_policy_is_immutable_and_conservative(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_INTEGRITY_POLICY.max_issues = 99  # type: ignore[misc]
        self.assertTrue(DEFAULT_INTEGRITY_POLICY.block_suspicious)
        self.assertFalse(DEFAULT_INTEGRITY_POLICY.advisories_in_feedback)

    def test_content_beyond_the_bound_is_not_guessed_at(self) -> None:
        policy = IntegrityPolicy(max_inspect_chars=32)
        result = check("const a = {\n" + "// pad\n" * 40, "src/a.ts", policy=policy)
        self.assertFalse(result.conclusive)
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)

    def test_maximum_inspected_lines_has_no_off_by_one(self) -> None:
        policy = IntegrityPolicy(max_inspect_lines=2)
        at_limit = check("const a = 1;\nconst b = 2;\n", "src/a.ts", policy=policy)
        over_limit = check(
            "const a = 1;\nconst b = 2;\nconst c = 3;\n",
            "src/a.ts", policy=policy,
        )
        self.assertTrue(at_limit.conclusive)
        self.assertFalse(over_limit.conclusive)

    def test_suspicious_can_be_made_non_blocking(self) -> None:
        permissive = IntegrityPolicy(block_suspicious=False)
        result = check("import { Suspense } from", "src/router.tsx")
        self.assertIs(result.status, IntegrityStatus.SUSPICIOUS)
        self.assertTrue(result.blocking(DEFAULT_INTEGRITY_POLICY))
        self.assertFalse(result.blocking(permissive))


class TestModeDetection(unittest.TestCase):
    def test_path_beats_a_worker_supplied_language_hint(self) -> None:
        # A worker cannot relabel package.json as prose to dodge JSON validation.
        mode = detect_inspection_mode(
            path="package.json", kind=ArtifactKind.CONFIG_FILE, language="markdown"
        )
        self.assertIs(mode, InspectionMode.JSON)

    def test_directories_and_deletes_are_not_inspected(self) -> None:
        self.assertIs(
            detect_inspection_mode(path="src", kind=ArtifactKind.DIRECTORY),
            InspectionMode.NONE,
        )
        self.assertIs(
            detect_inspection_mode(path="src/old.ts", kind=ArtifactKind.SOURCE_FILE,
                                   operation=ArtifactOperation.DELETE),
            InspectionMode.NONE,
        )

    def test_patches_use_diff_inspection(self) -> None:
        self.assertIs(
            detect_inspection_mode(path="fix", kind=ArtifactKind.PATCH),
            InspectionMode.DIFF,
        )

    def test_unknown_extension_is_generic_text_never_invalid(self) -> None:
        mode = detect_inspection_mode(path="src/main.zig",
                                      kind=ArtifactKind.SOURCE_FILE)
        self.assertIs(mode, InspectionMode.TEXT)

    def test_media_type_is_honored_when_the_path_is_silent(self) -> None:
        mode = detect_inspection_mode(path="data", kind=ArtifactKind.DATA_FILE,
                                      media_type="application/json")
        self.assertIs(mode, InspectionMode.JSON)

    def test_document_kind_beats_a_source_looking_suffix(self) -> None:
        mode = detect_inspection_mode(path="guide.py", kind=ArtifactKind.DOCUMENT)
        self.assertIs(mode, InspectionMode.PROSE)


class TestFeedback(unittest.TestCase):
    def test_feedback_names_the_artifact_and_asks_for_the_whole_thing(self) -> None:
        result = check(APP_CUT_IN_ATTRIBUTE, "src/app/App.tsx")
        text = integrity_feedback(result)
        self.assertIn("src/app/App.tsx", text)
        self.assertIn("incomplete", text)
        self.assertIn("complete", text.lower())
        self.assertLessEqual(len(text), 600)

    def test_feedback_names_distinct_artifact_id_and_path(self) -> None:
        result = evaluate_artifact_integrity(
            "export function App() {\n",
            artifact_id="application-shell",
            path="src/App.tsx",
            kind=ArtifactKind.SOURCE_FILE,
        )
        text = integrity_feedback(result)
        self.assertIn("application-shell", text)
        self.assertIn("src/App.tsx", text)

    def test_a_valid_artifact_produces_no_feedback(self) -> None:
        result = check(CARD_BRACES_IN_STRINGS, "src/components/ui/Card.tsx")
        self.assertEqual(integrity_feedback(result), "")

    def test_advisories_stay_out_of_feedback_by_default(self) -> None:
        result = evaluate_artifact_integrity(
            "# notes\n\n```ts\nconst a = 1;\n```\n",
            path="docs/notes.md", kind=ArtifactKind.DOCUMENT,
        )
        self.assertEqual(integrity_feedback(result), "")


if __name__ == "__main__":
    unittest.main()
