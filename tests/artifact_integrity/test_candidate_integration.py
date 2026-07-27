"""Phase 4D Part I — integrity at the candidate-acceptance boundary.

The flow under test:
    worker entry → ownership/provenance → content bounds → INTEGRITY → accept
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from dozen.artifact_integrity import IntegrityStatus
from dozen.artifact_results import (
    CollectionErrorCode,
    DEFAULT_RESULT_POLICY,
    ResultPolicy,
    content_hash,
    parse_artifact_candidates,
)
from dozen.artifact_integrity import IntegrityPolicy

from .harness import (
    APP_CUT_IN_ATTRIBUTE,
    CARD_BRACES_IN_STRINGS,
    CONTENT,
    LAYOUT_UNCLOSED_TAG,
    PACKAGE_JSON_MISSING_BRACE,
    ROUTER_CUT_AFTER_IMPORT,
    TEST_UNCLOSED_TEMPLATE,
    broken_entry,
    collect_for,
    entry,
    envelope,
    scope_for,
    react_work_plan,
)

APP = "src/app/App.tsx"
ROUTER = "src/app/router.tsx"
LAYOUT = "src/components/layout/DashboardLayout.tsx"
CARD = "src/components/ui/Card.tsx"
TESTS = "tests/dashboard.test.tsx"
PKG = "package.json"
TSCONFIG = "tsconfig.json"
BUILD = "build-result"


def codes(collection) -> list[CollectionErrorCode]:
    return [rejection.code for rejection in collection.rejected]


class TestRequiredDashboardCases(unittest.TestCase):
    """The fourteen approved React-dashboard integrity cases."""

    def test_1_a_complete_app_is_accepted(self) -> None:
        collection = collect_for("shell", envelope(
            entry(APP), entry(ROUTER), entry(LAYOUT),
        ))
        self.assertTrue(collection.satisfied)
        self.assertEqual(len(collection.accepted), 3)

    def test_2_an_app_cut_inside_a_jsx_attribute_is_rejected(self) -> None:
        collection = collect_for("shell", envelope(
            broken_entry(APP, APP_CUT_IN_ATTRIBUTE), entry(ROUTER), entry(LAYOUT),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertNotIn(APP, [c.artifact_id for c in collection.accepted])
        self.assertIn(APP, collection.missing_required_artifact_ids)
        self.assertEqual(collection.integrity_rejected_artifact_ids, (APP,))

    def test_3_a_router_cut_after_an_import_is_rejected_as_suspected(self) -> None:
        # Explicit policy: SUSPICIOUS blocks acceptance (block_suspicious=True).
        collection = collect_for("shell", envelope(
            entry(APP), broken_entry(ROUTER, ROUTER_CUT_AFTER_IMPORT), entry(LAYOUT),
        ))
        self.assertIn(CollectionErrorCode.TRUNCATION_SUSPECTED, codes(collection))
        self.assertNotIn(ROUTER, [c.artifact_id for c in collection.accepted])

    def test_4_a_package_json_missing_its_final_brace_is_rejected(self) -> None:
        collection = collect_for("foundation", envelope(
            broken_entry(PKG, PACKAGE_JSON_MISSING_BRACE),
            entry(TSCONFIG), entry("src/main.tsx"),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertEqual(collection.integrity_rejected_artifact_ids, (PKG,))

    def test_5_a_valid_tsconfig_is_accepted(self) -> None:
        collection = collect_for("foundation", envelope(
            entry(PKG), entry(TSCONFIG), entry("src/main.tsx"),
        ))
        self.assertTrue(collection.satisfied)
        accepted = {c.artifact_id for c in collection.accepted}
        self.assertIn(TSCONFIG, accepted)

    def test_6_a_layout_with_an_unclosed_jsx_tag_is_rejected(self) -> None:
        collection = collect_for("shell", envelope(
            entry(APP), entry(ROUTER), broken_entry(LAYOUT, LAYOUT_UNCLOSED_TAG),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertIn("never closed", collection.rejected[0].message)

    def test_7_a_card_with_braces_inside_strings_is_accepted(self) -> None:
        collection = collect_for("shared-ui", envelope(
            entry(CARD, content=CARD_BRACES_IN_STRINGS),
        ))
        self.assertTrue(collection.satisfied)
        self.assertEqual(collection.accepted[0].content, CARD_BRACES_IN_STRINGS)

    def test_8_a_test_file_with_an_unclosed_template_is_rejected(self) -> None:
        collection = collect_for("tests", envelope(
            broken_entry(TESTS, TEST_UNCLOSED_TEMPLATE),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertEqual(collection.accepted, ())

    def test_9_a_build_result_with_a_truncation_marker_is_rejected(self) -> None:
        collection = collect_for("validation", envelope(
            broken_entry(BUILD, "vite build\n[... output truncated ...]"),
            entry("test-result"),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertEqual(collection.integrity_rejected_artifact_ids, (BUILD,))

    def test_10_an_ordinary_ellipsis_inside_a_string_is_accepted(self) -> None:
        body = 'export function Card() {\n  return <p>Loading...</p>;\n}\n'
        collection = collect_for("shared-ui", envelope(entry(CARD, content=body)))
        self.assertTrue(collection.satisfied)

    def test_11_complete_true_cannot_override_a_truncated_body(self) -> None:
        collection = collect_for("shell", envelope(
            entry(APP, content=APP_CUT_IN_ATTRIBUTE, complete=True),
            entry(ROUTER), entry(LAYOUT),
        ))
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertEqual(len(collection.accepted), 2)

    def test_12_complete_false_on_valid_content_is_an_incomplete_declaration(self) -> None:
        collection = collect_for("shared-ui", envelope(
            entry(CARD, complete=False),
        ))
        self.assertEqual(codes(collection),
                         [CollectionErrorCode.INCOMPLETE_DECLARATION])
        self.assertEqual(collection.accepted, ())
        self.assertFalse(collection.satisfied)


class TestIntegrityAtTheBoundary(unittest.TestCase):
    def test_an_accepted_candidate_carries_its_integrity_evidence(self) -> None:
        collection = collect_for("shared-ui", envelope(entry(CARD)))
        candidate = collection.accepted[0]
        self.assertIsNotNone(candidate.integrity)
        self.assertIs(candidate.integrity.status, IntegrityStatus.VALID)
        self.assertTrue(candidate.integrity.structurally_complete)

    def test_the_exact_submitted_content_is_preserved(self) -> None:
        body = "\n" + CONTENT[CARD] + "\n"
        collection = collect_for("shared-ui", envelope(entry(CARD, content=body)))
        self.assertEqual(collection.accepted[0].content, body)

    def test_the_hash_still_covers_the_exact_submitted_content(self) -> None:
        body = "\n" + CONTENT[CARD] + "\n"
        collection = collect_for("shared-ui", envelope(entry(CARD, content=body)))
        self.assertEqual(collection.accepted[0].content_hash, content_hash(body))

    def test_a_markdown_wrapped_raw_file_never_enters_assembly(self) -> None:
        body = "```tsx\n" + CONTENT[CARD] + "```\n"
        collection = collect_for("shared-ui", envelope(entry(CARD, content=body)))
        self.assertEqual(collection.accepted, ())
        self.assertEqual(codes(collection),
                         [CollectionErrorCode.STRUCTURALLY_INVALID])

    def test_a_rejected_body_is_not_silently_dropped(self) -> None:
        collection = collect_for("shell", envelope(
            broken_entry(APP, APP_CUT_IN_ATTRIBUTE), entry(ROUTER), entry(LAYOUT),
        ))
        rejection = collection.rejected[0]
        self.assertEqual(rejection.artifact_id, APP)
        self.assertEqual(rejection.path, APP)
        self.assertIn("incomplete", rejection.message)

    def test_an_unsupported_language_is_accepted_without_strong_evidence(self) -> None:
        # A .tsx path is forced by the manifest, so use the RESULT artifact whose
        # body is free-form evidence text — an unknown shape must still pass.
        collection = collect_for("validation", envelope(
            entry(BUILD, content="cargo build --release :: Finished in 4.2s"),
            entry("test-result"),
        ))
        self.assertTrue(collection.satisfied)

    def test_integrity_runs_after_ownership_not_before(self) -> None:
        # A truncated body for an artifact this package does not own is rejected
        # for OWNERSHIP, not integrity: the trust boundary comes first.
        collection = collect_for("dashboard", envelope(
            entry("src/features/dashboard/DashboardPage.tsx"),
            broken_entry(PKG, PACKAGE_JSON_MISSING_BRACE),
        ))
        self.assertEqual(codes(collection), [CollectionErrorCode.OUT_OF_SCOPE])

    def test_worker_metadata_cannot_override_trusted_manifest_language(self) -> None:
        plan = react_work_plan()
        artifacts = tuple(
            replace(spec, language="json") if spec.id == BUILD else spec
            for spec in plan.manifest.artifacts
        )
        trusted_plan = replace(
            plan, manifest=replace(plan.manifest, artifacts=artifacts)
        )
        collection = collect_for(
            "validation",
            envelope(
                entry(BUILD, content="not json", language="text"),
                entry("test-result"),
            ),
            work_plan=trusted_plan,
        )
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID, codes(collection))
        self.assertNotIn(BUILD, [candidate.artifact_id
                                for candidate in collection.accepted])

    def test_directory_and_delete_artifacts_keep_phase_4c_behavior(self) -> None:
        # Nothing in 4D inspects a body that must not exist in the first place.
        collection = collect_for("shared-ui", envelope(entry(CARD, content="")))
        self.assertEqual(codes(collection), [CollectionErrorCode.MISSING_CONTENT])

    def test_suspicious_acceptance_follows_one_explicit_policy(self) -> None:
        permissive = ResultPolicy(
            integrity=IntegrityPolicy(block_suspicious=False)
        )
        blocked = collect_for(
            "shell",
            envelope(entry(ROUTER, content=ROUTER_CUT_AFTER_IMPORT)),
        )
        allowed = collect_for(
            "shell",
            envelope(entry(ROUTER, content=ROUTER_CUT_AFTER_IMPORT)),
            policy=permissive,
        )
        self.assertIn(CollectionErrorCode.TRUNCATION_SUSPECTED, codes(blocked))
        self.assertEqual(codes(allowed), [])
        self.assertIs(
            allowed.accepted[0].integrity.status, IntegrityStatus.SUSPICIOUS
        )

    def test_the_default_policy_blocks_suspicious_candidates(self) -> None:
        self.assertTrue(DEFAULT_RESULT_POLICY.integrity.block_suspicious)

    def test_parsing_is_deterministic(self) -> None:
        scope = scope_for("shell")
        payload = envelope(
            broken_entry(APP, APP_CUT_IN_ATTRIBUTE), entry(ROUTER), entry(LAYOUT),
        )
        first = parse_artifact_candidates(payload, scope)
        second = parse_artifact_candidates(payload, scope)
        self.assertEqual(first, second)

    def test_a_rejected_candidate_never_enters_the_accepted_set(self) -> None:
        collection = collect_for("shell", envelope(
            broken_entry(APP, APP_CUT_IN_ATTRIBUTE),
            broken_entry(LAYOUT, LAYOUT_UNCLOSED_TAG),
            entry(ROUTER),
        ))
        self.assertEqual([c.artifact_id for c in collection.accepted], [ROUTER])
        self.assertEqual(
            collection.integrity_rejected_artifact_ids, (APP, LAYOUT)
        )

    def test_feedback_names_every_broken_artifact_once(self) -> None:
        collection = collect_for("shell", envelope(
            broken_entry(APP, APP_CUT_IN_ATTRIBUTE), entry(ROUTER), entry(LAYOUT),
        ))
        feedback = collection.feedback()
        self.assertIn(APP, feedback)
        self.assertEqual(feedback.count(APP), 1)
        self.assertLessEqual(len(feedback), DEFAULT_RESULT_POLICY.max_feedback_chars)
        self.assertNotIn(ROUTER, feedback)


if __name__ == "__main__":
    unittest.main()
