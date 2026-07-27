"""Phase 4E Part G — parsing a repair reply against the narrow repair scope.

Phase 4C/4D validation is never loosened here: path, ownership, operation,
bounds, duplicates and structural integrity all still run.
"""

from __future__ import annotations

import unittest

from dozen.artifact_repair import (
    ArtifactRepairPolicy,
    collect_repair_artifacts,
    derive_repair_scope,
)
from dozen.artifact_results import CollectionErrorCode

from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    PAGE,
    PKG,
    ROUTER,
    decide,
    entry,
    envelope,
    envelope_with,
    first_attempt,
    scope_for,
)

TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


def setup(payload=None):
    """Attempt 1 leaves DashboardLayout truncated; App and router are preserved."""
    state = first_attempt(
        payload or envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})
    )
    request = decide(state).request
    return state, request, derive_repair_scope(scope_for("shell"), request)


def parse(payload, *, policy=None, attempt=2):
    _state, request, repair_scope = setup()
    return collect_repair_artifacts(
        payload, repair_scope, request,
        policy=policy or ArtifactRepairPolicy(),
        attempt=attempt, producer_subtask_id="shell",
    )


def codes(collection):
    return [rejection.code for rejection in collection.rejected]


class TestValidRepairResponse(unittest.TestCase):
    def test_a_targeted_response_is_accepted(self) -> None:
        collection = parse(envelope(entry(LAYOUT)))
        self.assertEqual([c.artifact_id for c in collection.accepted], [LAYOUT])
        self.assertEqual(collection.rejected, ())
        self.assertTrue(collection.satisfied)

    def test_the_repaired_content_is_preserved_exactly(self) -> None:
        collection = parse(envelope(entry(LAYOUT)))
        self.assertEqual(collection.accepted[0].content, CONTENT[LAYOUT])

    def test_runtime_provenance_wins_over_the_worker(self) -> None:
        collection = parse(envelope(entry(
            LAYOUT, attempt=99, subtask_id="spoofed", package_id="spoofed",
            producer_subtask_id="spoofed", recursion_depth=7,
        )))
        candidate = collection.accepted[0]
        self.assertEqual(candidate.attempt, 2)
        self.assertEqual(candidate.subtask_id, "shell")
        self.assertEqual(candidate.package_id, "application-shell")
        self.assertEqual(candidate.recursion_depth, 0)


class TestPreservedArtifacts(unittest.TestCase):
    def test_a_changed_preserved_artifact_is_rejected_as_an_overwrite(self) -> None:
        collection = parse(envelope(
            entry(APP, content="export default function App() { return null; }\n"),
            entry(LAYOUT),
        ))
        self.assertEqual(codes(collection), [CollectionErrorCode.PRESERVED_RESUBMISSION])
        self.assertEqual([c.artifact_id for c in collection.accepted], [LAYOUT])
        self.assertFalse(collection.satisfied)

    def test_an_identical_preserved_echo_is_tolerated_not_accepted(self) -> None:
        collection = parse(envelope(entry(APP), entry(LAYOUT)))
        self.assertEqual(collection.rejected, ())
        self.assertEqual([c.artifact_id for c in collection.accepted], [LAYOUT])
        self.assertTrue(any("already accepted" in w for w in collection.warnings))

    def test_policy_can_refuse_every_preserved_resubmission(self) -> None:
        collection = parse(
            envelope(entry(APP), entry(LAYOUT)),
            policy=ArtifactRepairPolicy(tolerate_identical_preserved_echo=False),
        )
        self.assertEqual(codes(collection), [CollectionErrorCode.PRESERVED_RESUBMISSION])

    def test_identical_body_under_a_modified_path_is_rejected(self) -> None:
        collection = parse(envelope(
            entry(APP, path="src/app/renamed-App.tsx"), entry(LAYOUT),
        ))
        self.assertEqual(codes(collection), [CollectionErrorCode.PRESERVED_RESUBMISSION])

    def test_identical_body_with_operation_spoofing_is_rejected(self) -> None:
        collection = parse(envelope(
            entry(APP, operation="delete"), entry(LAYOUT),
        ))
        self.assertEqual(codes(collection), [CollectionErrorCode.PRESERVED_RESUBMISSION])

    def test_a_preserved_body_under_the_target_id_is_rejected(self) -> None:
        collection = parse(envelope(entry(LAYOUT, content=CONTENT[APP])))
        self.assertEqual(codes(collection), [CollectionErrorCode.PRESERVED_RESUBMISSION])
        self.assertEqual(collection.accepted, ())

    def test_policy_can_ignore_preserved_resubmissions_entirely(self) -> None:
        collection = parse(
            envelope(entry(APP, content="different\n"), entry(LAYOUT)),
            policy=ArtifactRepairPolicy(allow_preserved_in_response=True),
        )
        self.assertEqual(collection.rejected, ())
        self.assertEqual([c.artifact_id for c in collection.accepted], [LAYOUT])


class TestUnrelatedAndInvalidTargets(unittest.TestCase):
    def test_an_unrelated_package_artifact_is_rejected(self) -> None:
        collection = parse(envelope(entry(LAYOUT), entry(PAGE)))
        self.assertEqual(codes(collection), [CollectionErrorCode.OUT_OF_SCOPE])
        self.assertEqual([c.artifact_id for c in collection.accepted], [LAYOUT])

    def test_an_unknown_artifact_id_is_rejected(self) -> None:
        collection = parse(envelope(
            entry(LAYOUT), entry("src/made-up.tsx", path="src/made-up.tsx"),
        ))
        self.assertEqual(codes(collection), [CollectionErrorCode.UNKNOWN_ARTIFACT_ID])

    def test_a_wrong_path_is_rejected(self) -> None:
        collection = parse(envelope(entry(LAYOUT, path="src/Layout.tsx")))
        self.assertEqual(codes(collection), [CollectionErrorCode.PATH_MISMATCH])
        self.assertEqual(collection.accepted, ())

    def test_a_duplicate_target_is_rejected(self) -> None:
        collection = parse(envelope(entry(LAYOUT), entry(LAYOUT)))
        self.assertEqual(codes(collection), [CollectionErrorCode.DUPLICATE_IN_RESPONSE])
        self.assertEqual(len(collection.accepted), 1)

    def test_a_still_truncated_repair_is_rejected_by_the_integrity_gate(self) -> None:
        collection = parse(envelope(entry(LAYOUT, content=TRUNCATED_LAYOUT)))
        self.assertIn(
            codes(collection)[0],
            (CollectionErrorCode.STRUCTURALLY_INVALID,
             CollectionErrorCode.TRUNCATION_SUSPECTED),
        )
        self.assertEqual(collection.accepted, ())

    def test_an_incomplete_declaration_is_still_rejected(self) -> None:
        collection = parse(envelope(entry(LAYOUT, complete=False)))
        self.assertEqual(codes(collection), [CollectionErrorCode.INCOMPLETE_DECLARATION])

    def test_an_empty_repair_list_leaves_the_target_missing(self) -> None:
        collection = parse(envelope())
        self.assertEqual(collection.accepted, ())
        self.assertEqual(collection.missing_required_artifact_ids, (LAYOUT,))
        self.assertFalse(collection.satisfied)

    def test_a_partial_repair_response_reports_what_is_still_missing(self) -> None:
        state = first_attempt(envelope(entry(APP)))
        request = decide(state).request
        repair_scope = derive_repair_scope(scope_for("shell"), request)
        collection = collect_repair_artifacts(
            envelope(entry(ROUTER)), repair_scope, request, attempt=2,
        )
        self.assertEqual([c.artifact_id for c in collection.accepted], [ROUTER])
        self.assertEqual(collection.missing_required_artifact_ids, (LAYOUT,))

    def test_a_legacy_envelope_on_a_repair_attempt_does_not_crash(self) -> None:
        collection = parse({"summary": "done", "artifacts": {"x.tsx": "y"}})
        self.assertFalse(collection.engaged)
        self.assertEqual(collection.accepted, ())
        self.assertEqual(collection.missing_required_artifact_ids, (LAYOUT,))


if __name__ == "__main__":
    unittest.main()
