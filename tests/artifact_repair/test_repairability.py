"""Phase 4E Part A — which failures targeted regeneration may and may not fix."""

from __future__ import annotations

import unittest

from dozen.artifact_repair import RepairReason, classify_rejection, is_repairable
from dozen.artifact_results import CandidateRejection, CollectionErrorCode

from .harness import LAYOUT


def rejection(code: CollectionErrorCode, artifact_id: str = LAYOUT) -> CandidateRejection:
    return CandidateRejection(
        code=code, message="x", artifact_id=artifact_id, subtask_id="shell"
    )


class TestRepairable(unittest.TestCase):
    def test_structural_invalidity_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.STRUCTURALLY_INVALID)),
            RepairReason.INTEGRITY_INVALID,
        )

    def test_suspected_truncation_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.TRUNCATION_SUSPECTED)),
            RepairReason.TRUNCATION_SUSPECTED,
        )

    def test_an_incomplete_declaration_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.INCOMPLETE_DECLARATION)),
            RepairReason.INCOMPLETE_DECLARATION,
        )

    def test_missing_content_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.MISSING_CONTENT)),
            RepairReason.MISSING_CONTENT,
        )

    def test_invalid_content_type_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.INVALID_CONTENT_TYPE)),
            RepairReason.INVALID_CONTENT,
        )

    def test_a_wrong_declared_path_for_an_owned_artifact_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.PATH_MISMATCH)),
            RepairReason.PATH_MISMATCH,
        )

    def test_a_malformed_entry_for_a_known_target_is_repairable(self) -> None:
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.MALFORMED_ENTRY)),
            RepairReason.RESPONSE_FORMAT,
        )

    def test_an_oversized_package_reply_is_repairable_by_narrowing_it(self) -> None:
        # Targeted repair is the direct remedy: the next reply carries only the
        # outstanding files, so the aggregate bound is no longer the obstacle.
        self.assertIs(
            classify_rejection(rejection(CollectionErrorCode.AGGREGATE_TOO_LARGE)),
            RepairReason.INVALID_CONTENT,
        )


class TestNonRepairable(unittest.TestCase):
    def assert_unsafe(self, code: CollectionErrorCode) -> None:
        self.assertIs(classify_rejection(rejection(code)), RepairReason.NON_REPAIRABLE)
        self.assertFalse(is_repairable(rejection(code)))

    def test_an_unknown_artifact_id_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.UNKNOWN_ARTIFACT_ID)

    def test_an_out_of_scope_artifact_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.OUT_OF_SCOPE)

    def test_wrong_package_ownership_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.WRONG_PACKAGE)

    def test_a_wrong_root_subtask_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.WRONG_SUBTASK)

    def test_external_artifact_production_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.EXTERNAL_ARTIFACT)

    def test_a_directory_body_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.DIRECTORY_CONTENT)

    def test_manifest_operation_spoofing_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.INVALID_OPERATION)

    def test_a_provenance_violation_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.INVALID_PROVENANCE)

    def test_an_invalid_path_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.INVALID_PATH)

    def test_a_duplicate_submission_is_not_repairable(self) -> None:
        # Competing/duplicated candidates are a CONFLICT, and Phase 4E resolves
        # no conflicts.
        self.assert_unsafe(CollectionErrorCode.DUPLICATE_IN_RESPONSE)

    def test_an_oversized_single_artifact_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.CONTENT_TOO_LARGE)

    def test_a_preserved_resubmission_is_not_repairable(self) -> None:
        self.assert_unsafe(CollectionErrorCode.PRESERVED_RESUBMISSION)

    def test_an_entry_with_no_artifact_id_creates_no_target(self) -> None:
        # There is nothing to aim a repair at; the owed artifact is simply still
        # missing, and is picked up as MISSING_REQUIRED instead.
        self.assert_unsafe(CollectionErrorCode.MISSING_ARTIFACT_ID)

    def test_every_collection_code_is_classified(self) -> None:
        for code in CollectionErrorCode:
            self.assertIsInstance(classify_rejection(rejection(code)), RepairReason)


if __name__ == "__main__":
    unittest.main()
