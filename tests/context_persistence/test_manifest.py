"""ManifestManager: backup rotation, corruption recovery, self-healing,
version gates."""

from __future__ import annotations

import json
import unittest

from dozen.context.domain.enums import ErrorCode
from dozen.context.domain.types import ConversationId

from .base import PersistenceTestCase, make_manifest


class TestManifestBackupAndRecovery(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = self.create_conversation()
        self.cid = self.manifest.conversation.id
        self.storage = self.provider.storage
        self.manifests = self.provider._manifests  # white-box on purpose

    def _corrupt_live_manifest(self) -> None:
        self.storage.manifest_path(self.cid).write_text("{ half a json", encoding="utf-8")

    def test_first_write_has_no_backup(self) -> None:
        self.assertFalse(self.storage.manifest_backup_path(self.cid).is_file())

    def test_second_write_creates_backup_of_previous_state(self) -> None:
        self.manifest.conversation.title = "renamed"
        updated = self.conversations.update_manifest(self.manifest, expected_version=1)
        self.assertTrue(updated.ok)
        backup = self.storage.manifest_backup_path(self.cid)
        self.assertTrue(backup.is_file())
        previous = json.loads(backup.read_text(encoding="utf-8"))
        self.assertEqual(previous["conversation"]["title"], "Test conversation")

    def test_corrupt_manifest_recovers_from_backup_and_self_heals(self) -> None:
        self.manifest.conversation.title = "renamed"
        self.conversations.update_manifest(self.manifest, expected_version=1)
        self._corrupt_live_manifest()

        read = self.conversations.read_manifest(self.cid)
        self.assertTrue(read.ok, getattr(read, "error", None))
        # Backup held the pre-update state — that is the recovered content.
        self.assertEqual(read.unwrap().conversation.title, "Test conversation")
        # And the live file has been restored to valid JSON on disk.
        healed = json.loads(self.storage.manifest_path(self.cid).read_text(encoding="utf-8"))
        self.assertEqual(healed["conversation"]["title"], "Test conversation")

    def test_both_copies_corrupt_reports_serialization_failure(self) -> None:
        self.manifest.conversation.title = "renamed"
        self.conversations.update_manifest(self.manifest, expected_version=1)
        self._corrupt_live_manifest()
        self.storage.manifest_backup_path(self.cid).write_text("also broken", encoding="utf-8")
        read = self.conversations.read_manifest(self.cid)
        self.assertFalse(read.ok)
        self.assertEqual(read.error.code, ErrorCode.SERIALIZATION_FAILED)

    def test_corrupt_manifest_without_backup_fails_cleanly(self) -> None:
        self._corrupt_live_manifest()
        read = self.conversations.read_manifest(self.cid)
        self.assertFalse(read.ok)
        self.assertEqual(read.error.code, ErrorCode.SERIALIZATION_FAILED)

    def test_manifest_for_wrong_conversation_is_rejected(self) -> None:
        other = make_manifest()
        path = self.storage.manifest_path(self.cid)
        path.write_text(json.dumps(other.to_dict()), encoding="utf-8")
        read = self.conversations.read_manifest(self.cid)
        self.assertFalse(read.ok)
        self.assertEqual(read.error.code, ErrorCode.SERIALIZATION_FAILED)


class TestManifestVersionGates(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = self.create_conversation()
        self.cid = self.manifest.conversation.id
        self.storage = self.provider.storage

    def _rewrite_raw(self, mutate) -> None:
        path = self.storage.manifest_path(self.cid)
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_newer_storage_version_is_readable(self) -> None:
        self._rewrite_raw(lambda d: d.__setitem__("storage_format_version", 99))
        read = self.conversations.read_manifest(self.cid)
        self.assertTrue(read.ok)
        self.assertEqual(read.unwrap().storage_format_version, 99)

    def test_newer_storage_version_is_not_writable(self) -> None:
        self._rewrite_raw(lambda d: d.__setitem__("storage_format_version", 99))
        newer = self.conversations.read_manifest(self.cid).unwrap()
        result = self.conversations.update_manifest(newer, expected_version=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VERSION_UNSUPPORTED)

    def test_unsupported_old_version_is_rejected_on_read(self) -> None:
        # -1 because the tolerant model parser coerces a raw 0 to the default
        # version 1; any value below MIN_READABLE must hit the gate.
        self._rewrite_raw(lambda d: d.__setitem__("storage_format_version", -1))
        read = self.conversations.read_manifest(self.cid)
        self.assertFalse(read.ok)
        self.assertEqual(read.error.code, ErrorCode.VERSION_UNSUPPORTED)

    def test_unknown_future_fields_survive_read_and_rewrite(self) -> None:
        self._rewrite_raw(lambda d: d.__setitem__("field_from_the_future", {"nested": True}))
        read = self.conversations.read_manifest(self.cid).unwrap()
        self.assertEqual(read.to_dict()["field_from_the_future"], {"nested": True})
        updated = self.conversations.update_manifest(read, expected_version=1)
        self.assertTrue(updated.ok)
        again = self.conversations.read_manifest(self.cid).unwrap()
        self.assertEqual(again.to_dict()["field_from_the_future"], {"nested": True})


class TestManifestMissing(PersistenceTestCase):
    def test_missing_conversation_is_not_found(self) -> None:
        read = self.conversations.read_manifest(ConversationId("01AAAAAAAAAAAAAAAAAAAAAAAA"))
        self.assertFalse(read.ok)
        self.assertEqual(read.error.code, ErrorCode.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
