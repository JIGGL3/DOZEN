"""Repository behavior: the Phase 1.1 port contract reproduced on the real
filesystem, plus the extended surface (exists, search, tail, stats)."""

from __future__ import annotations

import json
import unittest

from dozen.context.domain.enums import ConversationStatus, ErrorCode, MessageRole
from dozen.context.domain.results import Failure
from dozen.context.domain.types import ConversationId, MessageId, Timestamp

from .base import PersistenceTestCase, make_manifest, make_message, make_summary


class TestConversationCrud(PersistenceTestCase):
    def test_create_then_read(self) -> None:
        manifest = self.create_conversation()
        read = self.conversations.read_manifest(manifest.conversation.id)
        self.assertTrue(read.ok)
        self.assertEqual(read.unwrap().conversation.id, manifest.conversation.id)
        self.assertEqual(read.unwrap().conversation.title, "Test conversation")

    def test_duplicate_create_fails_without_exception(self) -> None:
        manifest = self.create_conversation()
        dup = self.conversations.create_conversation(manifest)
        self.assertIsInstance(dup, Failure)
        self.assertEqual(dup.error.code, ErrorCode.ALREADY_EXISTS)

    def test_read_missing_is_not_found(self) -> None:
        missing = self.conversations.read_manifest(ConversationId("nope-not-here"))
        self.assertIsInstance(missing, Failure)
        self.assertEqual(missing.error.code, ErrorCode.NOT_FOUND)

    def test_path_unsafe_id_is_rejected(self) -> None:
        manifest = make_manifest()
        manifest.conversation.id = ConversationId("../escape/attempt")
        created = self.conversations.create_conversation(manifest)
        self.assertFalse(created.ok)
        self.assertEqual(created.error.code, ErrorCode.VALIDATION_FAILED)
        # Nothing escaped the storage root.
        self.assertFalse((self.tmp / "escape").exists())

    def test_optimistic_versioning(self) -> None:
        manifest = self.create_conversation()
        updated = self.conversations.update_manifest(manifest, expected_version=1)
        self.assertTrue(updated.ok)
        self.assertEqual(updated.unwrap().conversation.version, 2)
        stale = self.conversations.update_manifest(manifest, expected_version=1)
        self.assertIsInstance(stale, Failure)
        self.assertEqual(stale.error.code, ErrorCode.CONFLICT)

    def test_update_missing_is_not_found(self) -> None:
        manifest = make_manifest()
        result = self.conversations.update_manifest(manifest, expected_version=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NOT_FOUND)

    def test_delete_removes_everything_and_is_idempotent(self) -> None:
        manifest = self.create_conversation()
        cid = manifest.conversation.id
        self.messages.append_messages(cid, [make_message(cid)])
        self.assertTrue(self.conversations.delete_conversation(cid).ok)
        self.assertFalse(self.conversations.exists(cid))
        self.assertFalse(self.provider.storage.conversation_dir(cid).exists())
        # Second delete: still success (idempotent).
        self.assertTrue(self.conversations.delete_conversation(cid).ok)

    def test_exists(self) -> None:
        manifest = self.create_conversation()
        self.assertTrue(self.conversations.exists(manifest.conversation.id))
        self.assertFalse(self.conversations.exists(ConversationId("01B00000000000000000000000")))


class TestConversationListingAndSearch(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifests = [self.create_conversation(f"Conversation {i}") for i in range(5)]
        archived = self.manifests[1]
        archived.conversation.status = ConversationStatus.ARCHIVED
        self.conversations.update_manifest(archived, expected_version=1)

    def test_list_all_sorted_by_id(self) -> None:
        listed = self.conversations.list_conversations(limit=50).unwrap()
        ids = [m.conversation.id for m in listed]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(listed), 5)

    def test_list_filters_by_status(self) -> None:
        archived = self.conversations.list_conversations(status=ConversationStatus.ARCHIVED).unwrap()
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].conversation.title, "Conversation 1")

    def test_list_pagination(self) -> None:
        page = self.conversations.list_conversations(limit=2, offset=2).unwrap()
        self.assertEqual(len(page), 2)
        everything = self.conversations.list_conversations(limit=50).unwrap()
        self.assertEqual(
            [m.conversation.id for m in page],
            [m.conversation.id for m in everything[2:4]],
        )

    def test_corrupt_conversation_does_not_poison_listing(self) -> None:
        victim = self.manifests[2].conversation.id
        self.provider.storage.manifest_path(victim).write_text("broken", encoding="utf-8")
        listed = self.conversations.list_conversations(limit=50).unwrap()
        self.assertEqual(len(listed), 4)

    def test_search_by_title(self) -> None:
        hits = self.conversations.search(title_contains="conversation 3").unwrap()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].conversation.title, "Conversation 3")

    def test_search_by_metadata(self) -> None:
        tagged = self.manifests[4]
        tagged.conversation.metadata.values["workflow.name"] = "stress-test"
        self.conversations.update_manifest(tagged, expected_version=1)
        hits = self.conversations.search(metadata_filters={"workflow.name": "stress-test"}).unwrap()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].conversation.id, tagged.conversation.id)

    def test_search_by_timestamp_window(self) -> None:
        target = self.manifests[0].conversation
        hits = self.conversations.search(
            created_from=target.created_at, created_to=target.created_at
        ).unwrap()
        self.assertIn(target.id, [m.conversation.id for m in hits])
        none = self.conversations.search(created_to=Timestamp("1999-01-01T00:00:00.000Z")).unwrap()
        self.assertEqual(none, [])


class TestMessageRepository(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = self.create_conversation()
        self.cid = self.manifest.conversation.id

    def test_append_preserves_order_and_pagination(self) -> None:
        msgs = [make_message(self.cid) for _ in range(5)]
        self.assertEqual(self.messages.append_messages(self.cid, msgs).unwrap(), 5)
        all_msgs = self.messages.read_messages(self.cid).unwrap()
        self.assertEqual([m.id for m in all_msgs], [m.id for m in msgs])
        after = self.messages.read_messages(self.cid, after_id=msgs[2].id).unwrap()
        self.assertEqual([m.id for m in after], [m.id for m in msgs[3:]])
        self.assertEqual(self.messages.count_messages(self.cid).unwrap(), 5)

    def test_unknown_after_id_yields_empty_page(self) -> None:
        self.messages.append_messages(self.cid, [make_message(self.cid)])
        page = self.messages.read_messages(self.cid, after_id=MessageId("01ZZZZZZZZZZZZZZZZZZZZZZZZ")).unwrap()
        self.assertEqual(page, [])

    def test_limit_is_respected(self) -> None:
        self.messages.append_messages(self.cid, [make_message(self.cid) for _ in range(10)])
        self.assertEqual(len(self.messages.read_messages(self.cid, limit=4).unwrap()), 4)

    def test_append_to_missing_conversation_is_not_found(self) -> None:
        ghost = ConversationId("01C00000000000000000000000")
        result = self.messages.append_messages(ghost, [make_message(ghost)])
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NOT_FOUND)

    def test_foreign_message_is_rejected(self) -> None:
        other = self.create_conversation("other")
        stray = make_message(other.conversation.id)
        result = self.messages.append_messages(self.cid, [stray])
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_roles_round_trip(self) -> None:
        msg = make_message(self.cid, role=MessageRole.SUMMARY)
        self.messages.append_messages(self.cid, [msg])
        stored = self.messages.read_messages(self.cid).unwrap()[0]
        self.assertEqual(stored.role, MessageRole.SUMMARY)

    def test_unknown_role_from_future_schema_survives(self) -> None:
        msg = make_message(self.cid)
        record = msg.to_dict()
        record["role"] = "critic"
        with open(self.provider.storage.messages_path(self.cid), "ab") as handle:
            handle.write((json.dumps(record) + "\n").encode("utf-8"))
        stored = self.messages.read_messages(self.cid).unwrap()[0]
        self.assertEqual(stored.role, MessageRole.UNKNOWN)
        self.assertEqual(stored.extra["role__raw"], "critic")

    def test_get_message_by_id(self) -> None:
        msgs = [make_message(self.cid, content=f"m{i}") for i in range(3)]
        self.messages.append_messages(self.cid, msgs)
        found = self.messages.get_message(self.cid, msgs[1].id)
        self.assertTrue(found.ok)
        self.assertEqual(found.unwrap().content, "m1")
        missing = self.messages.get_message(self.cid, MessageId("01D00000000000000000000000"))
        self.assertFalse(missing.ok)
        self.assertEqual(missing.error.code, ErrorCode.NOT_FOUND)

    def test_read_tail(self) -> None:
        msgs = [make_message(self.cid, content=f"m{i}") for i in range(10)]
        self.messages.append_messages(self.cid, msgs)
        tail = self.messages.read_tail(self.cid, 3).unwrap()
        self.assertEqual([m.content for m in tail], ["m7", "m8", "m9"])

    def test_search_by_role(self) -> None:
        self.messages.append_messages(self.cid, [
            make_message(self.cid, role=MessageRole.USER),
            make_message(self.cid, role=MessageRole.ASSISTANT),
            make_message(self.cid, role=MessageRole.USER),
        ])
        users = self.messages.search_messages(self.cid, role=MessageRole.USER).unwrap()
        self.assertEqual(len(users), 2)

    def test_search_by_timestamp_window(self) -> None:
        early = make_message(self.cid, created_at=Timestamp("2026-01-01T00:00:00.000Z"))
        late = make_message(self.cid, created_at=Timestamp("2026-06-01T00:00:00.000Z"))
        self.messages.append_messages(self.cid, [early, late])
        hits = self.messages.search_messages(
            self.cid,
            created_from=Timestamp("2026-03-01T00:00:00.000Z"),
        ).unwrap()
        self.assertEqual([m.id for m in hits], [late.id])

    def test_search_by_metadata(self) -> None:
        tagged = make_message(self.cid)
        tagged.metadata.values["agent.name"] = "researcher"
        self.messages.append_messages(self.cid, [make_message(self.cid), tagged])
        hits = self.messages.search_messages(
            self.cid, metadata_filters={"agent.name": "researcher"}
        ).unwrap()
        self.assertEqual([m.id for m in hits], [tagged.id])

    def test_search_pagination(self) -> None:
        self.messages.append_messages(self.cid, [make_message(self.cid, content=f"m{i}") for i in range(6)])
        page = self.messages.search_messages(self.cid, limit=2, offset=2).unwrap()
        self.assertEqual([m.content for m in page], ["m2", "m3"])

    def test_corrupt_line_is_contained_and_reported(self) -> None:
        self.messages.append_messages(self.cid, [make_message(self.cid, content="before")])
        with open(self.provider.storage.messages_path(self.cid), "ab") as handle:
            handle.write(b"{torn line no json\n")
        self.messages.append_messages(self.cid, [make_message(self.cid, content="after")])
        contents = [m.content for m in self.messages.read_messages(self.cid).unwrap()]
        self.assertEqual(contents, ["before", "after"])
        report = self.messages.integrity_report(self.cid).unwrap()
        self.assertTrue(report.corrupt)
        self.assertEqual(len(report.errors), 1)

    def test_append_updates_manifest_statistics_without_version_bump(self) -> None:
        msgs = [make_message(self.cid, content="hello world " * 10) for _ in range(3)]
        self.messages.append_messages(self.cid, msgs)
        manifest = self.conversations.read_manifest(self.cid).unwrap()
        stats = manifest.conversation.stats
        self.assertEqual(stats.message_count, 3)
        self.assertGreater(stats.approx_tokens, 0)
        self.assertEqual(stats.last_message_at, msgs[-1].created_at)
        # Bookkeeping must not consume the optimistic-concurrency version.
        self.assertEqual(manifest.conversation.version, 1)
        updated = self.conversations.update_manifest(manifest, expected_version=1)
        self.assertTrue(updated.ok)


class TestSummaryRepository(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = self.create_conversation()
        self.cid = self.manifest.conversation.id

    def test_append_and_read(self) -> None:
        summary = make_summary(self.cid, content="the story so far")
        self.assertTrue(self.summaries.append_summary(self.cid, summary).ok)
        stored = self.summaries.read_summaries(self.cid).unwrap()
        self.assertEqual([s.id for s in stored], [summary.id])
        self.assertEqual(stored[0].content, "the story so far")
        self.assertEqual(self.summaries.count_summaries(self.cid).unwrap(), 1)

    def test_append_to_missing_conversation_is_not_found(self) -> None:
        ghost = ConversationId("01E00000000000000000000000")
        result = self.summaries.append_summary(ghost, make_summary(ghost))
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NOT_FOUND)

    def test_foreign_summary_is_rejected(self) -> None:
        other = self.create_conversation("other")
        result = self.summaries.append_summary(self.cid, make_summary(other.conversation.id))
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_summary_count_reaches_manifest_stats(self) -> None:
        self.summaries.append_summary(self.cid, make_summary(self.cid))
        self.summaries.append_summary(self.cid, make_summary(self.cid))
        manifest = self.conversations.read_manifest(self.cid).unwrap()
        self.assertEqual(manifest.conversation.stats.summary_count, 2)


if __name__ == "__main__":
    unittest.main()
