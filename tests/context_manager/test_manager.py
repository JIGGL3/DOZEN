"""ConversationManager façade: lifecycle, message recording, ordering,
optimistic mutation, and event publishing."""

from __future__ import annotations

import unittest

from dozen.context.domain.enums import (
    ConversationStatus,
    ErrorCode,
    EventType,
    MessageRole,
    OriginKind,
)
from dozen.context.domain.types import ConversationId

from .base import ManagerTestCase


class TestConversationLifecycle(ManagerTestCase):
    def test_create_and_get(self) -> None:
        manifest = self.create("hello world")
        found = self.manager.get_conversation(manifest.conversation.id)
        self.assertTrue(found.ok)
        self.assertEqual(found.unwrap().conversation.title, "hello world")
        self.assertEqual(self.recorded.of_type(EventType.CONVERSATION_CREATED)[0].payload["title"],
                         "hello world")

    def test_create_with_metadata(self) -> None:
        created = self.manager.create_conversation(
            title="t", metadata={"user.locale": "en"}
        ).unwrap()
        again = self.manager.get_conversation(created.conversation.id).unwrap()
        self.assertEqual(again.conversation.metadata.values["user.locale"], "en")

    def test_exists(self) -> None:
        manifest = self.create()
        self.assertTrue(self.manager.conversation_exists(manifest.conversation.id))
        self.assertFalse(self.manager.conversation_exists(ConversationId("01AAAAAAAAAAAAAAAAAAAAAAAA")))

    def test_get_missing_is_not_found(self) -> None:
        missing = self.manager.get_conversation(ConversationId("01AAAAAAAAAAAAAAAAAAAAAAAA"))
        self.assertFalse(missing.ok)
        self.assertEqual(missing.error.code, ErrorCode.NOT_FOUND)

    def test_rename(self) -> None:
        manifest = self.create("old name")
        cid = manifest.conversation.id
        renamed = self.manager.rename_conversation(cid, "new name")
        self.assertTrue(renamed.ok)
        # Persisted (fresh manager = fresh cache = disk truth):
        other = self.new_manager()
        self.assertEqual(other.get_conversation(cid).unwrap().conversation.title, "new name")
        event = self.recorded.of_type(EventType.CONVERSATION_RENAMED)[0]
        self.assertEqual(event.payload, {"old_title": "old name", "new_title": "new name"})

    def test_rename_rejects_empty_title(self) -> None:
        manifest = self.create()
        result = self.manager.rename_conversation(manifest.conversation.id, "   ")
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_archive(self) -> None:
        manifest = self.create()
        cid = manifest.conversation.id
        archived = self.manager.archive_conversation(cid)
        self.assertTrue(archived.ok)
        self.assertEqual(archived.unwrap().conversation.status, ConversationStatus.ARCHIVED)
        listed = self.manager.list_conversations(status=ConversationStatus.ARCHIVED).unwrap()
        self.assertEqual([m.conversation.id for m in listed], [cid])
        self.assertEqual(len(self.recorded.of_type(EventType.CONVERSATION_ARCHIVED)), 1)

    def test_delete(self) -> None:
        manifest = self.create()
        cid = manifest.conversation.id
        self.manager.append_user_message(cid, "will be deleted")
        self.assertTrue(self.manager.delete_conversation(cid).ok)
        self.assertFalse(self.manager.conversation_exists(cid))
        self.assertFalse(self.manager.get_conversation(cid).ok)
        self.assertEqual(len(self.recorded.of_type(EventType.CONVERSATION_DELETED)), 1)

    def test_derive_title(self) -> None:
        self.assertEqual(self.manager.derive_title("  hello \n world  "), "hello world")
        self.assertEqual(self.manager.derive_title(""), "Untitled conversation")
        long_title = self.manager.derive_title("x" * 500)
        self.assertLessEqual(len(long_title), 80)
        self.assertTrue(long_title.endswith("…"))


class TestMessageRecording(ManagerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.create().conversation.id

    def test_user_and_assistant_messages(self) -> None:
        user = self.manager.append_user_message(self.cid, "Hello").unwrap()
        assistant = self.manager.append_assistant_message(
            self.cid, "Hi! How can I help?", provider=None, agent_name="synthesizer"
        ).unwrap()
        stored = self.manager.read_messages(self.cid).unwrap()
        self.assertEqual([m.id for m in stored], [user.id, assistant.id])
        self.assertEqual(stored[0].role, MessageRole.USER)
        self.assertEqual(stored[0].origin.kind, OriginKind.USER)
        self.assertEqual(stored[1].role, MessageRole.ASSISTANT)
        self.assertEqual(stored[1].origin.kind, OriginKind.SYNTHESIZER)

    def test_empty_content_rejected(self) -> None:
        for method in (self.manager.append_user_message, self.manager.append_assistant_message):
            result = method(self.cid, "   ")
            self.assertFalse(result.ok)
            self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_append_to_missing_conversation(self) -> None:
        ghost = ConversationId("01BBBBBBBBBBBBBBBBBBBBBBBB")
        result = self.manager.append_user_message(ghost, "into the void")
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NOT_FOUND)

    def test_message_ordering_is_total(self) -> None:
        for i in range(25):
            role_method = (
                self.manager.append_user_message if i % 2 == 0
                else self.manager.append_assistant_message
            )
            role_method(self.cid, f"message {i}")
        stored = self.manager.read_messages(self.cid, limit=100).unwrap()
        self.assertEqual([m.content for m in stored], [f"message {i}" for i in range(25)])
        ids = [m.id for m in stored]
        self.assertEqual(ids, sorted(ids))  # append order == id order

    def test_pagination_via_manager(self) -> None:
        appended = [self.manager.append_user_message(self.cid, f"m{i}").unwrap() for i in range(6)]
        page = self.manager.read_messages(self.cid, limit=2, after_id=appended[1].id).unwrap()
        self.assertEqual([m.content for m in page], ["m2", "m3"])
        self.assertEqual(self.manager.count_messages(self.cid).unwrap(), 6)

    def test_append_refreshes_manifest_stats(self) -> None:
        self.manager.append_user_message(self.cid, "one")
        self.manager.append_user_message(self.cid, "two")
        manifest = self.manager.get_conversation(self.cid).unwrap()
        self.assertEqual(manifest.conversation.stats.message_count, 2)

    def test_message_added_events(self) -> None:
        self.manager.append_user_message(self.cid, "Hello")
        events = self.recorded.of_type(EventType.MESSAGE_APPENDED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["role"], "user")
        self.assertEqual(events[0].conversation_id, self.cid)


class TestOptimisticMutation(ManagerTestCase):
    def test_concurrent_writers_do_not_lose_updates(self) -> None:
        """Manager A holds a stale cached view while manager B renames; A's
        rename must retry against the fresh version and win, not clobber."""
        manifest = self.create("original")
        cid = manifest.conversation.id
        other = self.new_manager()
        other.rename_conversation(cid, "renamed by B")
        result = self.manager.rename_conversation(cid, "renamed by A")
        self.assertTrue(result.ok, getattr(result, "error", None))
        final = self.new_manager().get_conversation(cid).unwrap()
        self.assertEqual(final.conversation.title, "renamed by A")
        self.assertGreaterEqual(final.conversation.version, 3)

    def test_mutate_missing_conversation(self) -> None:
        result = self.manager.rename_conversation(
            ConversationId("01CCCCCCCCCCCCCCCCCCCCCCCC"), "nope"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
