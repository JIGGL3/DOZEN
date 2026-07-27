"""ConversationSession / ConversationRegistry lifecycle and the
ConversationMetadataManager surface."""

from __future__ import annotations

import unittest

from dozen.context.domain.enums import ErrorCode, EventType
from dozen.context.domain.types import ConversationId, RunId
from dozen.context.manager import ConversationRegistry
from dozen.context.manager.factory import SystemClock

from .base import ManagerTestCase


class TestSessions(ManagerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.create().conversation.id

    def test_session_lifecycle_via_manager(self) -> None:
        started = self.manager.start_session(self.cid, run_id=RunId("run-1"), workflow_id=None)
        self.assertTrue(started.ok)
        session = started.unwrap()
        self.assertEqual(session.conversation_id, self.cid)
        self.assertEqual(session.run_id, "run-1")
        self.assertTrue(self.manager.cache.is_pinned(self.cid))  # active => pinned
        self.assertEqual(self.manager.registry.count(), 1)

        ended = self.manager.end_session(session.session_id)
        self.assertIsNotNone(ended)
        self.assertEqual(self.manager.registry.count(), 0)
        self.assertFalse(self.manager.cache.is_pinned(self.cid))  # unpinned again
        types = self.recorded.types()
        self.assertIn(EventType.SESSION_STARTED, types)
        self.assertIn(EventType.SESSION_ENDED, types)

    def test_session_for_missing_conversation(self) -> None:
        started = self.manager.start_session(ConversationId("01DDDDDDDDDDDDDDDDDDDDDDDD"))
        self.assertFalse(started.ok)
        self.assertEqual(started.error.code, ErrorCode.NOT_FOUND)

    def test_appends_touch_active_sessions(self) -> None:
        session = self.manager.start_session(self.cid, run_id=RunId("run-2")).unwrap()
        before = session.last_activity
        self.manager.append_user_message(self.cid, "activity!")
        self.assertGreaterEqual(session.last_activity, before)

    def test_pin_survives_until_last_session_ends(self) -> None:
        first = self.manager.start_session(self.cid).unwrap()
        second = self.manager.start_session(self.cid).unwrap()
        self.manager.end_session(first.session_id)
        self.assertTrue(self.manager.cache.is_pinned(self.cid))  # one still active
        self.manager.end_session(second.session_id)
        self.assertFalse(self.manager.cache.is_pinned(self.cid))

    def test_provider_and_agent_tracking(self) -> None:
        clock = SystemClock()
        session = self.manager.start_session(self.cid).unwrap()
        session.record_provider("openai", clock)
        session.record_provider("openai", clock)      # consecutive duplicate collapses
        session.record_provider("anthropic", clock)
        session.record_agent("researcher", clock)
        self.assertEqual(session.provider_history, ["openai", "anthropic"])
        self.assertEqual(session.current_provider, "anthropic")
        self.assertEqual(session.current_agent, "researcher")
        payload = session.to_dict()
        self.assertEqual(payload["provider_history"], ["openai", "anthropic"])


class TestRegistryUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ConversationRegistry()
        self.cid = ConversationId("01EEEEEEEEEEEEEEEEEEEEEEEE")

    def test_indexes_both_ways(self) -> None:
        one = self.registry.start_session(self.cid, run_id=RunId("r1"))
        two = self.registry.start_session(self.cid, run_id=RunId("r2"))
        self.assertEqual(self.registry.count(), 2)
        self.assertEqual(
            {s.session_id for s in self.registry.sessions_for(self.cid)},
            {one.session_id, two.session_id},
        )
        self.assertEqual(len(self.registry.active_sessions()), 2)

    def test_end_session_idempotent(self) -> None:
        session = self.registry.start_session(self.cid)
        self.assertIsNotNone(self.registry.end_session(session.session_id))
        self.assertIsNone(self.registry.end_session(session.session_id))
        self.assertEqual(self.registry.sessions_for(self.cid), [])

    def test_end_sessions_for_conversation(self) -> None:
        self.registry.start_session(self.cid)
        self.registry.start_session(self.cid)
        other = ConversationId("01FFFFFFFFFFFFFFFFFFFFFFFF")
        keep = self.registry.start_session(other)
        ended = self.registry.end_sessions_for(self.cid)
        self.assertEqual(len(ended), 2)
        self.assertEqual(self.registry.count(), 1)
        self.assertEqual(self.registry.active_sessions()[0].session_id, keep.session_id)


class TestMetadataManager(ManagerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.create().conversation.id

    def test_set_and_get(self) -> None:
        result = self.manager.metadata.set_values(
            self.cid, {"workflow.name": "stress", "user.locale": "en"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.manager.metadata.get(self.cid, "workflow.name").unwrap(), "stress")
        self.assertEqual(self.manager.metadata.get(self.cid, "missing", "fallback").unwrap(),
                         "fallback")
        # Persisted for a cold reader too:
        cold = self.new_manager()
        self.assertEqual(cold.metadata.get_all(self.cid).unwrap()["user.locale"], "en")

    def test_namespace_filter(self) -> None:
        self.manager.metadata.set_values(
            self.cid,
            {"workflow.a": 1, "workflow.b": 2, "agent.name": "researcher"},
        )
        workflow_only = self.manager.metadata.namespace(self.cid, "workflow.").unwrap()
        self.assertEqual(workflow_only, {"workflow.a": 1, "workflow.b": 2})

    def test_delete_keys(self) -> None:
        self.manager.metadata.set_values(self.cid, {"user.a": 1, "user.b": 2})
        self.manager.metadata.delete_keys(self.cid, ["user.a", "never-existed"])
        remaining = self.manager.metadata.get_all(self.cid).unwrap()
        self.assertNotIn("user.a", remaining)
        self.assertEqual(remaining["user.b"], 2)

    def test_invalid_keys_rejected(self) -> None:
        result = self.manager.metadata.set_values(self.cid, {"": "empty key"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_concurrent_metadata_writers_merge(self) -> None:
        """Two managers write different keys — optimistic retry must merge
        both, never lose one (the classic lost-update case)."""
        other = self.new_manager()
        self.manager.metadata.set_values(self.cid, {"user.first": 1})
        other.metadata.set_values(self.cid, {"user.second": 2})
        final = self.new_manager().metadata.get_all(self.cid).unwrap()
        self.assertEqual(final.get("user.first"), 1)
        self.assertEqual(final.get("user.second"), 2)


if __name__ == "__main__":
    unittest.main()
