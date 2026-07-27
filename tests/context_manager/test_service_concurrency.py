"""ConversationService (the server's entry point), restart persistence, and
concurrent access through the manager."""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from dozen.context.domain.enums import ErrorCode
from dozen.context.domain.types import ConversationId
from dozen.context.manager import ConversationService
from dozen.context.utils import is_ulid

from .base import ManagerTestCase


class ServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-convsvc-"))
        self.root = str(self.tmp / "conversations")
        self.service = ConversationService(root_path=self.root, fsync=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestEnsureConversation(ServiceTestCase):
    def test_no_id_creates_new(self) -> None:
        cid, created = self.service.ensure_conversation(None, title_hint="Hello world")
        self.assertTrue(created)
        self.assertTrue(is_ulid(cid))
        manifest = self.service.manager.get_conversation(cid).unwrap()
        self.assertEqual(manifest.conversation.title, "Hello world")

    def test_existing_id_is_reused(self) -> None:
        cid, _ = self.service.ensure_conversation(None, title_hint="first")
        again, created = self.service.ensure_conversation(str(cid), title_hint="second")
        self.assertEqual(again, cid)
        self.assertFalse(created)

    def test_malformed_id_falls_back_to_new(self) -> None:
        cid, created = self.service.ensure_conversation("not-a-ulid!!", title_hint="x")
        self.assertTrue(created)
        self.assertTrue(is_ulid(cid))

    def test_unknown_valid_ulid_falls_back_to_new(self) -> None:
        ghost = "01AAAAAAAAAAAAAAAAAAAAAAAA"
        cid, created = self.service.ensure_conversation(ghost, title_hint="x")
        self.assertTrue(created)
        self.assertNotEqual(str(cid), ghost)


class TestRecording(ServiceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cid, _ = self.service.ensure_conversation(None, title_hint="chat")

    def test_full_run_flow(self) -> None:
        """The exact sequence the server performs for one run."""
        stored = self.service.record_user_message(
            self.cid, "Hello", run_id="run-1", context="ctx", desired_output="markdown"
        )
        self.assertTrue(stored.ok)
        session = self.service.begin_run(self.cid, run_id="run-1")
        self.assertIsNotNone(session)
        answer = self.service.record_assistant_message(
            self.cid, "Hi! I am the orchestrator.", run_id="run-1"
        )
        self.assertTrue(answer.ok)
        self.service.finish_run(session)
        self.assertEqual(self.service.manager.registry.count(), 0)

        history = self.service.get_history(str(self.cid)).unwrap()
        self.assertEqual([m["role"] for m in history["messages"]], ["user", "assistant"])
        self.assertEqual(history["messages"][0]["content"], "Hello")
        self.assertEqual(history["message_count"], 2)
        # Prompt context/desired-output ride along as metadata:
        first = self.service.manager.read_messages(self.cid).unwrap()[0]
        self.assertEqual(first.metadata.values["workflow.context"], "ctx")
        self.assertEqual(first.metadata.values["workflow.desired_output"], "markdown")

    def test_failed_run_records_marker(self) -> None:
        result = self.service.record_assistant_message(
            self.cid, "", run_id="run-2", error="provider timeout", cancelled=True
        )
        self.assertTrue(result.ok)
        message = self.service.manager.read_messages(self.cid).unwrap()[-1]
        self.assertIn("no answer produced", message.content)
        self.assertEqual(message.metadata.values["workflow.error"], "provider timeout")
        self.assertTrue(message.metadata.values["workflow.cancelled"])

    def test_get_history_rejects_bad_ids(self) -> None:
        bad = self.service.get_history("../../etc/passwd")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.error.code, ErrorCode.VALIDATION_FAILED)
        ghost = self.service.get_history("01AAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertFalse(ghost.ok)
        self.assertEqual(ghost.error.code, ErrorCode.NOT_FOUND)


class TestRestartPersistence(ServiceTestCase):
    def test_history_survives_service_restart(self) -> None:
        cid, _ = self.service.ensure_conversation(None, title_hint="persistent chat")
        self.service.record_user_message(cid, "Hello")
        self.service.record_assistant_message(cid, "Hi there!")
        self.service.close()

        reborn = ConversationService(root_path=self.root, fsync=False)  # "restart"
        same, created = reborn.ensure_conversation(str(cid), title_hint="ignored")
        self.assertEqual(same, cid)
        self.assertFalse(created)
        history = reborn.get_history(str(cid)).unwrap()
        self.assertEqual(
            [(m["role"], m["content"]) for m in history["messages"]],
            [("user", "Hello"), ("assistant", "Hi there!")],
        )

    def test_multi_turn_accumulates_across_restarts(self) -> None:
        """Acceptance test 2/3 as code: two runs, a restart between them."""
        cid, _ = self.service.ensure_conversation(None, title_hint="Hello")
        self.service.record_user_message(cid, "Hello")
        self.service.record_assistant_message(cid, "Answer one")
        self.service.close()

        second = ConversationService(root_path=self.root, fsync=False)
        cid2, created = second.ensure_conversation(str(cid), title_hint="How are you?")
        self.assertFalse(created)
        second.record_user_message(cid2, "How are you?")
        second.record_assistant_message(cid2, "Answer two")
        history = second.get_history(str(cid)).unwrap()
        self.assertEqual(
            [m["content"] for m in history["messages"]],
            ["Hello", "Answer one", "How are you?", "Answer two"],
        )


class TestGracefulDegradation(ServiceTestCase):
    def test_broken_storage_root_never_raises(self) -> None:
        # A FILE where the root should be: every disk op will fail.
        blocked = self.tmp / "blocked"
        blocked.write_text("i am a file", encoding="utf-8")
        broken = ConversationService(root_path=str(blocked / "conversations"), fsync=False)
        cid, created = broken.ensure_conversation(None, title_hint="doomed")
        self.assertTrue(created)          # ephemeral id so the run can proceed
        self.assertTrue(is_ulid(cid))
        stored = broken.record_user_message(cid, "will not persist")
        self.assertFalse(stored.ok)       # failure reported, not raised
        self.assertIsNone(broken.begin_run(cid, run_id="r"))
        broken.finish_run(None)           # no-op, no exception


class TestConcurrentAccess(ManagerTestCase):
    def test_parallel_appends_through_manager(self) -> None:
        cid = self.create().conversation.id
        errors: list[str] = []

        def worker(w: int) -> None:
            for i in range(15):
                result = self.manager.append_user_message(cid, f"w{w}-m{i}")
                if not result.ok:
                    errors.append(result.error.message)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        stored = self.manager.read_messages(cid, limit=1000).unwrap()
        self.assertEqual(len(stored), 90)
        ids = [m.id for m in stored]
        self.assertEqual(ids, sorted(ids))          # total order preserved
        self.assertEqual(len({m.content for m in stored}), 90)  # nothing lost

    def test_rename_races_appends(self) -> None:
        cid = self.create("start").conversation.id
        failures: list[str] = []

        def renamer() -> None:
            for i in range(10):
                result = self.manager.rename_conversation(cid, f"title-{i}")
                if not result.ok:
                    failures.append(result.error.message)

        def appender() -> None:
            for i in range(10):
                result = self.manager.append_user_message(cid, f"msg-{i}")
                if not result.ok:
                    failures.append(result.error.message)

        threads = [threading.Thread(target=renamer), threading.Thread(target=appender)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(failures, [])
        final = self.new_manager().get_conversation(cid).unwrap()
        self.assertEqual(final.conversation.title, "title-9")
        self.assertEqual(final.conversation.stats.message_count, 10)


if __name__ == "__main__":
    unittest.main()
