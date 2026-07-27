"""Server/workflow integration: the exact code path ``/api/run`` executes,
driven through ``webllm.conversations`` (no browser, no live orchestrator —
the workflow step is simulated by calling finish_run with a result, which is
precisely what the run worker thread does)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from webllm import conversations as convmod
from webllm.conversations import finish_run, get_history, prepare_run, reset_service


class ServerIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-srvconv-"))
        self._old_env = os.environ.get(convmod._ENV_ROOT)
        os.environ[convmod._ENV_ROOT] = str(self.tmp / "conversations")
        reset_service()

    def tearDown(self) -> None:
        reset_service()
        if self._old_env is None:
            os.environ.pop(convmod._ENV_ROOT, None)
        else:
            os.environ[convmod._ENV_ROOT] = self._old_env
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestRunFlow(ServerIntegrationTestCase):
    def test_first_run_creates_conversation_and_records_both_sides(self) -> None:
        handle = prepare_run(prompt="Hello", conversation_id=None, run_id="run-1")
        self.assertTrue(handle.created)
        self.assertTrue(handle.recording)
        self.assertIsNotNone(handle.session)

        # ... the orchestrator runs here (unchanged) ...
        finish_run(handle, final_answer="Hi! I'm DOZEN.")

        payload, error = get_history(handle.conversation_id)
        self.assertIsNone(error)
        self.assertEqual(
            [(m["role"], m["content"]) for m in payload["messages"]],
            [("user", "Hello"), ("assistant", "Hi! I'm DOZEN.")],
        )
        self.assertEqual(payload["title"], "Hello")

    def test_second_run_reuses_conversation(self) -> None:
        """Acceptance test 2: two prompts -> four ordered messages."""
        first = prepare_run(prompt="Hello", conversation_id=None, run_id="run-1")
        finish_run(first, final_answer="Answer one")
        second = prepare_run(
            prompt="How are you?", conversation_id=first.conversation_id, run_id="run-2"
        )
        self.assertFalse(second.created)
        self.assertEqual(second.conversation_id, first.conversation_id)
        finish_run(second, final_answer="Answer two")

        payload, _ = get_history(first.conversation_id)
        self.assertEqual(
            [m["content"] for m in payload["messages"]],
            ["Hello", "Answer one", "How are you?", "Answer two"],
        )

    def test_history_survives_server_restart(self) -> None:
        """Acceptance test 3: reset_service() == process restart."""
        handle = prepare_run(prompt="Hello", conversation_id=None, run_id="run-1")
        finish_run(handle, final_answer="Answer")
        reset_service()  # everything in memory is gone; disk remains

        payload, error = get_history(handle.conversation_id)
        self.assertIsNone(error)
        self.assertEqual(len(payload["messages"]), 2)
        again = prepare_run(
            prompt="still here?", conversation_id=handle.conversation_id, run_id="run-2"
        )
        self.assertFalse(again.created)  # same conversation picked right up
        finish_run(again, final_answer="yes")

    def test_stale_id_starts_fresh_without_failing(self) -> None:
        handle = prepare_run(
            prompt="Hello", conversation_id="completely-bogus-id!!", run_id="run-1"
        )
        self.assertTrue(handle.created)
        self.assertTrue(handle.recording)
        finish_run(handle, final_answer="fine")
        payload, _ = get_history(handle.conversation_id)
        self.assertEqual(len(payload["messages"]), 2)

    def test_failed_and_cancelled_runs_still_close_cleanly(self) -> None:
        handle = prepare_run(prompt="doomed run", conversation_id=None, run_id="run-1")
        finish_run(handle, final_answer="", error="executor exploded", cancelled=True)
        payload, _ = get_history(handle.conversation_id)
        self.assertEqual(payload["messages"][0]["content"], "doomed run")
        self.assertIn("no answer produced", payload["messages"][1]["content"])
        # Session closed despite the failure:
        self.assertEqual(convmod.get_service().manager.registry.count(), 0)

    def test_get_history_errors_are_tuples_not_exceptions(self) -> None:
        payload, error = get_history("01AAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertIsNone(payload)
        self.assertIn("not_found", error)
        payload, error = get_history("garbage")
        self.assertIsNone(payload)
        self.assertIn("validation_failed", error)


class TestServerRequestModel(unittest.TestCase):
    def test_run_request_backwards_compatible(self) -> None:
        """Old clients that send no conversation_id must keep working."""
        from webllm.server import RunRequest

        old_style = RunRequest(prompt="hi")
        self.assertEqual(old_style.conversation_id, "")
        new_style = RunRequest(prompt="hi", conversation_id="01ABC")
        self.assertEqual(new_style.conversation_id, "01ABC")


if __name__ == "__main__":
    unittest.main()
