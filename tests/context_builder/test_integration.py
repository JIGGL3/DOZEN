"""Phase 1.4 integration: prepare_run builds context, compose_task_context
frames it, the worker prompt carries it. Includes the manual acceptance tests
as code.

NOTE: DOZEN's /api/run path is now ONE-SHOT — ``prepare_run`` defaults to
``inject_history=False`` and never feeds prior turns back into a prompt. The
acceptance tests below therefore pass ``inject_history=True`` explicitly: they
verify the context BUILDER still works (recall, restart persistence, per-
conversation isolation, budget trimming), not the default run behaviour.
``TestOneShotDefault`` covers the default.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from dozen.models import SubTask, Task
from dozen.prompts import build_planner_messages, build_worker_messages
from webllm import conversations as convmod
from webllm.conversations import (
    compose_task_context,
    finish_run,
    get_history,
    prepare_run,
    reset_service,
)


class IntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-ctx14-"))
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


class TestPrepareRunContext(IntegrationTestCase):
    def test_first_run_has_no_history(self) -> None:
        handle = prepare_run(prompt="My name is John.", conversation_id=None, run_id="r1")
        self.assertEqual(handle.formatted_history, "")
        self.assertEqual(handle.context_messages, 0)

    def test_acceptance_1_name_recall(self) -> None:
        """'My name is John.' then 'What is my name?' — the second run's task
        context must contain the first exchange."""
        first = prepare_run(prompt="My name is John.", conversation_id=None, run_id="r1")
        finish_run(first, final_answer="Nice to meet you, John!")

        second = prepare_run(
            prompt="What is my name?", conversation_id=first.conversation_id,
            run_id="r2", inject_history=True,
        )
        self.assertEqual(second.context_messages, 2)
        self.assertEqual(
            second.formatted_history,
            "User: My name is John.\n\nAssistant: Nice to meet you, John!",
        )
        # The current prompt must NOT appear in its own history:
        self.assertNotIn("What is my name?", second.formatted_history)

        # And the exact Task the server would build carries it everywhere:
        task = Task(prompt="What is my name?",
                    context=compose_task_context(second.formatted_history))
        brief = task.to_brief()
        self.assertIn("My name is John.", brief)                    # planner + synthesizer
        planner_user = build_planner_messages(task, 0, 2, "gpt (general)")[1].content
        self.assertIn("My name is John.", planner_user)
        worker_user = build_worker_messages(
            task, SubTask(title="answer", instruction="Answer the user's question."), {}
        )[1].content
        self.assertIn("My name is John.", worker_user)              # workers too

    def test_acceptance_2_restart_keeps_context(self) -> None:
        first = prepare_run(prompt="My name is John.", conversation_id=None, run_id="r1")
        finish_run(first, final_answer="Hello John!")
        reset_service()  # server restart

        second = prepare_run(
            prompt="What is my name?", conversation_id=first.conversation_id,
            run_id="r2", inject_history=True,
        )
        self.assertIn("User: My name is John.", second.formatted_history)

    def test_acceptance_3_histories_never_mix(self) -> None:
        a = prepare_run(prompt="Secret alpha", conversation_id=None, run_id="a1")
        finish_run(a, final_answer="noted alpha")
        b = prepare_run(prompt="Secret beta", conversation_id=None, run_id="b1")
        finish_run(b, final_answer="noted beta")

        a2 = prepare_run(prompt="continue", conversation_id=a.conversation_id,
                         run_id="a2", inject_history=True)
        b2 = prepare_run(prompt="continue", conversation_id=b.conversation_id,
                         run_id="b2", inject_history=True)
        self.assertIn("Secret alpha", a2.formatted_history)
        self.assertNotIn("Secret beta", a2.formatted_history)
        self.assertIn("Secret beta", b2.formatted_history)
        self.assertNotIn("Secret alpha", b2.formatted_history)

    def test_acceptance_4_budget_drops_oldest_first(self) -> None:
        first = prepare_run(prompt="oldest message marker", conversation_id=None, run_id="r0")
        finish_run(first, final_answer="oldest answer marker")
        cid = first.conversation_id
        for i in range(30):
            h = prepare_run(prompt=f"filler question {i} " + "x" * 400,
                            conversation_id=cid, run_id=f"r{i+1}")
            finish_run(h, final_answer=f"filler answer {i} " + "y" * 400)

        os.environ[convmod._ENV_BUDGET] = "2000"
        try:
            reset_service()  # rebuild the builder with the small budget
            probe = prepare_run(prompt="latest", conversation_id=cid,
                                run_id="probe", inject_history=True)
        finally:
            os.environ.pop(convmod._ENV_BUDGET, None)
            reset_service()
        self.assertTrue(probe.context_truncated)
        self.assertNotIn("oldest message marker", probe.formatted_history)  # oldest dropped
        self.assertIn("filler answer 29", probe.formatted_history)          # newest kept
        self.assertLessEqual(probe.context_tokens, 2000)

    def test_stale_id_runs_stateless_not_failing(self) -> None:
        handle = prepare_run(prompt="hello", conversation_id="broken-id!!", run_id="r1")
        self.assertTrue(handle.created)            # new conversation
        self.assertEqual(handle.formatted_history, "")  # no phantom history
        self.assertTrue(handle.recording)


class TestOneShotDefault(IntegrationTestCase):
    """Every run is independent: history is recorded, never fed back.

    Injecting prior turns let each prompt accumulate every earlier exchange —
    including long runs of failure notices — until the planner's prompt was
    mostly transcript and models answered the noise instead of the request.
    """

    def test_history_exists_but_is_not_injected_by_default(self) -> None:
        first = prepare_run(prompt="My name is John.", conversation_id=None, run_id="r1")
        finish_run(first, final_answer="Nice to meet you, John!")

        second = prepare_run(
            prompt="What is my name?", conversation_id=first.conversation_id, run_id="r2"
        )
        # Same conversation, and the earlier turn IS on disk...
        self.assertEqual(second.conversation_id, first.conversation_id)
        stored, err = get_history(first.conversation_id)
        self.assertIsNone(err)
        self.assertGreaterEqual(len(stored["messages"]), 2)
        # ...but none of it reaches the prompt.
        self.assertEqual(second.formatted_history, "")
        self.assertEqual(second.context_messages, 0)
        self.assertEqual(second.context_tokens, 0)
        self.assertFalse(second.context_truncated)
        self.assertIsNone(second.context_fallback_reason)

    def test_task_context_carries_only_the_callers_own_context(self) -> None:
        first = prepare_run(prompt="Secret alpha", conversation_id=None, run_id="a1")
        finish_run(first, final_answer="noted alpha")
        second = prepare_run(
            prompt="continue", conversation_id=first.conversation_id, run_id="a2"
        )

        # Exactly what the server builds for the orchestrator.
        task = Task(prompt="continue",
                    context=compose_task_context(second.formatted_history, "release notes v2"))
        self.assertNotIn("Secret alpha", task.context)
        self.assertNotIn("PREVIOUS CONVERSATION", task.context)
        self.assertEqual(task.context, "release notes v2")
        # And nothing leaks through the planner prompt either.
        planner_user = build_planner_messages(task, 0, 2, "gpt (general)")[1].content
        self.assertNotIn("Secret alpha", planner_user)
        self.assertNotIn("PREVIOUS CONVERSATION", planner_user)

    def test_a_long_failure_history_never_reaches_the_planner(self) -> None:
        """The reported failure mode, as a test."""
        cid = None
        for i in range(20):
            h = prepare_run(prompt="hey", conversation_id=cid, run_id=f"r{i}")
            cid = h.conversation_id
            finish_run(h, final_answer="", error="Planning failed: provider is quarantined")

        latest = prepare_run(prompt="hey", conversation_id=cid, run_id="final")
        self.assertEqual(latest.formatted_history, "")
        task = Task(prompt="hey", context=compose_task_context(latest.formatted_history))
        planner_user = build_planner_messages(task, 0, 2, "gpt (general)")[1].content
        self.assertNotIn("quarantined", planner_user)
        self.assertNotIn("no answer produced", planner_user)


class TestComposeTaskContext(unittest.TestCase):
    def test_empty_everything(self) -> None:
        self.assertEqual(compose_task_context("", ""), "")

    def test_history_only_is_framed(self) -> None:
        composed = compose_task_context("User: hi\n\nAssistant: hello")
        self.assertTrue(composed.startswith("PREVIOUS CONVERSATION"))
        self.assertIn("User: hi\n\nAssistant: hello", composed)

    def test_user_context_only_passes_through(self) -> None:
        self.assertEqual(compose_task_context("", "release notes v2"), "release notes v2")

    def test_history_precedes_user_context(self) -> None:
        composed = compose_task_context("User: hi", "extra docs")
        self.assertLess(composed.index("User: hi"), composed.index("extra docs"))

    def test_empty_context_keeps_pre_14_worker_prompt(self) -> None:
        """Backwards compatibility: no context -> byte-identical worker prompt
        to Phase 1.3 (no BACKGROUND CONTEXT block)."""
        task = Task(prompt="do the thing", context="")
        user = build_worker_messages(task, SubTask(title="t", instruction="i"), {})[1].content
        self.assertNotIn("BACKGROUND CONTEXT", user)


if __name__ == "__main__":
    unittest.main()
