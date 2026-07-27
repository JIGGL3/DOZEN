"""Worker-queue integration — the ten required race scenarios, driven through
real per-provider worker threads with a browser-free ``_send_prompt`` seam.
"""

from __future__ import annotations

import time
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_jobs import JobState
from webllm.providers import ProviderError

from ._util import ControllableBrowserManager, Gate


def _poll_state(mgr, job_id, want: JobState, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mgr._registry.state(job_id) is want:
            return True
        time.sleep(0.005)
    return False


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    # --- Scenario 1 — normal completion --------------------------------- #
    def test_s1_normal_completion(self) -> None:
        h = self.bm.submit_prompt("openai", "q1")
        self.assertEqual(h.result(timeout=5), "reply:q1")
        self.assertIs(h.state(), JobState.COMPLETED)
        self.assertEqual(self.bm.prompts_sent().count("q1"), 1)  # sent once

    # --- Scenario 2 — queued timeout, never submitted ------------------- #
    def test_s2_queued_timeout_never_submitted(self) -> None:
        blocker = Gate(value="a")
        self.bm.gates["blocker"] = blocker
        h_block = self.bm.submit_prompt("openai", "blocker")
        self.assertTrue(blocker.started.wait(2))  # worker busy on the blocker

        follow = Gate(value="b")
        self.bm.gates["follow"] = follow
        h2 = self.bm.submit_prompt("openai", "follow")  # queues behind blocker
        # The queued follower's caller times out before it ever starts.
        self.assertTrue(h2.mark_caller_timeout())
        self.assertIs(h2.state(), JobState.ABANDONED)
        self.assertTrue(h2.snapshot().physical_settled)
        self.assertTrue(h2.timed_out())

        blocker.release.set()                      # worker moves to the follower
        self.assertTrue(_poll_state(self.bm, h2.job_id, JobState.ABANDONED))
        self.assertFalse(follow.started.is_set())  # provider prompt NEVER sent
        self.assertNotIn("follow", self.bm.prompts_sent())
        h_block.result(timeout=5)

    # --- Scenario 3 — running timeout + late completion discarded ------- #
    def test_s3_running_timeout_late_completion_discarded(self) -> None:
        gate = Gate(value="late")
        self.bm.gates["slow"] = gate
        h = self.bm.submit_prompt("openai", "slow")
        self.assertTrue(gate.started.wait(2))
        with self.assertRaises(TimeoutError):
            h.result(timeout=0.2)
        self.assertIs(h.state(), JobState.TIMED_OUT)
        gate.release.set()
        self.assertTrue(_poll_state(self.bm, h.job_id, JobState.ABANDONED))
        snap = h.snapshot()
        self.assertTrue(snap.late_result_discarded)
        self.assertFalse(snap.result_delivered)

    # --- Scenario 4 — completion races timeout (worker level) ----------- #
    def test_s4_completion_wins_when_not_timed_out(self) -> None:
        # Completion path (no caller timeout) always yields COMPLETED + delivery.
        for i in range(25):
            h = self.bm.submit_prompt("openai", f"c{i}")
            self.assertEqual(h.result(timeout=5), f"reply:c{i}")
            self.assertIs(h.state(), JobState.COMPLETED)

    # --- Scenario 5 — cancellation while queued ------------------------- #
    def test_s5_cancel_while_queued_skips_send(self) -> None:
        gate = Gate(value="x")
        self.bm.gates["cancel-me"] = gate
        # should_cancel already true: worker must skip submission entirely.
        h = self.bm.submit_prompt("openai", "cancel-me", should_cancel=lambda: True)
        self.assertTrue(_poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertTrue(h.snapshot().physical_settled)
        self.assertTrue(h.cancelled())
        self.assertFalse(gate.started.is_set())
        self.assertNotIn("cancel-me", self.bm.prompts_sent())
        with self.assertRaises(CancelledError):
            h.result(timeout=1)

    # --- Scenario 6 — cancellation while running ------------------------ #
    def test_s6_cancel_while_running_discards_late_response(self) -> None:
        # The worker observes cancellation via a raised CancelledError.
        self.bm.handlers["run-cancel"] = lambda p, pr, sc: (_ for _ in ()).throw(
            CancelledError("stop")
        )
        h = self.bm.submit_prompt("openai", "run-cancel")
        with self.assertRaises(CancelledError):
            h.result(timeout=5)
        self.assertIs(h.state(), JobState.ABANDONED)
        self.assertTrue(h.cancelled())

    # --- Scenario 7 — stale release cannot clear a successor ------------ #
    def test_s7_stale_release(self) -> None:
        first = Gate(value="1")
        self.bm.gates["first"] = first
        h1 = self.bm.submit_prompt("openai", "first")
        self.assertTrue(first.started.wait(2))
        # h1's caller times out while it runs.
        with self.assertRaises(TimeoutError):
            h1.result(timeout=0.2)
        first.release.set()
        self.assertTrue(_poll_state(self.bm, h1.job_id, JobState.ABANDONED))
        # A successor now runs and owns the provider slot.
        h2 = self.bm.submit_prompt("openai", "second")
        self.assertEqual(h2.result(timeout=5), "reply:second")
        # h1's late settlement never disturbed h2's ownership/state.
        self.assertIs(h2.state(), JobState.COMPLETED)

    # --- Scenario 8 — provider failure ---------------------------------- #
    def test_s8_provider_failure(self) -> None:
        self.bm.handlers["fail"] = lambda p, pr, sc: (_ for _ in ()).throw(
            ProviderError("upstream 500")
        )
        h = self.bm.submit_prompt("openai", "fail")
        with self.assertRaises(ProviderError):
            h.result(timeout=5)
        self.assertIs(h.state(), JobState.FAILED)
        # Failure summarized once, no duplicate transition.
        self.assertEqual(h.snapshot().failure_category, "ProviderError")

    # --- Scenario 9 — manager shutdown --------------------------------- #
    def test_s9_shutdown_releases_queued_waiters(self) -> None:
        blocker = Gate(value="a")
        self.bm.gates["hold"] = blocker
        h_block = self.bm.submit_prompt("openai", "hold")
        self.assertTrue(blocker.started.wait(2))
        queued = [self.bm.submit_prompt("openai", f"queued{i}") for i in range(3)]
        blocker.release.set()
        self.bm.shutdown()
        # Every queued waiter is released (never blocks) with a compatible error.
        for h in queued:
            with self.assertRaises((RuntimeError, TimeoutError)):
                h.result(timeout=5)
            self.assertIn(
                h.state(), (JobState.ABANDONED, JobState.COMPLETED)
            )

    # --- Scenario 10 — multi-provider independence ---------------------- #
    def test_s10_multi_provider_independence(self) -> None:
        slow = Gate(value="slow")
        self.bm.gates["A-slow"] = slow
        ha = self.bm.submit_prompt("openai", "A-slow")
        self.assertTrue(slow.started.wait(2))
        with self.assertRaises(TimeoutError):
            ha.result(timeout=0.2)                 # provider A timed out, running
        self.assertIs(ha.state(), JobState.TIMED_OUT)

        # Provider B completes normally, wholly independent of A.
        hb = self.bm.submit_prompt("google", "B-fast")
        self.assertEqual(hb.result(timeout=5), "reply:B-fast")
        self.assertIs(hb.state(), JobState.COMPLETED)
        self.assertIsNone(self.bm.active_browser_job("google"))
        # A retains physical ownership until the blocked browser call unwinds.
        self.assertEqual(self.bm.active_browser_job("openai"), ha.job_id)
        self.assertIs(ha.state(), JobState.TIMED_OUT)
        slow.release.set()
        self.assertTrue(_poll_state(self.bm, ha.job_id, JobState.ABANDONED))


class TestWorkerClaim(unittest.TestCase):
    """Focused claim / cleanup behaviors."""

    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_queued_to_running_claim_then_complete(self) -> None:
        h = self.bm.submit_prompt("openai", "q")
        h.result(timeout=5)
        snap = h.snapshot()
        self.assertIsNotNone(snap.started_at)
        self.assertIsNotNone(snap.finished_at)
        self.assertGreaterEqual(snap.finished_at, snap.started_at)

    def test_ownership_released_after_completion(self) -> None:
        h = self.bm.submit_prompt("openai", "q")
        h.result(timeout=5)
        self.assertIsNone(self.bm.active_browser_job("openai"))

    def test_exception_safe_cleanup_releases_ownership(self) -> None:
        self.bm.handlers["boom"] = lambda p, pr, sc: (_ for _ in ()).throw(
            ProviderError("x")
        )
        h = self.bm.submit_prompt("openai", "boom")
        with self.assertRaises(ProviderError):
            h.result(timeout=5)
        self.assertIsNone(self.bm.active_browser_job("openai"))


if __name__ == "__main__":
    unittest.main()
