"""Queue-wait expiry tests: bounded deadline, worker-claim race, cause
preservation, immediate capacity release, provider operation never invoked."""

from __future__ import annotations

import time
import unittest

from webllm.browser_jobs import JobState, TerminalCause
from webllm.browser_queue import QueuePolicy

from ._util import QueueTestManager


def _poll(fn, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.005)
    return False


class TestQueueWaitExpiry(unittest.TestCase):
    def test_worker_claim_of_expired_prompt_never_sends(self) -> None:
        bm = QueueTestManager(queue_policy=QueuePolicy(max_queue_wait_s=0.05))
        try:
            blocker = bm.install_blocker("openai")
            h = bm.submit_prompt("openai", "expired-follower")
            time.sleep(0.15)  # outwait the 0.05s queue deadline
            blocker.release.set()  # worker moves to the follower
            self.assertTrue(_poll(lambda: h.state() in (JobState.ABANDONED,)))
            # The expired prompt was NEVER submitted to the provider.
            self.assertNotIn("expired-follower", bm.prompts_sent())
            # Compatible timeout behavior and the TIMED_OUT cause are preserved.
            self.assertTrue(h.timed_out())
            with self.assertRaises(TimeoutError):
                h.result(timeout=1)
        finally:
            bm.shutdown()

    def test_admission_sweep_frees_capacity_immediately(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=1, max_queue_wait_s=0.05
            )
        )
        try:
            bm.install_blocker("openai")
            first = bm.submit_prompt("openai", "first")  # fills the single slot
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 1)
            time.sleep(0.15)  # first's queue wait expires
            # A new submission sweeps the expired entry and takes the freed slot
            # instead of being rejected as provider-full.
            second = bm.submit_prompt("openai", "second")
            self.assertEqual(second.state(), JobState.QUEUED)
            self.assertTrue(_poll(lambda: first.timed_out()))
            self.assertNotIn("first", bm.prompts_sent())
        finally:
            bm.shutdown()

    def test_worker_claim_wins_when_not_expired(self) -> None:
        # A long deadline: the worker claims and runs it (worker wins the race).
        bm = QueueTestManager(queue_policy=QueuePolicy(max_queue_wait_s=120.0))
        try:
            blocker = bm.install_blocker("openai")
            h = bm.submit_prompt("openai", "fresh")
            blocker.release.set()
            self.assertEqual(h.result(timeout=5), "reply:fresh")
            self.assertIs(h.state(), JobState.COMPLETED)
            self.assertIn("fresh", bm.prompts_sent())
        finally:
            bm.shutdown()

    def test_expiry_preserves_original_cause_over_late_settlement(self) -> None:
        bm = QueueTestManager(queue_policy=QueuePolicy(max_queue_wait_s=0.05))
        try:
            blocker = bm.install_blocker("openai")
            h = bm.submit_prompt("openai", "cause-check")
            time.sleep(0.15)
            blocker.release.set()
            self.assertTrue(_poll(lambda: h.snapshot().physical_settled))
            self.assertEqual(
                h.snapshot().terminal_cause, TerminalCause.TIMED_OUT.value
            )
        finally:
            bm.shutdown()


if __name__ == "__main__":
    unittest.main()
