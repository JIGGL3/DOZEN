"""The fifteen required Phase 5C scenarios, driven through the real per-provider
worker threads with a browser-free send seam."""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_jobs import JobState
from webllm.browser_queue import (
    AdmissionOutcome,
    BrowserQueueRejectedError,
    QueuePolicy,
)

from ._util import QueueTestManager


def _poll(fn, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.005)
    return False


class TestScenarios(unittest.TestCase):
    def _mgr(self, **policy) -> QueueTestManager:
        bm = QueueTestManager(queue_policy=QueuePolicy(**policy))
        self.addCleanup(bm.shutdown)
        return bm

    # Scenario 1 — capacity boundary
    def test_s1_capacity_boundary(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=8, max_total_queued_prompts=64)
        bm.install_blocker("openai")
        for i in range(8):
            bm.submit_prompt("openai", f"q{i}")
        with self.assertRaises(BrowserQueueRejectedError) as ctx:
            bm.submit_prompt("openai", "ninth")
        self.assertIs(ctx.exception.outcome, AdmissionOutcome.REJECTED_PROVIDER_FULL)

    # Scenario 2 — concurrent admission
    def test_s2_concurrent_admission(self) -> None:
        cap = 5
        bm = self._mgr(max_queued_prompts_per_provider=cap, max_total_queued_prompts=64)
        bm.install_blocker("openai")
        accepted, rejected = [], []
        lock = threading.Lock()
        barrier = threading.Barrier(30)

        def submit(i: int) -> None:
            barrier.wait()
            try:
                bm.submit_prompt("openai", f"c{i}")
                with lock:
                    accepted.append(i)
            except BrowserQueueRejectedError:
                with lock:
                    rejected.append(i)

        ts = [threading.Thread(target=submit, args=(i,)) for i in range(30)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=5)
        self.assertEqual(len(accepted), cap)
        self.assertEqual(len(rejected), 30 - cap)

    # Scenario 3 — cancellation frees capacity immediately
    def test_s3_cancellation_frees_capacity(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=1)
        bm.install_blocker("openai")
        h1 = bm.submit_prompt("openai", "first")
        self.assertTrue(h1.request_cancel())  # queued cancel removes the entry
        # The slot is free WITHOUT waiting for the worker to dequeue h1.
        h2 = bm.submit_prompt("openai", "second")
        self.assertEqual(h2.state(), JobState.QUEUED)
        self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 1)

    # Scenario 4 — queued timeout frees capacity
    def test_s4_queued_timeout_frees_capacity(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=1)
        bm.install_blocker("openai")
        h1 = bm.submit_prompt("openai", "first")
        self.assertTrue(h1.mark_caller_timeout())
        h2 = bm.submit_prompt("openai", "second")
        self.assertEqual(h2.state(), JobState.QUEUED)
        self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 1)

    # Scenario 5 — running job does not consume queued capacity
    def test_s5_running_excluded_from_queue_capacity(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=3)
        bm.install_blocker("openai")  # RUNNING
        handles = [bm.submit_prompt("openai", f"q{i}") for i in range(3)]  # all queued
        self.assertEqual(len(handles), 3)
        snap = bm.queue_snapshot("openai")
        self.assertEqual(snap.prompt_depth, 3)
        self.assertIsNotNone(snap.running_job_id)

    # Scenario 6 — provider isolation
    def test_s6_provider_isolation(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=2, max_total_queued_prompts=32)
        bm.install_blocker("openai")
        bm.submit_prompt("openai", "o0")
        bm.submit_prompt("openai", "o1")
        with self.assertRaises(BrowserQueueRejectedError):
            bm.submit_prompt("openai", "o2")
        self.assertEqual(bm.submit_prompt("google", "g0").result(timeout=5), "reply:g0")

    # Scenario 7 — global capacity
    def test_s7_global_capacity(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=4, max_total_queued_prompts=6)
        bm.install_blocker("openai")
        bm.install_blocker("google")
        for i in range(4):
            bm.submit_prompt("openai", f"o{i}")
        for i in range(2):
            bm.submit_prompt("google", f"g{i}")
        with self.assertRaises(BrowserQueueRejectedError) as ctx:
            bm.submit_prompt("google", "overflow")
        self.assertIs(ctx.exception.outcome, AdmissionOutcome.REJECTED_GLOBAL_FULL)

    # Scenario 8 — quarantined provider
    def test_s8_quarantined_provider(self) -> None:
        bm = self._mgr()
        bm._quarantine_provider("openai", reason="test")
        with self.assertRaises(BrowserQueueRejectedError) as ctx:
            bm.submit_prompt("openai", "blocked")
        self.assertIs(
            ctx.exception.outcome, AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED
        )
        self.assertNotIn("blocked", bm.prompts_sent())

    # Scenario 9 — FIFO
    def test_s9_fifo(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=8)
        blocker = bm.install_blocker("openai")
        handles = [bm.submit_prompt("openai", f"q{i}") for i in range(5)]
        blocker.release.set()
        for h in handles:
            h.result(timeout=5)
        # The provider executed the blocker first, then q0..q4 in order.
        order = [p for p in bm.prompts_sent() if p.startswith("q")]
        self.assertEqual(order, [f"q{i}" for i in range(5)])

    # Scenario 10 — removed middle job preserves order
    def test_s10_removed_middle_preserves_order(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=8)
        blocker = bm.install_blocker("openai")
        handles = [bm.submit_prompt("openai", f"q{i}") for i in range(5)]
        self.assertTrue(handles[2].request_cancel())  # cancel the middle one
        blocker.release.set()
        for i, h in enumerate(handles):
            if i == 2:
                continue
            h.result(timeout=5)
        order = [p for p in bm.prompts_sent() if p.startswith("q")]
        self.assertEqual(order, ["q0", "q1", "q3", "q4"])  # q2 never sent, order kept

    # Scenario 11 — shutdown race
    def test_s11_shutdown_race(self) -> None:
        for _ in range(15):
            bm = QueueTestManager()
            barrier = threading.Barrier(2)
            handles: list = []

            def submit() -> None:
                barrier.wait()
                handles.append(bm.submit_prompt("openai", "race"))

            def shut() -> None:
                barrier.wait()
                bm.shutdown()

            ts = [threading.Thread(target=submit), threading.Thread(target=shut)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=3)
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0]._job.done.wait(1))
            try:
                handles[0].result(timeout=1)
            except (RuntimeError, TimeoutError):
                pass

    # Scenario 12 — control-job bounds
    def test_s12_control_bounds(self) -> None:
        bm = self._mgr(max_queued_control_jobs_per_provider=3)
        bm.install_blocker("openai")
        for _ in range(3):
            bm._enqueue("openai", lambda: "ok")
        with self.assertRaises(BrowserQueueRejectedError):
            bm._enqueue("openai", lambda: "ok")

    # Scenario 13 — queue expiry versus worker claim (exactly one wins)
    def test_s13_expiry_vs_claim(self) -> None:
        # Expired: never sent.
        bm = self._mgr(max_queue_wait_s=0.05)
        blocker = bm.install_blocker("openai")
        expired = bm.submit_prompt("openai", "expired")
        time.sleep(0.15)
        blocker.release.set()
        self.assertTrue(_poll(lambda: expired.timed_out()))
        self.assertNotIn("expired", bm.prompts_sent())

    def test_s13_claim_wins_when_fresh(self) -> None:
        bm = self._mgr(max_queue_wait_s=120.0)
        blocker = bm.install_blocker("openai")
        fresh = bm.submit_prompt("openai", "fresh")
        blocker.release.set()
        self.assertEqual(fresh.result(timeout=5), "reply:fresh")
        self.assertIn("fresh", bm.prompts_sent())

    # Scenario 14 — queue events accurate, no content
    def test_s14_events_accurate_no_content(self) -> None:
        events = []
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_prompts_per_provider=1),
            queue_event_sink=events.append,
        )
        self.addCleanup(bm.shutdown)
        bm.install_blocker("openai")
        bm.submit_prompt("openai", "secretA")
        try:
            bm.submit_prompt("openai", "secretB")
        except BrowserQueueRejectedError:
            pass
        kinds = {e.kind for e in events}
        self.assertIn("admission_accepted", kinds)
        self.assertIn("admission_rejected", kinds)
        for e in events:
            for value in e.to_dict().values():
                self.assertNotIn("secret", str(value))

    # Scenario 15 — long blocked provider, no unbounded tombstones
    def test_s15_repeated_cancellation_no_tombstones(self) -> None:
        bm = self._mgr(max_queued_prompts_per_provider=2)
        bm.install_blocker("openai")  # worker stays blocked the whole time
        for _ in range(300):
            h = bm.submit_prompt("openai", "churn")
            self.assertTrue(h.request_cancel())
            # Depth returns to zero every round: no physical residue accumulates.
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 0)
        worker = bm._peek_worker("openai")
        self.assertEqual(len(worker.jobs._entries), 0)


if __name__ == "__main__":
    unittest.main()
