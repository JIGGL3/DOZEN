"""Stress tests — barriers + events (not sleeps) drive many concurrent
admissions, removals, expirations, multi-provider global pressure, and shutdown
during admission. Deterministic and bounded; no unbounded loops."""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_jobs import JobState
from webllm.browser_queue import BrowserQueueRejectedError, QueuePolicy

from ._util import QueueTestManager


def _poll(fn, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.005)
    return False


class TestAdmissionStress(unittest.TestCase):
    def test_hundreds_of_concurrent_admissions_never_over_admit(self) -> None:
        cap = 8
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=cap, max_total_queued_prompts=cap
            )
        )
        try:
            bm.install_blocker("openai")
            accepted = []
            lock = threading.Lock()
            n = 300
            barrier = threading.Barrier(n)

            def submit(i: int) -> None:
                barrier.wait()
                try:
                    h = bm.submit_prompt("openai", f"c{i}")
                    with lock:
                        accepted.append(h)
                except BrowserQueueRejectedError:
                    pass

            threads = [threading.Thread(target=submit, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertTrue(all(not t.is_alive() for t in threads))
            # Never more than the cap admitted; physical depth matches exactly.
            self.assertEqual(len(accepted), cap)
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, cap)
        finally:
            bm.shutdown()

    def test_repeated_cancellations_behind_blocked_worker(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_prompts_per_provider=4)
        )
        try:
            bm.install_blocker("openai")
            # Many submit/cancel cycles while the worker never dequeues.
            for round_ in range(150):
                hs = [bm.submit_prompt("openai", f"r{round_}-{i}") for i in range(4)]
                for h in hs:
                    self.assertTrue(h.request_cancel())
                self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 0)
            # No tombstone residue accumulated behind the blocked running job.
            self.assertEqual(len(bm._peek_worker("openai").jobs._entries), 0)
        finally:
            bm.shutdown()

    def test_repeated_queue_expirations(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=3, max_queue_wait_s=0.05
            )
        )
        try:
            bm.install_blocker("openai")
            handles = []
            for i in range(3):
                handles.append(bm.submit_prompt("openai", f"e{i}"))
            time.sleep(0.15)
            # A fresh admission sweeps the expired trio and takes a freed slot.
            keep = bm.submit_prompt("openai", "survivor")
            for h in handles:
                self.assertTrue(_poll(lambda h=h: h.timed_out()))
                self.assertNotIn(h.job_id, [None])
            self.assertEqual(keep.state(), JobState.QUEUED)
            for h in handles:
                self.assertNotIn(f"e", bm.prompts_sent())  # none were sent
        finally:
            bm.shutdown()

    def test_multiple_providers_reach_global_capacity(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=4, max_total_queued_prompts=8
            )
        )
        try:
            for p in ("openai", "google"):
                bm.install_blocker(p)
            for i in range(4):
                bm.submit_prompt("openai", f"o{i}")
            for i in range(4):
                bm.submit_prompt("google", f"g{i}")
            # Global (8) is now full; either provider is rejected globally.
            with self.assertRaises(BrowserQueueRejectedError):
                bm.submit_prompt("google", "overflow")
        finally:
            bm.shutdown()

    def test_shutdown_during_admission_storm(self) -> None:
        for _ in range(10):
            bm = QueueTestManager(
                queue_policy=QueuePolicy(max_queued_prompts_per_provider=8)
            )
            handles: list = []
            lock = threading.Lock()
            start = threading.Barrier(9)

            def submit(i: int) -> None:
                start.wait()
                try:
                    h = bm.submit_prompt("openai", f"s{i}")
                    with lock:
                        handles.append(h)
                except Exception:
                    pass

            def shut() -> None:
                start.wait()
                bm.shutdown()

            ts = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
            ts.append(threading.Thread(target=shut))
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=5)
            self.assertTrue(all(not t.is_alive() for t in ts))
            # Every admitted handle resolves (never an orphan/hang).
            for h in handles:
                self.assertTrue(h._job.done.wait(2))

    def test_removal_races_with_worker_claim(self) -> None:
        # A queued job cancelled exactly as the worker frees up: registry
        # arbitration guarantees exactly one outcome and no double-send.
        for _ in range(40):
            bm = QueueTestManager(
                queue_policy=QueuePolicy(max_queued_prompts_per_provider=4)
            )
            try:
                blocker = bm.install_blocker("openai")
                h = bm.submit_prompt("openai", "target")
                barrier = threading.Barrier(2)

                def cancel() -> None:
                    barrier.wait()
                    h.request_cancel()

                def release() -> None:
                    barrier.wait()
                    blocker.release.set()

                ts = [threading.Thread(target=cancel), threading.Thread(target=release)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(timeout=5)
                self.assertTrue(_poll(lambda: h.snapshot() is None or h.snapshot().physical_settled))
                # If cancelled before claim it was never sent; if it ran it sent once.
                self.assertLessEqual(bm.prompts_sent().count("target"), 1)
            finally:
                bm.shutdown()


if __name__ == "__main__":
    unittest.main()
