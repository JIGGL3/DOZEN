"""Control-job tests: control capacity, prompt/control independence, shutdown
sentinel independence, login/status compatibility (browser-free)."""

from __future__ import annotations

import time
import unittest

from webllm.browser_queue import BrowserQueueRejectedError, QueuePolicy

from ._util import QueueTestManager


def _poll(fn, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.005)
    return False


class TestControlCapacity(unittest.TestCase):
    def test_control_jobs_are_bounded(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_control_jobs_per_provider=3)
        )
        try:
            bm.install_blocker("openai")  # keep the worker busy so nothing drains
            jobs = [bm._enqueue("openai", lambda: "ok") for _ in range(3)]
            self.assertEqual(len(jobs), 3)
            # The 4th control job exceeds the control bound.
            with self.assertRaises(BrowserQueueRejectedError):
                bm._enqueue("openai", lambda: "ok")
            self.assertEqual(bm.queue_snapshot("openai").control_depth, 3)
        finally:
            bm.shutdown()

    def test_control_and_prompt_bounds_are_independent(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=2,
                max_queued_control_jobs_per_provider=2,
            )
        )
        try:
            bm.install_blocker("openai")
            # Fill BOTH bounds; neither starves the other.
            bm.submit_prompt("openai", "p0")
            bm.submit_prompt("openai", "p1")
            bm._enqueue("openai", lambda: "c0")
            bm._enqueue("openai", lambda: "c1")
            snap = bm.queue_snapshot("openai")
            self.assertEqual(snap.prompt_depth, 2)
            self.assertEqual(snap.control_depth, 2)
            with self.assertRaises(BrowserQueueRejectedError):
                bm.submit_prompt("openai", "p2")  # prompt bound
            with self.assertRaises(BrowserQueueRejectedError):
                bm._enqueue("openai", lambda: "c2")  # control bound
        finally:
            bm.shutdown()

    def test_unlimited_control_calls_cannot_grow_queue(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_control_jobs_per_provider=4)
        )
        try:
            bm.install_blocker("openai")
            rejected = 0
            for _ in range(50):
                try:
                    bm._enqueue("openai", lambda: "ok")
                except BrowserQueueRejectedError:
                    rejected += 1
            self.assertGreater(rejected, 0)
            self.assertLessEqual(bm.queue_snapshot("openai").control_depth, 4)
        finally:
            bm.shutdown()


class TestShutdownSentinelIndependence(unittest.TestCase):
    def test_shutdown_stops_worker_even_with_full_queues(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=4,
                max_queued_control_jobs_per_provider=4,
            )
        )
        blocker = bm.install_blocker("openai")
        for i in range(4):
            bm.submit_prompt("openai", f"p{i}")
        for i in range(4):
            bm._enqueue("openai", lambda: "c")
        # Shutdown must wake and stop the worker without depending on a free slot
        # (ordinary user prompts can never consume the stop mechanism).
        blocker.release.set()
        start = time.time()
        bm.shutdown()
        self.assertLess(time.time() - start, 15)


class TestControlExecution(unittest.TestCase):
    def test_control_job_runs_and_returns(self) -> None:
        bm = QueueTestManager()
        try:
            job = bm._enqueue("openai", lambda: "control-result")
            self.assertTrue(job.done.wait(5))
            self.assertEqual(job.result, "control-result")
        finally:
            bm.shutdown()

    def test_login_status_is_browser_free_with_no_sessions(self) -> None:
        bm = QueueTestManager()
        try:
            self.assertEqual(bm.login_status(), {})
            self.assertEqual(bm.confirm_login(), {})
        finally:
            bm.shutdown()

    def test_prompt_after_control_still_runs(self) -> None:
        bm = QueueTestManager()
        try:
            ctrl = bm._enqueue("openai", lambda: "c")
            self.assertTrue(ctrl.done.wait(5))
            h = bm.submit_prompt("openai", "p")
            self.assertEqual(h.result(timeout=5), "reply:p")
        finally:
            bm.shutdown()


if __name__ == "__main__":
    unittest.main()
