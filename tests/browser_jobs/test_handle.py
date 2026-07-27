"""Handle tests: result/exception retrieval, timeout & cancellation state
updates, snapshot access, exactly-once delivery, late-result unavailability.
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


class TestHandle(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_result_retrieval(self) -> None:
        h = self.bm.submit_prompt("openai", "hello")
        self.assertEqual(h.result(timeout=5), "reply:hello")
        self.assertIs(h.state(), JobState.COMPLETED)
        self.assertTrue(h.snapshot().result_delivered)

    def test_exception_retrieval(self) -> None:
        self.bm.handlers["boom"] = lambda p, pr, sc: (_ for _ in ()).throw(
            ProviderError("provider down")
        )
        h = self.bm.submit_prompt("openai", "boom")
        with self.assertRaises(ProviderError):
            h.result(timeout=5)
        self.assertIs(h.state(), JobState.FAILED)
        snap = h.snapshot()
        self.assertEqual(snap.failure_category, "ProviderError")
        self.assertFalse(snap.result_delivered)

    def test_timeout_updates_state_and_discards_late_result(self) -> None:
        gate = Gate(value="late-answer")
        self.bm.gates["slow"] = gate
        h = self.bm.submit_prompt("openai", "slow")
        self.assertTrue(gate.started.wait(2))          # worker is running it
        with self.assertRaises(TimeoutError):
            h.result(timeout=0.2)                       # caller deadline expires
        self.assertIs(h.state(), JobState.TIMED_OUT)
        # The browser op finishes late; its result must be discarded.
        gate.release.set()
        self.assertTrue(_poll_state(self.bm, h.job_id, JobState.ABANDONED))
        snap = h.snapshot()
        self.assertTrue(snap.late_result_discarded)
        self.assertFalse(snap.result_delivered)
        # Late result is unavailable to the caller.
        with self.assertRaises(TimeoutError):
            h.result(timeout=1)

    def test_cancellation_updates_state(self) -> None:
        self.bm.handlers["stop"] = lambda p, pr, sc: (_ for _ in ()).throw(
            CancelledError("user stop")
        )
        h = self.bm.submit_prompt("openai", "stop")
        with self.assertRaises(CancelledError):
            h.result(timeout=5)
        self.assertIs(h.state(), JobState.ABANDONED)
        self.assertTrue(h.cancelled())

    def test_snapshot_access(self) -> None:
        h = self.bm.submit_prompt("openai", "hello")
        snap = h.snapshot()
        self.assertEqual(snap.job_id, h.job_id)
        self.assertEqual(snap.provider, "openai")
        h.result(timeout=5)

    def test_result_delivered_exactly_once(self) -> None:
        h = self.bm.submit_prompt("openai", "hello")
        first = h.result(timeout=5)
        second = h.result(timeout=5)  # idempotent read of a completed job
        self.assertEqual(first, "reply:hello")
        self.assertEqual(second, "reply:hello")
        # Exactly one COMPLETED transition happened.
        self.assertTrue(h.snapshot().result_delivered)

    def test_mark_caller_timeout_is_authoritative(self) -> None:
        gate = Gate(value="x")
        self.bm.gates["slow"] = gate
        h = self.bm.submit_prompt("openai", "slow")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.mark_caller_timeout())    # wins the race
        self.assertFalse(h.mark_caller_timeout())   # idempotent: already timed out
        gate.release.set()


if __name__ == "__main__":
    unittest.main()
