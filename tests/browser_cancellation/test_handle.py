"""Handle tests for Phase 5B: immediate cancellation/timeout delivery, the
read-only interruption snapshot, multi-reader behaviour and repeated result
calls, and eviction not altering the recorded outcome.
"""

from __future__ import annotations

import threading
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_cancellation import InterruptionOutcome, InterruptionPolicy
from webllm.browser_jobs import JobState

from ._util import CooperativeGate, Gate, InterruptibleBrowserManager, poll_state

_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class TestHandle(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager(interruption_policy=_FAST)

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_immediate_cancellation_delivery(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        done = threading.Event()

        def waiter() -> None:
            try:
                h.result(timeout=5)
            except CancelledError:
                done.set()

        t = threading.Thread(target=waiter)
        t.start()
        self.assertTrue(h.request_cancel())
        self.assertTrue(done.wait(1))    # caller released immediately
        t.join()

    def test_immediate_timeout_delivery(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        with self.assertRaises(TimeoutError):
            h.result(timeout=0.15)

    def test_interruption_snapshot_is_read_only_view(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        # Before any request: present, not requested.
        pre = h.interruption_snapshot()
        self.assertIsNotNone(pre)
        self.assertFalse(pre.requested)
        self.assertEqual(pre.job_id, h.job_id)
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        post = h.interruption_snapshot()
        self.assertTrue(post.requested)
        self.assertEqual(post.outcome, InterruptionOutcome.STOPPED.value)
        # Snapshot is a frozen copy — mutating internals is impossible.
        with self.assertRaises(Exception):
            post.requested = False  # type: ignore[misc]

    def test_multi_reader_shares_one_caller_outcome(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        outcomes: list[type[BaseException]] = []
        barrier = threading.Barrier(3)

        def reader() -> None:
            barrier.wait()
            try:
                h.result(timeout=2)
            except BaseException as exc:
                outcomes.append(type(exc))

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(outcomes, [CancelledError, CancelledError, CancelledError])

    def test_repeated_result_calls_are_stable(self) -> None:
        h = self.bm.submit_prompt("openai", "q")
        self.assertEqual(h.result(timeout=5), "reply:q")
        self.assertEqual(h.result(timeout=5), "reply:q")
        self.assertIs(h.state(), JobState.COMPLETED)

    def test_eviction_does_not_alter_interruption_outcome(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        outcome = h.interruption_snapshot().outcome
        # Churn many terminal jobs to force registry eviction of older records.
        for i in range(300):
            self.bm.submit_prompt("google", f"g{i}").result(timeout=5)
        # The interruption snapshot lives on the handle, not the registry.
        self.assertEqual(h.interruption_snapshot().outcome, outcome)


if __name__ == "__main__":
    unittest.main()
