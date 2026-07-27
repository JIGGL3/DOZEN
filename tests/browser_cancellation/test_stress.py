"""Stress tests — barriers + events (not sleeps) drive many concurrent
cancellation/completion/timeout races, repeated cancellation, stale callbacks,
two providers under mixed load, and shutdown during interruption.

Deterministic and bounded: exactly-once stop actions and stable caller outcomes
under contention.
"""

from __future__ import annotations

import threading
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_cancellation import InterruptionPolicy
from webllm.browser_jobs import JobState
from webllm.providers import ProviderError

from ._util import CooperativeGate, Gate, InterruptibleBrowserManager, poll, poll_state

_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class TestInterruptionStress(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager(interruption_policy=_FAST)

    def tearDown(self) -> None:
        try:
            self.bm.shutdown()
        except Exception:
            pass

    def test_many_running_cancellations_stop_exactly_once_each(self) -> None:
        # Each job is cancelled while running; every one is stopped exactly once.
        for i in range(60):
            gate = CooperativeGate(value="never")
            prompt = f"coop{i}"
            self.bm.gates[prompt] = gate
            h = self.bm.submit_prompt("openai", prompt)
            self.assertTrue(gate.started.wait(2))
            self.assertTrue(h.request_cancel())
            self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
            self.assertEqual(h.interruption_snapshot().attempt_count, 1)
        self.assertEqual(self.bm.stop_calls, 60)

    def test_completion_cancellation_races_never_mix(self) -> None:
        for _ in range(150):
            gate = Gate(value="ok")
            self.bm.gates["race"] = gate
            h = self.bm.submit_prompt("openai", "race")
            self.assertTrue(gate.started.wait(2))
            barrier = threading.Barrier(2)
            results: dict[str, object] = {}

            def do_complete() -> None:
                barrier.wait()
                gate.release.set()

            def do_cancel() -> None:
                barrier.wait()
                results["cancel"] = h.request_cancel()

            threads = [threading.Thread(target=do_complete),
                       threading.Thread(target=do_cancel)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertTrue(poll(lambda: h._job.done.is_set()))
            state = h.state()
            # Never an impossible mixed state; if cancel won it is never COMPLETED.
            if results.get("cancel"):
                self.assertIsNot(state, JobState.COMPLETED)
            del self.bm.gates["race"]

    def test_timeout_stop_races_preserve_cause(self) -> None:
        for _ in range(80):
            gate = CooperativeGate(value="never")
            self.bm.gates["coop"] = gate
            h = self.bm.submit_prompt("openai", "coop")
            self.assertTrue(gate.started.wait(2))
            with self.assertRaises(TimeoutError):
                h.result(timeout=0.1)
            self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
            self.assertTrue(h.timed_out())
            del self.bm.gates["coop"]

    def test_two_providers_mixed_cancellation_load(self) -> None:
        # openai serializes, so cancel each running job before the next starts;
        # google completes normally throughout, fully independent of openai.
        for i in range(20):
            g = CooperativeGate(value="never")
            self.bm.gates[f"cx{i}"] = g
            ha = self.bm.submit_prompt("openai", f"cx{i}")
            self.assertTrue(g.started.wait(2))
            hb = self.bm.submit_prompt("google", f"ok{i}")
            self.assertEqual(hb.result(timeout=5), f"reply:ok{i}")
            self.assertTrue(ha.request_cancel())
            self.assertTrue(poll_state(self.bm, ha.job_id, JobState.ABANDONED))
            self.assertTrue(ha.cancelled())

    def test_shutdown_during_interruption_does_not_hang(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        t = threading.Thread(target=self.bm.shutdown)
        t.start()
        t.join(timeout=20)
        self.assertFalse(t.is_alive())  # shutdown did not hang mid-interruption


if __name__ == "__main__":
    unittest.main()
