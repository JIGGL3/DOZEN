"""Stress tests — barriers + events (not sleeps) drive many concurrent races,
registrations, repeated stale completions, and shutdown under load.

Kept deterministic and fast: bounded rounds, bounded waits, no unbounded loops.
"""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_jobs import BrowserJobRegistry, JobState

from ._util import ControllableBrowserManager, Gate


def _poll_state(mgr, job_id, want: JobState, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mgr._registry.state(job_id) is want:
            return True
        time.sleep(0.005)
    return False


class TestRegistryStress(unittest.TestCase):
    def test_many_completion_timeout_races(self) -> None:
        """1000 concurrent completion-vs-timeout races: never both, never mixed."""
        for _ in range(1000):
            reg = BrowserJobRegistry()
            jid = reg.register("openai")
            reg.mark_running(jid)
            barrier = threading.Barrier(3)
            outcomes: dict[str, bool] = {}

            def contend(name: str, target: JobState) -> None:
                barrier.wait()
                outcomes[name] = reg.transition(jid, target).transitioned

            threads = [
                threading.Thread(target=contend, args=("complete", JobState.COMPLETED)),
                threading.Thread(target=contend, args=("timeout", JobState.TIMED_OUT)),
                threading.Thread(target=contend, args=("cancel", JobState.CANCELLED)),
            ]
            # The barrier has exactly 3 parties: the three contenders release
            # together. The main thread only joins (it must NOT wait on the
            # barrier, or it would be a 4th party and deadlock).
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            winners = [k for k, v in outcomes.items() if v]
            self.assertEqual(len(winners), 1)  # exactly one winner, always
            self.assertIn(
                reg.state(jid),
                (JobState.COMPLETED, JobState.TIMED_OUT, JobState.CANCELLED),
            )

    def test_repeated_stale_completions_are_inert(self) -> None:
        reg = BrowserJobRegistry()
        jid = reg.register("openai")
        reg.mark_running(jid)
        reg.mark_completed(jid)
        threads = [
            threading.Thread(target=lambda: reg.mark_completed(jid))
            for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIs(reg.state(jid), JobState.COMPLETED)
        self.assertFalse(reg.snapshot(jid).result_delivered)
        self.assertTrue(reg.mark_result_delivered(jid))
        self.assertTrue(reg.snapshot(jid).result_delivered)

    def test_many_registrations_bounded(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=64)
        for _ in range(5000):
            jid = reg.register("openai")
            reg.mark_running(jid)
            reg.mark_completed(jid)
        self.assertLessEqual(len(reg), 64)


class TestManagerStress(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        try:
            self.bm.shutdown()
        except Exception:
            pass

    def test_two_providers_under_load(self) -> None:
        # Phase 5C bounds each provider queue, so sustained load is driven in
        # within-bound rounds that drain as they go (the same cross-provider load
        # intent, now honoring admission backpressure instead of an unbounded
        # queue). Every admitted job still completes and both tabs settle idle.
        batch = self.bm._queue_policy.max_queued_prompts_per_provider - 1
        completed = 0
        for rnd in range(6):
            handles = []
            for i in range(batch):
                handles.append(self.bm.submit_prompt("openai", f"o{rnd}-{i}"))
                handles.append(self.bm.submit_prompt("google", f"g{rnd}-{i}"))
            for h in handles:
                self.assertTrue(h.result(timeout=10).startswith("reply:"))
                self.assertIs(h.state(), JobState.COMPLETED)
                completed += 1
        self.assertEqual(completed, 6 * batch * 2)
        self.assertIsNone(self.bm.active_browser_job("openai"))
        self.assertIsNone(self.bm.active_browser_job("google"))

    def test_shutdown_during_running_and_queued_work(self) -> None:
        gate = Gate(value="running")
        self.bm.gates["run"] = gate
        h_run = self.bm.submit_prompt("openai", "run")
        self.assertTrue(gate.started.wait(2))
        queued = [self.bm.submit_prompt("openai", f"q{i}") for i in range(5)]
        # Release the running job and shut down concurrently.
        t = threading.Thread(target=self.bm.shutdown)
        t.start()
        gate.release.set()
        t.join(timeout=20)
        self.assertFalse(t.is_alive())  # shutdown did not hang
        # No queued waiter is left blocked.
        for h in queued:
            with self.assertRaises((RuntimeError, TimeoutError)):
                h.result(timeout=5)


if __name__ == "__main__":
    unittest.main()
