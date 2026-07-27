"""Transition-table tests: every valid transition, every invalid terminal one,
duplicate/stale attempts, and the completion/timeout & completion/cancel races.

These run against the registry directly (no browser, no worker threads) so the
compare-and-transition semantics are exercised in isolation.
"""

from __future__ import annotations

import threading
import unittest

from webllm.browser_jobs import (
    BrowserJobRegistry,
    JobState,
    allowed_transition,
    is_terminal,
)

VALID = [
    (JobState.QUEUED, JobState.RUNNING),
    (JobState.QUEUED, JobState.CANCELLED),
    (JobState.QUEUED, JobState.TIMED_OUT),
    (JobState.QUEUED, JobState.ABANDONED),
    (JobState.RUNNING, JobState.COMPLETED),
    (JobState.RUNNING, JobState.FAILED),
    (JobState.RUNNING, JobState.CANCELLED),
    (JobState.RUNNING, JobState.TIMED_OUT),
    (JobState.RUNNING, JobState.ABANDONED),
    (JobState.TIMED_OUT, JobState.ABANDONED),
    (JobState.CANCELLED, JobState.ABANDONED),
]

INVALID_TERMINAL = [
    (JobState.COMPLETED, JobState.RUNNING),
    (JobState.FAILED, JobState.COMPLETED),
    (JobState.TIMED_OUT, JobState.COMPLETED),
    (JobState.ABANDONED, JobState.COMPLETED),
    (JobState.COMPLETED, JobState.FAILED),
    (JobState.CANCELLED, JobState.COMPLETED),
]


def _drive(reg: BrowserJobRegistry, job_id, path: list[JobState]) -> None:
    """Force a job through ``path`` (each step assumed valid)."""
    for st in path:
        res = reg.transition(job_id, st)
        assert res.transitioned, (st, res.reason)


class TestTransitionTable(unittest.TestCase):
    def test_table_matches_helper(self) -> None:
        for old, new in VALID:
            self.assertTrue(allowed_transition(old, new), f"{old}->{new}")

    def test_terminal_states_admit_nothing(self) -> None:
        for term in (JobState.COMPLETED, JobState.FAILED, JobState.ABANDONED):
            self.assertTrue(is_terminal(term))
            for target in JobState:
                self.assertFalse(allowed_transition(term, target))


class TestRegistryTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = BrowserJobRegistry()

    def _new(self):
        return self.reg.register("openai")

    def test_every_valid_transition(self) -> None:
        # Reach each source state fresh, then apply the target.
        prefixes = {
            JobState.QUEUED: [],
            JobState.RUNNING: [JobState.RUNNING],
            JobState.TIMED_OUT: [JobState.RUNNING, JobState.TIMED_OUT],
            JobState.CANCELLED: [JobState.RUNNING, JobState.CANCELLED],
        }
        for old, new in VALID:
            reg = BrowserJobRegistry()
            jid = reg.register("openai")
            _drive(reg, jid, prefixes[old])
            res = reg.transition(jid, new)
            self.assertTrue(res.transitioned, f"{old}->{new}: {res.reason}")
            self.assertEqual(res.state, new)

    def test_invalid_terminal_transitions_rejected(self) -> None:
        paths = {
            JobState.COMPLETED: [JobState.RUNNING, JobState.COMPLETED],
            JobState.FAILED: [JobState.RUNNING, JobState.FAILED],
            JobState.TIMED_OUT: [JobState.RUNNING, JobState.TIMED_OUT],
            JobState.ABANDONED: [JobState.RUNNING, JobState.ABANDONED],
            JobState.CANCELLED: [JobState.RUNNING, JobState.CANCELLED],
        }
        for old, new in INVALID_TERMINAL:
            reg = BrowserJobRegistry()
            jid = reg.register("openai")
            _drive(reg, jid, paths[old])
            res = reg.transition(jid, new)
            self.assertFalse(res.transitioned, f"{old}->{new} should be rejected")
            self.assertEqual(res.state, old)  # state unchanged
            self.assertEqual(reg.state(jid), old)

    def test_duplicate_completion_rejected(self) -> None:
        jid = self._new()
        _drive(self.reg, jid, [JobState.RUNNING])
        self.assertTrue(self.reg.mark_completed(jid).transitioned)
        self.assertFalse(self.reg.mark_completed(jid).transitioned)
        self.assertEqual(self.reg.state(jid), JobState.COMPLETED)

    def test_duplicate_failure_rejected(self) -> None:
        jid = self._new()
        _drive(self.reg, jid, [JobState.RUNNING])
        self.assertTrue(self.reg.mark_failed(jid).transitioned)
        self.assertFalse(self.reg.mark_failed(jid).transitioned)

    def test_unknown_job_transition_is_safe(self) -> None:
        res = self.reg.transition("does-not-exist", JobState.RUNNING)
        self.assertFalse(res.transitioned)
        self.assertEqual(res.reason, "unknown-job")

    def test_timed_out_can_only_go_to_abandoned(self) -> None:
        jid = self._new()
        _drive(self.reg, jid, [JobState.RUNNING, JobState.TIMED_OUT])
        self.assertFalse(self.reg.mark_completed(jid).transitioned)
        self.assertTrue(self.reg.mark_abandoned(jid).transitioned)

    def test_stale_transition_does_not_corrupt_state(self) -> None:
        jid = self._new()
        _drive(self.reg, jid, [JobState.RUNNING, JobState.COMPLETED])
        # A stale worker tries several late transitions; all must be no-ops.
        for st in (JobState.RUNNING, JobState.FAILED, JobState.TIMED_OUT,
                   JobState.CANCELLED, JobState.ABANDONED):
            self.assertFalse(self.reg.transition(jid, st).transitioned)
        self.assertEqual(self.reg.state(jid), JobState.COMPLETED)


class TestRaces(unittest.TestCase):
    """Completion vs timeout / cancel races.

    Both winners are exercised *deterministically* by forcing each order, and a
    barrier-synchronized concurrent variant then verifies — over many rounds —
    that exactly one competitor ever wins and no impossible mixed state arises.
    """

    def _run_first(self, loser_state: JobState) -> None:
        # Deterministic: completion clearly first -> COMPLETED, loser rejected.
        reg = BrowserJobRegistry()
        jid = reg.register("openai")
        reg.mark_running(jid)
        self.assertTrue(reg.mark_completed(jid).transitioned)
        self.assertFalse(reg.transition(jid, loser_state).transitioned)
        self.assertEqual(reg.state(jid), JobState.COMPLETED)

    def _loser_first(self, loser_state: JobState) -> None:
        # Deterministic: timeout/cancel first -> loser wins, completion rejected.
        reg = BrowserJobRegistry()
        jid = reg.register("openai")
        reg.mark_running(jid)
        self.assertTrue(reg.transition(jid, loser_state).transitioned)
        self.assertFalse(reg.mark_completed(jid).transitioned)  # never COMPLETED
        self.assertEqual(reg.state(jid), loser_state)

    def _concurrent(self, loser_state: JobState, rounds: int = 500) -> None:
        for _ in range(rounds):
            reg = BrowserJobRegistry()
            jid = reg.register("openai")
            reg.mark_running(jid)
            barrier = threading.Barrier(2)
            results: dict[str, bool] = {}

            def do_complete() -> None:
                barrier.wait()
                results["complete"] = reg.mark_completed(jid).transitioned

            def do_loser() -> None:
                barrier.wait()
                results["loser"] = reg.transition(jid, loser_state).transitioned

            t1 = threading.Thread(target=do_complete)
            t2 = threading.Thread(target=do_loser)
            t1.start(); t2.start(); t1.join(); t2.join()

            # EXACTLY one competing transition wins; the final state is one of
            # the two candidates and never an impossible mix.
            self.assertNotEqual(results["complete"], results["loser"])
            final = reg.state(jid)
            self.assertIn(final, (JobState.COMPLETED, loser_state))
            self.assertEqual(final == JobState.COMPLETED, results["complete"])

    def test_completion_races_timeout(self) -> None:
        self._run_first(JobState.TIMED_OUT)
        self._loser_first(JobState.TIMED_OUT)
        self._concurrent(JobState.TIMED_OUT)

    def test_completion_races_cancellation(self) -> None:
        self._run_first(JobState.CANCELLED)
        self._loser_first(JobState.CANCELLED)
        self._concurrent(JobState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
