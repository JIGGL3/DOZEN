"""Worker-integration tests — the sixteen required Phase 5B scenarios, driven
through real per-provider worker threads with browser-free send and stop-action
seams.
"""

from __future__ import annotations

import threading
import time
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_cancellation import (
    InterruptionActionResult,
    InterruptionOutcome,
    InterruptionPolicy,
)
from webllm.browser_jobs import JobState, TerminalCause
from webllm.providers import ProviderError

from ._util import CooperativeGate, Gate, InterruptibleBrowserManager, poll, poll_state

_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager(interruption_policy=_FAST)
        self.main_ident = threading.get_ident()

    def tearDown(self) -> None:
        self.bm.shutdown()

    # --- Scenario 1 — normal completion --------------------------------- #
    def test_s1_normal_completion_no_stop_action(self) -> None:
        h = self.bm.submit_prompt("openai", "q1")
        self.assertEqual(h.result(timeout=5), "reply:q1")
        self.assertIs(h.state(), JobState.COMPLETED)
        self.assertEqual(self.bm.stop_calls, 0)
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.NOT_REQUESTED.value
        )

    # --- Scenario 2 — queued cancellation, never submitted -------------- #
    def test_s2_queued_cancellation_skips_send_and_stop(self) -> None:
        blocker = Gate(value="a")
        self.bm.gates["blocker"] = blocker
        hb = self.bm.submit_prompt("openai", "blocker")
        self.assertTrue(blocker.started.wait(2))
        queued = self.bm.submit_prompt("openai", "queued")
        self.assertTrue(queued.request_cancel())
        with self.assertRaises(CancelledError):
            queued.result(timeout=1)
        blocker.release.set()
        self.assertTrue(poll_state(self.bm, queued.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_calls, 0)
        self.assertNotIn("queued", self.bm.prompts_sent())
        hb.result(timeout=5)

    # --- Scenario 3 — running explicit cancellation --------------------- #
    def test_s3_running_cancellation_stops_once_on_worker_thread(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))

        # Caller receives CancelledError immediately.
        self.assertTrue(h.request_cancel())
        with self.assertRaises(CancelledError):
            h.result(timeout=1)

        # Ownership retained until the physical op settles.
        self.assertEqual(self.bm.active_browser_job("openai"), h.job_id)

        # Stop action runs exactly once, on the provider worker thread.
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_calls, 1)
        self.assertEqual(self.bm.stop_idents[0], gate.worker_ident)
        self.assertNotEqual(self.bm.stop_idents[0], self.main_ident)
        # Partial output discarded; result holder empty; terminal cause CANCELLED.
        self.assertIsNone(h._job.result)
        self.assertTrue(h.cancelled())
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.STOPPED.value
        )

        # Successor starts only after the interrupted job settles.
        self.assertIsNone(self.bm.active_browser_job("openai"))
        h2 = self.bm.submit_prompt("openai", "after")
        self.assertEqual(h2.result(timeout=5), "reply:after")

    # --- Scenario 4 — running caller timeout ---------------------------- #
    def test_s4_running_caller_timeout_stops_once_cause_preserved(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        with self.assertRaises(TimeoutError):
            h.result(timeout=0.2)
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_calls, 1)
        self.assertTrue(h.timed_out())
        self.assertEqual(h.snapshot().terminal_cause, TerminalCause.TIMED_OUT.value)
        self.assertEqual(
            h.interruption_snapshot().reason, "caller_timeout"
        )

    # --- Scenario 5 — completion beats cancellation --------------------- #
    def test_s5_completion_beats_cancellation(self) -> None:
        gate = Gate(value="done")
        self.bm.gates["race"] = gate
        h = self.bm.submit_prompt("openai", "race")
        self.assertTrue(gate.started.wait(2))
        gate.release.set()
        self.assertTrue(h._job.done.wait(2))
        self.assertEqual(h.result(timeout=1), "done")
        self.assertIs(h.state(), JobState.COMPLETED)
        # Cancellation now loses the race and triggers no stop action.
        self.assertFalse(h.request_cancel())
        self.assertEqual(self.bm.stop_calls, 0)

    # --- Scenario 6 — cancellation beats completion --------------------- #
    def test_s6_cancellation_beats_completion(self) -> None:
        gate = Gate(value="late-value")
        self.bm.gates["race"] = gate
        h = self.bm.submit_prompt("openai", "race")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())          # cancellation wins
        gate.release.set()                            # late completion arrives
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        # Never COMPLETED; late value discarded.
        self.assertIsNot(h.state(), JobState.COMPLETED)
        self.assertTrue(h.cancelled())
        self.assertTrue(h.snapshot().late_result_discarded)
        with self.assertRaises(CancelledError):
            h.result(timeout=1)

    # --- Scenario 7 — repeated cancellation, exactly one stop ----------- #
    def test_s7_repeated_cancellation_exactly_one_stop(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop", should_cancel=lambda: False)
        self.assertTrue(gate.started.wait(2))
        barrier = threading.Barrier(6)

        def hammer() -> None:
            barrier.wait()
            h.request_cancel()
            h.mark_caller_timeout()

        threads = [threading.Thread(target=hammer) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_calls, 1)  # exactly once, despite many signals
        self.assertEqual(h.interruption_snapshot().attempt_count, 1)

    # --- Scenario 8 — unsupported provider ------------------------------ #
    def test_s8_unsupported_provider_still_cancels(self) -> None:
        self.bm._stop_result = InterruptionActionResult.unsupported("no-stop")
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.UNSUPPORTED.value
        )
        self.assertEqual(self.bm.active_browser_job("openai"), h.job_id)
        self.assertFalse(h.snapshot().physical_settled)

    # --- Scenario 9 — stop action failure ------------------------------- #
    def test_s9_stop_action_failure_is_diagnostic_only(self) -> None:
        self.bm._stop_exc = RuntimeError("stop boom")
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        # Caller still receives cancellation — never a provider failure.
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.FAILED.value
        )
        self.assertEqual(self.bm.active_browser_job("openai"), h.job_id)
        self.assertFalse(h.snapshot().physical_settled)

    # --- Scenario 10 — generation already settled (no stale click) ------ #
    def test_s10_already_idle_reports_not_needed(self) -> None:
        self.bm._stop_result = InterruptionActionResult.already_idle("settled")
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.NOT_NEEDED.value
        )

    # --- Scenario 11 — stale ownership cannot stop a successor ---------- #
    def test_s11_stale_job_never_stops_successor(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        a = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(a.request_cancel())
        self.assertTrue(poll_state(self.bm, a.job_id, JobState.ABANDONED))
        # Successor runs and completes normally.
        b = self.bm.submit_prompt("openai", "b")
        self.assertEqual(b.result(timeout=5), "reply:b")
        # The one stop action only ever targeted A — never B.
        self.assertEqual(self.bm.stop_jobs, [a.job_id])
        self.assertNotIn(b.job_id, self.bm.stop_jobs)

    # --- Scenario 12 — same-thread Playwright action -------------------- #
    def test_s12_stop_runs_on_worker_thread(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        self.assertEqual(self.bm.stop_idents[0], gate.worker_ident)
        self.assertNotEqual(self.bm.stop_idents[0], self.main_ident)

    # --- Scenario 13 — multi-provider independence ---------------------- #
    def test_s13_multi_provider_independence(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["A-coop"] = gate
        ha = self.bm.submit_prompt("openai", "A-coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(ha.request_cancel())

        hb = self.bm.submit_prompt("google", "B-fast")
        self.assertEqual(hb.result(timeout=5), "reply:B-fast")
        self.assertIs(hb.state(), JobState.COMPLETED)

        self.assertTrue(poll_state(self.bm, ha.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_jobs, [ha.job_id])  # only A stopped

    # --- Scenario 14 — external cancellation callback ------------------- #
    def test_s14_external_token_triggers_one_interruption(self) -> None:
        flag = {"v": False}
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop", should_cancel=lambda: flag["v"])
        self.assertTrue(gate.started.wait(2))
        flag["v"] = True                               # external orchestration cancel
        self.assertTrue(poll(lambda: self.bm.stop_calls == 1))
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        self.assertEqual(self.bm.stop_calls, 1)
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        snap = h.interruption_snapshot()
        self.assertEqual(snap.reason, "cancelled")
        self.assertEqual(snap.source, "external-token")

    # --- Scenario 15 — provider timeout is NOT a caller interruption ---- #
    def test_s15_provider_timeout_is_ordinary_failure(self) -> None:
        self.bm.handlers["boom"] = lambda p, pr, sc: (_ for _ in ()).throw(
            TimeoutError("provider generation timeout")
        )
        h = self.bm.submit_prompt("openai", "boom")
        with self.assertRaises(TimeoutError):
            h.result(timeout=5)
        self.assertIs(h.state(), JobState.FAILED)
        self.assertEqual(self.bm.stop_calls, 0)
        self.assertEqual(
            h.interruption_snapshot().outcome, InterruptionOutcome.NOT_REQUESTED.value
        )

    # --- Scenario 16 — partial provider output never leaks -------------- #
    def test_s16_partial_output_never_enters_result_or_next_job(self) -> None:
        gate = CooperativeGate(value="PARTIAL-LEFTOVER")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        # Result holder empty; caller never receives partial text.
        self.assertIsNone(h._job.result)
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        # No response text field exists anywhere in the content-free snapshots.
        self.assertNotIn("PARTIAL", str(h.snapshot().to_dict()))
        self.assertNotIn("PARTIAL", str(h.interruption_snapshot().to_dict()))
        # The next job on the same provider observes a clean boundary.
        h2 = self.bm.submit_prompt("openai", "next")
        self.assertEqual(h2.result(timeout=5), "reply:next")


if __name__ == "__main__":
    unittest.main()
