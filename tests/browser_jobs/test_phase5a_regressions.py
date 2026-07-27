"""Regression coverage for independently reproduced Phase 5A lifecycle races."""

from __future__ import annotations

import threading
import time
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_jobs import (
    BrowserJobId,
    BrowserJobRegistry,
    BrowserJobSnapshot,
    JobState,
    TerminalCause,
)
from webllm.providers import ProviderError

from ._util import ControllableBrowserManager, Gate


class TestCallerLifecycleVsPhysicalOwnership(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_running_timeout_retains_owner_and_blocks_successor(self) -> None:
        first = Gate(value="late")
        second = Gate(value="next")
        self.bm.gates.update(first=first, second=second)
        h1 = self.bm.submit_prompt("openai", "first")
        self.assertTrue(first.started.wait(2))
        with self.assertRaises(TimeoutError):
            h1.result(timeout=0.05)

        h2 = self.bm.submit_prompt("openai", "second")
        snap = h1.snapshot()
        self.assertEqual(self.bm.active_browser_job("openai"), h1.job_id)
        self.assertFalse(snap.physical_settled)
        self.assertEqual(snap.terminal_cause, TerminalCause.TIMED_OUT.value)
        self.assertFalse(second.started.wait(0.1))

        first.release.set()
        self.assertTrue(second.started.wait(2))
        second.release.set()
        self.assertEqual(h2.result(timeout=2), "next")
        self.assertTrue(h1.snapshot().physical_settled)
        self.assertTrue(h1.timed_out())

    def test_cancel_releases_existing_and_later_waiters_as_cancelled(self) -> None:
        gate = Gate(value="late")
        self.bm.gates["cancel"] = gate
        h = self.bm.submit_prompt("openai", "cancel")
        self.assertTrue(gate.started.wait(2))
        outcomes: list[type[BaseException]] = []
        waiter_done = threading.Event()

        def waiter() -> None:
            try:
                h.result(timeout=5)
            except BaseException as exc:  # test records exact public outcome
                outcomes.append(type(exc))
            finally:
                waiter_done.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        self.assertTrue(h.request_cancel())
        self.assertTrue(waiter_done.wait(0.5))
        self.assertEqual(outcomes, [CancelledError])
        self.assertEqual(self.bm.active_browser_job("openai"), h.job_id)
        self.assertFalse(h.snapshot().physical_settled)

        gate.release.set()
        self.assertTrue(h._job.physical_done.wait(2))
        thread.join()
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        self.assertIs(h.state(), JobState.ABANDONED)
        self.assertTrue(h.cancelled())
        self.assertTrue(h.snapshot().late_result_discarded)

    def test_late_exception_cannot_reclassify_user_cancellation(self) -> None:
        gate = Gate(exc=ProviderError("late provider failure"))
        self.bm.gates["cancel-fail"] = gate
        h = self.bm.submit_prompt("openai", "cancel-fail")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        gate.release.set()
        self.assertTrue(h._job.physical_done.wait(2))
        with self.assertRaises(CancelledError):
            h.result(timeout=1)
        self.assertEqual(h.snapshot().terminal_cause, TerminalCause.CANCELLED.value)

    def test_handle_is_multi_reader_with_one_shared_caller_outcome(self) -> None:
        gate = Gate(value="late")
        self.bm.gates["shared"] = gate
        h = self.bm.submit_prompt("openai", "shared")
        self.assertTrue(gate.started.wait(2))
        outcomes: list[type[BaseException]] = []
        barrier = threading.Barrier(2)

        def wait(timeout: float) -> None:
            barrier.wait()
            try:
                h.result(timeout=timeout)
            except BaseException as exc:
                outcomes.append(type(exc))

        threads = [
            threading.Thread(target=wait, args=(0.05,)),
            threading.Thread(target=wait, args=(5,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(outcomes.count(TimeoutError), 2)
        gate.release.set()

    def test_queued_cancel_settles_immediately_and_is_never_sent(self) -> None:
        blocker = Gate(value="blocker")
        queued_gate = Gate(value="must-not-run")
        self.bm.gates.update(blocker=blocker, queued=queued_gate)
        first = self.bm.submit_prompt("openai", "blocker")
        self.assertTrue(blocker.started.wait(2))
        queued = self.bm.submit_prompt("openai", "queued")
        self.assertTrue(queued.request_cancel())
        with self.assertRaises(CancelledError):
            queued.result(timeout=0.2)
        self.assertTrue(queued.snapshot().physical_settled)
        self.assertIs(queued.state(), JobState.ABANDONED)
        self.assertFalse(queued_gate.started.is_set())
        blocker.release.set()
        self.assertEqual(first.result(timeout=2), "blocker")
        self.assertTrue(queued._job.done.wait(2))
        self.assertNotIn("queued", self.bm.prompts_sent())


class TestRegistrySettlementAndSerialization(unittest.TestCase):
    def test_unsettled_cancelled_job_is_not_evictable(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=2)
        held = reg.register("openai")
        reg.mark_running(held)
        reg.mark_cancelled(held)
        for _ in range(20):
            jid = reg.register("google")
            reg.mark_running(jid)
            reg.mark_completed(jid)
        snap = reg.snapshot(held)
        self.assertIsNotNone(snap)
        self.assertFalse(snap.physical_settled)
        self.assertEqual(reg.active_job("openai"), held.value)
        reg.mark_physical_settled(held)
        self.assertEqual(reg.state(held), JobState.ABANDONED)

    def test_snapshot_rejects_malformed_serialized_fields(self) -> None:
        valid = BrowserJobSnapshot(
            job_id=BrowserJobId.new().value,
            provider="openai",
            state="queued",
            created_at=time.time(),
        ).to_dict()
        for field, value, error in (
            ("job_id", "bad", ValueError),
            ("state", "future-state", ValueError),
            ("result_delivered", "false", TypeError),
            ("physical_settled", 1, TypeError),
        ):
            data = dict(valid)
            data[field] = value
            with self.subTest(field=field), self.assertRaises(error):
                BrowserJobSnapshot.from_dict(data)

    def test_registry_never_retains_raw_provider_failure_text(self) -> None:
        reg = BrowserJobRegistry()
        jid = reg.register("openai", correlation="run 7\nunsafe")
        reg.mark_running(jid)
        reg.mark_failed(
            jid,
            failure_category="ProviderError\ntrace",
            failure_message="password=hunter2 <html>prompt and response</html>",
        )
        snap = reg.snapshot(jid)
        self.assertEqual(snap.correlation, "run_7_unsafe")
        self.assertEqual(snap.failure_category, "ProviderError_trace")
        self.assertIsNone(snap.failure_message)

    def test_result_delivered_means_a_handle_actually_read_it(self) -> None:
        bm = ControllableBrowserManager()
        try:
            gate = Gate(value="answer")
            bm.gates["answer"] = gate
            h = bm.submit_prompt("openai", "answer")
            self.assertTrue(gate.started.wait(2))
            gate.release.set()
            self.assertTrue(h._job.done.wait(2))
            self.assertFalse(h.snapshot().result_delivered)
            self.assertEqual(h.result(timeout=1), "answer")
            self.assertTrue(h.snapshot().result_delivered)
            self.assertEqual(h.result(timeout=1), "answer")
        finally:
            bm.shutdown()

    def test_registry_eviction_does_not_invalidate_completed_handle(self) -> None:
        bm = ControllableBrowserManager()
        try:
            # Phase 5C bounds admission, so 300 completions are produced in
            # within-bound rounds that drain as they go. This still overruns the
            # registry's terminal-history bound (evicting the earliest record)
            # while its already-completed handle keeps delivering its cached
            # result — the invariant this regression guards.
            batch = bm._queue_policy.max_queued_prompts_per_provider - 1
            handles = []
            i = 0
            while i < 300:
                round_handles = []
                for _ in range(min(batch, 300 - i)):
                    round_handles.append(bm.submit_prompt("openai", f"evict-{i}"))
                    i += 1
                for h in round_handles:
                    self.assertTrue(h._job.done.wait(5))
                handles.extend(round_handles)
            self.assertIsNone(handles[0].snapshot())  # earliest record evicted
            self.assertEqual(handles[0].result(timeout=1), "reply:evict-0")
            self.assertEqual(handles[-1].result(timeout=1), "reply:evict-299")
        finally:
            bm.shutdown()

    def test_shutdown_drains_queued_job_behind_running_call(self) -> None:
        bm = ControllableBrowserManager()
        gate = Gate(value="running")
        bm.gates["running"] = gate
        running = bm.submit_prompt("openai", "running")
        self.assertTrue(gate.started.wait(2))
        queued = bm.submit_prompt("openai", "queued")
        shutdown = threading.Thread(target=bm.shutdown)
        shutdown.start()
        try:
            self.assertTrue(queued._job.done.wait(1))
            with self.assertRaises(RuntimeError):
                queued.result(timeout=1)
            self.assertEqual(
                queued.snapshot().terminal_cause, TerminalCause.SHUTDOWN.value
            )
            self.assertTrue(queued.snapshot().physical_settled)
        finally:
            gate.release.set()
            shutdown.join(timeout=5)
            self.assertFalse(shutdown.is_alive())
            try:
                running.result(timeout=1)
            except (RuntimeError, TimeoutError):
                pass

    def test_submit_shutdown_race_never_orphans_a_handle(self) -> None:
        for _ in range(25):
            bm = ControllableBrowserManager()
            barrier = threading.Barrier(2)
            handles = []

            def submit() -> None:
                barrier.wait()
                handles.append(bm.submit_prompt("openai", "race"))

            def shutdown() -> None:
                barrier.wait()
                bm.shutdown()

            threads = [threading.Thread(target=submit), threading.Thread(target=shutdown)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0]._job.done.wait(1))
            try:
                handles[0].result(timeout=1)
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main()
