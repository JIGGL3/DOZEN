"""Atomic admission tests: per-provider capacity, global capacity, shutdown,
quarantine, concurrent submitters, registration rollback / no orphans."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from webllm.browser_jobs import JobState
from webllm.browser_queue import (
    AdmissionOutcome,
    BrowserQueueRejectedError,
    QueuePolicy,
)
from webllm.providers import ProviderError

from ._util import QueueTestManager


class TestProviderCapacity(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=4, max_total_queued_prompts=32
            )
        )

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_exact_boundary(self) -> None:
        self.bm.install_blocker("openai")  # occupies the running slot
        handles = [self.bm.submit_prompt("openai", f"q{i}") for i in range(4)]
        self.assertEqual(len(handles), 4)
        # The 5th exceeds the per-provider bound and is rejected immediately.
        with self.assertRaises(BrowserQueueRejectedError) as ctx:
            self.bm.submit_prompt("openai", "overflow")
        self.assertIs(ctx.exception.outcome, AdmissionOutcome.REJECTED_PROVIDER_FULL)
        self.assertEqual(ctx.exception.admission.provider_queued_depth, 4)
        # The rejected prompt was never registered as a QUEUED job.
        self.assertNotIn("overflow", self.bm.prompts_sent())

    def test_running_job_does_not_consume_queued_capacity(self) -> None:
        # One running job PLUS the full per-provider queue is valid.
        self.bm.install_blocker("openai")
        handles = [self.bm.submit_prompt("openai", f"q{i}") for i in range(4)]
        snap = self.bm.queue_snapshot("openai")
        self.assertEqual(snap.prompt_depth, 4)
        self.assertIsNotNone(snap.running_job_id)
        self.assertEqual(len(handles), 4)


class TestGlobalCapacity(unittest.TestCase):
    def test_global_boundary_rejects_next(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=4, max_total_queued_prompts=6
            )
        )
        try:
            bm.install_blocker("openai")
            bm.install_blocker("google")
            # Fill 4 + 2 = 6 queued globally (openai hits its own cap at 4).
            for i in range(4):
                bm.submit_prompt("openai", f"o{i}")
            for i in range(2):
                bm.submit_prompt("google", f"g{i}")
            # Global is now full; google still has provider room but global is 6.
            with self.assertRaises(BrowserQueueRejectedError) as ctx:
                bm.submit_prompt("google", "one-too-many")
            self.assertIs(ctx.exception.outcome, AdmissionOutcome.REJECTED_GLOBAL_FULL)
        finally:
            bm.shutdown()

    def test_expired_other_provider_entries_are_swept_globally(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=2,
                max_total_queued_prompts=2,
                max_queue_wait_s=0.05,
            )
        )
        try:
            bm.install_blocker("openai", "block-openai")
            bm.install_blocker("google", "block-google")
            expired = [
                bm.submit_prompt("openai", f"expired-{i}", timeout=10)
                for i in range(2)
            ]
            time.sleep(0.1)
            admitted = bm.submit_prompt("google", "after-sweep", timeout=10)
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 0)
            self.assertEqual(bm.queue_snapshot("google").prompt_depth, 1)
            self.assertTrue(all(h.timed_out() for h in expired))
            self.assertIs(admitted.state(), JobState.QUEUED)
        finally:
            bm.shutdown()


class TestProviderIsolation(unittest.TestCase):
    def test_full_provider_does_not_block_other(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=3, max_total_queued_prompts=32
            )
        )
        try:
            bm.install_blocker("openai")
            for i in range(3):
                bm.submit_prompt("openai", f"o{i}")
            with self.assertRaises(BrowserQueueRejectedError):
                bm.submit_prompt("openai", "o-overflow")
            # Provider B is entirely unaffected and completes normally.
            h = bm.submit_prompt("google", "g0")
            self.assertEqual(h.result(timeout=5), "reply:g0")
        finally:
            bm.shutdown()


class TestShutdownAdmission(unittest.TestCase):
    def test_submit_after_shutdown_returns_settled_handle(self) -> None:
        bm = QueueTestManager()
        bm.shutdown()
        # Shutdown is a deterministic rejection via a settled handle (preserving
        # the Phase 5A/5B contract), never a hang and never an orphan.
        h = bm.submit_prompt("openai", "late")
        self.assertTrue(h._job.done.wait(1))
        with self.assertRaises(RuntimeError):
            h.result(timeout=1)
        self.assertNotIn("late", bm.prompts_sent())

    def test_concurrent_submit_and_shutdown_no_orphan(self) -> None:
        for _ in range(20):
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
            self.assertTrue(all(not t.is_alive() for t in ts))
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0]._job.done.wait(1))
            try:
                handles[0].result(timeout=1)
            except (RuntimeError, TimeoutError):
                pass


class TestQuarantineAdmission(unittest.TestCase):
    def test_quarantined_provider_rejects_prompt(self) -> None:
        bm = QueueTestManager()
        try:
            # Force quarantine directly (Phase 5B leaves a provider unsafe; here we
            # exercise the admission consequence in isolation).
            bm._quarantine_provider("openai", reason="test")
            self.assertTrue(bm.is_quarantined("openai"))
            with self.assertRaises(BrowserQueueRejectedError) as ctx:
                bm.submit_prompt("openai", "blocked")
            self.assertIs(
                ctx.exception.outcome,
                AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED,
            )
            self.assertNotIn("blocked", bm.prompts_sent())
            # A different provider is unaffected.
            self.assertEqual(bm.submit_prompt("google", "ok").result(timeout=5), "reply:ok")
            # Explicit teardown clears quarantine.
            bm.remove_provider("openai")
            self.assertFalse(bm.is_quarantined("openai"))
        finally:
            bm.shutdown()

    def test_failed_teardown_does_not_clear_live_worker_quarantine(self) -> None:
        bm = QueueTestManager()
        try:
            bm.install_blocker("openai")
            bm._quarantine_provider("openai", reason="test")
            with patch.object(
                bm, "_submit_to", side_effect=RuntimeError("control admission failed")
            ):
                with self.assertRaises(RuntimeError):
                    bm.remove_provider("openai")
            self.assertTrue(bm.is_quarantined("openai"))
        finally:
            bm.shutdown()


class TestConcurrentAdmission(unittest.TestCase):
    def test_exactly_capacity_accepted_under_race(self) -> None:
        cap = 8
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=cap, max_total_queued_prompts=64
            )
        )
        try:
            bm.install_blocker("openai")  # keep the worker busy so nothing drains
            accepted: list = []
            rejected: list = []
            lock = threading.Lock()
            barrier = threading.Barrier(40)

            def submit(i: int) -> None:
                barrier.wait()
                try:
                    h = bm.submit_prompt("openai", f"c{i}")
                    with lock:
                        accepted.append(h)
                except BrowserQueueRejectedError:
                    with lock:
                        rejected.append(i)

            threads = [threading.Thread(target=submit, args=(i,)) for i in range(40)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertEqual(len(accepted), cap)
            self.assertEqual(len(rejected), 40 - cap)
            # No over-admission: physical queue depth equals the cap exactly.
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, cap)
        finally:
            bm.shutdown()


class TestNoOrphans(unittest.TestCase):
    def test_rejected_admission_registers_no_queued_record(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(
                max_queued_prompts_per_provider=2, max_total_queued_prompts=32
            )
        )
        try:
            bm.install_blocker("openai")
            bm.submit_prompt("openai", "a")
            bm.submit_prompt("openai", "b")
            before = len(bm.browser_job_snapshots())
            with self.assertRaises(ProviderError):
                bm.submit_prompt("openai", "rejected")
            after = len(bm.browser_job_snapshots())
            # No new browser-job record was created for the rejected submission.
            self.assertEqual(after, before)
            # And no record is in QUEUED state without a physical queue entry.
            queued = [
                s for s in bm.browser_job_snapshots() if s.state == JobState.QUEUED.value
            ]
            self.assertLessEqual(len(queued), bm.queue_snapshot("openai").prompt_depth)
        finally:
            bm.shutdown()

    def test_post_registration_failure_rolls_back_record_and_queue(self) -> None:
        bm = QueueTestManager()
        try:
            before = len(bm.browser_job_snapshots())
            with patch(
                "webllm.browser_manager.InterruptionRequest",
                side_effect=RuntimeError("injected-after-register"),
            ):
                with self.assertRaises(BrowserQueueRejectedError) as ctx:
                    bm.submit_prompt("openai", "must-not-leak", timeout=5)
            self.assertIs(
                ctx.exception.outcome,
                AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE,
            )
            self.assertEqual(len(bm.browser_job_snapshots()), before)
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 0)
        finally:
            bm.shutdown()

    def test_enqueue_failure_after_append_rolls_back_both_sides(self) -> None:
        bm = QueueTestManager()
        try:
            with bm._admission_lock:
                worker = bm._ensure_worker_locked("openai")
            original = worker.jobs.enqueue_prompt

            def append_then_fail(entry) -> None:
                original(entry)
                raise RuntimeError("injected-after-append")

            with patch.object(worker.jobs, "enqueue_prompt", append_then_fail):
                with self.assertRaises(BrowserQueueRejectedError):
                    bm.submit_prompt("openai", "must-not-leak", timeout=5)
            self.assertEqual(bm.browser_job_snapshots(), [])
            self.assertEqual(worker.jobs.prompt_depth(), 0)
        finally:
            bm.shutdown()

    def test_snapshot_failure_after_enqueue_rolls_back_both_sides(self) -> None:
        bm = QueueTestManager()
        try:
            with patch.object(
                bm,
                "_admission_snapshot_locked",
                side_effect=RuntimeError("injected-snapshot-failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected-snapshot-failure"):
                    bm.submit_prompt("openai", "must-not-leak", timeout=5)
            self.assertEqual(bm.browser_job_snapshots(), [])
            self.assertEqual(bm.queue_snapshot("openai").prompt_depth, 0)
        finally:
            bm.shutdown()

    def test_publication_failure_returns_settled_non_orphan_handle(self) -> None:
        bm = QueueTestManager()
        try:
            with bm._admission_lock:
                worker = bm._ensure_worker_locked("openai")
            with patch.object(
                worker.jobs,
                "publish_prompt",
                side_effect=RuntimeError("injected-publication-failure"),
            ):
                handle = bm.submit_prompt("openai", "must-not-run", timeout=5)
            with self.assertRaises(ProviderError):
                handle.result(timeout=1)
            self.assertTrue(handle.snapshot().physical_settled)
            self.assertEqual(worker.jobs.prompt_depth(), 0)
            self.assertNotIn("must-not-run", bm.prompts_sent())
        finally:
            bm.shutdown()

    def test_worker_creation_failure_is_typed_unavailable(self) -> None:
        bm = QueueTestManager()
        try:
            with patch(
                "webllm.browser_manager._Worker",
                side_effect=RuntimeError("injected-worker-start"),
            ):
                with self.assertRaises(BrowserQueueRejectedError) as ctx:
                    bm.submit_prompt("openai", "must-not-leak", timeout=5)
            self.assertIs(
                ctx.exception.outcome,
                AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE,
            )
            self.assertEqual(bm.browser_job_snapshots(), [])
        finally:
            bm.shutdown()

    def test_unknown_provider_is_rejected_before_registration(self) -> None:
        bm = QueueTestManager()
        try:
            with self.assertRaises(BrowserQueueRejectedError) as ctx:
                bm.submit_prompt("not-a-provider", "must-not-leak", timeout=5)
            self.assertIs(
                ctx.exception.outcome,
                AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE,
            )
            self.assertEqual(bm.browser_job_snapshots(), [])
            self.assertIsNone(bm.queue_snapshot("not-a-provider"))
        finally:
            bm.shutdown()

    def test_non_finite_or_non_positive_timeout_is_rejected_before_admission(self) -> None:
        bm = QueueTestManager()
        try:
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(timeout=value), self.assertRaises(ValueError):
                    bm.submit_prompt("openai", "invalid-timeout", timeout=value)
            self.assertEqual(bm.browser_job_snapshots(), [])
            self.assertIsNone(bm.queue_snapshot("openai"))
        finally:
            bm.shutdown()


class TestSnapshotBound(unittest.TestCase):
    def test_bulk_snapshot_respects_policy_bound(self) -> None:
        bm = QueueTestManager(queue_policy=QueuePolicy(max_queue_snapshot_entries=2))
        try:
            providers = ["openai", "google", "grok", "groq", "deepseek"]
            handles = [
                bm.submit_prompt(provider, f"snapshot-{i}", timeout=5)
                for i, provider in enumerate(providers)
            ]
            for handle in handles:
                handle.result(timeout=2)
            self.assertEqual(len(bm.queue_snapshots()), 2)
        finally:
            bm.shutdown()


if __name__ == "__main__":
    unittest.main()
