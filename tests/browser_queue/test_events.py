"""Queue event tests: acceptance, rejection, depth changes, expiration,
callback outside lock, callback failure isolation, no content leakage."""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_queue import (
    AdmissionOutcome,
    QueueEvent,
    QueueEventKind,
    QueuePolicy,
)

from ._util import QueueTestManager


def _poll(fn, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.005)
    return False


class _Collector:
    def __init__(self) -> None:
        self.events: list[QueueEvent] = []
        self._lock = threading.Lock()

    def __call__(self, ev: QueueEvent) -> None:
        with self._lock:
            self.events.append(ev)

    def kinds(self) -> list[str]:
        with self._lock:
            return [e.kind for e in self.events]

    def of_kind(self, kind: QueueEventKind) -> list[QueueEvent]:
        with self._lock:
            return [e for e in self.events if e.kind == kind.value]


class TestQueueEvents(unittest.TestCase):
    def test_acceptance_event(self) -> None:
        col = _Collector()
        bm = QueueTestManager(queue_event_sink=col)
        try:
            h = bm.submit_prompt("openai", "hello-secret-prompt")
            h.result(timeout=5)
            accepted = col.of_kind(QueueEventKind.ADMISSION_ACCEPTED)
            self.assertTrue(accepted)
            ev = accepted[0]
            self.assertEqual(ev.outcome, AdmissionOutcome.ACCEPTED.value)
            self.assertEqual(ev.provider, "openai")
            self.assertEqual(ev.job_id, h.job_id)
        finally:
            bm.shutdown()

    def test_rejection_event(self) -> None:
        col = _Collector()
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_prompts_per_provider=1),
            queue_event_sink=col,
        )
        try:
            bm.install_blocker("openai")
            bm.submit_prompt("openai", "a")
            with self.assertRaises(Exception):
                bm.submit_prompt("openai", "rejected")
            rej = col.of_kind(QueueEventKind.ADMISSION_REJECTED)
            self.assertTrue(rej)
            self.assertEqual(
                rej[-1].outcome, AdmissionOutcome.REJECTED_PROVIDER_FULL.value
            )
        finally:
            bm.shutdown()

    def test_depth_changed_event_on_dequeue(self) -> None:
        col = _Collector()
        bm = QueueTestManager(queue_event_sink=col)
        try:
            h = bm.submit_prompt("openai", "q")
            h.result(timeout=5)
            self.assertTrue(col.of_kind(QueueEventKind.QUEUE_DEPTH_CHANGED))
        finally:
            bm.shutdown()

    def test_expiration_event(self) -> None:
        col = _Collector()
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queue_wait_s=0.05), queue_event_sink=col
        )
        try:
            blocker = bm.install_blocker("openai")
            h = bm.submit_prompt("openai", "expire-me")
            time.sleep(0.15)
            blocker.release.set()
            self.assertTrue(_poll(lambda: h.timed_out()))
            self.assertTrue(_poll(lambda: col.of_kind(QueueEventKind.QUEUE_WAIT_EXPIRED)))
        finally:
            bm.shutdown()

    def test_removed_event_on_queued_cancel(self) -> None:
        col = _Collector()
        bm = QueueTestManager(queue_event_sink=col)
        try:
            bm.install_blocker("openai")
            h = bm.submit_prompt("openai", "cancel-me")
            self.assertTrue(h.request_cancel())
            self.assertTrue(_poll(lambda: col.of_kind(QueueEventKind.QUEUED_JOB_REMOVED)))
        finally:
            bm.shutdown()

    def test_no_prompt_content_in_events(self) -> None:
        col = _Collector()
        secret = "TOP-SECRET-PROMPT-BODY-12345"
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_prompts_per_provider=1),
            queue_event_sink=col,
        )
        try:
            bm.install_blocker("openai")
            bm.submit_prompt("openai", secret)
            try:
                bm.submit_prompt("openai", secret + "-rejected")
            except Exception:
                pass
            with col._lock:
                for ev in col.events:
                    for value in ev.to_dict().values():
                        self.assertNotIn(secret, str(value))
        finally:
            bm.shutdown()

    def test_callback_failure_is_isolated(self) -> None:
        def bad_sink(ev: QueueEvent) -> None:
            raise RuntimeError("observer blew up")

        bm = QueueTestManager(queue_event_sink=bad_sink)
        try:
            # A raising sink must never break admission.
            h = bm.submit_prompt("openai", "q")
            self.assertEqual(h.result(timeout=5), "reply:q")
        finally:
            bm.shutdown()

    def test_callback_runs_outside_locks(self) -> None:
        # A sink that re-enters the manager (reads a queue snapshot / submits)
        # would deadlock if it ran under the admission or queue lock.
        seen: list = []

        def reentrant(ev: QueueEvent) -> None:
            seen.append(bm.queue_snapshot(ev.provider))

        bm = QueueTestManager(queue_event_sink=reentrant)
        try:
            h = bm.submit_prompt("openai", "q")
            h.result(timeout=5)
            self.assertTrue(seen)  # no deadlock
        finally:
            bm.shutdown()

    def test_admission_callback_can_request_shutdown_without_deadlock(self) -> None:
        events: list[str] = []

        def sink(event: QueueEvent) -> None:
            events.append(event.kind)
            if event.kind == QueueEventKind.ADMISSION_ACCEPTED.value:
                bm.shutdown()

        bm = QueueTestManager(queue_event_sink=sink)
        handles: list = []
        submitter = threading.Thread(
            target=lambda: handles.append(
                bm.submit_prompt("openai", "shutdown-from-callback", timeout=5)
            )
        )
        submitter.start()
        submitter.join(timeout=2)
        try:
            self.assertFalse(submitter.is_alive())
            self.assertEqual(len(handles), 1)
            with self.assertRaises(RuntimeError):
                handles[0].result(timeout=1)
            self.assertIn(QueueEventKind.PROVIDER_QUEUE_CLOSED.value, events)
            self.assertNotIn("shutdown-from-callback", bm.prompts_sent())
        finally:
            bm.shutdown()
            submitter.join(timeout=2)

    def test_fast_worker_cannot_overtake_admission_and_queued_events(self) -> None:
        order: list[str] = []
        accepted_entered = threading.Event()
        release_sink = threading.Event()

        def queue_sink(event: QueueEvent) -> None:
            order.append(f"queue:{event.kind}")
            if event.kind == QueueEventKind.ADMISSION_ACCEPTED.value:
                accepted_entered.set()
                release_sink.wait(timeout=2)

        def job_sink(event) -> None:
            order.append(f"job:{event.new_state}")

        bm = QueueTestManager(
            queue_event_sink=queue_sink, job_event_sink=job_sink
        )
        handles: list = []
        submitter = threading.Thread(
            target=lambda: handles.append(
                bm.submit_prompt("openai", "fast", timeout=5)
            )
        )
        submitter.start()
        try:
            self.assertTrue(accepted_entered.wait(1))
            time.sleep(0.05)
            self.assertNotIn("job:running", order)
            release_sink.set()
            submitter.join(timeout=2)
            self.assertFalse(submitter.is_alive())
            self.assertEqual(handles[0].result(timeout=2), "reply:fast")
            self.assertLess(
                order.index(f"queue:{QueueEventKind.ADMISSION_ACCEPTED.value}"),
                order.index("job:queued"),
            )
            self.assertLess(order.index("job:queued"), order.index("job:running"))
        finally:
            release_sink.set()
            bm.shutdown()
            submitter.join(timeout=2)

    def test_depth_change_reports_physical_global_depth(self) -> None:
        col = _Collector()
        bm = QueueTestManager(queue_event_sink=col)
        try:
            bm.install_blocker("openai", "block-openai")
            bm.install_blocker("google", "block-google")
            removed = bm.submit_prompt("openai", "remove-me", timeout=5)
            bm.submit_prompt("google", "keep-me", timeout=5)
            self.assertTrue(removed.request_cancel())
            events = col.of_kind(QueueEventKind.QUEUED_JOB_REMOVED)
            self.assertTrue(events)
            self.assertEqual(events[-1].global_depth, 1)
        finally:
            bm.shutdown()

    def test_expiry_event_is_suppressed_when_cancellation_won(self) -> None:
        col = _Collector()
        bm = QueueTestManager(queue_event_sink=col)
        try:
            bm.install_blocker("openai")
            handle = bm.submit_prompt("openai", "race", timeout=5)
            worker = bm._peek_worker("openai")
            entry = worker.jobs.remove_prompt(handle.job_id)
            self.assertIsNotNone(entry)
            self.assertTrue(handle.request_cancel())
            bm._expire_queued_prompt(entry, reason="injected-expiry-race")
            self.assertEqual(
                col.of_kind(QueueEventKind.QUEUE_WAIT_EXPIRED), []
            )
            self.assertTrue(handle.cancelled())
        finally:
            bm.shutdown()


if __name__ == "__main__":
    unittest.main()
