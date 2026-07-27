"""InterruptionRequest + CancellationObservation unit tests:
idempotency, exactly-once begin_action, first-reason-wins arbitration,
composite external/internal observation, fail-safe callbacks, identity.
"""

from __future__ import annotations

import threading
import unittest

from webllm.browser_cancellation import (
    CancellationObservation,
    InterruptionOutcome,
    InterruptionReason,
    InterruptionRequest,
)
from webllm.browser_jobs import BrowserJobId


def _request(**kw) -> InterruptionRequest:
    return InterruptionRequest(job_id=BrowserJobId.new().value, provider="openai", **kw)


class TestInterruptionRequest(unittest.TestCase):
    def test_starts_unrequested(self) -> None:
        req = _request()
        self.assertFalse(req.is_requested())
        self.assertIs(req.outcome, InterruptionOutcome.NOT_REQUESTED)
        snap = req.snapshot()
        self.assertFalse(snap.requested)
        self.assertEqual(snap.attempt_count, 0)

    def test_first_request_wins_and_is_idempotent(self) -> None:
        req = _request()
        self.assertTrue(req.request(InterruptionReason.CANCELLED, "handle"))
        # A second request (even a different reason) does not overwrite.
        self.assertFalse(req.request(InterruptionReason.CALLER_TIMEOUT, "other"))
        self.assertIs(req.reason, InterruptionReason.CANCELLED)
        self.assertEqual(req.snapshot().source, "handle")

    def test_require_physical_can_be_upgraded_but_reason_is_stable(self) -> None:
        req = _request()
        req.request(InterruptionReason.CANCELLED, "handle", require_physical=False)
        self.assertFalse(req.snapshot().physical_interruption_required)
        req.request(InterruptionReason.CANCELLED, "handle", require_physical=True)
        self.assertTrue(req.snapshot().physical_interruption_required)
        req.mark_required()
        self.assertTrue(req.snapshot().physical_interruption_required)

    def test_begin_action_returns_true_at_most_once(self) -> None:
        req = _request()
        req.request(InterruptionReason.CANCELLED, "handle")
        self.assertTrue(req.begin_action())
        self.assertFalse(req.begin_action())
        self.assertFalse(req.begin_action())
        self.assertEqual(req.snapshot().attempt_count, 1)

    def test_begin_action_is_thread_safe_exactly_once(self) -> None:
        for _ in range(200):
            req = _request()
            req.request(InterruptionReason.CANCELLED, "handle")
            winners: list[bool] = []
            lock = threading.Lock()
            barrier = threading.Barrier(8)

            def contend() -> None:
                barrier.wait()
                won = req.begin_action()
                with lock:
                    winners.append(won)

            threads = [threading.Thread(target=contend) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(sum(1 for w in winners if w), 1)
            self.assertEqual(req.snapshot().attempt_count, 1)

    def test_complete_records_outcome_and_bounded_diagnostic(self) -> None:
        req = _request()
        req.request(InterruptionReason.CANCELLED, "handle")
        req.begin_action()
        req.complete(InterruptionOutcome.STOPPED, "line\nz" * 200)
        snap = req.snapshot()
        self.assertEqual(snap.outcome, InterruptionOutcome.STOPPED.value)
        self.assertIsNotNone(snap.completed_at)
        self.assertNotIn("\n", snap.diagnostic)
        self.assertLessEqual(len(snap.diagnostic), req.policy.max_diagnostic_length)

    def test_snapshot_never_carries_content(self) -> None:
        req = _request()
        req.request(InterruptionReason.CANCELLED, "handle")
        d = req.snapshot().to_dict()
        self.assertNotIn("prompt", d)
        self.assertNotIn("response", d)


class TestCancellationObservation(unittest.TestCase):
    def _obs(self, external=None) -> tuple[CancellationObservation, InterruptionRequest]:
        req = _request()
        obs = CancellationObservation(
            job_id=req.job_id, provider="openai", request=req, external=external
        )
        return obs, req

    def test_reports_internal_request(self) -> None:
        obs, req = self._obs()
        self.assertFalse(obs())
        req.request(InterruptionReason.CANCELLED, "handle")
        self.assertTrue(obs())

    def test_external_trip_records_cancel_exactly_once(self) -> None:
        flag = {"v": False}
        obs, req = self._obs(external=lambda: flag["v"])
        self.assertFalse(obs())
        self.assertFalse(req.is_requested())
        flag["v"] = True
        self.assertTrue(obs())
        self.assertTrue(req.is_requested())
        self.assertIs(req.reason, InterruptionReason.CANCELLED)
        # Repeated polling does not create a second request / change the reason.
        self.assertTrue(obs())
        self.assertEqual(req.snapshot().source, "external-token")

    def test_external_callback_exception_fails_safe(self) -> None:
        def boom() -> bool:
            raise RuntimeError("bad predicate")

        obs, req = self._obs(external=boom)
        self.assertFalse(obs())  # exception treated as "not cancelled"
        self.assertFalse(req.is_requested())

    def test_identity_available_for_owner_checks(self) -> None:
        obs, req = self._obs()
        self.assertEqual(obs.job_id, req.job_id)
        self.assertEqual(obs.provider, "openai")
        self.assertIs(obs.request, req)


if __name__ == "__main__":
    unittest.main()
