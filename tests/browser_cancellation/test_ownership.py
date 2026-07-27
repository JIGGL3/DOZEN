"""Ownership tests for the interruption owner check: exact owner, stale job,
queued/running successors, ownership change before click, physical ownership
retained until settlement.
"""

from __future__ import annotations

import unittest

from webllm.browser_cancellation import (
    InterruptionActionResult,
    InterruptionPolicy,
    InterruptionReason,
    InterruptionRequest,
)
from webllm.browser_jobs import JobState

from ._util import CooperativeGate, Gate, InterruptibleBrowserManager, poll_state

_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class _FakeJob:
    def __init__(self, job_id, provider):
        self.job_id = job_id
        self.provider = provider
        self.interruption = InterruptionRequest(job_id=job_id, provider=provider)
        self.cancel_observation = None


class TestOwnerCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager(interruption_policy=_FAST)

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_exact_running_owner_passes(self) -> None:
        gate = Gate(value="x")
        self.bm.gates["run"] = gate
        h = self.bm.submit_prompt("openai", "run")
        self.assertTrue(gate.started.wait(2))
        job = _FakeJob(h.job_id, "openai")
        self.assertTrue(self.bm._interruption_owner_ok(job))
        gate.release.set()
        h.result(timeout=5)

    def test_never_owner_fails(self) -> None:
        # A job id that never started / is not the provider's active owner.
        reg = self.bm._registry
        jid = reg.register("openai")
        job = _FakeJob(jid.value, "openai")
        self.assertFalse(self.bm._interruption_owner_ok(job))

    def test_settled_owner_fails(self) -> None:
        h = self.bm.submit_prompt("openai", "q")
        h.result(timeout=5)  # completes and settles
        job = _FakeJob(h.job_id, "openai")
        self.assertFalse(self.bm._interruption_owner_ok(job))

    def test_successor_running_makes_prior_stale(self) -> None:
        first = Gate(value="1")
        self.bm.gates["first"] = first
        h1 = self.bm.submit_prompt("openai", "first")
        self.assertTrue(first.started.wait(2))
        with self.assertRaises(TimeoutError):
            h1.result(timeout=0.2)          # h1 times out, still running
        first.release.set()
        self.assertTrue(poll_state(self.bm, h1.job_id, JobState.ABANDONED))
        # Successor now owns the provider.
        second = Gate(value="2")
        self.bm.gates["second"] = second
        h2 = self.bm.submit_prompt("openai", "second")
        self.assertTrue(second.started.wait(2))
        # h1 is stale: an owner check for it must fail (never stop h2).
        stale = _FakeJob(h1.job_id, "openai")
        self.assertFalse(self.bm._interruption_owner_ok(stale))
        # h2 is the true owner.
        live = _FakeJob(h2.job_id, "openai")
        self.assertTrue(self.bm._interruption_owner_ok(live))
        second.release.set()
        h2.result(timeout=5)

    def test_physical_ownership_retained_until_settlement(self) -> None:
        gate = CooperativeGate(value="never")
        self.bm.gates["coop"] = gate
        h = self.bm.submit_prompt("openai", "coop")
        self.assertTrue(gate.started.wait(2))
        self.assertTrue(h.request_cancel())
        # Ownership is still held right after the caller is released.
        self.assertEqual(self.bm.active_browser_job("openai"), h.job_id)
        self.assertTrue(poll_state(self.bm, h.job_id, JobState.ABANDONED))
        # Released only once the physical op settles.
        self.assertIsNone(self.bm.active_browser_job("openai"))
        self.assertTrue(h.snapshot().physical_settled)

    def test_perform_stop_action_without_session_is_unsupported(self) -> None:
        # The REAL _perform_stop_action (no override) with no live session must
        # report UNSUPPORTED and never raise.
        import tempfile

        from webllm.browser_manager import BrowserManager, _Job

        real = BrowserManager(profiles_dir=tempfile.mkdtemp(), headless=True)
        try:
            reg = real._registry
            jid = reg.register("openai")
            reg.mark_running(jid)
            job = _Job(fn=lambda: "x", job_id=jid.value, provider="openai")
            job.interruption = InterruptionRequest(job_id=jid.value, provider="openai")
            job.interruption.request(InterruptionReason.CANCELLED, "handle")
            result = real._perform_stop_action(job)
            self.assertIs(result, result)  # returned, did not raise
            self.assertEqual(result.status.value, "unsupported")
        finally:
            real.shutdown()


if __name__ == "__main__":
    unittest.main()
