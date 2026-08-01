"""Quarantine recovery: a stopped run must not disable a provider forever.

A Stop whose quiescence could not be PROVEN quarantines the provider so new
prompts are never queued behind a possibly-still-generating tab. Before this
path existed that verdict lasted for the life of the session — one Stop made
the provider reject every later run with "provider is quarantined" until it was
removed and re-added.

Recovery is observation-only: it re-probes the tab and lifts the quarantine
only when the tab is demonstrably idle. Browser-free — the adapter's
``probe_reusable`` seam is stubbed, so the genuine admission/worker machinery
runs without launching Chromium.
"""

from __future__ import annotations

import unittest

from webllm.browser_manager import _Session
from webllm.browser_queue import AdmissionOutcome, BrowserQueueRejectedError

from ._util import QueueTestManager


class _FakePage:
    def __init__(self, closed: bool = False) -> None:
        self._closed = closed

    def is_closed(self) -> bool:
        return self._closed


class _FakeAdapter:
    """Minimal adapter stand-in exposing only what recovery touches."""

    def __init__(self, key: str, reusable: bool = True, raises: bool = False) -> None:
        self.key = key
        self.name = key.title()
        self.reusable = reusable
        self.raises = raises
        self.probes = 0

    def probe_reusable(self, page, policy) -> bool:
        self.probes += 1
        if self.raises:
            raise RuntimeError("probe blew up")
        return self.reusable


def _install_session(bm, provider: str, adapter: _FakeAdapter, closed: bool = False):
    sess = _Session(
        adapter=adapter, context=object(), page=_FakePage(closed), logged_in=True
    )
    with bm._sessions_lock:
        bm._sessions[provider] = sess
    return sess


class TestQuarantineRecovery(unittest.TestCase):
    def test_idle_tab_recovers_and_accepts_work_again(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=True)
            _install_session(bm, "openai", adapter)
            bm._quarantine_provider("openai", reason="interruption-unsafe")

            # Precondition: the reported symptom — every prompt is rejected.
            with self.assertRaises(BrowserQueueRejectedError) as ctx:
                bm.submit_prompt("openai", "blocked")
            self.assertIs(
                ctx.exception.outcome, AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED
            )

            result = bm.recover_provider("openai")
            self.assertTrue(result["recovered"])
            self.assertEqual(result["reason"], "probed-idle")
            self.assertEqual(adapter.probes, 1)
            self.assertFalse(bm.is_quarantined("openai"))

            # The provider is genuinely usable again, not merely un-flagged.
            self.assertEqual(
                bm.submit_prompt("openai", "after").result(timeout=5), "reply:after"
            )
        finally:
            bm.shutdown()

    def test_busy_tab_stays_quarantined(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=False)
            _install_session(bm, "openai", adapter)
            bm._quarantine_provider("openai", reason="interruption-unsafe")

            result = bm.recover_provider("openai")
            self.assertFalse(result["recovered"])
            self.assertEqual(result["reason"], "still-busy")
            self.assertTrue(bm.is_quarantined("openai"))
            with self.assertRaises(BrowserQueueRejectedError):
                bm.submit_prompt("openai", "still-blocked")
        finally:
            bm.shutdown()

    def test_probe_error_does_not_lift_quarantine(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", raises=True)
            _install_session(bm, "openai", adapter)
            bm._quarantine_provider("openai", reason="interruption-unsafe")

            result = bm.recover_provider("openai")
            self.assertFalse(result["recovered"])
            self.assertTrue(result["reason"].startswith("probe-error:"))
            self.assertTrue(bm.is_quarantined("openai"))
        finally:
            bm.shutdown()

    def test_dead_tab_recovers_without_probing(self) -> None:
        """No live tab means nothing can still be generating on it."""
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai")
            _install_session(bm, "openai", adapter, closed=True)
            bm._quarantine_provider("openai", reason="interruption-unsafe")

            result = bm.recover_provider("openai")
            self.assertTrue(result["recovered"])
            self.assertEqual(result["reason"], "no-live-session")
            self.assertEqual(adapter.probes, 0)
            self.assertFalse(bm.is_quarantined("openai"))
        finally:
            bm.shutdown()

    def test_recovering_a_healthy_provider_is_a_noop(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai")
            _install_session(bm, "openai", adapter)
            result = bm.recover_provider("openai")
            self.assertTrue(result["recovered"])
            self.assertEqual(result["reason"], "not-quarantined")
            self.assertEqual(adapter.probes, 0)
        finally:
            bm.shutdown()

    def test_recover_quarantined_reports_every_provider(self) -> None:
        bm = QueueTestManager()
        try:
            good = _FakeAdapter("openai", reusable=True)
            bad = _FakeAdapter("google", reusable=False)
            _install_session(bm, "openai", good)
            _install_session(bm, "google", bad)
            bm._quarantine_provider("openai", reason="interruption-unsafe")
            bm._quarantine_provider("google", reason="interruption-unsafe")
            self.assertEqual(bm.quarantined_providers(), ["google", "openai"])

            outcomes = bm.recover_quarantined()
            self.assertEqual(set(outcomes), {"openai", "google"})
            self.assertTrue(outcomes["openai"]["recovered"])
            self.assertFalse(outcomes["google"]["recovered"])
            # Only the provably-idle one is released.
            self.assertEqual(bm.quarantined_providers(), ["google"])
        finally:
            bm.shutdown()

    def test_recover_quarantined_is_safe_when_nothing_is_quarantined(self) -> None:
        bm = QueueTestManager()
        try:
            self.assertEqual(bm.recover_quarantined(), {})
        finally:
            bm.shutdown()


class TestStaleProviderOwnership(unittest.TestCase):
    """A terminal job that never physically settled keeps owning the tab.

    Provider ownership is released only by physical settlement, so such a job
    blocks every later prompt at the ``mark_running`` claim — which the manager
    used to report to the caller as "Browser job timed out" even though nothing
    ever waited.
    """

    def _leave_stale_owner(self, bm, provider: str) -> str:
        """Register a job, run it, cancel it, and skip physical settlement."""
        job_id, _ = bm._registry.register_deferred(provider)
        bm._registry.mark_running(job_id)
        bm._registry.mark_cancelled(job_id, reason="stopped-by-user")
        self.assertEqual(bm._registry.active_job(provider), job_id.value)
        return job_id.value

    def test_stale_owner_blocks_new_work_with_an_honest_error(self) -> None:
        bm = QueueTestManager()
        try:
            self._leave_stale_owner(bm, "openai")
            handle = bm.submit_prompt("openai", "blocked-by-stale-owner")
            with self.assertRaises(Exception) as ctx:
                handle.result(timeout=5)
            message = str(ctx.exception)
            # The old behaviour reported this as a timeout, which was false.
            self.assertNotIsInstance(ctx.exception, TimeoutError)
            self.assertIn("still busy", message)
            self.assertIn("Recover this model", message)
            # The prompt was genuinely never sent.
            self.assertNotIn("blocked-by-stale-owner", bm.prompts_sent())
        finally:
            bm.shutdown()

    def test_recovery_releases_the_stale_owner_and_unblocks_the_provider(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=True)
            _install_session(bm, "openai", adapter)
            stale = self._leave_stale_owner(bm, "openai")

            # Needs recovery even though nothing was ever quarantined.
            self.assertFalse(bm.is_quarantined("openai"))
            self.assertEqual(bm.blocked_providers(), ["openai"])

            result = bm.recover_provider("openai")
            self.assertTrue(result["recovered"])
            self.assertEqual(result["released_job"], stale)
            self.assertIsNone(bm._registry.active_job("openai"))
            self.assertEqual(bm.blocked_providers(), [])

            # The provider genuinely accepts work again.
            self.assertEqual(
                bm.submit_prompt("openai", "after").result(timeout=5), "reply:after"
            )
        finally:
            bm.shutdown()

    def test_recovery_never_steals_a_tab_from_a_running_job(self) -> None:
        """A RUNNING owner may be mid-Playwright-call; it must not be released."""
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=True)
            _install_session(bm, "openai", adapter)
            job_id, _ = bm._registry.register_deferred("openai")
            bm._registry.mark_running(job_id)

            self.assertIsNone(bm._stale_owner("openai"))
            self.assertEqual(bm.blocked_providers(), [])
            self.assertEqual(bm.recover_quarantined(), {})
            # Ownership is untouched.
            self.assertEqual(bm._registry.active_job("openai"), job_id.value)
        finally:
            bm.shutdown()

    def test_quarantine_recovery_also_clears_stale_ownership(self) -> None:
        """Both symptoms of one stopped run must clear together.

        Lifting the quarantine alone would just move the failure from admission
        ("provider is quarantined") to the running claim ("provider-busy").
        """
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=True)
            _install_session(bm, "openai", adapter)
            self._leave_stale_owner(bm, "openai")
            bm._quarantine_provider("openai", reason="interruption-unsafe")

            outcomes = bm.recover_quarantined()
            self.assertTrue(outcomes["openai"]["recovered"])
            self.assertFalse(bm.is_quarantined("openai"))
            self.assertIsNone(bm._registry.active_job("openai"))
            self.assertEqual(
                bm.submit_prompt("openai", "after").result(timeout=5), "reply:after"
            )
        finally:
            bm.shutdown()

    def test_busy_tab_keeps_its_stale_owner(self) -> None:
        bm = QueueTestManager()
        try:
            adapter = _FakeAdapter("openai", reusable=False)
            _install_session(bm, "openai", adapter)
            stale = self._leave_stale_owner(bm, "openai")

            result = bm.recover_provider("openai")
            self.assertFalse(result["recovered"])
            self.assertEqual(result["reason"], "still-busy")
            self.assertEqual(bm._registry.active_job("openai"), stale)
        finally:
            bm.shutdown()


if __name__ == "__main__":
    unittest.main()
