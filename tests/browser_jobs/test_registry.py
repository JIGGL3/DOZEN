"""Registry tests: registration, snapshots, provider ownership, owner-checked
release, bounded retention, eviction safety, event-outside-lock, callback
failure isolation, and multi-provider independence.
"""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_jobs import (
    BrowserJobEvent,
    BrowserJobRegistry,
    JobState,
)


class TestRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = BrowserJobRegistry()

    def test_register_creates_queued_snapshot(self) -> None:
        jid = self.reg.register("openai", correlation="run-7")
        snap = self.reg.snapshot(jid)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.state, JobState.QUEUED.value)
        self.assertEqual(snap.provider, "openai")
        self.assertEqual(snap.correlation, "run-7")
        self.assertEqual(snap.job_id, jid.value)
        self.assertIsNotNone(snap.queued_at)

    def test_snapshot_unknown_returns_none(self) -> None:
        self.assertIsNone(self.reg.snapshot("nope"))
        self.assertIsNone(self.reg.state("nope"))

    def test_unpublished_deferred_registration_can_be_rolled_back(self) -> None:
        jid, _event = self.reg.register_deferred("openai")
        self.assertTrue(self.reg.rollback_deferred(jid))
        self.assertIsNone(self.reg.snapshot(jid))
        self.assertFalse(self.reg.rollback_deferred(jid))


class TestOwnership(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = BrowserJobRegistry()

    def test_running_claims_provider_slot(self) -> None:
        jid = self.reg.register("openai")
        self.assertIsNone(self.reg.active_job("openai"))
        self.reg.mark_running(jid)
        self.assertEqual(self.reg.active_job("openai"), jid.value)

    def test_completion_releases_slot(self) -> None:
        jid = self.reg.register("openai")
        self.reg.mark_running(jid)
        self.reg.mark_completed(jid)
        self.assertIsNone(self.reg.active_job("openai"))

    def test_owner_checked_release_is_noop_for_non_owner(self) -> None:
        a = self.reg.register("openai")
        self.reg.mark_running(a)
        b = self.reg.register("openai")  # queued, not owner
        self.assertFalse(self.reg.release_provider(b))
        self.assertEqual(self.reg.active_job("openai"), a.value)  # untouched

    def test_unsettled_job_blocks_successor_ownership(self) -> None:
        a = self.reg.register("openai")
        self.reg.mark_running(a)
        self.reg.mark_timed_out(a)
        b = self.reg.register("openai")
        blocked = self.reg.mark_running(b)
        self.assertFalse(blocked.transitioned)
        self.assertEqual(self.reg.active_job("openai"), a.value)
        self.reg.mark_physical_settled(a)
        self.assertTrue(self.reg.mark_running(b).transitioned)
        self.assertEqual(self.reg.active_job("openai"), b.value)
        # A's stale release must NOT clear B's ownership.
        self.assertFalse(self.reg.release_provider(a))
        self.assertEqual(self.reg.active_job("openai"), b.value)


class TestRetention(unittest.TestCase):
    def test_terminal_history_is_bounded(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=10)
        ids = []
        for _ in range(200):
            jid = reg.register("openai")
            reg.mark_running(jid)
            reg.mark_completed(jid)
            ids.append(jid)
        self.assertLessEqual(len(reg), 10)
        # Oldest evicted, newest retained.
        self.assertIsNone(reg.snapshot(ids[0]))
        self.assertIsNotNone(reg.snapshot(ids[-1]))

    def test_active_jobs_never_evicted(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=5)
        running = reg.register("openai")
        reg.mark_running(running)
        for _ in range(100):
            jid = reg.register("google")
            reg.mark_running(jid)
            reg.mark_completed(jid)
        # The still-RUNNING job survives regardless of terminal churn.
        self.assertEqual(reg.state(running), JobState.RUNNING)
        self.assertIsNotNone(reg.snapshot(running))

    def test_settling_states_not_evicted_until_terminal(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=3)
        timed = reg.register("openai")
        reg.mark_running(timed)
        reg.mark_timed_out(timed)  # not fully terminal yet
        for _ in range(50):
            jid = reg.register("google")
            reg.mark_running(jid)
            reg.mark_completed(jid)
        self.assertIsNotNone(reg.snapshot(timed))  # survives while TIMED_OUT
        reg.mark_abandoned(timed)                  # now fully terminal

    def test_age_policy_evicts_on_read_but_not_before_settlement(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=10, max_terminal_age_s=0.01)
        finished = reg.register("google")
        reg.mark_running(finished)
        reg.mark_completed(finished)
        unsettled = reg.register("openai")
        reg.mark_running(unsettled)
        reg.mark_timed_out(unsettled)
        time.sleep(0.02)
        self.assertIsNone(reg.snapshot(finished))
        self.assertIsNotNone(reg.snapshot(unsettled))


class TestEvents(unittest.TestCase):
    def test_events_emitted_for_lifecycle(self) -> None:
        events: list[BrowserJobEvent] = []
        reg = BrowserJobRegistry(event_sink=events.append)
        jid = reg.register("openai")
        reg.mark_running(jid)
        reg.mark_completed(jid)
        kinds = [(e.old_state, e.new_state) for e in events]
        self.assertIn((None, "queued"), kinds)
        self.assertIn(("queued", "running"), kinds)
        self.assertIn(("running", "completed"), kinds)

    def test_callback_runs_outside_lock(self) -> None:
        """The sink can safely call back INTO the registry without deadlock."""
        reg = BrowserJobRegistry()
        seen: list[str] = []

        def sink(ev: BrowserJobEvent) -> None:
            # Re-entrant read: would deadlock if emitted under the lock.
            seen.append(reg.state(ev.job_id).value if reg.state(ev.job_id) else "?")

        reg._event_sink = sink  # install after construction for clarity
        jid = reg.register("openai")
        reg.mark_running(jid)
        self.assertTrue(seen)  # no deadlock, callback observed live state

    def test_callback_failure_is_isolated(self) -> None:
        def bad_sink(ev: BrowserJobEvent) -> None:
            raise RuntimeError("observer blew up")

        reg = BrowserJobRegistry(event_sink=bad_sink)
        jid = reg.register("openai")            # sink raises here
        self.assertTrue(reg.mark_running(jid).transitioned)  # job unaffected
        self.assertEqual(reg.state(jid), JobState.RUNNING)

    def test_late_result_discard_emits_event(self) -> None:
        events: list[BrowserJobEvent] = []
        reg = BrowserJobRegistry(event_sink=events.append)
        jid = reg.register("openai")
        reg.mark_running(jid)
        reg.mark_timed_out(jid)
        events.clear()
        reg.record_late_result_discarded(jid, reason="late-result")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].late_result_discarded)
        self.assertTrue(reg.snapshot(jid).late_result_discarded)


class TestMultiProvider(unittest.TestCase):
    def test_providers_are_independent(self) -> None:
        reg = BrowserJobRegistry()
        a = reg.register("openai")
        b = reg.register("google")
        reg.mark_running(a)
        reg.mark_running(b)
        self.assertEqual(reg.active_job("openai"), a.value)
        self.assertEqual(reg.active_job("google"), b.value)
        reg.mark_timed_out(a)  # A times out
        # B is entirely unaffected.
        self.assertEqual(reg.active_job("google"), b.value)
        self.assertTrue(reg.mark_completed(b).transitioned)
        self.assertEqual(reg.state(b), JobState.COMPLETED)

    def test_concurrent_registration_is_safe(self) -> None:
        reg = BrowserJobRegistry(max_terminal_jobs=10_000)
        ids: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            local = []
            for _ in range(200):
                jid = reg.register("openai")
                local.append(jid.value)
            with lock:
                ids.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(ids), 1600)
        self.assertEqual(len(set(ids)), 1600)  # all unique, none lost


if __name__ == "__main__":
    unittest.main()
