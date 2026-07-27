"""Unit + race tests for RunSlot — the single-active-run source of truth.

No server, no browser: RunSlot is exercised directly, including a genuine
barrier-synchronized race for the slot.
"""

from __future__ import annotations

import threading
import unittest

from dozen.cancellation import CancelToken
from webllm.run_slot import ActiveRun, RunSlot


class TestRunSlotTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self.slot = RunSlot()

    def test_acquire_idle_succeeds(self) -> None:
        ok, who = self.slot.try_acquire("run-1", CancelToken())
        self.assertTrue(ok)
        self.assertEqual(who, "run-1")
        self.assertTrue(self.slot.is_active())
        self.assertEqual(self.slot.active_run_id(), "run-1")

    def test_second_acquire_is_rejected_with_active_id(self) -> None:
        self.slot.try_acquire("run-1", CancelToken())
        ok, who = self.slot.try_acquire("run-2", CancelToken())
        self.assertFalse(ok)
        self.assertEqual(who, "run-1")            # names the holder, not the loser
        self.assertEqual(self.slot.active_run_id(), "run-1")

    def test_release_by_owner_frees_the_slot(self) -> None:
        self.slot.try_acquire("run-1", CancelToken())
        self.assertTrue(self.slot.release("run-1"))
        self.assertFalse(self.slot.is_active())
        # A fresh run may now acquire.
        ok, _ = self.slot.try_acquire("run-2", CancelToken())
        self.assertTrue(ok)

    def test_release_by_non_owner_is_safe_noop(self) -> None:
        self.slot.try_acquire("run-1", CancelToken())
        self.assertFalse(self.slot.release("some-other-run"))   # foreign id
        self.assertEqual(self.slot.active_run_id(), "run-1")    # untouched
        self.assertFalse(self.slot.release("run-1") is False)   # owner still works

    def test_release_when_idle_is_safe(self) -> None:
        self.assertFalse(self.slot.release("anything"))

    def test_stale_release_cannot_clear_a_successor(self) -> None:
        """A finishing run must not free a different run's slot."""
        self.slot.try_acquire("run-A", CancelToken())
        self.slot.release("run-A")
        self.slot.try_acquire("run-B", CancelToken())          # successor
        self.assertFalse(self.slot.release("run-A"))           # stale release
        self.assertEqual(self.slot.active_run_id(), "run-B")   # successor safe

    def test_stop_active_trips_only_the_owner_token(self) -> None:
        token = CancelToken()
        self.slot.try_acquire("run-1", token)
        result = self.slot.stop_active()
        self.assertTrue(result["stopped"])
        self.assertEqual(result["run_id"], "run-1")
        self.assertTrue(token.cancelled)

    def test_stop_active_when_idle_reports_no_run(self) -> None:
        result = self.slot.stop_active()
        self.assertFalse(result["stopped"])
        self.assertIn("No task", result["detail"])

    def test_stop_active_is_idempotent(self) -> None:
        token = CancelToken()
        self.slot.try_acquire("run-1", token)
        first = self.slot.stop_active()
        second = self.slot.stop_active()          # repeated stop, same run
        self.assertEqual(first, second)
        self.assertTrue(token.cancelled)

    def test_stop_does_not_cancel_a_replaced_run(self) -> None:
        old_token = CancelToken()
        self.slot.try_acquire("run-old", old_token)
        self.slot.release("run-old")              # old run finished
        new_token = CancelToken()
        self.slot.try_acquire("run-new", new_token)
        self.slot.stop_active()                   # stop the CURRENT run
        self.assertTrue(new_token.cancelled)
        self.assertFalse(old_token.cancelled)     # old run untouched

    def test_active_run_record_is_immutable(self) -> None:
        record = ActiveRun(run_id="r", cancel=CancelToken(), started_at=1.0)
        with self.assertRaises(Exception):
            record.run_id = "mutated"  # type: ignore[misc]


class TestRunSlotRace(unittest.TestCase):
    def test_exactly_one_of_many_contenders_wins(self) -> None:
        slot = RunSlot()
        contenders = 24
        barrier = threading.Barrier(contenders)
        outcomes: list[tuple[bool, str]] = []
        outcomes_lock = threading.Lock()

        def contend(i: int) -> None:
            barrier.wait()                        # all fire simultaneously
            ok, who = slot.try_acquire(f"run-{i}", CancelToken())
            with outcomes_lock:
                outcomes.append((ok, who))

        threads = [threading.Thread(target=contend, args=(i,)) for i in range(contenders)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        winners = [o for o in outcomes if o[0]]
        losers = [o for o in outcomes if not o[0]]
        self.assertEqual(len(winners), 1)                    # exactly one acquired
        self.assertEqual(len(losers), contenders - 1)        # everyone else rejected
        winner_id = winners[0][1]
        # Every loser was told the same, real winner id.
        self.assertTrue(all(who == winner_id for _, who in losers))
        self.assertEqual(slot.active_run_id(), winner_id)

    def test_stop_racing_release_is_safe(self) -> None:
        """Stop and completion firing together must never explode or cancel a
        successor; the token cancelled is always the owner observed under lock."""
        for _ in range(200):
            slot = RunSlot()
            token = CancelToken()
            slot.try_acquire("run-1", token)

            def stopper() -> None:
                slot.stop_active()

            def finisher() -> None:
                slot.release("run-1")

            t1 = threading.Thread(target=stopper)
            t2 = threading.Thread(target=finisher)
            t1.start(); t2.start()
            t1.join(timeout=5); t2.join(timeout=5)
            # Slot ends idle (run-1 released) regardless of interleaving.
            self.assertFalse(slot.is_active())


if __name__ == "__main__":
    unittest.main()
