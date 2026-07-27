"""ProviderJobQueue unit tests: FIFO, capacity, removal, expiry, close/wake,
snapshot, no tombstone growth, stable ordering."""

from __future__ import annotations

import threading
import time
import unittest

from webllm.browser_queue import ProviderJobQueue, QueueEntry, QueuePolicy


class _FakeJob:
    def __init__(self, job_id: str, provider: str = "openai") -> None:
        self.job_id = job_id
        self.provider = provider


def _prompt(job_id: str, *, deadline: float = None, enq: float = None) -> QueueEntry:
    now = time.monotonic()
    return QueueEntry(
        job=_FakeJob(job_id),
        is_prompt=True,
        job_id=job_id,
        enqueued_mono=enq if enq is not None else now,
        deadline_mono=deadline,
    )


class TestFifoAndDepth(unittest.TestCase):
    def setUp(self) -> None:
        self.q = ProviderJobQueue("openai", QueuePolicy())

    def test_enqueue_dequeue_fifo(self) -> None:
        for i in range(5):
            self.q.enqueue_prompt(_prompt(f"{i:032x}"))
        self.assertEqual(self.q.prompt_depth(), 5)
        got = [self.q.get().job_id for _ in range(5)]
        self.assertEqual(got, [f"{i:032x}" for i in range(5)])
        self.assertEqual(self.q.prompt_depth(), 0)

    def test_control_capacity_enforced(self) -> None:
        cap = self.q.policy.max_queued_control_jobs_per_provider
        for _ in range(cap):
            self.assertTrue(self.q.enqueue_control(object()))
        self.assertFalse(self.q.enqueue_control(object()))  # bound reached
        self.assertEqual(self.q.control_depth(), cap)

    def test_prompt_capacity_is_enforced_by_physical_queue(self) -> None:
        q = ProviderJobQueue(
            "openai",
            QueuePolicy(
                max_queued_prompts_per_provider=1,
                max_total_queued_prompts=1,
            ),
        )
        q.enqueue_prompt(_prompt("a" * 32))
        with self.assertRaises(OverflowError):
            q.enqueue_prompt(_prompt("b" * 32))
        self.assertEqual(q.prompt_depth(), 1)

    def test_unpublished_prompt_cannot_be_claimed(self) -> None:
        entry = _prompt("a" * 32)
        entry.claimable = False
        self.q.enqueue_prompt(entry)
        claimed: list[QueueEntry | None] = []
        thread = threading.Thread(target=lambda: claimed.append(self.q.get()))
        thread.start()
        thread.join(timeout=0.05)
        self.assertTrue(thread.is_alive())
        self.assertTrue(self.q.publish_prompt(entry.job_id))
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(claimed, [entry])

    def test_prompt_and_control_interleave_in_fifo(self) -> None:
        self.q.enqueue_prompt(_prompt("a" * 32))
        self.q.enqueue_control("ctrl")
        self.q.enqueue_prompt(_prompt("b" * 32))
        self.assertEqual(self.q.prompt_depth(), 2)
        self.assertEqual(self.q.control_depth(), 1)
        order = [self.q.get() for _ in range(3)]
        self.assertEqual(order[0].job_id, "a" * 32)
        self.assertFalse(order[1].is_prompt)
        self.assertEqual(order[2].job_id, "b" * 32)


class TestRemovalAndExpiry(unittest.TestCase):
    def setUp(self) -> None:
        self.q = ProviderJobQueue("openai", QueuePolicy())

    def test_remove_by_job_id_preserves_order(self) -> None:
        ids = [f"{i:032x}" for i in range(5)]
        for j in ids:
            self.q.enqueue_prompt(_prompt(j))
        removed = self.q.remove_prompt(ids[2])  # middle
        self.assertIsNotNone(removed)
        self.assertEqual(removed.job_id, ids[2])
        remaining = [self.q.get().job_id for _ in range(4)]
        self.assertEqual(remaining, [ids[0], ids[1], ids[3], ids[4]])

    def test_remove_unknown_returns_none(self) -> None:
        self.q.enqueue_prompt(_prompt("a" * 32))
        self.assertIsNone(self.q.remove_prompt("f" * 32))
        self.assertEqual(self.q.prompt_depth(), 1)

    def test_collect_expired_only_expired_prompts(self) -> None:
        now = time.monotonic()
        self.q.enqueue_prompt(_prompt("a" * 32, deadline=now - 1))  # expired
        self.q.enqueue_prompt(_prompt("b" * 32, deadline=now + 100))  # fresh
        self.q.enqueue_control("ctrl")  # never expired
        expired = self.q.collect_expired(now)
        self.assertEqual([e.job_id for e in expired], ["a" * 32])
        self.assertEqual(self.q.prompt_depth(), 1)
        self.assertEqual(self.q.control_depth(), 1)

    def test_no_tombstone_growth_under_repeated_removal(self) -> None:
        # Physically removing cancelled entries leaves ZERO residue behind a
        # (conceptually) blocked worker: depth tracks live entries exactly.
        for round_ in range(200):
            jid = f"{round_:032x}"
            self.q.enqueue_prompt(_prompt(jid))
            self.assertEqual(self.q.prompt_depth(), 1)
            self.assertIsNotNone(self.q.remove_prompt(jid))
            self.assertEqual(self.q.prompt_depth(), 0)
        # Internal storage is empty (no tombstones accumulated).
        self.assertEqual(len(self.q._entries), 0)


class TestCloseAndWake(unittest.TestCase):
    def test_get_blocks_then_wakes_on_enqueue(self) -> None:
        q = ProviderJobQueue("openai", QueuePolicy())
        out: list = []

        def worker() -> None:
            out.append(q.get())

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)  # worker is blocked in get()
        self.assertFalse(out)
        q.enqueue_prompt(_prompt("a" * 32))
        t.join(timeout=2)
        self.assertFalse(t.is_alive())
        self.assertEqual(out[0].job_id, "a" * 32)

    def test_close_wakes_blocked_worker_with_sentinel(self) -> None:
        q = ProviderJobQueue("openai", QueuePolicy())
        out: list = []

        def worker() -> None:
            out.append(q.get())

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        q.close()
        t.join(timeout=2)
        self.assertFalse(t.is_alive())
        self.assertIsNone(out[0])  # deterministic shutdown sentinel

    def test_enqueue_after_close_raises(self) -> None:
        q = ProviderJobQueue("openai", QueuePolicy())
        q.close()
        with self.assertRaises(RuntimeError):
            q.enqueue_prompt(_prompt("a" * 32))

    def test_drain_returns_all_and_empties(self) -> None:
        q = ProviderJobQueue("openai", QueuePolicy())
        q.enqueue_prompt(_prompt("a" * 32))
        q.enqueue_control("c")
        drained = q.drain()
        self.assertEqual(len(drained), 2)
        self.assertEqual(q.prompt_depth(), 0)
        self.assertEqual(q.control_depth(), 0)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_fields(self) -> None:
        q = ProviderJobQueue("openai", QueuePolicy())
        q.enqueue_prompt(_prompt("a" * 32, enq=time.monotonic() - 2.0))
        q.enqueue_control("c")
        snap = q.snapshot(
            global_queued_depth=1, quarantined=True, running_job_id="b" * 32
        )
        self.assertEqual(snap.provider, "openai")
        self.assertEqual(snap.prompt_depth, 1)
        self.assertEqual(snap.control_depth, 1)
        self.assertTrue(snap.quarantined)
        self.assertEqual(snap.running_job_id, "b" * 32)
        self.assertGreaterEqual(snap.oldest_queued_age_s, 1.5)


if __name__ == "__main__":
    unittest.main()
