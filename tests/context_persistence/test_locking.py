"""Concurrency: many readers / single writer, cross-process lock files,
stale-lock recovery, and end-to-end thread safety of the repositories."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import (
    FileLock,
    LockTimeoutError,
    ReadWriteLock,
)
from dozen.context.domain.enums import MessageRole

from .base import PersistenceTestCase, make_message


class TestReadWriteLock(unittest.TestCase):
    def test_readers_share(self) -> None:
        lock = ReadWriteLock()
        entered = threading.Barrier(2, timeout=5)

        def reader() -> None:
            with lock.reading():
                entered.wait()  # both readers inside simultaneously or Barrier times out

        threads = [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertFalse(any(t.is_alive() for t in threads))

    def test_writer_excludes_readers_and_writers(self) -> None:
        lock = ReadWriteLock()
        active = []
        violations = []

        def worker(kind: str) -> None:
            ctx = lock.writing() if kind == "w" else lock.reading()
            with ctx:
                if kind == "w" and active:
                    violations.append(f"writer entered alongside {active}")
                if kind == "r" and "w" in active:
                    violations.append("reader entered alongside a writer")
                active.append(kind)
                time.sleep(0.005)
                active.remove(kind)

        threads = [threading.Thread(target=worker, args=("w" if i % 3 == 0 else "r",))
                   for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(violations, [])


class TestFileLock(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-lock-"))
        self.lock_path = self.tmp / ".writer.lock"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_acquire_release_cycle(self) -> None:
        lock = FileLock(self.lock_path, timeout=1.0)
        with lock:
            self.assertTrue(self.lock_path.exists())
        self.assertFalse(self.lock_path.exists())

    def test_second_holder_times_out(self) -> None:
        with FileLock(self.lock_path, timeout=1.0, stale_after=60.0):
            second = FileLock(self.lock_path, timeout=0.3, stale_after=60.0)
            started = time.monotonic()
            with self.assertRaises(LockTimeoutError):
                second.acquire()
            self.assertLess(time.monotonic() - started, 5.0)

    def test_stale_lock_is_broken(self) -> None:
        """A crashed writer's lock (old mtime, never released) is reclaimed."""
        self.lock_path.write_text('{"pid": 999999, "acquired_at": 0}', encoding="utf-8")
        import os
        ancient = time.time() - 3600
        os.utime(self.lock_path, (ancient, ancient))
        lock = FileLock(self.lock_path, timeout=1.0, stale_after=30.0)
        lock.acquire()  # must not raise
        lock.release()

    def test_release_without_acquire_is_harmless(self) -> None:
        FileLock(self.lock_path).release()  # no exception, no file


class TestConcurrentRepositoryAccess(PersistenceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = self.create_conversation()
        self.cid = self.manifest.conversation.id

    def test_parallel_appends_all_land_intact(self) -> None:
        thread_count, per_thread = 8, 25
        errors: list[str] = []

        def appender(worker: int) -> None:
            for i in range(per_thread):
                msg = make_message(self.cid, content=f"w{worker}-m{i}")
                result = self.messages.append_messages(self.cid, [msg])
                if not result.ok:
                    errors.append(result.error.message)

        threads = [threading.Thread(target=appender, args=(w,)) for w in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(self.messages.count_messages(self.cid).unwrap(), thread_count * per_thread)
        # Zero interleaving damage: every line parses.
        report = self.messages.integrity_report(self.cid).unwrap()
        self.assertFalse(report.corrupt)
        contents = {m.content for m in self.messages.read_messages(self.cid, limit=10_000).unwrap()}
        self.assertEqual(len(contents), thread_count * per_thread)

    def test_readers_see_valid_state_during_writes(self) -> None:
        stop = threading.Event()
        problems: list[str] = []

        def writer() -> None:
            i = 0
            while not stop.is_set():
                self.messages.append_messages(
                    self.cid, [make_message(self.cid, content=f"live-{i}")]
                )
                i += 1

        def reader() -> None:
            while not stop.is_set():
                result = self.messages.read_messages(self.cid, limit=10_000)
                if not result.ok:
                    problems.append(result.error.message)
                    continue
                for msg in result.unwrap():
                    if msg.role is not MessageRole.USER or not msg.content.startswith("live-"):
                        problems.append(f"torn read: {msg.content!r}")

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        writer_thread.start()
        for t in reader_threads:
            t.start()
        time.sleep(0.7)
        stop.set()
        writer_thread.join(timeout=10)
        for t in reader_threads:
            t.join(timeout=10)
        self.assertEqual(problems, [])

    def test_concurrent_manifest_updates_are_serialized(self) -> None:
        """Optimistic concurrency under real threads: exactly one writer wins
        each version, no update is silently lost."""
        wins, conflicts = [], []
        lock = threading.Lock()

        def updater(worker: int) -> None:
            for _ in range(10):
                current = self.conversations.read_manifest(self.cid)
                if not current.ok:
                    continue
                manifest = current.unwrap()
                manifest.conversation.metadata.values["user.last_writer"] = worker
                result = self.conversations.update_manifest(
                    manifest, expected_version=manifest.conversation.version
                )
                with lock:
                    (wins if result.ok else conflicts).append(worker)

        threads = [threading.Thread(target=updater, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        final = self.conversations.read_manifest(self.cid).unwrap()
        # Version advanced exactly once per successful update — none lost.
        self.assertEqual(final.conversation.version, 1 + len(wins))
        self.assertGreater(len(wins), 0)


if __name__ == "__main__":
    unittest.main()
