"""Concurrency control for the filesystem backend.

Model: many readers, one writer — per conversation.

* Readers never block each other and never take filesystem locks: the storage
  format makes lock-free reads safe (manifest replacement is atomic, JSONL is
  append-only, and a torn final line is skipped by the reader).
* Writers take BOTH an in-process readers-writer lock (threads inside this
  orchestrator) and an advisory cross-process lock file (a second process, a
  future multi-agent runner, or a future distributed worker on a shared
  volume).

The lock file is portable (``O_CREAT | O_EXCL`` is atomic on POSIX and
Windows) and self-healing: locks older than ``stale_after`` seconds are
broken, so a crashed writer cannot wedge a conversation forever. PID liveness
probing is deliberately avoided — ``os.kill(pid, 0)`` is not safe on Windows.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LockTimeoutError(OSError):
    """Raised when the cross-process writer lock cannot be acquired in time."""


class ReadWriteLock:
    """In-process readers-writer lock. Writer-preferring: once a writer is
    waiting, new readers queue behind it so the writer cannot starve."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer_active or self._writers_waiting:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._condition:
            self._writer_active = False
            self._condition.notify_all()

    @contextmanager
    def reading(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def writing(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()


class FileLock:
    """Advisory cross-process exclusive lock backed by an ``O_EXCL`` file.

    The file body records owner pid and acquisition time for diagnostics and
    staleness judgment. Not reentrant; hold it briefly (single write batches).
    """

    def __init__(
        self,
        path: Path,
        timeout: float = 10.0,
        poll_interval: float = 0.02,
        stale_after: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self._held = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        body = json.dumps({"pid": os.getpid(), "acquired_at": time.time()}).encode("utf-8")
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, body)
                finally:
                    os.close(fd)
                self._held = True
                return
            except FileExistsError:
                self._break_if_stale()
            except PermissionError:
                # Windows: O_CREAT|O_EXCL against a lock file that another
                # process is concurrently unlinking (delete-pending state)
                # surfaces as EACCES, not EEXIST. It's the same condition —
                # someone else holds/held the lock — so poll, don't crash.
                pass
            if time.monotonic() >= deadline:
                raise LockTimeoutError(
                    f"could not acquire writer lock {self.path} within {self.timeout}s"
                )
            time.sleep(self.poll_interval)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            os.unlink(self.path)
        except OSError:
            pass  # already broken as stale, or the directory was deleted

    def _break_if_stale(self) -> None:
        try:
            age = time.time() - os.stat(self.path).st_mtime
        except OSError:
            return  # holder released between our attempts — retry immediately
        if age > self.stale_after:
            try:
                os.unlink(self.path)  # crashed writer: reclaim
            except OSError:
                pass  # someone else broke it first; the retry loop handles it

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class ConversationLockRegistry:
    """Per-conversation lock coordinator used by every repository.

    ``read(cid)`` — in-process shared access.
    ``write(cid, lock_file)`` — in-process exclusive access plus the advisory
    cross-process file lock next to the conversation's data files.
    """

    def __init__(self, lock_timeout: float = 10.0, stale_after: float = 30.0) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, ReadWriteLock] = {}
        self._lock_timeout = lock_timeout
        self._stale_after = stale_after

    def _lock_for(self, conversation_id: str) -> ReadWriteLock:
        with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = ReadWriteLock()
                self._locks[conversation_id] = lock
            return lock

    @contextmanager
    def read(self, conversation_id: str) -> Iterator[None]:
        with self._lock_for(conversation_id).reading():
            yield

    @contextmanager
    def write(self, conversation_id: str, lock_file: Path) -> Iterator[None]:
        with self._lock_for(conversation_id).writing():
            file_lock = FileLock(
                lock_file, timeout=self._lock_timeout, stale_after=self._stale_after
            )
            with file_lock:
                yield

    def forget(self, conversation_id: str) -> None:
        """Drop the in-process lock of a deleted conversation."""
        with self._guard:
            self._locks.pop(conversation_id, None)
