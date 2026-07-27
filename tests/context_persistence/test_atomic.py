"""AtomicFileWriter: replacement is all-or-nothing, failures leave no debris."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dozen.context.adapters.filesystem import AtomicFileWriter
from dozen.context.domain.enums import ErrorCode


class TestAtomicFileWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-atomic-"))
        self.writer = AtomicFileWriter(fsync=False)
        self.target = self.tmp / "manifest.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_new_file(self) -> None:
        result = self.writer.write_text(self.target, '{"v": 1}')
        self.assertTrue(result.ok)
        self.assertEqual(self.target.read_text(encoding="utf-8"), '{"v": 1}')

    def test_replaces_existing_content(self) -> None:
        self.writer.write_text(self.target, "old")
        result = self.writer.write_text(self.target, "new")
        self.assertTrue(result.ok)
        self.assertEqual(self.target.read_text(encoding="utf-8"), "new")

    def test_creates_missing_parent_directories(self) -> None:
        nested = self.tmp / "a" / "b" / "file.json"
        self.assertTrue(self.writer.write_text(nested, "x").ok)
        self.assertEqual(nested.read_text(encoding="utf-8"), "x")

    def test_failed_write_keeps_original_and_removes_temp(self) -> None:
        self.writer.write_text(self.target, "original")
        with mock.patch("os.replace", side_effect=OSError(28, "No space left on device")):
            result = self.writer.write_text(self.target, "would be lost")
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.IO_ERROR)
        self.assertEqual(result.error.details.get("reason"), "disk_full")
        # Original untouched…
        self.assertEqual(self.target.read_text(encoding="utf-8"), "original")
        # …and no orphaned temp file remains.
        leftovers = [p for p in self.tmp.iterdir() if p.name != self.target.name]
        self.assertEqual(leftovers, [])

    def test_unwritable_parent_reports_failure(self) -> None:
        bogus = self.tmp / "not-a-dir"
        bogus.write_text("i am a file", encoding="utf-8")
        result = self.writer.write_text(bogus / "child.json", "x")
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.IO_ERROR)

    def test_fsync_enabled_still_writes(self) -> None:
        durable = AtomicFileWriter(fsync=True)
        self.assertTrue(durable.write_bytes(self.target, b"durable").ok)
        self.assertEqual(self.target.read_bytes(), b"durable")

    def test_utf8_content_round_trips(self) -> None:
        text = "emoji 🚀 and ünïcode — preserved"
        self.writer.write_text(self.target, text)
        self.assertEqual(self.target.read_text(encoding="utf-8"), text)

    def test_interrupted_write_before_rename_is_invisible(self) -> None:
        """A crash while writing the temp file must not disturb the target."""
        self.writer.write_text(self.target, "stable")

        real_fdopen = os.fdopen

        def exploding_fdopen(fd: int, *args: object, **kwargs: object):
            handle = real_fdopen(fd, *args, **kwargs)  # type: ignore[arg-type]
            original_write = handle.write

            def write_then_die(data):  # simulate power loss mid-write
                original_write(data[: len(data) // 2])
                raise OSError(5, "Input/output error")

            handle.write = write_then_die  # type: ignore[method-assign]
            return handle

        with mock.patch("os.fdopen", side_effect=exploding_fdopen):
            result = self.writer.write_text(self.target, "half-written garbage")
        self.assertFalse(result.ok)
        self.assertEqual(self.target.read_text(encoding="utf-8"), "stable")


if __name__ == "__main__":
    unittest.main()
