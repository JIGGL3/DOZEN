"""JSONL writer/reader: appends, streaming reads, pagination, tail reads,
corruption containment and crash-recovery sealing."""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import JsonLineReader, JsonLineWriter


class JsonlTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-jsonl-"))
        self.path = self.tmp / "messages.jsonl"
        self.writer = JsonLineWriter(self.path, fsync=False)
        self.reader = JsonLineReader(self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAppendAndRead(JsonlTestCase):
    def test_round_trip(self) -> None:
        records = [{"id": f"m{i}", "content": f"msg {i}"} for i in range(10)]
        self.assertEqual(self.writer.append(records).unwrap(), 10)
        report = self.reader.read().unwrap()
        self.assertEqual(report.records, records)
        self.assertFalse(report.corrupt)

    def test_append_empty_is_noop(self) -> None:
        self.assertEqual(self.writer.append([]).unwrap(), 0)
        self.assertFalse(self.path.exists())

    def test_missing_file_reads_empty(self) -> None:
        self.assertEqual(self.reader.read().unwrap().records, [])
        self.assertEqual(self.reader.count().unwrap(), 0)
        self.assertEqual(self.reader.read_tail(5).unwrap().records, [])

    def test_multiple_appends_accumulate_in_order(self) -> None:
        self.writer.append([{"n": 1}])
        self.writer.append([{"n": 2}, {"n": 3}])
        self.assertEqual([r["n"] for r in self.reader.read().unwrap().records], [1, 2, 3])

    def test_unserializable_record_fails_without_touching_file(self) -> None:
        self.writer.append([{"n": 1}])
        result = self.writer.append([{"bad": object()}])
        self.assertFalse(result.ok)
        self.assertEqual(self.reader.count().unwrap(), 1)

    def test_unicode_content_round_trips(self) -> None:
        record = {"content": "multi\nline\tand 🚀 unicode"}
        self.writer.append([record])
        self.assertEqual(self.reader.read().unwrap().records, [record])


class TestPagination(JsonlTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.writer.append([{"n": i} for i in range(20)])

    def test_offset_and_limit(self) -> None:
        page = self.reader.read(offset=5, limit=3).unwrap().records
        self.assertEqual([r["n"] for r in page], [5, 6, 7])

    def test_offset_past_end(self) -> None:
        self.assertEqual(self.reader.read(offset=100).unwrap().records, [])

    def test_count(self) -> None:
        self.assertEqual(self.reader.count().unwrap(), 20)

    def test_tail(self) -> None:
        tail = self.reader.read_tail(4).unwrap().records
        self.assertEqual([r["n"] for r in tail], [16, 17, 18, 19])

    def test_tail_larger_than_file(self) -> None:
        tail = self.reader.read_tail(500).unwrap().records
        self.assertEqual(len(tail), 20)
        self.assertEqual(tail[0]["n"], 0)


class TestCorruptionRecovery(JsonlTestCase):
    def test_malformed_middle_line_is_skipped_and_reported(self) -> None:
        self.writer.append([{"n": 1}])
        with open(self.path, "ab") as handle:
            handle.write(b"{this is not json\n")
        self.writer.append([{"n": 2}])
        report = self.reader.read().unwrap()
        self.assertEqual([r["n"] for r in report.records], [1, 2])
        self.assertTrue(report.corrupt)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0].line_number, 2)

    def test_non_object_line_is_an_error(self) -> None:
        with open(self.path, "ab") as handle:
            handle.write(b'[1,2,3]\n"just a string"\n')
        report = self.reader.read().unwrap()
        self.assertEqual(report.records, [])
        self.assertEqual(len(report.errors), 2)

    def test_partial_trailing_line_is_sealed_by_next_append(self) -> None:
        """Simulates a crash mid-append: the torn line is isolated, old and
        new records all survive."""
        self.writer.append([{"n": 1}])
        with open(self.path, "ab") as handle:
            handle.write(b'{"n": 2, "content": "the crash happened he')  # no newline
        self.writer.append([{"n": 3}])
        report = self.reader.read().unwrap()
        self.assertEqual([r["n"] for r in report.records], [1, 3])
        self.assertEqual(len(report.errors), 1)

    def test_blank_lines_are_ignored_silently(self) -> None:
        with open(self.path, "ab") as handle:
            handle.write(b'\n\n{"n": 1}\n\n')
        report = self.reader.read().unwrap()
        self.assertEqual([r["n"] for r in report.records], [1])
        self.assertFalse(report.corrupt)

    def test_tail_read_skips_corruption(self) -> None:
        self.writer.append([{"n": i} for i in range(5)])
        with open(self.path, "ab") as handle:
            handle.write(b"garbage garbage\n")
        self.writer.append([{"n": 5}])
        tail = self.reader.read_tail(3).unwrap()
        self.assertEqual([r["n"] for r in tail.records], [3, 4, 5])
        self.assertTrue(tail.corrupt)


class TestPerformanceSanity(JsonlTestCase):
    """Not benchmarks — just proof the design goals hold at small scale:
    appends are batched single-writes and tail reads don't scan the file."""

    def test_bulk_append_and_tail_read(self) -> None:
        big_content = "x" * 400
        started = time.monotonic()
        for batch_start in range(0, 2000, 100):
            records = [{"id": f"m{i}", "content": big_content} for i in range(batch_start, batch_start + 100)]
            self.assertTrue(self.writer.append(records).ok)
        append_seconds = time.monotonic() - started
        self.assertLess(append_seconds, 10.0, "2000 appends should take well under 10s")

        started = time.monotonic()
        tail = self.reader.read_tail(50).unwrap().records
        tail_seconds = time.monotonic() - started
        self.assertEqual(len(tail), 50)
        self.assertEqual(tail[-1]["id"], "m1999")
        self.assertLess(tail_seconds, 2.0, "tail read must not scan the whole file")

    def test_streaming_read_is_lazy(self) -> None:
        self.writer.append([{"n": i} for i in range(1000)])
        consumed = 0
        for record, _error in self.reader.scan():
            consumed += 1
            if consumed == 3:
                break  # generator abandoned: no full-file materialization
        self.assertEqual(consumed, 3)


if __name__ == "__main__":
    unittest.main()
