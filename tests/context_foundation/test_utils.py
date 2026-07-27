"""Pure-utility tests: ULID properties, hashing, timestamps."""

from __future__ import annotations

import unittest

from dozen.context.utils import (
    content_hash,
    format_iso,
    generate_ulid,
    is_ulid,
    parse_iso,
    utc_now_iso,
)


class TestUlid(unittest.TestCase):
    def test_shape(self) -> None:
        u = generate_ulid()
        self.assertEqual(len(u), 26)
        self.assertTrue(is_ulid(u))
        self.assertFalse(is_ulid("not-a-ulid"))
        self.assertFalse(is_ulid("I" * 26))  # I is not in the Crockford alphabet

    def test_uniqueness_and_monotonic_sort(self) -> None:
        ids = [generate_ulid() for _ in range(500)]
        self.assertEqual(len(set(ids)), 500)
        self.assertEqual(ids, sorted(ids))  # generation order == sort order

    def test_time_ordering_across_milliseconds(self) -> None:
        early = generate_ulid(timestamp_ms=1_000_000)
        late = generate_ulid(timestamp_ms=2_000_000)
        self.assertLess(early[:10], late[:10])


class TestHashing(unittest.TestCase):
    def test_deterministic_and_distinct(self) -> None:
        self.assertEqual(content_hash("abc"), content_hash("abc"))
        self.assertNotEqual(content_hash("abc"), content_hash("abd"))
        self.assertEqual(len(content_hash("abc")), 32)


class TestTime(unittest.TestCase):
    def test_canonical_format_round_trip(self) -> None:
        ts = utc_now_iso()
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        self.assertEqual(format_iso(parse_iso(ts)), ts)


if __name__ == "__main__":
    unittest.main()
