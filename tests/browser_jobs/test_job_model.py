"""Model tests: job id, immutable snapshots, serialization, safety."""

from __future__ import annotations

import dataclasses
import unittest

from webllm.browser_jobs import (
    FAILURE_MESSAGE_MAX,
    SCHEMA_VERSION,
    BrowserJobEvent,
    BrowserJobId,
    BrowserJobSnapshot,
    JobState,
    _bounded_message,
)


class TestBrowserJobId(unittest.TestCase):
    def test_ids_are_unique(self) -> None:
        ids = {BrowserJobId.new().value for _ in range(10_000)}
        self.assertEqual(len(ids), 10_000)

    def test_id_is_full_hex_not_truncated(self) -> None:
        jid = BrowserJobId.new()
        self.assertEqual(len(jid.value), 32)  # full uuid4 hex, no truncation
        int(jid.value, 16)  # parses as hex

    def test_id_compares_and_serializes_as_string(self) -> None:
        jid = BrowserJobId.new()
        self.assertEqual(str(jid), jid.value)
        self.assertEqual(BrowserJobId(jid.value), jid)  # value equality
        self.assertEqual(hash(BrowserJobId(jid.value)), hash(jid))

    def test_id_is_frozen(self) -> None:
        jid = BrowserJobId.new()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            jid.value = "x"  # type: ignore[misc]

    def test_malformed_ids_are_rejected(self) -> None:
        for value in ("", "j", "g" * 32, BrowserJobId.new().value.upper()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BrowserJobId(value)

    def test_id_generation_does_not_depend_on_per_job_rng_success(self) -> None:
        # The process nonce is chosen once; each job uses the authoritative
        # monotonic counter, so a failing/repeated RNG cannot reuse an id.
        first = BrowserJobId.new()
        second = BrowserJobId.new()
        self.assertNotEqual(first, second)
        self.assertEqual(first.value[:16], second.value[:16])


class TestSnapshot(unittest.TestCase):
    def _snap(self, **kw) -> BrowserJobSnapshot:
        base = dict(
            job_id=BrowserJobId.new().value,
            provider="openai",
            state="queued",
            created_at=100.0,
        )
        base.update(kw)
        return BrowserJobSnapshot(**base)

    def test_snapshot_is_immutable(self) -> None:
        snap = self._snap()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.state = "running"  # type: ignore[misc]

    def test_round_trip_serialization(self) -> None:
        snap = self._snap(
            state="failed",
            started_at=101.0,
            finished_at=102.0,
            failure_category="ProviderError",
            failure_message="boom",
            result_delivered=False,
        )
        again = BrowserJobSnapshot.from_dict(snap.to_dict())
        self.assertEqual(snap, again)

    def test_schema_version_present(self) -> None:
        self.assertEqual(self._snap().schema_version, SCHEMA_VERSION)

    def test_rejects_finished_before_started(self) -> None:
        with self.assertRaises(ValueError):
            self._snap(started_at=200.0, finished_at=150.0)

    def test_rejects_started_before_created(self) -> None:
        with self.assertRaises(ValueError):
            self._snap(started_at=50.0)  # created_at is 100.0

    def test_no_sensitive_fields_exist(self) -> None:
        fields = {f.name for f in dataclasses.fields(BrowserJobSnapshot)}
        for banned in ("prompt", "response", "text", "cookies", "dom",
                       "traceback", "credentials", "password"):
            self.assertNotIn(banned, fields)


class TestBoundedMessage(unittest.TestCase):
    def test_long_message_is_truncated(self) -> None:
        out = _bounded_message("x" * (FAILURE_MESSAGE_MAX + 500))
        self.assertLessEqual(len(out), FAILURE_MESSAGE_MAX)

    def test_newlines_flattened(self) -> None:
        self.assertNotIn("\n", _bounded_message("a\nb\r\nc"))

    def test_none_stays_none(self) -> None:
        self.assertIsNone(_bounded_message(None))


class TestJobStateEnum(unittest.TestCase):
    def test_state_values_are_strings(self) -> None:
        self.assertEqual(JobState.RUNNING.value, "running")
        self.assertEqual(str(JobState.RUNNING), "running")

    def test_unknown_state_lookup_raises(self) -> None:
        with self.assertRaises(ValueError):
            JobState("not-a-real-state")


class TestEvent(unittest.TestCase):
    def test_event_is_immutable_and_content_free(self) -> None:
        ev = BrowserJobEvent(
            job_id=BrowserJobId.new().value, provider="openai", old_state="queued",
            new_state="running", timestamp=1.0, reason="claimed",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ev.reason = "x"  # type: ignore[misc]
        fields = {f.name for f in dataclasses.fields(BrowserJobEvent)}
        for banned in ("prompt", "response", "text", "body"):
            self.assertNotIn(banned, fields)
        self.assertEqual(BrowserJobEvent.from_dict(ev.to_dict()), ev)


if __name__ == "__main__":
    unittest.main()
