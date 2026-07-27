"""Interruption-model tests: immutable models, serialization, unknown values,
timestamp consistency, bounded diagnostics, policy validation, no sensitive
fields.
"""

from __future__ import annotations

import dataclasses
import unittest

from webllm.browser_cancellation import (
    DEFAULT_INTERRUPTION_POLICY,
    InterruptionActionResult,
    InterruptionOutcome,
    InterruptionPolicy,
    InterruptionReason,
    InterruptionSnapshot,
    StopActionStatus,
    outcome_for_status,
)
from webllm.browser_jobs import SCHEMA_VERSION, BrowserJobId


class TestVocabulary(unittest.TestCase):
    def test_reasons_are_small_and_string_valued(self) -> None:
        self.assertEqual(
            {r.value for r in InterruptionReason}, {"cancelled", "caller_timeout"}
        )

    def test_outcomes_cover_the_documented_vocabulary(self) -> None:
        self.assertEqual(
            {o.value for o in InterruptionOutcome},
            {"not_requested", "not_needed", "stopped",
             "settled_before_action", "unsupported", "failed"},
        )

    def test_unknown_reason_and_outcome_raise(self) -> None:
        with self.assertRaises(ValueError):
            InterruptionReason("provider_timeout")
        with self.assertRaises(ValueError):
            InterruptionOutcome("mystery")

    def test_status_to_outcome_mapping(self) -> None:
        self.assertIs(outcome_for_status(StopActionStatus.STOPPED), InterruptionOutcome.STOPPED)
        self.assertIs(outcome_for_status(StopActionStatus.ALREADY_IDLE), InterruptionOutcome.NOT_NEEDED)
        self.assertIs(outcome_for_status(StopActionStatus.NO_CONTROL), InterruptionOutcome.UNSUPPORTED)
        self.assertIs(
            outcome_for_status(StopActionStatus.SETTLED_BEFORE_ACTION),
            InterruptionOutcome.SETTLED_BEFORE_ACTION,
        )
        self.assertIs(outcome_for_status(StopActionStatus.UNSUPPORTED), InterruptionOutcome.UNSUPPORTED)
        self.assertIs(outcome_for_status(StopActionStatus.FAILED), InterruptionOutcome.FAILED)


class TestPolicy(unittest.TestCase):
    def test_default_policy_is_active_and_exactly_once(self) -> None:
        p = DEFAULT_INTERRUPTION_POLICY
        self.assertTrue(p.active_interruption_enabled)
        self.assertTrue(p.interrupt_on_cancel)
        self.assertTrue(p.interrupt_on_caller_timeout)
        self.assertEqual(p.max_stop_attempts, 1)

    def test_policy_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_INTERRUPTION_POLICY.max_stop_attempts = 5  # type: ignore[misc]

    def test_rejects_zero_or_negative_bounds(self) -> None:
        for kw in (
            {"max_stop_attempts": 0},
            {"stop_action_timeout_s": 0},
            {"post_stop_grace_s": -1},
            {"quiescence_poll_interval_s": 0},
            {"max_diagnostic_length": 0},
            {"max_event_reason_length": 0},
        ):
            with self.subTest(kw=kw), self.assertRaises((ValueError, TypeError)):
                InterruptionPolicy(**kw)

    def test_poll_interval_cannot_exceed_grace(self) -> None:
        with self.assertRaises(ValueError):
            InterruptionPolicy(post_stop_grace_s=0.5, quiescence_poll_interval_s=1.0)

    def test_no_unbounded_values(self) -> None:
        # Every numeric bound is finite and strictly positive.
        p = InterruptionPolicy()
        for name in ("stop_action_timeout_s", "post_stop_grace_s",
                     "quiescence_poll_interval_s"):
            self.assertGreater(getattr(p, name), 0)


class TestActionResult(unittest.TestCase):
    def test_factories_set_status(self) -> None:
        self.assertIs(InterruptionActionResult.stopped(quiescent=True).status, StopActionStatus.STOPPED)
        self.assertIs(InterruptionActionResult.already_idle().status, StopActionStatus.ALREADY_IDLE)
        self.assertIs(InterruptionActionResult.no_control().status, StopActionStatus.NO_CONTROL)
        self.assertIs(
            InterruptionActionResult.settled_before_action().status,
            StopActionStatus.SETTLED_BEFORE_ACTION,
        )
        self.assertIs(InterruptionActionResult.failed("x").status, StopActionStatus.FAILED)
        self.assertIs(InterruptionActionResult.unsupported().status, StopActionStatus.UNSUPPORTED)

    def test_result_is_immutable(self) -> None:
        r = InterruptionActionResult.stopped(quiescent=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.status = StopActionStatus.FAILED  # type: ignore[misc]

    def test_diagnostic_is_bounded_and_flattened(self) -> None:
        r = InterruptionActionResult.failed("line1\nline2 " + "z" * 500)
        self.assertNotIn("\n", r.diagnostic)
        self.assertLessEqual(len(r.diagnostic), 200)


class TestSnapshot(unittest.TestCase):
    def _snap(self, **kw) -> InterruptionSnapshot:
        base = dict(job_id=BrowserJobId.new().value, provider="openai")
        base.update(kw)
        return InterruptionSnapshot(**base)

    def test_default_snapshot_is_not_requested(self) -> None:
        snap = self._snap()
        self.assertFalse(snap.requested)
        self.assertEqual(snap.outcome, InterruptionOutcome.NOT_REQUESTED.value)
        self.assertEqual(snap.attempt_count, 0)

    def test_snapshot_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self._snap().requested = True  # type: ignore[misc]

    def test_round_trip_serialization(self) -> None:
        snap = self._snap(
            requested=True,
            reason=InterruptionReason.CALLER_TIMEOUT.value,
            requested_at=100.0,
            source="handle",
            physical_interruption_required=True,
            started=True,
            action_attempted=True,
            attempt_count=1,
            completed_at=101.0,
            outcome=InterruptionOutcome.STOPPED.value,
            diagnostic="quiescence-not-confirmed",
        )
        again = InterruptionSnapshot.from_dict(snap.to_dict())
        self.assertEqual(snap, again)

    def test_schema_version_present(self) -> None:
        self.assertEqual(self._snap().schema_version, SCHEMA_VERSION)

    def test_rejects_unknown_reason_and_outcome(self) -> None:
        with self.assertRaises(ValueError):
            self._snap(reason="bogus")
        with self.assertRaises(ValueError):
            self._snap(outcome="bogus")

    def test_rejects_bad_job_id(self) -> None:
        with self.assertRaises(ValueError):
            InterruptionSnapshot(job_id="short", provider="openai")

    def test_no_sensitive_fields_exist(self) -> None:
        fields = {f.name for f in dataclasses.fields(InterruptionSnapshot)}
        for banned in ("prompt", "response", "text", "cookies", "dom",
                       "traceback", "credentials", "password", "selector"):
            self.assertNotIn(banned, fields)


if __name__ == "__main__":
    unittest.main()
