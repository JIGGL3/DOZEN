"""Policy / model tests: immutable policy, validation, admission outcomes,
serialization, bounds, no sensitive fields, rejection-exception compatibility."""

from __future__ import annotations

import dataclasses
import math
import unittest

from webllm.browser_jobs import SCHEMA_VERSION
from webllm.browser_queue import (
    ADMISSION_DIAGNOSTIC_MAX,
    AdmissionOutcome,
    AdmissionSnapshot,
    BrowserQueueRejectedError,
    QueueEvent,
    QueueEventKind,
    QueuePolicy,
    QueueSnapshot,
    make_admission_snapshot,
)
from webllm.providers import ProviderError


class TestQueuePolicy(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        p = QueuePolicy()
        self.assertEqual(p.max_queued_prompts_per_provider, 8)
        self.assertEqual(p.max_total_queued_prompts, 32)
        self.assertEqual(p.max_queued_control_jobs_per_provider, 4)
        self.assertEqual(p.max_queue_wait_s, 120.0)
        self.assertTrue(p.fail_fast_when_full)
        self.assertTrue(p.reject_quarantined_provider)
        self.assertEqual(p.schema_version, SCHEMA_VERSION)

    def test_immutable(self) -> None:
        p = QueuePolicy()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.max_queued_prompts_per_provider = 99  # type: ignore[misc]

    def test_validation_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            QueuePolicy(max_queued_prompts_per_provider=0)
        with self.assertRaises(ValueError):
            QueuePolicy(max_total_queued_prompts=1, max_queued_prompts_per_provider=8)
        with self.assertRaises(ValueError):
            QueuePolicy(max_queue_wait_s=0)
        with self.assertRaises(TypeError):
            QueuePolicy(max_queue_wait_s="soon")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            QueuePolicy(fail_fast_when_full=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            QueuePolicy(schema_version=999)

    def test_validation_rejects_non_finite_and_disabled_safety(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(max_queue_wait_s=value), self.assertRaises(ValueError):
                QueuePolicy(max_queue_wait_s=value)
        for field in (
            "fail_fast_when_full",
            "release_capacity_on_queued_cancel",
            "release_capacity_on_queued_timeout",
            "reject_quarantined_provider",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                QueuePolicy(**{field: False})
        with self.assertRaises(ValueError):
            QueuePolicy(
                max_admission_diagnostic_length=ADMISSION_DIAGNOSTIC_MAX + 1
            )

    def test_large_finite_capacity_is_valid(self) -> None:
        policy = QueuePolicy(
            max_queued_prompts_per_provider=100_000,
            max_total_queued_prompts=1_000_000,
            max_queued_control_jobs_per_provider=50_000,
            max_queue_wait_s=1_000_000.0,
            max_queue_snapshot_entries=100_000,
        )
        self.assertEqual(policy.max_total_queued_prompts, 1_000_000)

    def test_roundtrip_dict(self) -> None:
        p = QueuePolicy(max_queued_prompts_per_provider=3, max_total_queued_prompts=9)
        self.assertEqual(p.to_dict()["max_queued_prompts_per_provider"], 3)
        self.assertEqual(p.to_dict()["max_total_queued_prompts"], 9)


class TestAdmissionOutcome(unittest.TestCase):
    def test_all_outcomes_present(self) -> None:
        values = {o.value for o in AdmissionOutcome}
        self.assertIn("accepted", values)
        for expected in (
            "rejected_provider_full",
            "rejected_global_full",
            "rejected_provider_quarantined",
            "rejected_manager_shutting_down",
            "rejected_provider_unavailable",
            "expired_before_start",
        ):
            self.assertIn(expected, values)

    def test_accepted_flag(self) -> None:
        self.assertTrue(AdmissionOutcome.ACCEPTED.accepted)
        self.assertFalse(AdmissionOutcome.REJECTED_PROVIDER_FULL.accepted)


class TestAdmissionSnapshot(unittest.TestCase):
    def _snap(self, outcome=AdmissionOutcome.ACCEPTED) -> AdmissionSnapshot:
        return make_admission_snapshot(
            outcome,
            "openai",
            provider_queued_depth=3,
            provider_capacity=8,
            global_queued_depth=5,
            global_capacity=32,
        )

    def test_serialization_roundtrip(self) -> None:
        snap = self._snap(AdmissionOutcome.REJECTED_GLOBAL_FULL)
        back = AdmissionSnapshot.from_dict(snap.to_dict())
        self.assertEqual(back, snap)
        self.assertFalse(back.accepted)

    def test_unknown_outcome_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AdmissionSnapshot.from_dict({**self._snap().to_dict(), "outcome": "bogus"})

    def test_negative_depth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AdmissionSnapshot(
                outcome="accepted",
                provider="openai",
                timestamp=1.0,
                provider_queued_depth=-1,
                provider_capacity=8,
                global_queued_depth=0,
                global_capacity=32,
            )

    def test_reason_is_bounded_and_flat(self) -> None:
        snap = make_admission_snapshot(
            AdmissionOutcome.REJECTED_PROVIDER_FULL,
            "openai",
            provider_queued_depth=8,
            provider_capacity=8,
            global_queued_depth=8,
            global_capacity=32,
            reason="x" * 5000 + "\nsecret",
        )
        self.assertLessEqual(len(snap.reason), ADMISSION_DIAGNOSTIC_MAX)
        self.assertNotIn("\n", snap.reason)

    def test_carries_no_sensitive_fields(self) -> None:
        # The dataclass fields are a fixed, content-free set — no prompt/response/
        # cookie/credential/traceback field can exist.
        fields = {f.name for f in dataclasses.fields(AdmissionSnapshot)}
        for banned in ("prompt", "response", "cookie", "credential", "traceback", "body"):
            self.assertNotIn(banned, fields)


class TestQueueSnapshot(unittest.TestCase):
    def test_valid_and_serializable(self) -> None:
        s = QueueSnapshot(
            provider="openai",
            prompt_depth=2,
            control_depth=1,
            provider_capacity=8,
            global_queued_depth=4,
            quarantined=False,
            running_job_id=None,
            oldest_queued_age_s=1.5,
        )
        self.assertEqual(s.to_dict()["prompt_depth"], 2)

    def test_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            QueueSnapshot(
                provider="openai",
                prompt_depth=-1,
                control_depth=0,
                provider_capacity=8,
                global_queued_depth=0,
            )


class TestQueueEvent(unittest.TestCase):
    def test_valid_and_serializable(self) -> None:
        e = QueueEvent(
            kind=QueueEventKind.ADMISSION_REJECTED.value,
            provider="openai",
            timestamp=1.0,
            outcome=AdmissionOutcome.REJECTED_PROVIDER_FULL.value,
            prompt_depth=8,
            global_depth=8,
            reason="provider queue is full",
        )
        self.assertEqual(e.to_dict()["kind"], "admission_rejected")

    def test_bad_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QueueEvent(kind="nope", provider="openai", timestamp=1.0)

    def test_carries_no_sensitive_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(QueueEvent)}
        for banned in ("prompt", "response", "cookie", "credential", "traceback", "body"):
            self.assertNotIn(banned, fields)


class TestRejectionException(unittest.TestCase):
    def test_is_a_provider_error(self) -> None:
        snap = make_admission_snapshot(
            AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED,
            "openai",
            provider_queued_depth=0,
            provider_capacity=8,
            global_queued_depth=0,
            global_capacity=32,
        )
        err = BrowserQueueRejectedError(snap)
        self.assertIsInstance(err, ProviderError)
        self.assertIs(err.outcome, AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED)
        self.assertIs(err.admission, snap)
        self.assertIn("openai", str(err))
        # Existing callers catch ProviderError — this must be caught by it.
        try:
            raise err
        except ProviderError as caught:
            self.assertIsInstance(caught, BrowserQueueRejectedError)


if __name__ == "__main__":
    unittest.main()
