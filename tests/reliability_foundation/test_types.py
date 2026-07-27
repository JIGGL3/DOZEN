"""Enum completeness and forward-compatibility (SADD-002 taxonomies)."""

from __future__ import annotations

import unittest

from dozen.reliability import (
    AttemptStatus,
    CheckpointKind,
    FailureType,
    HealthState,
    RecoveryAction,
    RecoveryStatus,
    parse_enum,
)


class TestEnumCompleteness(unittest.TestCase):
    def test_failure_type_members(self) -> None:
        expected = {
            "TIMEOUT", "GENERATION_STALLED", "DOM_CHANGED", "RATE_LIMIT",
            "MODEL_BUSY", "LOGIN_REQUIRED", "CAPTCHA", "NETWORK_ERROR",
            "BROWSER_CRASH", "TAB_CLOSED", "PROMPT_REJECTED",
            "OUTPUT_CORRUPTED", "UNEXPECTED_UI", "UNKNOWN",
        }
        self.assertEqual({m.name for m in FailureType}, expected)
        self.assertEqual(len(FailureType), 14)

    def test_health_state_members(self) -> None:
        expected = {
            "UNKNOWN", "HEALTHY", "DEGRADED", "SUSPECT",
            "RECOVERING", "QUARANTINED", "NEEDS_HUMAN", "OFFLINE",
        }
        self.assertEqual({m.name for m in HealthState}, expected)
        self.assertEqual(len(HealthState), 8)

    def test_recovery_action_members(self) -> None:
        expected = {
            "RETRY", "REFRESH", "NEW_CHAT", "RECOVER_TAB", "RESTART_WORKER",
            "REAUTH", "WAIT", "SHRINK_PROMPT", "DELEGATE", "ESCALATE",
            "ABORT", "UNKNOWN",
        }
        self.assertEqual({m.name for m in RecoveryAction}, expected)

    def test_supporting_enums_exist(self) -> None:
        self.assertIn(RecoveryStatus.RECOVERED, RecoveryStatus)
        self.assertIn(AttemptStatus.SUCCEEDED, AttemptStatus)
        self.assertIn(CheckpointKind.SUBTASK_COMPLETED, CheckpointKind)

    def test_every_enum_has_unknown(self) -> None:
        for enum_cls in (
            FailureType, HealthState, RecoveryAction,
            RecoveryStatus, AttemptStatus, CheckpointKind,
        ):
            with self.subTest(enum=enum_cls.__name__):
                self.assertEqual(enum_cls("unknown").name, "UNKNOWN")

    def test_values_are_stable_strings(self) -> None:
        # Serialized values are the contract — spot-check the wire format.
        self.assertEqual(FailureType.LOGIN_REQUIRED.value, "login_required")
        self.assertEqual(HealthState.NEEDS_HUMAN.value, "needs_human")
        self.assertEqual(RecoveryAction.RESTART_WORKER.value, "restart_worker")
        self.assertEqual(CheckpointKind.RUN_STARTED.value, "run_started")
        for enum_cls in (FailureType, HealthState, RecoveryAction):
            for member in enum_cls:
                self.assertIsInstance(member.value, str)


class TestForwardCompatibility(unittest.TestCase):
    def test_parse_known(self) -> None:
        self.assertIs(parse_enum(FailureType, "captcha"), FailureType.CAPTCHA)
        self.assertIs(parse_enum(HealthState, "degraded"), HealthState.DEGRADED)

    def test_parse_future_value_maps_to_unknown(self) -> None:
        self.assertIs(parse_enum(FailureType, "quantum_decoherence"), FailureType.UNKNOWN)
        self.assertIs(parse_enum(RecoveryAction, "summon_intern"), RecoveryAction.UNKNOWN)
        self.assertIs(parse_enum(HealthState, 42), HealthState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
