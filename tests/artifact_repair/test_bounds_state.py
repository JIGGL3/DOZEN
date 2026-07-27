"""Phase 4E Parts H/O — the preserved-candidate state, its bounds and immutability."""

from __future__ import annotations

import dataclasses
import unittest

from dozen.artifact_repair import (
    ArtifactRepairError,
    ArtifactRepairState,
    begin_repair_state,
    submitted_content_hashes,
)
from dozen.artifact_results import content_hash

from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    ROUTER,
    entry,
    envelope,
    envelope_with,
    first_attempt,
)

TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


class TestSubmittedHashes(unittest.TestCase):
    def test_every_typed_entry_is_hashed(self) -> None:
        payload = envelope(entry(APP), entry(LAYOUT, content=TRUNCATED_LAYOUT))
        hashes = dict(submitted_content_hashes(payload))
        self.assertEqual(hashes[APP], content_hash(CONTENT[APP]))
        self.assertEqual(hashes[LAYOUT], content_hash(TRUNCATED_LAYOUT))

    def test_a_legacy_envelope_yields_no_hashes(self) -> None:
        self.assertEqual(submitted_content_hashes({"artifacts": {"a.tsx": "x"}}), ())

    def test_non_mapping_payloads_are_safe(self) -> None:
        self.assertEqual(submitted_content_hashes(None), ())
        self.assertEqual(submitted_content_hashes("nope"), ())

    def test_hashes_are_deterministic_and_deduplicated(self) -> None:
        payload = envelope(entry(APP), entry(APP))
        hashes = submitted_content_hashes(payload)
        self.assertEqual(len(hashes), 1)


class TestRepairState(unittest.TestCase):
    def test_a_seed_state_records_the_first_attempt(self) -> None:
        payload = envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})
        state = begin_repair_state(
            payload and first_attempt(payload).collection,
            submitted_hashes=submitted_content_hashes(payload),
        )
        self.assertEqual(state.attempts_used, 1)
        self.assertEqual(state.repair_attempts, 0)
        # Only the REJECTED body's hash is retained (App/router were accepted).
        self.assertEqual(state.invalid_hash_for(LAYOUT), content_hash(TRUNCATED_LAYOUT))
        self.assertEqual(state.invalid_hash_for(APP), "")

    def test_state_requires_a_collection(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            ArtifactRepairState(collection=object())  # type: ignore[arg-type]

    def test_state_is_immutable(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.attempts_used = 9  # type: ignore[misc]

    def test_preserved_exposes_the_accepted_candidates(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        self.assertEqual(
            {c.artifact_id for c in state.preserved}, {APP, ROUTER}
        )

    def test_counters_canonicalize_from_a_mapping(self) -> None:
        state = ArtifactRepairState(
            collection=first_attempt(envelope()).collection,
            attempt_counts={APP: 2, ROUTER: 1},
            invalid_hashes={APP: "sha256:x"},
        )
        self.assertEqual(state.attempts_for(APP), 2)
        self.assertEqual(state.attempts_for(ROUTER), 1)
        self.assertEqual(state.invalid_hash_for(APP), "sha256:x")
        self.assertIsInstance(state.attempt_counts, tuple)

    def test_malformed_counters_are_rejected(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            ArtifactRepairState(
                collection=first_attempt(envelope()).collection,
                attempt_counts=[("a", "not-an-int")],
            )

    def test_diagnostics_are_bounded(self) -> None:
        state = ArtifactRepairState(
            collection=first_attempt(envelope()).collection,
            diagnostics=tuple(f"d{i}" for i in range(50)),
        )
        self.assertLessEqual(len(state.diagnostics), 8)


if __name__ == "__main__":
    unittest.main()
