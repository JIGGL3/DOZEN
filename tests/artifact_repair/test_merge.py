"""Phase 4E Parts H/K — the deterministic preserved-candidate merge and
no-progress detection. No latest-wins, no mutation, no overwrite of a preserved
candidate.
"""

from __future__ import annotations

import unittest

from dozen.artifact_repair import (
    ArtifactRepairPolicy,
    RepairStatus,
    begin_repair_state,
    submitted_content_hashes,
)

from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    ROUTER,
    cycle,
    decide,
    entry,
    envelope,
    envelope_with,
    first_attempt,
    repair_attempt,
    scope_for,
)

TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"
TRUNCATED_APP = 'import { Router } from "./router";\n\nexport default function App() {\n  return (\n    <button className="'


def broken_layout():
    return envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})


class TestPreservedCandidateMerge(unittest.TestCase):
    def test_a_preserved_candidate_passes_through_byte_identically(self) -> None:
        state = first_attempt(broken_layout())
        original = state.collection.accepted_by_artifact()[APP][0]
        decision = decide(state)
        merged = repair_attempt(state, decision, envelope(entry(LAYOUT)))
        preserved = merged.collection.accepted_by_artifact()[APP][0]
        self.assertIs(preserved, original)  # the exact same object
        self.assertEqual(preserved.content, CONTENT[APP])
        self.assertEqual(preserved.content_hash, original.content_hash)

    def test_a_repair_candidate_fills_the_missing_target(self) -> None:
        state = first_attempt(broken_layout())
        decision = decide(state)
        merged = repair_attempt(state, decision, envelope(entry(LAYOUT)))
        ids = [c.artifact_id for c in merged.collection.accepted]
        self.assertEqual(ids, [APP, ROUTER, LAYOUT])
        self.assertTrue(merged.collection.satisfied)
        self.assertEqual(merged.repaired_artifact_ids, (LAYOUT,))

    def test_a_repaired_candidate_carries_its_new_attempt_and_provenance(self) -> None:
        state = first_attempt(broken_layout())
        decision = decide(state)
        merged = repair_attempt(
            state, decision, envelope(entry(LAYOUT)), attempt=2,
            producer_subtask_id="nested", recursion_depth=1,
        )
        repaired = merged.collection.accepted_by_artifact()[LAYOUT][0]
        self.assertEqual(repaired.attempt, 2)
        self.assertEqual(repaired.producer_subtask_id, "nested")
        self.assertEqual(repaired.recursion_depth, 1)
        preserved = merged.collection.accepted_by_artifact()[APP][0]
        self.assertEqual(preserved.attempt, 1)

    def test_a_repair_cannot_replace_a_preserved_candidate(self) -> None:
        # A reply that resubmits App with different content: the preserved App
        # wins, the overwrite is rejected, and the real target still merges.
        state = first_attempt(broken_layout())
        decision = decide(state)
        merged = repair_attempt(state, decision, envelope(
            entry(APP, content="export default function App() { return null; }\n"),
            entry(LAYOUT),
        ))
        self.assertEqual(
            merged.collection.accepted_by_artifact()[APP][0].content, CONTENT[APP]
        )

    def test_a_rejected_repair_does_not_delete_preserved_candidates(self) -> None:
        state = first_attempt(broken_layout())
        decision = decide(state)
        merged = repair_attempt(
            state, decision, envelope(entry(LAYOUT, content=TRUNCATED_LAYOUT))
        )
        ids = {c.artifact_id for c in merged.collection.accepted}
        self.assertEqual(ids, {APP, ROUTER})
        self.assertIn(LAYOUT, merged.collection.missing_required_artifact_ids)

    def test_manifest_ordering_is_stable(self) -> None:
        state = first_attempt(envelope(entry(APP)))  # router + layout missing
        decision = decide(state)
        merged = repair_attempt(
            state, decision, envelope(entry(LAYOUT), entry(ROUTER)),
        )
        self.assertEqual(
            [c.artifact_id for c in merged.collection.accepted],
            [APP, ROUTER, LAYOUT],
        )

    def test_no_latest_wins_behavior(self) -> None:
        # The repair reply cannot make a NEW body for an already-valid artifact
        # "win": order of arrival is irrelevant, the preserved body stands.
        state = first_attempt(broken_layout())
        decision = decide(state)
        forward = repair_attempt(state, decision, envelope(
            entry(APP, content="rival\n"), entry(LAYOUT),
        ))
        self.assertEqual(
            forward.collection.accepted_by_artifact()[APP][0].content, CONTENT[APP]
        )

    def test_fatal_repair_response_order_does_not_change_the_result(self) -> None:
        state = first_attempt(broken_layout())
        decision = decide(state)
        valid = entry(LAYOUT)
        first_fatal = entry(
            "unknown-a", path="unknown-a.tsx", content="export const a = 1;\n"
        )
        second_fatal = entry(
            "unknown-b", path="unknown-b.tsx", content="export const b = 1;\n"
        )
        forward = repair_attempt(
            state, decision, envelope(valid, first_fatal, second_fatal)
        )
        reverse = repair_attempt(
            state, decision, envelope(second_fatal, valid, first_fatal)
        )
        self.assertEqual(forward.collection, reverse.collection)
        self.assertEqual(
            decide(forward, attempt=2).reason,
            decide(reverse, attempt=2).reason,
        )


class TestMultipleRepairAttempts(unittest.TestCase):
    def test_a_partial_repair_narrows_the_next_attempt(self) -> None:
        state = first_attempt(envelope(entry(APP)))  # router + layout missing
        decision = decide(state, attempt=1, max_attempts=3)
        self.assertEqual(decision.request.target_artifact_ids, (ROUTER, LAYOUT))
        # Attempt 2 only fixes the router.
        state = repair_attempt(state, decision, envelope(entry(ROUTER)), attempt=2)
        decision = decide(state, attempt=2, max_attempts=3)
        self.assertIs(decision.status, RepairStatus.REPAIRABLE)
        self.assertEqual(decision.request.target_artifact_ids, (LAYOUT,))
        self.assertEqual(decision.request.preserved_artifact_ids, (APP, ROUTER))

    def test_the_full_lifecycle_repairs_one_damaged_file(self) -> None:
        state, decision, history = cycle([
            broken_layout(),
            envelope(entry(LAYOUT)),
        ], max_attempts=3)
        self.assertIs(decision.status, RepairStatus.SATISFIED)
        self.assertTrue(state.collection.satisfied)
        self.assertEqual(state.repaired_artifact_ids, (LAYOUT,))
        self.assertEqual(state.repair_attempts, 1)


class TestNoProgress(unittest.TestCase):
    def test_repeated_identical_invalid_content_is_recorded(self) -> None:
        payload = broken_layout()
        state = first_attempt(payload)
        decision = decide(state)
        merged = repair_attempt(state, decision, payload)  # same broken body again
        self.assertIn(LAYOUT, merged.repeated_hash_artifact_ids)
        self.assertGreaterEqual(merged.no_progress_attempts, 1)

    def test_a_repeated_hash_stops_further_attempts(self) -> None:
        payload = broken_layout()
        state, decision, _ = cycle([payload, payload, payload], max_attempts=3)
        self.assertIs(decision.status, RepairStatus.NO_PROGRESS)
        # App and router stay preserved; the shell is simply incomplete.
        self.assertEqual(
            {c.artifact_id for c in state.collection.accepted}, {APP, ROUTER}
        )
        self.assertFalse(state.collection.satisfied)

    def test_an_empty_repair_counts_as_no_progress(self) -> None:
        state = first_attempt(broken_layout())
        decision = decide(state)
        merged = repair_attempt(state, decision, envelope())  # nothing came back
        self.assertGreaterEqual(merged.no_progress_attempts, 1)
        decision2 = decide(merged, attempt=2, max_attempts=3)
        self.assertIs(decision2.status, RepairStatus.NO_PROGRESS)

    def test_progress_resets_the_stall_signal(self) -> None:
        # A different (still-broken) body is not "no progress by repetition":
        # only an ACCEPTED artifact or a changed hash matters.
        state = first_attempt(broken_layout())
        decision = decide(state)
        other = "export function DashboardLayout() {\n  return <section>"
        merged = repair_attempt(state, decision, envelope(entry(LAYOUT, content=other)))
        self.assertNotIn(LAYOUT, merged.repeated_hash_artifact_ids)

    def test_no_progress_is_deterministic(self) -> None:
        payload = broken_layout()
        a, da, _ = cycle([payload, payload, payload], max_attempts=3)
        b, db, _ = cycle([payload, payload, payload], max_attempts=3)
        self.assertEqual(da.status, db.status)
        self.assertEqual(a.repeated_hash_artifact_ids, b.repeated_hash_artifact_ids)


if __name__ == "__main__":
    unittest.main()
