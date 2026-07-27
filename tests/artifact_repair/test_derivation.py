"""Phase 4E Parts D/E — deriving the bounded request and the narrow repair scope."""

from __future__ import annotations

import dataclasses
import unittest

from dozen.artifact_repair import (
    ArtifactRepairPolicy,
    RepairReason,
    RepairStatus,
    begin_repair_state,
    derive_repair_scope,
    plan_repair,
    submitted_content_hashes,
)
from dozen.decomposition import derive_execution_scope

from ..artifact_decomposition.harness import react_work_plan
from ..artifact_results.harness import collect_subtask_artifacts
from .harness import (
    APP,
    CARD,
    LAYOUT,
    PAGE,
    PKG,
    ROUTER,
    collect_for,
    decide,
    entry,
    envelope,
    envelope_with,
    first_attempt,
    scope_for,
    worker_envelope_for,
)

TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


def optional_layout_plan():
    """The approved plan with the shell's DashboardLayout marked OPTIONAL."""
    plan = react_work_plan()
    manifest = dataclasses.replace(
        plan.manifest,
        artifacts=tuple(
            dataclasses.replace(spec, required=False) if spec.id == LAYOUT else spec
            for spec in plan.manifest.artifacts
        ),
    )
    return dataclasses.replace(plan, manifest=manifest)


def optional_state(payload, plan=None):
    plan = plan or optional_layout_plan()
    scope = derive_execution_scope(plan, "shell")
    collection = collect_subtask_artifacts(payload, scope, attempt=1, work_plan=plan)
    return scope, begin_repair_state(
        collection, submitted_hashes=submitted_content_hashes(payload)
    )


class TestTargetSelection(unittest.TestCase):
    def test_a_satisfied_collection_asks_for_nothing(self) -> None:
        decision = decide(first_attempt(worker_envelope_for("shell")))
        self.assertIs(decision.status, RepairStatus.SATISFIED)
        self.assertIsNone(decision.request)
        self.assertFalse(decision.should_retry)

    def test_only_the_damaged_artifact_is_targeted(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        decision = decide(state)
        self.assertIs(decision.status, RepairStatus.REPAIRABLE)
        self.assertEqual(decision.request.target_artifact_ids, (LAYOUT,))
        self.assertEqual(decision.request.attempt, 2)

    def test_valid_candidates_are_preserved_not_re_requested(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        decision = decide(state)
        self.assertEqual(decision.request.preserved_artifact_ids, (APP, ROUTER))
        self.assertNotIn(APP, decision.request.target_artifact_ids)
        self.assertNotIn(ROUTER, decision.request.target_artifact_ids)

    def test_one_missing_and_one_damaged_artifact(self) -> None:
        state = first_attempt(envelope(
            entry(APP), entry(LAYOUT, content=TRUNCATED_LAYOUT),
        ))
        decision = decide(state)
        self.assertEqual(decision.request.target_artifact_ids, (ROUTER, LAYOUT))
        self.assertEqual(decision.request.preserved_artifact_ids, (APP,))

    def test_targets_follow_manifest_order(self) -> None:
        state = first_attempt(envelope())  # nothing at all came back
        decision = decide(state)
        self.assertEqual(
            decision.request.target_artifact_ids, (APP, ROUTER, LAYOUT)
        )

    def test_reasons_are_typed_per_target(self) -> None:
        state = first_attempt(envelope(
            entry(ROUTER), entry(LAYOUT, content=TRUNCATED_LAYOUT),
        ))
        reasons = {t.artifact_id: t.reason for t in decide(state).request.targets}
        self.assertIs(reasons[APP], RepairReason.MISSING_REQUIRED)
        self.assertIn(
            reasons[LAYOUT],
            (RepairReason.INTEGRITY_INVALID, RepairReason.TRUNCATION_SUSPECTED),
        )

    def test_relevant_diagnostics_and_criteria_travel_with_the_request(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        request = decide(state).request
        self.assertTrue(request.targets[0].evidence)
        self.assertTrue(request.diagnostics)
        self.assertEqual(request.package_ids, ("application-shell",))
        self.assertEqual(
            request.origin_output_artifact_ids, (APP, ROUTER, LAYOUT)
        )

    def test_the_target_count_is_bounded(self) -> None:
        state = first_attempt(envelope())
        policy = ArtifactRepairPolicy(max_targets_per_attempt=1)
        decision = decide(state, policy=policy)
        self.assertEqual(len(decision.request.targets), 1)

    def test_custom_diagnostic_and_schema_bounds_reach_the_request(self) -> None:
        state = first_attempt(envelope())
        policy = ArtifactRepairPolicy(
            max_diagnostics=1, max_diagnostic_chars=24, schema_version=2,
        )
        request = decide(state, policy=policy).request
        self.assertEqual(len(request.diagnostics), 1)
        self.assertLessEqual(len(request.diagnostics[0]), 24)
        self.assertLessEqual(len(request.completion_criteria), 1)
        self.assertEqual(request.schema_version, 2)

    def test_an_artifact_may_not_exceed_its_attempt_budget(self) -> None:
        state = first_attempt(envelope())
        policy = ArtifactRepairPolicy(max_attempts_per_artifact=1)
        decision = decide(state, policy=policy)
        self.assertIs(decision.status, RepairStatus.NON_REPAIRABLE)
        self.assertEqual(
            decision.unrepairable_artifact_ids, (APP, ROUTER, LAYOUT)
        )

    def test_the_existing_attempt_budget_is_never_exceeded(self) -> None:
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        decision = decide(state, attempt=1, max_attempts=1)
        self.assertIs(decision.status, RepairStatus.EXHAUSTED)
        self.assertIsNone(decision.request)


class TestNonRepairableDerivation(unittest.TestCase):
    def test_an_out_of_scope_submission_makes_repair_unsafe(self) -> None:
        state = first_attempt(envelope(
            entry(APP), entry(ROUTER), entry(LAYOUT), entry(PKG),
        ))
        decision = decide(state)
        self.assertIs(decision.status, RepairStatus.NON_REPAIRABLE)
        self.assertIsNone(decision.request)
        self.assertIn(PKG, decision.unrepairable_artifact_ids)

    def test_a_mixed_repairable_and_fatal_rejection_fails_safely(self) -> None:
        # A truncated owned file AND a spoofed unknown id: the reply crossed the
        # manifest trust boundary, so nothing is "repaired" on top of it.
        state = first_attempt(envelope(
            entry(APP), entry(ROUTER),
            entry(LAYOUT, content=TRUNCATED_LAYOUT),
            entry("src/evil.tsx", path="src/evil.tsx", content="boom\n"),
        ))
        decision = decide(state)
        self.assertIs(decision.status, RepairStatus.NON_REPAIRABLE)
        self.assertIsNone(decision.request)
        self.assertIn("src/evil.tsx", decision.unrepairable_artifact_ids)

    def test_a_path_mismatch_is_still_repairable(self) -> None:
        state = first_attempt(envelope(
            entry(APP), entry(ROUTER),
            entry(LAYOUT, path="src/components/Layout.tsx"),
        ))
        decision = decide(state)
        self.assertIs(decision.status, RepairStatus.REPAIRABLE)
        self.assertEqual(decision.request.target_artifact_ids, (LAYOUT,))
        self.assertIs(decision.request.targets[0].reason, RepairReason.PATH_MISMATCH)


class TestOptionalArtifacts(unittest.TestCase):
    def test_optional_targets_cannot_displace_required_targets(self) -> None:
        plan = optional_layout_plan()
        manifest = dataclasses.replace(
            plan.manifest,
            artifacts=tuple(
                dataclasses.replace(spec, required=False)
                if spec.id == APP else
                dataclasses.replace(spec, required=True)
                if spec.id == LAYOUT else spec
                for spec in plan.manifest.artifacts
            ),
        )
        plan = dataclasses.replace(plan, manifest=manifest)
        scope, state = optional_state(envelope(), plan=plan)
        decision = plan_repair(
            scope, state, attempt=1, max_attempts=3,
            policy=ArtifactRepairPolicy(
                max_targets_per_attempt=1, repair_missing_optional=True,
            ),
        )
        self.assertTrue(decision.request.targets[0].required)
        self.assertEqual(decision.request.target_artifact_ids, (ROUTER,))

    def test_a_missing_optional_artifact_does_not_trigger_repair(self) -> None:
        scope, state = optional_state(envelope(entry(APP), entry(ROUTER)))
        decision = plan_repair(scope, state, attempt=1, max_attempts=3)
        self.assertIs(decision.status, RepairStatus.SATISFIED)
        self.assertIsNone(decision.request)

    def test_a_submitted_but_invalid_optional_artifact_is_repaired(self) -> None:
        scope, state = optional_state(envelope(
            entry(APP), entry(ROUTER), entry(LAYOUT, content=TRUNCATED_LAYOUT),
        ))
        decision = plan_repair(scope, state, attempt=1, max_attempts=3)
        self.assertIs(decision.status, RepairStatus.REPAIRABLE)
        self.assertEqual(decision.request.target_artifact_ids, (LAYOUT,))
        self.assertFalse(decision.request.targets[0].required)

    def test_policy_can_refuse_to_repair_a_rejected_optional_artifact(self) -> None:
        scope, state = optional_state(envelope(
            entry(APP), entry(ROUTER), entry(LAYOUT, content=TRUNCATED_LAYOUT),
        ))
        decision = plan_repair(
            scope, state, attempt=1, max_attempts=3,
            policy=ArtifactRepairPolicy(repair_rejected_optional=False),
        )
        self.assertIs(decision.status, RepairStatus.NON_REPAIRABLE)


class TestNarrowRepairScope(unittest.TestCase):
    def scope_and_request(self):
        state = first_attempt(envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}))
        request = decide(state).request
        return scope_for("shell"), request

    def test_the_repair_scope_owns_only_the_targets(self) -> None:
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        self.assertEqual(narrow.owned_artifact_ids, (LAYOUT,))
        self.assertEqual(narrow.output_artifact_ids, (LAYOUT,))
        self.assertEqual([s.id for s in narrow.owned_specs], [LAYOUT])

    def test_the_original_scope_is_not_mutated(self) -> None:
        scope, request = self.scope_and_request()
        before = dataclasses.astuple(scope)
        derive_repair_scope(scope, request)
        self.assertEqual(dataclasses.astuple(scope), before)
        self.assertEqual(scope.output_artifact_ids, (APP, ROUTER, LAYOUT))

    def test_identity_and_ownership_survive(self) -> None:
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        self.assertEqual(narrow.manifest_id, scope.manifest_id)
        self.assertEqual(narrow.subtask_id, "shell")
        self.assertEqual(narrow.package_ids, ("application-shell",))
        self.assertEqual(narrow.artifact_owners, ((LAYOUT, "application-shell"),))

    def test_preserved_artifacts_are_named_but_not_owned(self) -> None:
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        self.assertNotIn(APP, narrow.owned_artifact_ids)
        self.assertIn(APP, narrow.input_artifact_ids)
        self.assertIn("preserved", narrow.brief.lower())

    def test_the_manifest_id_space_is_retained_for_rejection_accuracy(self) -> None:
        # An unrelated file must be rejected as OUT_OF_SCOPE, not as an unknown
        # id: the narrow scope still knows the whole manifest.
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        self.assertIn(PKG, narrow.known_artifact_ids)
        self.assertIn(PAGE, narrow.known_artifact_ids)

    def test_criteria_and_validations_are_carried(self) -> None:
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        self.assertEqual(narrow.validation_ids, scope.validation_ids)
        self.assertEqual(narrow.completion_criteria, scope.completion_criteria)

    def test_the_rendered_scope_block_stays_bounded(self) -> None:
        scope, request = self.scope_and_request()
        narrow = derive_repair_scope(scope, request)
        block = narrow.to_worker_block()
        self.assertLessEqual(len(block), narrow.max_render_chars)
        self.assertIn(LAYOUT, block)
        self.assertNotIn(CARD, block)


if __name__ == "__main__":
    unittest.main()
