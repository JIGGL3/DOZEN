"""Phase 4F — finalization-mode selection from TRUSTED inputs only."""

from __future__ import annotations

import unittest

from dozen.decomposition import derive_execution_scope
from dozen.finalization import (
    DEFAULT_FINALIZATION_POLICY,
    FinalizationMode,
    FinalizationPolicy,
    decide_finalization,
    user_requested_polish,
)
from dozen.intent import resolve_contract

from ..artifact_decomposition.harness import react_work_plan
from .harness import FAILED_PROMPT


class TestModeSelection(unittest.TestCase):
    def test_artifact_manifest_forces_artifact_assembly(self) -> None:
        contract = resolve_contract("Build me a React dashboard with tests.")
        decision = decide_finalization(
            contract=contract, artifact_plan=react_work_plan()
        )
        self.assertIs(decision.mode, FinalizationMode.ARTIFACT_ASSEMBLY)
        self.assertTrue(decision.artifacts_required)
        self.assertTrue(decision.complete_assembly_required)
        self.assertFalse(decision.model_synthesis_allowed)
        self.assertTrue(decision.deterministic_fallback_required)

    def test_code_only_contract_forces_artifact_assembly(self) -> None:
        contract = resolve_contract(FAILED_PROMPT)
        self.assertTrue(contract.code_only)
        decision = decide_finalization(contract=contract, artifact_plan=None)
        self.assertIs(decision.mode, FinalizationMode.ARTIFACT_ASSEMBLY)
        self.assertTrue(decision.code_only)
        self.assertFalse(decision.prose_allowed)
        self.assertFalse(decision.model_synthesis_allowed)

    def test_prose_report_defaults_to_ordered_sections(self) -> None:
        contract = resolve_contract("Write a report on database replication.")
        decision = decide_finalization(contract=contract)
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.model_synthesis_allowed)

    def test_research_workflow_defaults_to_ordered_sections(self) -> None:
        contract = resolve_contract(
            "Research and compare four message queues for our workload."
        )
        decision = decide_finalization(contract=contract)
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)

    def test_explicit_polish_request_no_longer_enables_synthesis(self) -> None:
        # Phase 4G: model-based synthesis is eliminated. Even an explicit
        # "single polished narrative" request stays deterministic — the polish
        # signal is still recognized (kept for telemetry) but selects ordered
        # sections, never optional model synthesis.
        prompt = "Compare the systems and present a single polished narrative."
        self.assertTrue(user_requested_polish(prompt=prompt))
        decision = decide_finalization(
            contract=resolve_contract(prompt),
            polish_requested=user_requested_polish(prompt=prompt),
        )
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.model_synthesis_allowed)
        self.assertTrue(decision.deterministic_fallback_required)

    def test_trusted_polish_config_no_longer_enables_synthesis(self) -> None:
        # Phase 4G: even the trusted config flag cannot re-enable model-based
        # synthesis — the authoritative invariant overrides it.
        policy = FinalizationPolicy(allow_model_polish=True)
        decision = decide_finalization(
            contract=resolve_contract("Explain what a mutex does."),
            policy=policy,
        )
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.model_synthesis_allowed)

    def test_polish_never_overrides_an_artifact_contract(self) -> None:
        decision = decide_finalization(
            contract=resolve_contract("Build me a dashboard."),
            artifact_plan=react_work_plan(),
            polish_requested=True,
            policy=FinalizationPolicy(allow_model_polish=True),
        )
        self.assertIs(decision.mode, FinalizationMode.ARTIFACT_ASSEMBLY)
        self.assertFalse(decision.model_synthesis_allowed)

    def test_json_deliverable_stays_deterministic(self) -> None:
        contract = resolve_contract(
            "Return the current inventory as raw JSON only."
        )
        decision = decide_finalization(contract=contract)
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.model_synthesis_allowed)

    def test_missing_contract_defaults_to_ordered_sections(self) -> None:
        decision = decide_finalization(contract=None)
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)

    def test_contradictory_contract_resolves_safely(self) -> None:
        # "no code" outranks "code only": the prohibition wins and the run
        # remains a deterministic prose deliverable.
        contract = resolve_contract(
            "Design the system. Do not write code. Give code only."
        )
        self.assertFalse(contract.code_only)
        decision = decide_finalization(contract=contract)
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)

    def test_scoped_recursive_child_never_polishes(self) -> None:
        scope = derive_execution_scope(react_work_plan(), "shell")
        decision = decide_finalization(
            contract=resolve_contract("Build the shell."),
            execution_scope=scope,
            scope_output_required=True,
            polish_requested=True,
            policy=FinalizationPolicy(allow_model_polish=True),
        )
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.model_synthesis_allowed)
        self.assertTrue(decision.artifacts_required)

    def test_scoped_support_child_stays_deterministic(self) -> None:
        scope = derive_execution_scope(react_work_plan(), "shell")
        decision = decide_finalization(
            contract=resolve_contract("Research the shell."),
            execution_scope=scope,
            scope_output_required=False,
        )
        self.assertIs(decision.mode, FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(decision.artifacts_required)

    def test_workers_cannot_select_the_mode(self) -> None:
        # The decision function receives no worker output at all — a response
        # field like "finalization_mode" has no seam to enter. The signature
        # accepts only trusted runtime inputs.
        import inspect

        from dozen.finalization import decide_finalization as fn
        parameters = set(inspect.signature(fn).parameters)
        self.assertEqual(
            parameters,
            {"contract", "artifact_plan", "assembly", "execution_scope",
             "scope_output_required", "polish_requested", "policy"},
        )

    def test_policy_defaults_match_the_phase_contract(self) -> None:
        policy = DEFAULT_FINALIZATION_POLICY
        self.assertIs(policy.artifact_default_mode,
                      FinalizationMode.ARTIFACT_ASSEMBLY)
        self.assertIs(policy.prose_default_mode,
                      FinalizationMode.ORDERED_SECTIONS)
        self.assertFalse(policy.allow_model_polish)
        self.assertFalse(policy.allow_partial_artifact_delivery)
        self.assertFalse(policy.code_only_allows_prose)
        self.assertTrue(policy.synthesis_failure_falls_back)
        self.assertTrue(policy.render_failed_sections)
        self.assertGreater(policy.max_sections, 0)
        self.assertGreater(policy.max_section_title_chars, 0)
        self.assertGreater(policy.max_section_diagnostic_chars, 0)
        self.assertEqual(policy.schema_version, 1)


if __name__ == "__main__":
    unittest.main()
