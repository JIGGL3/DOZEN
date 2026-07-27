"""Phase 4F — deterministic artifact finalization (Part F): no synthesizer
call, exact content, no partial project presented as success."""

from __future__ import annotations

import unittest

from dozen.artifact_results import (
    AssemblyStatus,
    SubtaskCollectionResult,
    assemble_deliverable,
    merge_subtask_collections,
)
from dozen.finalization import (
    CODE_ONLY_FAILURE_FOOTER,
    CODE_ONLY_FAILURE_HEADLINE,
    decide_finalization,
    finalize_artifact_delivery,
    render_artifact_failure,
    render_code_only_deliverable,
)
from dozen.intent import resolve_contract

from ..artifact_results.harness import (
    CONTENT,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    react_work_plan,
    scope_for,
)

CARD = "src/components/ui/Card.tsx"


def two_branch_collection(payload_a, payload_b) -> SubtaskCollectionResult:
    """Two producing delegations inside the shared-ui package's recursion."""
    scope = scope_for("shared-ui")
    return merge_subtask_collections(
        [
            collect_for("shared-ui", payload_a, producer_subtask_id="inner-a",
                       recursion_depth=1),
            collect_for("shared-ui", payload_b, producer_subtask_id="inner-b",
                       recursion_depth=1),
        ],
        scope,
    )

CODE_ONLY = resolve_contract("Build me a React dashboard. (give code only)")
PLAIN_IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")


class TestCompleteAssembly(unittest.TestCase):
    def test_complete_assembly_renders_every_file_exactly_once(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(work_plan, collect_dashboard())
        self.assertEqual(assembly.status, AssemblyStatus.COMPLETE)
        decision = decide_finalization(
            contract=CODE_ONLY, artifact_plan=work_plan, assembly=assembly
        )
        text = finalize_artifact_delivery(assembly, decision)
        for path, content in CONTENT.items():
            self.assertEqual(text.count(f"### {path}"), 1, path)
            self.assertIn(content.strip(), text)

    def test_complete_assembly_preserves_exact_content(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(work_plan, collect_dashboard())
        text = render_code_only_deliverable(assembly)
        self.assertIn(CONTENT["src/app/App.tsx"], text)
        self.assertIn(CONTENT["package.json"], text)

    def test_no_headline_summary_or_confidence_in_code_only_render(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(work_plan, collect_dashboard())
        text = render_code_only_deliverable(assembly)
        for banned in ("Implemented the package.", "confidence", "summary",
                       "key_decisions", "All required artifacts were"):
            self.assertNotIn(banned, text)


class TestIncompleteAssembly(unittest.TestCase):
    def test_partial_assembly_never_presents_as_success(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(
            work_plan, collect_dashboard(overrides={"tests": None})
        )
        self.assertEqual(assembly.status, AssemblyStatus.PARTIAL)
        decision = decide_finalization(
            contract=CODE_ONLY, artifact_plan=work_plan, assembly=assembly
        )
        text = finalize_artifact_delivery(assembly, decision)
        self.assertTrue(text.startswith(CODE_ONLY_FAILURE_HEADLINE))
        self.assertIn("tests/dashboard.test.tsx", text)
        self.assertTrue(text.rstrip().endswith(CODE_ONLY_FAILURE_FOOTER))

    def test_missing_required_artifacts_are_named_by_id_only(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(
            work_plan,
            collect_dashboard(overrides={"shell": None, "tests": None}),
        )
        text = render_artifact_failure(assembly)
        self.assertIn("src/app/App.tsx", text)
        self.assertIn("tests/dashboard.test.tsx", text)
        self.assertIn("Missing required files:", text)

    def test_conflicted_assembly_never_presents_as_success(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(
            work_plan,
            collect_dashboard(overrides={"shared-ui": None}) + [
                two_branch_collection(
                    envelope(entry(CARD, content="export function Card() { return 1; }")),
                    envelope(entry(CARD, content="export function Card() { return 2; }")),
                )
            ],
        )
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        decision = decide_finalization(
            contract=CODE_ONLY, artifact_plan=work_plan, assembly=assembly
        )
        text = finalize_artifact_delivery(assembly, decision)
        self.assertTrue(text.startswith(CODE_ONLY_FAILURE_HEADLINE))
        self.assertIn("unresolved conflicting submissions", text.lower())

    def test_integrity_rejection_never_presents_as_success(self) -> None:
        work_plan = react_work_plan()
        truncated = envelope(entry(
            "src/app/App.tsx",
            content="export default function App() {\n  return <div>",
        ))
        assembly = assemble_deliverable(
            work_plan, collect_dashboard(overrides={
                "shell": [truncated,
                          *(entry(a) for a in (
                              "src/app/router.tsx",
                              "src/components/layout/DashboardLayout.tsx",
                          ))],
            }),
        )
        self.assertNotEqual(assembly.status, AssemblyStatus.COMPLETE)
        decision = decide_finalization(
            contract=CODE_ONLY, artifact_plan=work_plan, assembly=assembly
        )
        text = finalize_artifact_delivery(assembly, decision)
        self.assertTrue(text.startswith(CODE_ONLY_FAILURE_HEADLINE))

    def test_optional_missing_artifacts_do_not_block_success(self) -> None:
        # Every REQUIRED artifact present; nothing optional is declared in the
        # react fixture, so a complete run stays COMPLETE (regression guard —
        # optional-vs-required must not be conflated in the failure path).
        work_plan = react_work_plan()
        assembly = assemble_deliverable(work_plan, collect_dashboard())
        self.assertEqual(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(assembly.missing_optional_artifact_ids, ())

    def test_no_assembly_at_all_yields_one_clean_failure(self) -> None:
        decision = decide_finalization(contract=CODE_ONLY, artifact_plan=None)
        text = finalize_artifact_delivery(None, decision)
        self.assertIn(CODE_ONLY_FAILURE_HEADLINE, text)
        self.assertIn(CODE_ONLY_FAILURE_FOOTER, text)


class TestNoSynthesizerInvocation(unittest.TestCase):
    def test_finalize_artifact_delivery_takes_no_client(self) -> None:
        import inspect
        parameters = inspect.signature(finalize_artifact_delivery).parameters
        self.assertNotIn("client", parameters)
        self.assertNotIn("synthesizer", parameters)

    def test_render_functions_never_import_the_synthesizer(self) -> None:
        # The module borrows one shared JSON-fence helper from llm_client (the
        # same local-import pattern dozen.validation already uses to avoid a
        # cycle) — but never the Synthesizer class or a provider call.
        import dozen.finalization as mod
        source = inspect_source(mod)
        self.assertNotIn("Synthesizer", source)
        self.assertNotIn(".complete(", source)
        self.assertNotIn("import Synthesizer", source)


def inspect_source(module) -> str:
    import inspect
    return inspect.getsource(module)


class TestNoWorkerProseInCodeOnly(unittest.TestCase):
    def test_worker_summary_text_never_enters_the_code_only_render(self) -> None:
        work_plan = react_work_plan()
        assembly = assemble_deliverable(
            work_plan,
            collect_dashboard(overrides={
                "shell": envelope(
                    *(entry(a) for a in (
                        "src/app/App.tsx", "src/app/router.tsx",
                        "src/components/layout/DashboardLayout.tsx",
                    )),
                    summary="I have implemented the shell with great care.",
                ),
            }),
        )
        text = render_code_only_deliverable(assembly)
        self.assertNotIn("implemented the shell with great care", text)


if __name__ == "__main__":
    unittest.main()
