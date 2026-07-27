"""Phase 4D Parts J/K/L/M — the existing repair loop, root assembly, recursion
and synthesis, under structural-integrity rejection.

No new retry loop, no new provider call, no repair, no filesystem write.
"""

from __future__ import annotations

import dataclasses
import unittest

from dozen.artifact_results import (
    AssemblyStatus,
    CollectionErrorCode,
    assemble_deliverable,
    merge_subtask_collections,
    render_assembled_deliverable,
)
from dozen.models import TaskStatus

from ..artifact_decomposition.harness import react_work_plan
from ..artifact_results.test_executor_collection import (
    EnvelopeClient,
    all_envelopes,
    make_executor,
    run,
)
from .harness import (
    APP_CUT_IN_ATTRIBUTE,
    CARD_BRACES_IN_STRINGS,
    PACKAGE_JSON_MISSING_BRACE,
    ROUTER_CUT_AFTER_IMPORT,
    assemble_dashboard,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    envelope_with,
    scope_for,
    worker_envelope_for,
)

APP = "src/app/App.tsx"
CARD = "src/components/ui/Card.tsx"
PKG = "package.json"


def broken_shell() -> dict:
    return envelope_with("shell", **{APP: APP_CUT_IN_ATTRIBUTE})


def optional_test_result_plan():
    """The approved work plan with ``test-result`` marked OPTIONAL."""
    plan = react_work_plan()
    manifest = dataclasses.replace(
        plan.manifest,
        artifacts=tuple(
            dataclasses.replace(spec, required=False)
            if spec.id == "test-result" else spec
            for spec in plan.manifest.artifacts
        ),
    )
    return dataclasses.replace(plan, manifest=manifest)


class TestExistingRepairLoop(unittest.TestCase):
    def test_13_a_truncated_attempt_is_repaired_by_the_next_attempt(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), worker_envelope_for("shell")],
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertTrue(shell.artifact_collection.satisfied)
        self.assertEqual(len(shell.artifact_collection.accepted), 3)
        retry_prompt = [
            p for p in client.worker_prompts if "Do the shell work." in p
        ][1]
        self.assertIn("REVISION REQUIRED", retry_prompt)
        self.assertIn(APP, retry_prompt)
        self.assertIn("incomplete", retry_prompt)

    def test_14_a_repeated_truncated_attempt_fails_deterministically(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), broken_shell()],
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertFalse(shell.artifacts_satisfied)
        self.assertEqual(
            shell.artifact_collection.integrity_rejected_artifact_ids, (APP,)
        )
        # The prose answer is still a genuine answer; only the artifact failed.
        self.assertEqual(shell.status, TaskStatus.COMPLETED)

    def test_the_repair_feedback_is_focused_and_bounded(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), worker_envelope_for("shell")],
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        run(make_executor(client, repairs=1))
        retry_prompt = [
            p for p in client.worker_prompts if "Do the shell work." in p
        ][1]
        # It asks for this package's artifacts back — never unrelated files, and
        # never a fragment/continuation.
        self.assertNotIn(PKG, retry_prompt.split("REVISION REQUIRED")[1])
        self.assertNotIn("continue from", retry_prompt.lower())

    def test_no_extra_provider_call_is_made(self) -> None:
        client = EnvelopeClient({
            "shell": broken_shell(),
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        run(make_executor(client, repairs=0))
        shell_calls = [p for p in client.worker_prompts if "Do the shell work." in p]
        self.assertEqual(len(shell_calls), 1)

    def test_a_structurally_valid_worker_is_never_retried(self) -> None:
        client = EnvelopeClient(all_envelopes())
        run(make_executor(client, repairs=2))
        self.assertEqual(client.worker_calls, 6)

    def test_nothing_is_repaired_or_rewritten_by_the_detector(self) -> None:
        client = EnvelopeClient({
            "shell": broken_shell(),
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        shell = run(make_executor(client, repairs=0))["shell"]
        accepted = {c.artifact_id for c in shell.artifact_collection.accepted}
        self.assertNotIn(APP, accepted)          # not repaired into existence
        self.assertNotIn(APP_CUT_IN_ATTRIBUTE, [c.content for c in
                                                shell.artifact_collection.accepted])


class TestRootAssembly(unittest.TestCase):
    def test_a_structural_rejection_leaves_the_required_artifact_missing(self) -> None:
        assembly = assemble_dashboard({"shell": broken_shell()})
        self.assertIs(assembly.status, AssemblyStatus.PARTIAL)
        self.assertIn(APP, assembly.missing_required_artifact_ids)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, (APP,))

    def test_missing_and_truncated_are_distinguishable(self) -> None:
        # foundation omits package.json entirely; shell truncates App.tsx.
        assembly = assemble_dashboard({
            "foundation": envelope(entry("tsconfig.json"), entry("src/main.tsx")),
            "shell": broken_shell(),
        })
        self.assertIn(PKG, assembly.missing_required_artifact_ids)
        self.assertIn(APP, assembly.missing_required_artifact_ids)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, (APP,))
        self.assertNotIn(PKG, assembly.integrity_rejected_artifact_ids)

    def test_a_complete_run_of_valid_artifacts_stays_complete(self) -> None:
        assembly = assemble_dashboard()
        self.assertIs(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, ())

    def test_a_required_result_artifact_rejection_blocks_complete(self) -> None:
        assembly = assemble_dashboard({
            "validation": envelope(
                entry("build-result", content="build ok\n... output truncated"),
                entry("test-result"),
            ),
        })
        self.assertIn("build-result", assembly.integrity_rejected_artifact_ids)
        self.assertIs(assembly.status, AssemblyStatus.PARTIAL)

    def test_an_optional_artifact_rejection_is_reported_not_ignored(self) -> None:
        plan = optional_test_result_plan()
        assembly = assemble_dashboard(
            {
                "validation": envelope(
                    entry("build-result"),
                    entry("test-result", content="vitest: 4 passed\n[truncated]"),
                ),
            },
            work_plan=plan,
        )
        # An OPTIONAL artifact that came back truncated is never quietly dropped:
        # it is named, it stays out of the assembled set, and the run is not
        # allowed to call itself complete on the strength of a broken file.
        self.assertEqual(assembly.integrity_rejected_artifact_ids, ("test-result",))
        self.assertIn("test-result", assembly.missing_optional_artifact_ids)
        self.assertNotIn("test-result", [c.artifact_id for c in assembly.artifacts])
        self.assertIs(assembly.status, AssemblyStatus.PARTIAL)

    def test_an_invalid_candidate_cannot_participate_in_a_conflict(self) -> None:
        first = collect_for(
            "shared-ui", envelope(entry(CARD, content=CARD_BRACES_IN_STRINGS)),
            producer_subtask_id="branch-a", recursion_depth=1,
        )
        truncated = collect_for(
            "shared-ui",
            envelope(entry(CARD, content="export function Card() {\n  return <div>")),
            producer_subtask_id="branch-b", recursion_depth=1,
        )
        base = collect_dashboard({"shared-ui": None})
        assembly = assemble_deliverable(
            react_work_plan(), base + [first, truncated]
        )
        # One VALID candidate remains: no competing content, so no conflict.
        self.assertEqual(assembly.conflicts, ())
        self.assertEqual(
            assembly.by_artifact()[CARD].content, CARD_BRACES_IN_STRINGS
        )
        self.assertIn(CollectionErrorCode.STRUCTURALLY_INVALID,
                      [r.code for r in assembly.rejected])

    def test_two_valid_competing_candidates_still_conflict(self) -> None:
        first = collect_for(
            "shared-ui", envelope(entry(CARD, content=CARD_BRACES_IN_STRINGS)),
            producer_subtask_id="branch-a", recursion_depth=1,
        )
        second = collect_for(
            "shared-ui",
            envelope(entry(CARD, content="export const Card = () => <p>b</p>;\n")),
            producer_subtask_id="branch-b", recursion_depth=1,
        )
        base = collect_dashboard({"shared-ui": None})
        assembly = assemble_deliverable(react_work_plan(), base + [first, second])
        self.assertIs(assembly.status, AssemblyStatus.CONFLICTED)

    def test_assembly_is_order_independent(self) -> None:
        collections = collect_dashboard({"shell": broken_shell()})
        forward = assemble_deliverable(react_work_plan(), collections)
        reverse = assemble_deliverable(react_work_plan(), list(reversed(collections)))
        self.assertEqual(forward, reverse)

    def test_a_fatal_integrity_failure_is_not_a_fatal_ownership_failure(self) -> None:
        # Truncation is REPAIRABLE, so it must degrade to PARTIAL, never FAILED.
        assembly = assemble_dashboard({
            "foundation": envelope_with("foundation", **{
                PKG: PACKAGE_JSON_MISSING_BRACE,
            }),
        })
        self.assertIs(assembly.status, AssemblyStatus.PARTIAL)

    def test_the_summary_counts_integrity_rejections(self) -> None:
        assembly = assemble_dashboard({"shell": broken_shell()})
        self.assertEqual(assembly.summary()["integrity_rejected"], 1)

    def test_serialization_round_trips(self) -> None:
        from dozen.artifact_results import AssembledDeliverable

        assembly = assemble_dashboard({"shell": broken_shell()})
        self.assertEqual(
            AssembledDeliverable.from_dict(assembly.to_dict()), assembly
        )


class TestRecursion(unittest.TestCase):
    def test_a_nested_truncated_artifact_stays_rejected_at_the_parent(self) -> None:
        scope = scope_for("shell")
        child = collect_for(
            "shell", broken_shell(), producer_subtask_id="nested-writer",
            recursion_depth=2,
        )
        merged = merge_subtask_collections([child], scope)
        self.assertEqual(merged.integrity_rejected_artifact_ids, (APP,))
        self.assertIn(APP, merged.missing_required_artifact_ids)

    def test_recursive_provenance_survives_an_integrity_rejection(self) -> None:
        child = collect_for(
            "shell", broken_shell(), producer_subtask_id="nested-writer",
            recursion_depth=2,
        )
        self.assertEqual(child.rejected[0].subtask_id, "shell")
        surviving = {c.producer_subtask_id for c in child.accepted}
        self.assertEqual(surviving, {"nested-writer"})

    def test_recursive_synthesis_cannot_hide_an_integrity_failure(self) -> None:
        scope = scope_for("shell")
        merged = merge_subtask_collections(
            [collect_for("shell", broken_shell(), producer_subtask_id="deep",
                         recursion_depth=3)],
            scope,
        )
        self.assertFalse(merged.satisfied)
        base = collect_dashboard({"shell": None})
        assembly = assemble_deliverable(react_work_plan(), base + [merged])
        self.assertIsNot(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, (APP,))


class TestSynthesisRendering(unittest.TestCase):
    def test_a_rejected_body_is_never_rendered_as_a_delivered_file(self) -> None:
        assembly = assemble_dashboard({"shell": broken_shell()})
        text = render_assembled_deliverable(assembly)
        self.assertIn("INCOMPLETE DELIVERABLE", text)
        self.assertIn("Structurally incomplete artifacts", text)
        self.assertIn(APP, text)
        self.assertNotIn('<button className="', text)

    def test_the_defect_is_described_not_repaired(self) -> None:
        assembly = assemble_dashboard({"shell": broken_shell()})
        text = render_assembled_deliverable(assembly)
        self.assertIn("rejected, not repaired", text)
        self.assertIn("incomplete", text)

    def test_valid_artifacts_are_still_rendered_verbatim(self) -> None:
        assembly = assemble_dashboard({"shell": broken_shell()})
        text = render_assembled_deliverable(assembly)
        self.assertIn("### tsconfig.json", text)
        self.assertIn('"jsx": "react-jsx"', text)

    def test_a_complete_assembly_renders_exactly_as_before(self) -> None:
        text = render_assembled_deliverable(assemble_dashboard())
        self.assertIn("All required artifacts were produced", text)
        self.assertNotIn("Structurally incomplete", text)

    def test_rendering_is_deterministic(self) -> None:
        first = render_assembled_deliverable(assemble_dashboard({"shell": broken_shell()}))
        second = render_assembled_deliverable(assemble_dashboard({"shell": broken_shell()}))
        self.assertEqual(first, second)

    def test_diagnostics_never_carry_the_whole_body(self) -> None:
        big = "export const x = {\n" + "  // pad\n" * 2000
        assembly = assemble_dashboard({
            "shared-ui": envelope(entry(CARD, content=big)),
        })
        for rejection in assembly.rejected:
            self.assertLessEqual(len(rejection.message), 300)


if __name__ == "__main__":
    unittest.main()
