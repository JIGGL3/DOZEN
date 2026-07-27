"""Phase 4E Parts M/N — root assembly after repair, and the compact result trace.

Repair adds no new assembly status; a repaired file assembles normally, an
exhausted target stays missing, and root conflicts are NOT repaired in this phase.
"""

from __future__ import annotations

import json
import unittest

from dozen import Orchestrator
from dozen.agent_pool import AgentPool, AgentSpec
from dozen.artifact_results import (
    AssemblyStatus,
    assemble_deliverable,
    render_assembled_deliverable,
)
from dozen.config import OrchestratorConfig
from dozen.intent import resolve_contract
from dozen.llm_client import LLMClient, LLMResponse
from dozen.models import Task, TaskStatus

from ..artifact_decomposition.harness import (
    react_plan,
    react_plan_json,
    react_work_plan,
)
from ..artifact_results.test_executor_collection import (
    EnvelopeClient,
    make_executor,
    run,
)
from ..artifact_results.test_synthesis_and_result import REACT_INSTRUCTION
from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    ROUTER,
    entry,
    envelope,
    envelope_with,
    worker_envelope_for,
)

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")
TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


def other_envelopes():
    return {sid: worker_envelope_for(sid)
            for sid in ("foundation", "shared-ui", "dashboard", "tests", "validation")}


def assemble_from_executor(shell_payloads, *, repairs=1):
    client = EnvelopeClient({"shell": shell_payloads, **other_envelopes()})
    results = list(run(make_executor(client, repairs=repairs)).values())
    collections = [
        r.artifact_collection for r in results if r.artifact_collection is not None
    ]
    return assemble_deliverable(react_work_plan(), collections)


class TestRootAssemblyAfterRepair(unittest.TestCase):
    def test_a_repaired_package_assembles_completely(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        self.assertIs(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(len(assembly.artifacts), 11)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, ())

    def test_preserved_and_repaired_coexist_with_their_provenance(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        by_id = assembly.by_artifact()
        self.assertEqual(by_id[APP].attempt, 1)
        self.assertEqual(by_id[ROUTER].attempt, 1)
        self.assertEqual(by_id[LAYOUT].attempt, 2)
        self.assertEqual(by_id[LAYOUT].content, CONTENT[LAYOUT])

    def test_a_repaired_artifact_no_longer_appears_as_rejected(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        self.assertNotIn(LAYOUT, assembly.integrity_rejected_artifact_ids)
        self.assertNotIn(LAYOUT, assembly.missing_required_artifact_ids)

    def test_an_exhausted_target_remains_missing(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})],
        )
        self.assertIs(assembly.status, AssemblyStatus.PARTIAL)
        self.assertIn(LAYOUT, assembly.missing_required_artifact_ids)
        self.assertEqual(assembly.integrity_rejected_artifact_ids, (LAYOUT,))

    def test_a_preserved_overwrite_attempt_fails_the_assembly(self) -> None:
        assembly = assemble_from_executor([
            envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
            envelope(
                entry(APP, content="export default function App() { return null; }\n"),
                entry(LAYOUT),
            ),
        ])
        self.assertIs(assembly.status, AssemblyStatus.FAILED)
        self.assertEqual(assembly.by_artifact()[APP].content, CONTENT[APP])

    def test_no_repaired_status_exists(self) -> None:
        self.assertEqual(
            {s.value for s in AssemblyStatus},
            {"complete", "partial", "conflicted", "failed"},
        )

    def test_the_final_answer_contains_each_valid_file_once(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        text = render_assembled_deliverable(assembly)
        self.assertEqual(text.count(f"### {LAYOUT}"), 1)
        self.assertEqual(text.count(f"### {APP}"), 1)
        self.assertIn(CONTENT[LAYOUT], text)

    def test_invalid_bodies_are_never_rendered_as_delivered(self) -> None:
        assembly = assemble_from_executor(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})],
        )
        text = render_assembled_deliverable(assembly)
        self.assertIn("INCOMPLETE DELIVERABLE", text)
        self.assertNotIn("return <main>\n\nReply", text)


class MultiAttemptRunClient(LLMClient):
    """A full artifact-plan run whose workers may return a DIFFERENT scripted
    envelope per attempt (a list is consumed one entry per call)."""

    def __init__(self, envelopes: dict) -> None:
        super().__init__(mock=True)
        self.envelopes = envelopes
        self.calls: dict[str, int] = {}

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[1].content
        if "MANAGER" in system:
            return LLMResponse(text=json.dumps(react_plan_json()),
                               provider=provider, model=model)
        if "Worker" in system:
            for subtask_id, payload in self.envelopes.items():
                if REACT_INSTRUCTION[subtask_id] in user:
                    if isinstance(payload, list):
                        index = min(self.calls.get(subtask_id, 0), len(payload) - 1)
                        self.calls[subtask_id] = index + 1
                        payload = payload[index]
                    return LLMResponse(text=json.dumps(payload), provider=provider,
                                       model=model)
            return LLMResponse(text="```tsx\nexport const x = 1;\n```",
                               provider=provider, model=model)
        return LLMResponse(text="stitched", provider=provider, model=model)


class TestFullOrchestratorRun(unittest.TestCase):
    def orchestrator(self, shell_payloads, *, repairs=1):
        agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
        client = MultiAttemptRunClient({"shell": shell_payloads, **other_envelopes()})
        return Orchestrator(
            client=client, pool=AgentPool([agent]),
            config=OrchestratorConfig(
                max_parallelism=1, max_repair_attempts=repairs, verify_outputs=False,
                use_llm_router=False, max_depth=1,
            ),
        )

    def test_a_complete_repaired_project_reports_no_error(self) -> None:
        orch = self.orchestrator(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        result = orch.run("I want you to build me a React-based dashboard with tests.")
        self.assertEqual(result.error, "")
        self.assertIs(result.artifact_assembly.status, AssemblyStatus.COMPLETE)

    def test_the_compact_summary_carries_repair_counts(self) -> None:
        orch = self.orchestrator(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope(entry(LAYOUT))]
        )
        result = orch.run("I want you to build me a React-based dashboard with tests.")
        summary = result.summary()
        self.assertIn("artifact_repair", summary)
        repair = summary["artifact_repair"]
        self.assertTrue(repair["repair_attempted"])
        self.assertEqual(repair["artifacts_repaired"], 1)
        self.assertEqual(repair["artifacts_preserved"], 2)
        # Counts and ids only: no artifact content leaks into a summary.
        self.assertNotIn("export function", json.dumps(summary))

    def test_a_healthy_run_omits_the_repair_summary(self) -> None:
        orch = self.orchestrator(worker_envelope_for("shell"))
        result = orch.run("I want you to build me a React-based dashboard with tests.")
        self.assertNotIn("artifact_repair", result.summary())

    def test_an_exhausted_repair_reports_the_incomplete_deliverable(self) -> None:
        orch = self.orchestrator(
            [envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
             envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})],
        )
        result = orch.run("I want you to build me a React-based dashboard with tests.")
        self.assertTrue(result.error)
        self.assertIs(result.artifact_assembly.status, AssemblyStatus.PARTIAL)


class TestRootConflictNotRepaired(unittest.TestCase):
    def test_a_root_conflict_stays_conflicted(self) -> None:
        # Two producers submit different bodies for the same shared artifact.
        from ..artifact_results.harness import collect_for as collect
        first = collect(
            "shared-ui",
            envelope(entry("src/components/ui/Card.tsx", content="export const A = 1;\n")),
            producer_subtask_id="a", recursion_depth=1,
        )
        second = collect(
            "shared-ui",
            envelope(entry("src/components/ui/Card.tsx", content="export const B = 2;\n")),
            producer_subtask_id="b", recursion_depth=1,
        )
        from ..artifact_results.harness import collect_dashboard
        base = collect_dashboard({"shared-ui": None})
        assembly = assemble_deliverable(react_work_plan(), base + [first, second])
        # Phase 4E does not resolve or repair this — it remains conflicted.
        self.assertIs(assembly.status, AssemblyStatus.CONFLICTED)


if __name__ == "__main__":
    unittest.main()
