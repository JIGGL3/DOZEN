"""Phase 4C — synthesis presentation and terminal-result consistency."""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.artifact_results import AssemblyStatus, render_assembled_deliverable
from dozen.config import OrchestratorConfig
from dozen.models import Plan, SubTaskResult, Task, TaskStatus
from dozen.synthesizer import Synthesizer

from ..artifact_decomposition.harness import react_plan_json
from .harness import (
    CONTENT,
    assemble_dashboard,
    entry,
    envelope,
    worker_envelope_for,
)

CARD = "src/components/ui/Card.tsx"


class ExplodingClient(LLMClient):
    """Any synthesis call would fail — the deterministic path must not need one."""

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        raise RuntimeError("no model call should be needed to present an assembly")


def synthesizer(client=None) -> Synthesizer:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"writing": 0.9}, tier=3)
    return Synthesizer(client or ExplodingClient(mock=True), agent)


class TestSynthesisPresentation(unittest.TestCase):
    def synthesize(self, assembly) -> str:
        return synthesizer().synthesize(
            Task(prompt="Build me a React dashboard."),
            Plan(analysis="a", subtasks=[], synthesis_strategy="s"),
            [],
            assembly=assembly,
        )

    def test_complete_assembly_is_presented_with_paths_and_content(self) -> None:
        text = self.synthesize(assemble_dashboard())
        for path, content in CONTENT.items():
            self.assertIn(f"### {path}", text)
            self.assertIn(content.strip(), text)
        self.assertIn("All required artifacts were produced", text)

    def test_no_model_call_and_no_filesystem_write_are_needed(self) -> None:
        # ExplodingClient raises on any call; reaching here proves determinism.
        self.assertTrue(self.synthesize(assemble_dashboard()))

    def test_partial_assembly_is_clearly_represented(self) -> None:
        assembly = assemble_dashboard({
            "shell": envelope(entry("src/app/App.tsx"))
        })
        text = self.synthesize(assembly)
        self.assertEqual(assembly.status, AssemblyStatus.PARTIAL)
        self.assertIn("INCOMPLETE DELIVERABLE", text)
        self.assertIn("Missing required artifacts", text)
        self.assertIn("src/app/router.tsx", text)
        # Everything that WAS produced is still presented in full.
        self.assertIn(CONTENT["src/app/App.tsx"].strip(), text)

    def test_conflicted_content_is_never_presented_as_successful(self) -> None:
        from dozen.artifact_results import assemble_deliverable, merge_subtask_collections

        from .harness import collect_dashboard, collect_for, react_work_plan, scope_for

        conflicted = merge_subtask_collections(
            [
                collect_for("shared-ui", envelope(entry(CARD, content="VERSION ONE")),
                            producer_subtask_id="a", recursion_depth=1),
                collect_for("shared-ui", envelope(entry(CARD, content="VERSION TWO")),
                            producer_subtask_id="b", recursion_depth=1),
            ],
            scope_for("shared-ui"),
        )
        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard({"shared-ui": None}) + [conflicted]
        )
        text = self.synthesize(assembly)
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        self.assertIn("CONFLICTED DELIVERABLE", text)
        self.assertIn("Conflicting artifacts", text)
        self.assertNotIn("VERSION ONE", text)
        self.assertNotIn("VERSION TWO", text)
        self.assertNotIn(f"### {CARD}", text)

    def test_duplicates_and_rejections_are_listed(self) -> None:
        assembly = assemble_dashboard({
            "dashboard": envelope(
                entry("src/features/dashboard/DashboardPage.tsx"),
                entry("package.json"),
            )
        })
        text = self.synthesize(assembly)
        self.assertIn("Rejected artifact submissions", text)
        self.assertIn("out_of_scope", text)

    def test_rendering_is_deterministic_and_lossless(self) -> None:
        assembly = assemble_dashboard()
        self.assertEqual(
            render_assembled_deliverable(assembly),
            render_assembled_deliverable(assembly),
        )
        text = render_assembled_deliverable(assembly)
        for content in CONTENT.values():
            self.assertIn(content.strip(), text)

    def test_content_with_backticks_is_never_rewritten(self) -> None:
        from dozen.artifact_results import assemble_deliverable

        from .harness import collect_dashboard, collect_for, react_work_plan

        # A valid JS string containing Markdown fence characters. The renderer
        # must widen its presentation fence without rewriting the source body.
        tricky = 'const md = "```tsx\\nnested\\n```";\n'
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [
                collect_for("shared-ui", envelope(entry(CARD, content=tricky)))
            ],
        )
        text = render_assembled_deliverable(assembly)
        self.assertIn(tricky, text)
        self.assertEqual(assembly.by_artifact()[CARD].content, tricky)


class TestLegacySynthesisUnchanged(unittest.TestCase):
    def test_synthesis_without_an_assembly_is_untouched(self) -> None:
        class Client(LLMClient):
            def complete(self, *, provider, model, messages, **kwargs):
                return LLMResponse(text="stitched answer", provider=provider,
                                   model=model)

        results = [
            SubTaskResult(subtask_id="a", title="A", status=TaskStatus.COMPLETED,
                          output="alpha"),
            SubTaskResult(subtask_id="b", title="B", status=TaskStatus.COMPLETED,
                          output="beta"),
        ]
        answer = synthesizer(Client(mock=True)).synthesize(
            Task(prompt="Explain mutexes."),
            Plan(analysis="a", subtasks=[], synthesis_strategy="s"),
            results,
        )
        self.assertEqual(answer, "stitched answer")


# --------------------------------------------------------------------------- #
# Terminal result consistency (Part K)
# --------------------------------------------------------------------------- #
class ScriptedRunClient(LLMClient):
    """A full artifact-plan run whose six workers return scripted envelopes."""

    def __init__(self, envelopes: dict) -> None:
        super().__init__(mock=True)
        self.envelopes = envelopes

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[1].content
        if "MANAGER" in system:
            return LLMResponse(text=json.dumps(react_plan_json()),
                               provider=provider, model=model)
        if "Worker" in system:
            for subtask_id, payload in self.envelopes.items():
                marker = REACT_INSTRUCTION[subtask_id]
                if marker in user:
                    return LLMResponse(text=json.dumps(payload), provider=provider,
                                       model=model)
            return LLMResponse(text=_LEGACY_CODE, provider=provider, model=model)
        return LLMResponse(text=_LEGACY_CODE, provider=provider, model=model)


_LEGACY_CODE = """```tsx
export default function App() {
  return <div>dashboard</div>;
}
```"""


REACT_INSTRUCTION = {
    "foundation": "Write the complete project configuration",
    "shell": "Write the complete application shell",
    "shared-ui": "Write the complete shared UI",
    "dashboard": "Write the complete dashboard page",
    "tests": "Write complete unit tests",
    "validation": "Write a complete validation summary",
}


def run_dashboard(overrides=None) -> object:
    envelopes = {sid: worker_envelope_for(sid) for sid in REACT_INSTRUCTION}
    envelopes.update(overrides or {})
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
    return Orchestrator(
        client=ScriptedRunClient(envelopes), pool=AgentPool([agent]),
        config=OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                  verify_outputs=False, use_llm_router=False),
    ).run("I want you to build me a React-based dashboard with tests.")


class TestResultConsistency(unittest.TestCase):
    def test_a_complete_run_reports_success(self) -> None:
        result = run_dashboard()
        self.assertEqual(result.error, "")
        self.assertIsNotNone(result.artifact_assembly)
        self.assertEqual(result.artifact_assembly.status, AssemblyStatus.COMPLETE)
        summary = result.summary()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["artifacts"]["assembled"], 11)
        self.assertEqual(summary["artifacts"]["missing_required"], 0)
        self.assertEqual(summary["artifacts"]["conflicts"], 0)
        self.assertEqual(summary["artifacts"]["rejected"], 0)
        self.assertEqual(summary["artifacts"]["validation_results"], 2)
        for content in CONTENT.values():
            self.assertIn(content.strip(), result.final_answer)

    def test_a_partial_required_deliverable_reports_failure(self) -> None:
        result = run_dashboard({"shell": envelope(entry("src/app/App.tsx"))})
        self.assertEqual(result.artifact_assembly.status, AssemblyStatus.PARTIAL)
        self.assertIn("Artifact assembly partial", result.error)
        self.assertIn("src/app/router.tsx", result.error)
        self.assertEqual(result.summary()["status"], "failed")
        self.assertEqual(result.summary()["artifacts"]["missing_required"], 2)
        self.assertIn("INCOMPLETE DELIVERABLE", result.final_answer)

    def test_status_error_summary_and_events_agree(self) -> None:
        events: list[dict] = []
        envelopes = {sid: worker_envelope_for(sid) for sid in REACT_INSTRUCTION}
        envelopes["shell"] = envelope(entry("src/app/App.tsx"))
        agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
        result = Orchestrator(
            client=ScriptedRunClient(envelopes), pool=AgentPool([agent]),
            config=OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                      verify_outputs=False, use_llm_router=False),
        ).run("I want you to build me a React-based dashboard with tests.",
              on_event=events.append)
        self.assertTrue(result.error)
        self.assertEqual(result.summary()["status"], "failed")
        artifact_events = [e for e in events if e.get("phase") == "artifacts"]
        self.assertTrue(artifact_events)
        self.assertTrue(any(e.get("status") == "error" for e in artifact_events))
        self.assertTrue(
            any("partial" in str(e.get("message", "")) for e in artifact_events)
        )

    def test_a_non_artifact_run_is_unchanged(self) -> None:
        class PlainClient(LLMClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "MANAGER" in messages[0].content:
                    return LLMResponse(
                        text=json.dumps({
                            "analysis": "simple",
                            "subtasks": [{"id": "s1", "title": "Explain",
                                          "instruction": "Explain mutexes.",
                                          "assigned_model": "alpha"}],
                            "synthesis_strategy": "present it",
                        }),
                        provider=provider, model=model,
                    )
                return LLMResponse(text="A mutex guards shared state.",
                                   provider=provider, model=model)

        agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9}, tier=4)
        result = Orchestrator(
            client=PlainClient(mock=True), pool=AgentPool([agent]),
            config=OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                      verify_outputs=False, use_llm_router=False),
        ).run("Explain what a mutex does.")
        self.assertEqual(result.error, "")
        self.assertIsNone(result.artifact_assembly)
        self.assertIsNone(result.artifact_collection)
        self.assertNotIn("artifacts", result.summary())

    def test_scoped_workers_cannot_bypass_assembly_with_legacy_output(self) -> None:
        # No scoped worker uses the typed envelope: the root still assembles the
        # empty collections and fails instead of silently claiming legacy success.
        agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
        result = Orchestrator(
            client=ScriptedRunClient({}), pool=AgentPool([agent]),
            config=OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                      verify_outputs=False, use_llm_router=False),
        ).run("I want you to build me a React-based dashboard with tests.")
        self.assertIsNotNone(result.artifact_assembly)
        self.assertEqual(result.artifact_assembly.status, AssemblyStatus.FAILED)
        self.assertTrue(result.error)
        self.assertEqual(result.summary()["status"], "failed")
        self.assertEqual(result.summary()["artifacts"]["assembled"], 0)


if __name__ == "__main__":
    unittest.main()
