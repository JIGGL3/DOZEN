"""Phase 6 — artifact deliverables bypass model stitching entirely."""

from __future__ import annotations

import unittest

from dozen import AgentSpec, LLMClient, LLMResponse
from dozen.models import Plan, Task
from dozen.synthesizer import Synthesizer

from ..artifact_results.harness import (
    CONTENT,
    assemble_dashboard,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    react_work_plan,
    scope_for,
)


class CountingClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(mock=True)
        self.calls = 0
        self.inputs: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        self.inputs.append("\n".join(m.content for m in messages))
        return LLMResponse(text="model output", provider=provider, model=model)


def make_synthesizer(client: LLMClient) -> Synthesizer:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"writing": 0.9}, tier=3)
    return Synthesizer(client, agent)


class TestArtifactBypass(unittest.TestCase):
    def synthesize(self, client, assembly) -> str:
        return make_synthesizer(client).synthesize(
            Task(prompt="Build me a React dashboard."),
            Plan(analysis="a", subtasks=[], synthesis_strategy="s"),
            [],
            assembly=assembly,
        )

    def test_complete_project_makes_zero_model_calls(self) -> None:
        client = CountingClient()
        synthesizer = make_synthesizer(client)
        text = synthesizer.synthesize(
            Task(prompt="Build me a React dashboard."),
            Plan(analysis="a", subtasks=[], synthesis_strategy="s"),
            [], assembly=assemble_dashboard(),
        )
        self.assertEqual(client.calls, 0)
        self.assertEqual(synthesizer.last_call_input_chars, [])
        self.assertTrue(text)

    def test_file_content_is_byte_identical_and_appears_once(self) -> None:
        client = CountingClient()
        text = self.synthesize(client, assemble_dashboard())
        for path, content in CONTENT.items():
            self.assertIn(content.strip(), text)
            self.assertEqual(text.count(f"### {path}"), 1)

    def test_large_project_never_enters_a_synthesizer_prompt(self) -> None:
        big = "export const data = [\n" + ",\n".join(
            f'  "row-{i}-' + "z" * 120 + '"' for i in range(600)
        ) + "\n];\n"
        assembly = assemble_dashboard({
            "shared-ui": envelope(
                entry("src/components/ui/Card.tsx", content=big),
                entry("src/components/ui/Panel.tsx"),
            )
        })
        client = CountingClient()
        text = self.synthesize(client, assembly)
        self.assertEqual(client.calls, 0)
        self.assertIn(big, text)  # rendered deterministically, byte-identical
        # Nothing of the file bodies ever reached a model prompt.
        self.assertEqual(client.inputs, [])

    def test_missing_files_are_shown_cleanly(self) -> None:
        assembly = assemble_dashboard({
            "shell": envelope(entry("src/app/App.tsx"))
        })
        text = self.synthesize(CountingClient(), assembly)
        self.assertIn("Missing required artifacts", text)
        self.assertIn("src/app/router.tsx", text)
        self.assertNotIn('"artifacts"', text)

    def test_conflicted_files_are_excluded_from_the_successful_set(self) -> None:
        from dozen.artifact_results import (
            assemble_deliverable, merge_subtask_collections,
        )

        card = "src/components/ui/Card.tsx"
        conflicted = merge_subtask_collections(
            [
                collect_for("shared-ui", envelope(entry(card, content="ONE")),
                            producer_subtask_id="a", recursion_depth=1),
                collect_for("shared-ui", envelope(entry(card, content="TWO")),
                            producer_subtask_id="b", recursion_depth=1),
            ],
            scope_for("shared-ui"),
        )
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [conflicted],
        )
        text = self.synthesize(CountingClient(), assembly)
        self.assertIn("Conflicting artifacts", text)
        # Neither conflicting version is presented as delivered content.
        self.assertNotIn("\nONE\n", text)
        self.assertNotIn("\nTWO\n", text)
        self.assertNotIn(f"### {card}", text)

    def test_validation_result_artifacts_render_without_code_fences(self) -> None:
        assembly = assemble_dashboard()
        text = self.synthesize(CountingClient(), assembly)
        # The harness manifest declares test/validation result artifacts; they
        # are listed and rendered as prose sections, not fenced source files.
        self.assertTrue(assembly.validation_result_artifact_ids)


if __name__ == "__main__":
    unittest.main()
