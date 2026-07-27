"""Phase 6 — recursive workflows can never reintroduce raw protocol output.

A complex subtask re-orchestrates as a child run. Only the approved final
presentation (or the typed RecursiveArtifactOutput) may reach the parent —
never worker envelopes, synthesizer envelopes, confidence/key-decision fields,
or malformed protocol payloads.
"""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.cancellation import CancelledError, CancelToken
from dozen.config import OrchestratorConfig
from dozen.models import OrchestrationResult, SubTask, Task
from dozen.synthesis_scaling import FALLBACK_HEADLINE

from .harness import envelope

_PROTOCOL_TOKENS = ('"summary"', '"key_decisions"', '"artifacts"', '"confidence"')


def parent_plan() -> dict:
    return {
        "analysis": "one deep component, one summary",
        "direct_answer": None,
        "delegations": [
            {
                "id": "s1",
                "title": "Deep dive",
                "instruction": "Run the deep-dive analysis of the system.",
                "assigned_model": "alpha",
                "complex": True,
                "depends_on": [],
            },
            {
                "id": "s2",
                "title": "Summarize",
                "instruction": "Write the executive summary of the findings.",
                "assigned_model": "alpha",
                "depends_on": ["s1"],
            },
        ],
        "synthesis_strategy": "Present the deep dive, then the summary.",
    }


def child_plan() -> dict:
    return {
        "analysis": "two component fact-finding passes",
        "direct_answer": None,
        "delegations": [
            {
                "id": "c1",
                "title": "Alpha facts",
                "instruction": "List the core facts about the alpha component.",
                "assigned_model": "alpha",
                "depends_on": [],
            },
            {
                "id": "c2",
                "title": "Beta facts",
                "instruction": "List the core facts about the beta component.",
                "assigned_model": "alpha",
                "depends_on": [],
            },
        ],
        "synthesis_strategy": "Merge both component fact lists.",
    }


class RecursiveClient(LLMClient):
    """Scripted planner/worker/synthesizer for a two-level recursive run."""

    def __init__(self, child_workers: dict[str, str],
                 synth_replies: list[object],
                 child_plan_json: dict | None = None) -> None:
        super().__init__(mock=True)
        self.child_workers = child_workers
        self.synth_replies = list(synth_replies)
        self.child_plan_json = child_plan_json or child_plan()
        self.synth_call_sizes: list[int] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[-1].content
        size = sum(len(m.content) for m in messages)
        if "MANAGER" in system:
            plan = (self.child_plan_json if "deep-dive analysis" in user
                    else parent_plan())
            return LLMResponse(text=json.dumps(plan), provider=provider, model=model)
        if ("SYNTHESIZER" in system or "combining consecutive sections" in system
                or "condensing one part" in system):
            self.synth_call_sizes.append(size)
            reply = self.synth_replies.pop(0) if self.synth_replies else "Combined."
            if isinstance(reply, Exception):
                raise reply
            return LLMResponse(text=str(reply), provider=provider, model=model)
        text = "generic worker output with plenty of words to pass validation"
        for marker, reply in self.child_workers.items():
            if marker in user:
                text = reply
                break
        return LLMResponse(text=text, provider=provider, model=model)


def make_orchestrator(client: LLMClient, **overrides) -> Orchestrator:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "writing": 0.9}, tier=4)
    config = OrchestratorConfig(
        max_parallelism=1, max_repair_attempts=1, verify_outputs=False,
        use_llm_router=False, verbose=False, **overrides,
    )
    return Orchestrator(client=client, pool=AgentPool([agent]), config=config)


class TestRecursiveProtocolSafety(unittest.TestCase):
    def test_recursive_worker_envelopes_never_appear_raw(self) -> None:
        client = RecursiveClient(
            child_workers={
                "alpha component": envelope(
                    {"alpha.md": "Alpha stores the session state."},
                    decisions=["ALPHA-DECISION"], confidence=0.31,
                ),
                "beta component": envelope(
                    {"beta.md": "Beta routes the requests."},
                ),
                "executive summary": "The system pairs alpha state with beta routing.",
            },
            synth_replies=[
                "Alpha stores state; beta routes requests.",   # child synthesis
                "Full report: alpha state plus beta routing.",  # parent synthesis
            ],
        )
        result = make_orchestrator(client).run("Produce the full system report.")
        self.assertEqual(result.error, "")
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, result.final_answer)
        for r in result.subtask_results:
            for token in _PROTOCOL_TOKENS:
                self.assertNotIn(token, r.output)
        self.assertNotIn("0.31", result.final_answer)
        self.assertNotIn("ALPHA-DECISION", result.final_answer)

    def test_recursion_never_invokes_synthesis_even_when_enabled(self) -> None:
        # Phase 4G: model synthesis is eliminated at EVERY depth. Even with the
        # (inert) polish flag on, a recursive child returns deterministic
        # ordered sections; the scripted synth envelope is never consumed and no
        # protocol tokens leak.
        child_synth_envelope = envelope(
            {"report.md": "Decoded child synthesis content."}
        )
        client = RecursiveClient(
            child_workers={
                "alpha component": "Alpha component facts stated plainly here.",
                "beta component": "Beta component facts stated plainly here.",
                "executive summary": "Summary of both components in one line.",
            },
            synth_replies=[
                child_synth_envelope,
                "Final report using the decoded child content.",
            ],
        )
        result = make_orchestrator(client, enable_model_polish=True).run(
            "Produce the full system report."
        )
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_call_sizes, [])  # no synthesis at any depth
        deep_dive = result.subtask_results[0]
        self.assertIn("Alpha component facts", deep_dive.output)
        self.assertNotIn("Decoded child synthesis content.", deep_dive.output)
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, deep_dive.output)
            self.assertNotIn(token, result.final_answer)

    def test_nested_malformed_protocol_yields_clean_diagnostic(self) -> None:
        broken_child_plan = {
            "analysis": "direct",
            "delegations": [],
            "direct_answer": '{"summary": "s", "key_decisions": ["k"], '
                             '"artifacts": {"f.py": "def broken(',
            "synthesis_strategy": "",
        }
        client = RecursiveClient(
            child_workers={
                "executive summary": "Summary written despite the failure.",
            },
            synth_replies=["Parent synthesis over the surviving outputs."],
            child_plan_json=broken_child_plan,
        )
        result = make_orchestrator(client).run("Produce the full system report.")
        deep_dive = result.subtask_results[0]
        self.assertEqual(deep_dive.status.value, "failed")
        self.assertNotIn("def broken(", deep_dive.error)
        self.assertNotIn("def broken(", result.final_answer)
        self.assertNotIn('"artifacts"', result.final_answer)

    def test_large_recursive_prose_needs_no_synthesis(self) -> None:
        big_alpha = "Alpha fact sentence. " * 700   # ~15k chars
        big_beta = "Beta fact sentence. " * 700
        client = RecursiveClient(
            child_workers={
                "alpha component": big_alpha,
                "beta component": big_beta,
                "executive summary": "Short executive summary of the findings.",
            },
            synth_replies=["unused", "unused", "unused"],
        )
        # Phase 4G: large recursive prose is rendered deterministically as
        # ordered sections — no hierarchical model synthesis, at any input size.
        orchestrator = make_orchestrator(
            client, max_synthesis_input_chars=12000, enable_model_polish=True
        )
        result = orchestrator.run("Produce the full system report.")
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_call_sizes, [])  # no model merge step
        self.assertIn("Alpha fact sentence.", result.subtask_results[0].output)

    def test_recursive_deterministic_fallback_replaces_failed_synthesis(self) -> None:
        # Phase 4F: a failed OPTIONAL polish returns the deterministic ordered
        # sections unchanged (no capsule-fallback banner, no envelope, no loss).
        client = RecursiveClient(
            child_workers={
                "alpha component": "Alpha component facts stated plainly here.",
                "beta component": "Beta component facts stated plainly here.",
                "executive summary": "Summary of both components in one line.",
            },
            synth_replies=[
                RuntimeError("child synthesis provider unavailable"),
                "Parent synthesis over the child fallback.",
            ],
        )
        result = make_orchestrator(client, enable_model_polish=True).run(
            "Produce the full system report."
        )
        self.assertEqual(result.error, "")
        deep_dive = result.subtask_results[0]
        self.assertNotIn(FALLBACK_HEADLINE, deep_dive.output)
        self.assertIn("Alpha component facts", deep_dive.output)
        self.assertIn("Beta component facts", deep_dive.output)
        self.assertNotIn('"artifacts"', deep_dive.output)

    def test_recursion_completes_without_synthesis_cancellation(self) -> None:
        # Phase 4G: a synth reply scripted to raise CancelledError can never fire
        # because synthesis never runs. With an untripped token, recursion
        # completes deterministically instead of surfacing a synthesis cancel.
        client = RecursiveClient(
            child_workers={
                "alpha component": "Alpha component facts stated plainly here.",
                "beta component": "Beta component facts stated plainly here.",
            },
            synth_replies=[CancelledError("stop pressed")],
        )
        result = make_orchestrator(client, enable_model_polish=True).run(
            "Produce the full system report.", cancel=CancelToken()
        )
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_call_sizes, [])
        self.assertTrue(result.final_answer.strip())

    def test_recursive_artifact_output_stays_typed(self) -> None:
        # Unit-level: a child result carrying a typed collection reaches the
        # parent as RecursiveArtifactOutput (text + typed candidates), never as
        # stitched prose or an envelope.
        from dozen.artifact_results import RecursiveArtifactOutput

        from ..artifact_results.harness import collect_for, envelope as typed_envelope, entry

        collection = collect_for(
            "shared-ui",
            typed_envelope(entry("src/components/ui/Card.tsx"),
                           entry("src/components/ui/Panel.tsx")),
            producer_subtask_id="child", recursion_depth=1,
        )
        client = RecursiveClient(child_workers={}, synth_replies=[])
        orchestrator = make_orchestrator(client)
        fake_child = OrchestrationResult(
            task_id="child", final_answer="rendered child text",
            artifact_collection=collection,
        )
        orchestrator._orchestrate = lambda task, depth: fake_child  # type: ignore[method-assign]
        out = orchestrator._recurse(
            SubTask(title="Nested", instruction="produce the package"),
            Task(prompt="scoped child"), 1,
        )
        self.assertIsInstance(out, RecursiveArtifactOutput)
        self.assertEqual(out.text, "rendered child text")
        self.assertIs(out.collection, collection)


if __name__ == "__main__":
    unittest.main()
