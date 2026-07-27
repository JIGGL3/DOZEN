"""Phase 4E Part L — targeted repair for recursive designated producers.

Valid recursive artifacts are preserved; only the remaining targets are re-asked;
support children never receive repair ownership; nested provenance survives; a
recursive synthesizer cannot invent a repair; nested no-progress works.
"""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.artifact_repair import (
    RepairStatus,
    begin_repair_state,
    derive_repair_scope,
    merge_repair_collection,
    plan_repair,
    submitted_content_hashes,
)
from dozen.config import OrchestratorConfig
from dozen.decomposition import derive_execution_scope
from dozen.intent import resolve_contract
from dozen.models import Task, TaskStatus

from ..artifact_decomposition.harness import react_work_plan
from ..artifact_results.harness import collect_for
from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    ROUTER,
    decide,
    entry,
    envelope,
    envelope_with,
    first_attempt,
    repair_attempt,
    scope_for,
)

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")
REAL_CODE = "```tsx\nexport default function App() { return <div/>; }\n```"
TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


class TestRecursiveDerivationAndMerge(unittest.TestCase):
    def test_only_the_invalid_recursive_artifact_is_targeted(self) -> None:
        # A recursive producer returned two valid files and one truncated one.
        collection = collect_for(
            "shell", envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
            producer_subtask_id="nested-writer", recursion_depth=1,
        )
        state = begin_repair_state(collection)
        decision = plan_repair(scope_for("shell"), state, attempt=1, max_attempts=3)
        self.assertIs(decision.status, RepairStatus.REPAIRABLE)
        self.assertEqual(decision.request.target_artifact_ids, (LAYOUT,))
        self.assertEqual(decision.request.preserved_artifact_ids, (APP, ROUTER))

    def test_valid_recursive_artifacts_are_preserved_with_their_provenance(self) -> None:
        collection = collect_for(
            "shell", envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT}),
            producer_subtask_id="nested-writer", recursion_depth=2,
        )
        state = begin_repair_state(collection)
        decision = plan_repair(scope_for("shell"), state, attempt=1, max_attempts=3)
        merged = repair_attempt(
            state, decision, envelope(entry(LAYOUT)),
            producer_subtask_id="nested-writer-2", recursion_depth=2, attempt=2,
        )
        by_id = merged.collection.accepted_by_artifact()
        self.assertEqual(by_id[APP][0].producer_subtask_id, "nested-writer")
        self.assertEqual(by_id[APP][0].recursion_depth, 2)
        self.assertEqual(by_id[APP][0].content, CONTENT[APP])
        # The repaired file carries the NEW nested delegation's provenance.
        self.assertEqual(by_id[LAYOUT][0].producer_subtask_id, "nested-writer-2")

    def test_nested_no_progress_works(self) -> None:
        payload = envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})
        collection = collect_for(
            "shell", payload, producer_subtask_id="deep", recursion_depth=3
        )
        state = begin_repair_state(
            collection, submitted_hashes=submitted_content_hashes(payload)
        )
        decision = plan_repair(scope_for("shell"), state, attempt=1, max_attempts=3)
        merged = repair_attempt(
            state, decision, payload,  # same broken layout again
            producer_subtask_id="deep", recursion_depth=3, attempt=2,
        )
        decision2 = plan_repair(scope_for("shell"), merged, attempt=2, max_attempts=3)
        self.assertIs(decision2.status, RepairStatus.NO_PROGRESS)


class RecursiveRepairClient(LLMClient):
    """Plans the shell recursively; the producer truncates the layout on attempt 1
    and fixes it on the next attempt."""

    def __init__(self) -> None:
        super().__init__(mock=True)
        self.worker_prompts: list[str] = []
        self.producer_calls = 0

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[1].content
        if "MANAGER" in system:
            return LLMResponse(
                text=json.dumps({
                    "analysis": "split the shell",
                    "subtasks": [
                        {"id": "r", "title": "Research the shell",
                         "instruction": "Research routing constraints.",
                         "assigned_model": "alpha"},
                        {"id": "i", "title": "Implement the shell",
                         "instruction": "Write every owned shell file.",
                         "assigned_model": "alpha", "depends_on": ["r"],
                         "produces_parent_artifacts": True},
                    ],
                    "synthesis_strategy": "present the shell files",
                }),
                provider=provider, model=model,
            )
        if "Worker" in system:
            self.worker_prompts.append(user)
            if "Write every owned shell file." in user:
                self.producer_calls += 1
                if self.producer_calls == 1:
                    payload = envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})
                else:
                    payload = envelope(entry(LAYOUT))
                return LLMResponse(text=json.dumps(payload), provider=provider,
                                   model=model)
            if "Research routing constraints." in user:
                return LLMResponse(
                    text=json.dumps({
                        "summary": "Routing should be file-based.",
                        "key_decisions": [],
                        "artifacts": {"notes.md": "Use a route table."},
                        "confidence": 0.8,
                    }),
                    provider=provider, model=model,
                )
            return LLMResponse(text=REAL_CODE, provider=provider, model=model)
        return LLMResponse(text=REAL_CODE, provider=provider, model=model)


def scoped_child_task(subtask_id: str = "shell") -> Task:
    return Task(
        prompt="Build the application shell.",
        contract=IMPLEMENT,
        execution_scope=derive_execution_scope(react_work_plan(), subtask_id),
    )


def make_orchestrator(client, *, repairs: int = 1) -> Orchestrator:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
    return Orchestrator(
        client=client, pool=AgentPool([agent]),
        config=OrchestratorConfig(
            max_parallelism=1, max_repair_attempts=repairs, verify_outputs=False,
            use_llm_router=False, max_depth=2,
        ),
    )


class TestRecursionThroughTheOrchestrator(unittest.TestCase):
    def test_a_recursive_producer_is_repaired_in_place(self) -> None:
        client = RecursiveRepairClient()
        result = make_orchestrator(client, repairs=1)._orchestrate(
            scoped_child_task(), depth=1
        )
        collection = result.artifact_collection
        self.assertIsNotNone(collection)
        self.assertTrue(collection.satisfied)
        self.assertEqual(
            sorted(c.artifact_id for c in collection.accepted),
            sorted([APP, ROUTER, LAYOUT]),
        )
        # Exactly two producer calls: initial + one targeted repair.
        self.assertEqual(client.producer_calls, 2)

    def test_the_recursive_repair_prompt_is_targeted(self) -> None:
        client = RecursiveRepairClient()
        make_orchestrator(client, repairs=1)._orchestrate(
            scoped_child_task(), depth=1
        )
        producer_prompts = [
            p for p in client.worker_prompts
            if "Write every owned shell file." in p
        ]
        self.assertEqual(len(producer_prompts), 2)
        self.assertIn("TARGETED ARTIFACT REPAIR", producer_prompts[1])
        target_list = producer_prompts[1].split(
            "Return ONLY these repair artifacts"
        )[1].split("Return one typed artifact envelope")[0]
        self.assertIn(LAYOUT, target_list)
        self.assertNotIn(APP, target_list)

    def test_a_support_child_never_receives_repair_ownership(self) -> None:
        client = RecursiveRepairClient()
        make_orchestrator(client, repairs=1)._orchestrate(
            scoped_child_task(), depth=1
        )
        research = next(
            p for p in client.worker_prompts if "Research routing constraints." in p
        )
        self.assertNotIn("TARGETED ARTIFACT REPAIR", research)
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", research)

    def test_a_recursive_synthesizer_cannot_invent_a_repair(self) -> None:
        # The producer never returns Card.tsx; the merged collection cannot hold
        # an artifact no delegation actually produced.
        client = RecursiveRepairClient()
        result = make_orchestrator(client, repairs=1)._orchestrate(
            scoped_child_task(), depth=1
        )
        ids = {c.artifact_id for c in result.artifact_collection.accepted}
        self.assertNotIn("src/components/ui/Card.tsx", ids)


if __name__ == "__main__":
    unittest.main()
