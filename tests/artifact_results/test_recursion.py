"""Phase 4C — recursive execution: typed candidates reach the parent, support
children never leak, and synthesis prose can never invent an artifact.
"""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.artifact_results import (
    AssemblyStatus,
    ConflictKind,
    RecursiveArtifactOutput,
    collect_subtask_artifacts,
    merge_subtask_collections,
)
from dozen.config import OrchestratorConfig
from dozen.decomposition import derive_execution_scope
from dozen.executor import Executor
from dozen.intent import resolve_contract
from dozen.models import Plan, Task, TaskStatus
from dozen.router import Router
from dozen.verifier import Verifier

from ..artifact_decomposition.harness import react_plan, react_work_plan
from .harness import (
    CONTENT,
    entry,
    envelope,
    scope_for,
    worker_envelope_for,
)

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")
REAL_CODE = "```tsx\nexport default function App() { return <div/>; }\n```"


class ConcreteCodeClient(LLMClient):
    """Keeps prerequisite fixtures concrete under code-delivery enforcement."""

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        return LLMResponse(text=REAL_CODE, provider=provider, model=model)


class RecursiveClient(LLMClient):
    """Plans the shell as a research + implementation sub-plan; scripts workers."""

    def __init__(self, producer_payload) -> None:
        super().__init__(mock=True)
        self.producer_payload = producer_payload
        self.worker_prompts: list[str] = []

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
                return LLMResponse(text=json.dumps(self.producer_payload),
                                   provider=provider, model=model)
            if "Research routing constraints." in user:
                # A support child: prose only, and it must never become an artifact.
                return LLMResponse(
                    text=json.dumps({
                        "summary": "Routing should be file-based.",
                        "key_decisions": ["use a route table"],
                        "artifacts": {"notes.md": "Use a route table."},
                        "confidence": 0.8,
                    }),
                    provider=provider, model=model,
                )
            return LLMResponse(text=REAL_CODE, provider=provider, model=model)
        if "SYNTHESIZER" in system:
            # Synthesis tries to "helpfully" invent extra files. It must not win.
            return LLMResponse(
                text="### src/components/ui/Card.tsx\nexport function Card() {}\n",
                provider=provider, model=model,
            )
        return LLMResponse(text=REAL_CODE, provider=provider, model=model)


def scoped_child_task(subtask_id: str = "shell") -> Task:
    return Task(
        prompt="Build the application shell.",
        contract=IMPLEMENT,
        execution_scope=derive_execution_scope(react_work_plan(), subtask_id),
    )


def make_orchestrator(client, *, depth: int = 2) -> Orchestrator:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
    return Orchestrator(
        client=client, pool=AgentPool([agent]),
        config=OrchestratorConfig(
            max_parallelism=1, max_repair_attempts=0, verify_outputs=False,
            use_llm_router=False, max_depth=depth,
        ),
    )


class TestRecursiveCollection(unittest.TestCase):
    def test_designated_producer_artifacts_reach_the_parent(self) -> None:
        client = RecursiveClient(worker_envelope_for("shell"))
        result = make_orchestrator(client)._orchestrate(scoped_child_task(), depth=1)
        collection = result.artifact_collection
        self.assertIsNotNone(collection)
        self.assertTrue(collection.engaged)
        self.assertTrue(collection.satisfied)
        self.assertEqual(
            [c.artifact_id for c in collection.accepted],
            ["src/app/App.tsx", "src/app/router.tsx",
             "src/components/layout/DashboardLayout.tsx"],
        )
        self.assertEqual(collection.subtask_id, "shell")

    def test_support_child_artifacts_do_not_leak(self) -> None:
        client = RecursiveClient(worker_envelope_for("shell"))
        result = make_orchestrator(client)._orchestrate(scoped_child_task(), depth=1)
        # The researcher returned a legacy envelope with notes.md; it owns nothing,
        # so it produced no candidate and no scope block.
        ids = {c.artifact_id for c in result.artifact_collection.accepted}
        self.assertNotIn("notes.md", ids)
        self.assertEqual(len(ids), 3)
        research_prompt = next(
            p for p in client.worker_prompts if "Research routing constraints." in p
        )
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", research_prompt)
        self.assertNotIn('"artifact_id"', research_prompt)

    def test_recursive_synthesis_cannot_invent_artifacts(self) -> None:
        client = RecursiveClient(worker_envelope_for("shell"))
        result = make_orchestrator(client)._orchestrate(scoped_child_task(), depth=1)
        # Phase 4F strengthens the Phase 4C invariant: a scoped child finalizes
        # deterministically, so the synthesizer's invented Card.tsx block can
        # no longer even appear in the text — and the TYPED collection remains
        # the only artifact channel to the parent.
        self.assertNotIn("Card", result.final_answer)
        self.assertIn("src/app/App.tsx", result.final_answer)
        self.assertNotIn(
            "src/components/ui/Card.tsx",
            {c.artifact_id for c in result.artifact_collection.accepted},
        )

    def test_unrelated_root_packages_are_excluded(self) -> None:
        client = RecursiveClient(envelope(
            entry("src/app/App.tsx"),
            entry("src/app/router.tsx"),
            entry("src/components/layout/DashboardLayout.tsx"),
            entry("tests/dashboard.test.tsx"),  # another package's artifact
        ))
        result = make_orchestrator(client)._orchestrate(scoped_child_task(), depth=1)
        collection = result.artifact_collection
        self.assertEqual(collection.unexpected_artifact_ids,
                         ("tests/dashboard.test.tsx",))
        self.assertNotIn(
            "tests/dashboard.test.tsx",
            {c.artifact_id for c in collection.accepted},
        )

    def test_nested_provenance_is_preserved(self) -> None:
        client = RecursiveClient(worker_envelope_for("shell"))
        result = make_orchestrator(client)._orchestrate(scoped_child_task(), depth=1)
        producer = next(
            s.id for s in result.plan.subtasks if s.produces_parent_artifacts
        )
        self.assertNotEqual(producer, "shell")
        for candidate in result.artifact_collection.accepted:
            # The plan-mapped owner and the nested delegation that actually
            # emitted the file are BOTH retained.
            self.assertEqual(candidate.subtask_id, "shell")
            self.assertEqual(candidate.producer_subtask_id, producer)
            self.assertEqual(candidate.recursion_depth, 1)
            self.assertEqual(candidate.package_id, "application-shell")

    def test_duplicate_recursive_candidates_are_detected(self) -> None:
        scope = scope_for("shell")
        merged = merge_subtask_collections(
            [
                collect_subtask_artifacts(
                    worker_envelope_for("shell"), scope,
                    producer_subtask_id="a", recursion_depth=1,
                ),
                collect_subtask_artifacts(
                    envelope(entry("src/app/App.tsx", content="a rival App")),
                    scope, producer_subtask_id="b", recursion_depth=1,
                ),
            ],
            scope,
        )
        self.assertEqual(merged.duplicate_artifact_ids, ("src/app/App.tsx",))
        self.assertEqual(len(merged.accepted), 4)  # both rivals retained
        self.assertTrue(any("more than one delegation" in w for w in merged.warnings))


class TestRecursionThroughTheExecutor(unittest.TestCase):
    def executor_with(self, recurse_fn) -> Executor:
        agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
        pool = AgentPool([agent])
        client = ConcreteCodeClient(mock=True)
        return Executor(
            client=client, pool=pool,
            router=Router(client, pool, router_agent=agent, use_llm_router=False),
            verifier=Verifier(client, agent),
            config=OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                      verify_outputs=False, use_llm_router=False),
            recurse_fn=recurse_fn, log=lambda _m: None,
        )

    def recursive_plan(self) -> Plan:
        plan = react_plan()
        for subtask in plan.subtasks:
            if subtask.id == "shell":
                subtask.complex = True
        return plan

    def test_typed_recursive_output_is_attached_to_the_parent_result(self) -> None:
        collection = collect_subtask_artifacts(
            worker_envelope_for("shell"), scope_for("shell"),
            producer_subtask_id="inner", recursion_depth=1,
        )
        results = self.executor_with(
            lambda _st, _task, _d: RecursiveArtifactOutput(
                text="the shell is done", collection=collection
            )
        ).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            self.recursive_plan(), depth=0,
        )
        shell = next(r for r in results if r.subtask_id == "shell")
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertEqual(shell.output, "the shell is done")
        self.assertIs(shell.artifact_collection, collection)

    def test_a_legacy_string_recurse_fn_still_works(self) -> None:
        results = self.executor_with(lambda *_a: REAL_CODE).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            self.recursive_plan(), depth=0,
        )
        shell = next(r for r in results if r.subtask_id == "shell")
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertEqual(shell.output, REAL_CODE)
        self.assertIsNone(shell.artifact_collection)


class TestRecursivePackageInsideAFullRun(unittest.TestCase):
    def test_a_recursive_shell_package_assembles_at_the_root(self) -> None:
        # The shell recursion returns its three artifacts through the designated
        # producer; the other five packages answer directly.
        from .harness import assemble_dashboard, collect_for
        from dozen.artifact_results import assemble_deliverable

        recursive_shell = merge_subtask_collections(
            [collect_for("shell", worker_envelope_for("shell"),
                         producer_subtask_id="i", recursion_depth=1)],
            scope_for("shell"),
        )
        from .harness import collect_dashboard

        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shell": None}) + [recursive_shell],
        )
        self.assertEqual(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(len(assembly.artifacts), 11)
        self.assertEqual(assemble_dashboard().status, AssemblyStatus.COMPLETE)
        app = assembly.by_artifact()["src/app/App.tsx"]
        self.assertEqual(app.recursion_depth, 1)
        self.assertEqual(app.content, CONTENT["src/app/App.tsx"])
        self.assertEqual(
            [c.kind for c in assembly.conflicts], []
        )
        self.assertNotIn(ConflictKind.CONTENT, [c.kind for c in assembly.conflicts])


if __name__ == "__main__":
    unittest.main()
