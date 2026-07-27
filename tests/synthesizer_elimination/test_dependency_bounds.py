"""Phase 4G dependency-context bounds and typed artifact filtering."""

from __future__ import annotations

import unittest

from dozen import AgentPool, AgentSpec, LLMClient
from dozen.artifact_results import ArtifactCandidate, SubtaskCollectionResult
from dozen.config import OrchestratorConfig
from dozen.decomposition import ArtifactExecutionScope
from dozen.executor import Executor
from dozen.models import SubTask, SubTaskResult, Task, TaskStatus
from dozen.router import Router
from dozen.verifier import Verifier


AGENT = AgentSpec(
    name="small-context",
    provider="openai",
    model="small",
    strengths={"coding": 1.0},
    max_context_tokens=8_000,
    max_tokens=4_096,
)


def make_executor(client: LLMClient) -> Executor:
    pool = AgentPool([AGENT])
    config = OrchestratorConfig(
        max_parallelism=1,
        max_repair_attempts=0,
        verify_outputs=False,
        use_llm_router=False,
        verbose=False,
    )
    return Executor(
        client=client,
        pool=pool,
        router=Router(client, pool, router_agent=AGENT, use_llm_router=False),
        verifier=Verifier(client, AGENT),
        config=config,
    )


class TestAggregateInputBudget(unittest.TestCase):
    def test_several_large_dependencies_fit_selected_agent_budget(self) -> None:
        client = LLMClient(mock=True)
        executor = make_executor(client)
        outputs = {
            f"dependency-{index}": "dependency detail " * 600
            for index in range(4)
        }
        messages = executor._bounded_worker_messages(
            Task(prompt="Build the application."),
            SubTask(
                id="fan-in",
                title="Dependency wiring",
                instruction="Implement dependency injection wiring.",
                depends_on=[f"s{index}" for index in range(4)],
            ),
            outputs,
            AGENT,
            "",
            scope=None,
            repair=None,
        )
        budget = executor._worker_input_budget_chars(AGENT)
        self.assertLessEqual(client.measure_input_chars(messages), budget)
        self.assertIn("PREREQUISITE OUTPUTS", messages[-1].content)
        for title in outputs:
            self.assertIn(title, messages[-1].content)

    def test_recursive_dependency_bodies_share_one_aggregate_cap(self) -> None:
        outputs = {f"d{index}": "x" * 8_000 for index in range(6)}
        bounded = Executor._bound_dependency_bodies(outputs, 24_000)
        self.assertLessEqual(sum(map(len, bounded.values())), 24_000)
        self.assertEqual(list(bounded), list(outputs))


class TestTypedArtifactDependencyFiltering(unittest.TestCase):
    def test_unrelated_artifact_body_is_not_forwarded(self) -> None:
        wanted = ArtifactCandidate(
            artifact_id="wanted.py",
            path="wanted.py",
            subtask_id="producer",
            package_id="producer-package",
            content="WANTED_BODY",
        )
        unrelated = ArtifactCandidate(
            artifact_id="unrelated.py",
            path="unrelated.py",
            subtask_id="producer",
            package_id="producer-package",
            content="UNRELATED_SECRET_BODY",
        )
        dependency = SubTaskResult(
            subtask_id="producer",
            title="Producer",
            status=TaskStatus.COMPLETED,
            output="FLATTENED_OUTPUT_WITH_UNRELATED_SECRET_BODY",
            artifact_collection=SubtaskCollectionResult(
                subtask_id="producer",
                package_ids=("producer-package",),
                accepted=(wanted, unrelated),
                engaged=True,
            ),
        )
        scope = ArtifactExecutionScope(
            manifest_id="manifest",
            subtask_id="consumer",
            package_ids=("consumer-package",),
            input_artifact_ids=("wanted.py",),
        )
        text = Executor._dependency_content(dependency, scope)
        self.assertIn("ARTIFACT INPUT REFERENCE", text)
        self.assertIn("id=wanted.py", text)
        self.assertIn("WANTED_BODY", text)
        self.assertNotIn("UNRELATED_SECRET_BODY", text)


if __name__ == "__main__":
    unittest.main()
