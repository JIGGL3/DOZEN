"""Phase 4G — the planner and executor refuse synthesis delegations.

The planner rejects a synthesis/merge/integration/final-assembly delegation at
BOTH the root and every recursive depth, using the ONE existing corrective
re-plan; a plan that still contains one after the retry fails planning
deterministically. The executor is a defense-in-depth backstop: a forbidden
synthesis subtask that reaches dispatch is failed with a typed reason and NEVER
sent to a provider.
"""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, CancelledError, LLMClient, LLMResponse
from dozen.config import OrchestratorConfig
from dozen.models import Plan, SubTask, Task, TaskStatus
from dozen.planner import Planner, PlanError
from dozen.router import Router
from dozen.verifier import Verifier


AGENT = AgentSpec(name="alpha", provider="openai", model="gpt",
                  strengths={"reasoning": 0.9, "coding": 0.9, "writing": 0.9},
                  tier=4)


def _pool() -> AgentPool:
    return AgentPool([AGENT])


CLEAN_PLAN = {
    "analysis": "two independent modules",
    "delegations": [
        {"id": "s1", "title": "Auth", "instruction": "Write the auth module.",
         "assigned_model": "alpha"},
        {"id": "s2", "title": "DB", "instruction": "Write the db module.",
         "assigned_model": "alpha"},
    ],
    "synthesis_strategy": "auth then db",
}

SYNTHESIS_PLAN = {
    "analysis": "two modules then a merge",
    "delegations": [
        {"id": "s1", "title": "Auth", "instruction": "Write the auth module.",
         "assigned_model": "alpha"},
        {"id": "s2", "title": "DB", "instruction": "Write the db module.",
         "assigned_model": "alpha"},
        {"id": "s3", "title": "Final assembly",
         "instruction": "Merge the worker outputs into one final project.",
         "assigned_model": "alpha", "depends_on": ["s1", "s2"]},
    ],
    "synthesis_strategy": "merge everything",
}


class _ScriptedPlannerClient(LLMClient):
    """Returns a scripted plan dict per planner call (list consumed in order)."""

    def __init__(self, plans: list[dict]) -> None:
        super().__init__(mock=True)
        self._plans = list(plans)
        self.planner_calls = 0

    def complete_json(self, *, provider, model, messages, **kwargs) -> dict:
        index = min(self.planner_calls, len(self._plans) - 1)
        self.planner_calls += 1
        return dict(self._plans[index])


class TestPlannerRejectsSynthesis(unittest.TestCase):
    def _planner(self, plans: list[dict]) -> Planner:
        return Planner(_ScriptedPlannerClient(plans), AGENT, _pool())

    def test_root_synthesis_plan_fails_after_one_replan(self) -> None:
        client = _ScriptedPlannerClient([SYNTHESIS_PLAN, SYNTHESIS_PLAN])
        planner = Planner(client, AGENT, _pool())
        with self.assertRaises(PlanError) as ctx:
            planner.plan(Task(prompt="Do the work."), depth=0, max_depth=2)
        self.assertIn("synthesis", str(ctx.exception).lower())
        self.assertEqual(client.planner_calls, 2)  # exactly one corrective re-plan

    def test_recursive_synthesis_plan_is_also_rejected(self) -> None:
        client = _ScriptedPlannerClient([SYNTHESIS_PLAN, SYNTHESIS_PLAN])
        planner = Planner(client, AGENT, _pool())
        with self.assertRaises(PlanError):
            planner.plan(Task(prompt="Sub-problem."), depth=1, max_depth=2)

    def test_corrective_replan_to_clean_plan_succeeds(self) -> None:
        client = _ScriptedPlannerClient([SYNTHESIS_PLAN, CLEAN_PLAN])
        planner = Planner(client, AGENT, _pool())
        plan = planner.plan(Task(prompt="Do the work."), depth=0, max_depth=2)
        self.assertEqual(client.planner_calls, 2)
        titles = {s.title for s in plan.subtasks}
        self.assertEqual(titles, {"Auth", "DB"})
        self.assertNotIn("Final assembly", titles)

    def test_clean_plan_needs_no_replan(self) -> None:
        client = _ScriptedPlannerClient([CLEAN_PLAN])
        planner = Planner(client, AGENT, _pool())
        plan = planner.plan(Task(prompt="Do the work."), depth=0, max_depth=2)
        self.assertEqual(client.planner_calls, 1)
        self.assertEqual(len(plan.subtasks), 2)

    def test_cancellation_during_corrective_replan_propagates(self) -> None:
        class CancellingCorrectiveClient(_ScriptedPlannerClient):
            def complete_json(self, *, provider, model, messages, **kwargs) -> dict:
                if self.planner_calls == 1:
                    self.planner_calls += 1
                    raise CancelledError("cancelled during corrective planning")
                return super().complete_json(
                    provider=provider, model=model, messages=messages, **kwargs
                )

        client = CancellingCorrectiveClient([SYNTHESIS_PLAN])
        planner = Planner(client, AGENT, _pool())
        with self.assertRaises(CancelledError):
            planner.plan(Task(prompt="Do the work."), depth=0, max_depth=2)
        self.assertEqual(client.planner_calls, 2)


class _RecordingWorkerClient(LLMClient):
    """Records every worker user-prompt; returns a benign valid answer."""

    def __init__(self) -> None:
        super().__init__(mock=True)
        self.worker_prompts: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        self.worker_prompts.append(messages[-1].content)
        return LLMResponse(
            text="A complete, valid worker answer with enough words to pass.",
            provider=provider, model=model,
        )


class TestExecutorDispatchGuard(unittest.TestCase):
    def _executor(self, client):
        from dozen.executor import Executor
        pool = _pool()
        config = OrchestratorConfig(
            max_parallelism=1, max_repair_attempts=0, verify_outputs=False,
            use_llm_router=False, verbose=False,
        )
        return Executor(
            client=client, pool=pool,
            router=Router(client, pool, router_agent=AGENT, use_llm_router=False),
            verifier=Verifier(client, AGENT), config=config,
        )

    def test_forbidden_synthesis_subtask_is_never_dispatched(self) -> None:
        # A plan built directly (bypassing the planner guard) still cannot reach
        # a provider with a merge instruction: the executor refuses it.
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Work", instruction="Write the module.", id="s1"),
            SubTask(title="Merge",
                    instruction="Merge the worker outputs into one file.",
                    depends_on=["s1"], assigned_model="alpha", id="s2"),
        ])
        client = _RecordingWorkerClient()
        results = self._executor(client).run(Task(prompt="Do it."), plan, depth=0)
        by_id = {r.subtask_id: r for r in results}
        self.assertEqual(by_id["s1"].status, TaskStatus.COMPLETED)
        self.assertEqual(by_id["s2"].status, TaskStatus.FAILED)
        self.assertIn("synthesis", by_id["s2"].error.lower())
        # The merge instruction was NEVER sent to any provider.
        for prompt in client.worker_prompts:
            self.assertNotIn("Merge the worker outputs", prompt)

    def test_ordinary_subtasks_run_normally(self) -> None:
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Work", instruction="Write the module.", id="s1"),
        ])
        client = _RecordingWorkerClient()
        results = self._executor(client).run(Task(prompt="Do it."), plan, depth=0)
        self.assertEqual(results[0].status, TaskStatus.COMPLETED)
        self.assertEqual(len(client.worker_prompts), 1)

    def test_subtask_renamed_to_synthesis_after_planning_is_never_dispatched(self) -> None:
        # The plan can become stale or be mutated after planner validation. The
        # dispatch-time backstop evaluates the current object, not its history.
        subtask = SubTask(
            title="Review", instruction="Review the module independently.", id="s1"
        )
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[subtask])
        subtask.title = "Unified final response"
        subtask.instruction = "Combine all prior delegated results for the user."

        client = _RecordingWorkerClient()
        results = self._executor(client).run(Task(prompt="Do it."), plan, depth=0)
        self.assertEqual(results[0].status, TaskStatus.FAILED)
        self.assertIn("synthesis", results[0].error.lower())
        self.assertEqual(client.worker_prompts, [])


if __name__ == "__main__":
    unittest.main()
