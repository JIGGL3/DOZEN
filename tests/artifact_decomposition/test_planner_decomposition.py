"""Phase 4B — planner validation and the ONE shared corrective re-plan.

Fake clients only; no provider is ever contacted.
"""

from __future__ import annotations

import copy
import json
import unittest

from dozen import LLMClient, LLMResponse, Orchestrator
from dozen.intent import resolve_contract
from dozen.models import Task
from dozen.planner import PlanError
from dozen.prompts import build_planner_messages
from dozen.decomposition import derive_execution_scope

from .harness import (
    REACT_ARTIFACT_BLOCK,
    QueuedPlannerClient,
    make_planner,
    react_plan_json,
    react_work_plan,
)

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")


def task() -> Task:
    return Task(prompt="Build me a React dashboard with tests.",
                contract=IMPLEMENT)


def block(**mutate) -> dict:
    data = copy.deepcopy(REACT_ARTIFACT_BLOCK)
    for key, value in mutate.items():
        data[key] = value
    return data


def packages_of(data: dict) -> dict:
    return {package["id"]: package for package in data["packages"]}


class TestValidBoundedDecomposition(unittest.TestCase):
    def test_valid_bounded_artifact_decomposition_passes(self) -> None:
        client = QueuedPlannerClient(react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(plan.artifact_plan)
        self.assertEqual(len(plan.artifact_plan.packages), 6)
        # Every package maps to an INTERNAL subtask id.
        internal = {s.id for s in plan.subtasks}
        for _pid, sid in plan.artifact_plan.subtask_map:
            self.assertIn(sid, internal)

    def test_non_artifact_plans_remain_unchanged(self) -> None:
        client = QueuedPlannerClient(react_plan_json(artifact_block=None))
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan.artifact_plan)


class TestFatalDecompositionErrors(unittest.TestCase):
    """Each defect is retried ONCE, then fails deterministically."""

    def assert_one_corrective_replan(self, bad: dict, fragment: str) -> None:
        # Invalid then valid -> recovered in exactly two planner calls.
        client = QueuedPlannerClient(react_plan_json(bad), react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIsNotNone(plan.artifact_plan)
        self.assertIn("PREVIOUS PLAN WAS REJECTED", client.prompts[1])
        self.assertIn(fragment, client.prompts[1])

        # Invalid twice -> PlanError, still exactly two calls (no third).
        client = QueuedPlannerClient(react_plan_json(bad), react_plan_json(bad))
        with self.assertRaises(PlanError) as ctx:
            make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("decomposition", str(ctx.exception))

    def test_large_manifest_assigned_to_one_package_fails(self) -> None:
        data = block()
        by_id = packages_of(data)
        # Give the foundation package everything except the result artifacts.
        source_ids = [
            entry["path"] for entry in data["artifacts"]
            if entry["kind"] not in ("build_result", "test_result")
        ]
        by_id["project-foundation"]["owns"] = source_ids
        for pid in ("application-shell", "shared-ui", "dashboard-feature",
                    "dashboard-tests"):
            by_id[pid]["owns"] = []
            by_id[pid]["inputs"] = []
        self.assert_one_corrective_replan(data, "wholesale")

    def test_executable_package_without_subtask_mapping_fails(self) -> None:
        data = block()
        packages_of(data)["shared-ui"].pop("subtask_id")
        self.assert_one_corrective_replan(data, "has no mapped subtask")

    def test_package_dependency_missing_from_subtask_dag_fails(self) -> None:
        payload = react_plan_json()
        # s4 (dashboard) no longer depends on s3 (shared-ui), but the package
        # dashboard-feature still depends on package shared-ui.
        for delegation in payload["delegations"]:
            if delegation["id"] == "s4":
                delegation["depends_on"] = ["s2"]
        client = QueuedPlannerClient(payload, react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("does not depend", client.prompts[1])
        self.assertIsNotNone(plan.artifact_plan)

    def test_validation_package_ordered_before_producers_fails(self) -> None:
        payload = react_plan_json()
        for delegation in payload["delegations"]:
            if delegation["id"] == "s6":  # the validation subtask
                delegation["depends_on"] = []
        client = QueuedPlannerClient(payload, react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn(
            "validation package 'integration-validation' would execute before "
            "producer", client.prompts[1],
        )
        self.assertIsNotNone(plan.artifact_plan)

    def test_too_many_packages_in_one_subtask_fails(self) -> None:
        data = block()
        for package in data["packages"]:
            package["subtask_id"] = "s1"
        self.assert_one_corrective_replan(data, "share one subtask")

    def test_second_invalid_decomposition_fails_deterministically(self) -> None:
        data = block()
        packages_of(data)["shared-ui"].pop("subtask_id")
        client = QueuedPlannerClient(react_plan_json(data), react_plan_json(data))
        with self.assertRaises(PlanError) as ctx:
            make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("invalid artifact decomposition", str(ctx.exception))


class TestSharedCorrectiveReplan(unittest.TestCase):
    def test_phase_3_and_4a_errors_share_one_corrective_replan(self) -> None:
        # Phase 3 (prose-only plan) + Phase 4A (the artifact block references
        # subtask ids that no longer exist): still exactly ONE retry, total.
        payload = react_plan_json(block())
        payload["delegations"] = [
            {"id": "s1", "title": "Describe the architecture",
             "instruction": "Explain the component structure.",
             "assigned_model": "alpha"},
        ]
        client = QueuedPlannerClient(payload, react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIsNotNone(plan.artifact_plan)
        feedback = client.prompts[1]
        self.assertIn("no concrete implementation work", feedback)
        self.assertIn("unknown subtask", feedback)  # Phase 4A parse error

    def test_phase_3_and_4b_errors_share_one_corrective_replan(self) -> None:
        # Phase 3 (prose-only plan) + Phase 4B (an executable package with no
        # mapped subtask). The artifact block still PARSES, so 4B validation
        # runs — and both problems ride the SAME single corrective re-plan.
        payload = react_plan_json(block())
        for delegation in payload["delegations"]:
            delegation["title"] = f"Discussion of {delegation['id']}"
            delegation["instruction"] = (
                "Explain how this part of the system is expected to work."
            )
        packages_of(payload["artifact_plan"])["shared-ui"].pop("subtask_id")
        client = QueuedPlannerClient(payload, react_plan_json())
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 2)  # ONE corrective re-plan, total
        self.assertIsNotNone(plan.artifact_plan)
        feedback = client.prompts[1]
        self.assertIn("no concrete implementation work", feedback)   # Phase 3
        self.assertIn("has no mapped subtask", feedback)             # Phase 4B

    def test_advisory_warnings_never_trigger_an_extra_planner_call(self) -> None:
        # A package with only manifest-wide validation coverage is advisory.
        data = block()
        for package in data["packages"]:
            package.pop("completion", None)
        client = QueuedPlannerClient(react_plan_json(data))
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(plan.artifact_plan)

    def test_json_repair_then_invalid_plan_has_one_semantic_replan(self) -> None:
        bad = react_plan_json()
        packages_of(bad["artifact_plan"])["shared-ui"].pop("subtask_id")

        class MalformedThenSemanticClient(LLMClient):
            def __init__(self):
                super().__init__(mock=True, max_retries=2, retry_backoff_s=0)
                self.calls = 0
                self.prompts = []

            def complete(self, *, provider, model, messages, **kwargs):
                self.calls += 1
                self.prompts.append("\n".join(m.content for m in messages))
                payload = (
                    "not valid json" if self.calls == 1
                    else json.dumps(bad) if self.calls == 2
                    else json.dumps(react_plan_json())
                )
                return LLMResponse(text=payload, provider=provider, model=model)

        client = MalformedThenSemanticClient()
        plan = make_planner(client).plan(task(), 0, 2)
        self.assertIsNotNone(plan.artifact_plan)
        self.assertEqual(client.calls, 3)
        self.assertIn("could not be parsed as JSON", client.prompts[1])
        self.assertNotIn("PREVIOUS PLAN WAS REJECTED", client.prompts[1])
        self.assertIn("PREVIOUS PLAN WAS REJECTED", client.prompts[2])


class TestPlannerPromptRules(unittest.TestCase):
    def test_artifact_schema_appears_exactly_once(self) -> None:
        messages = build_planner_messages(task(), 0, 2, "alpha")
        combined = "\n".join(m.content for m in messages)
        self.assertEqual(combined.count("ARTIFACT PLAN"), 1)
        self.assertEqual(combined.count('"title": "short deliverable name"'), 1)
        self.assertEqual(combined.count('"packages": ['), 1)
        self.assertEqual(combined.count('"validations": ['), 1)
        # Phase 4B modified the Phase 4A schema in place; it did not append a
        # second one, so the schema block appears only in the system message.
        self.assertNotIn("ARTIFACT PLAN", messages[1].content)

    def test_corrective_prompt_does_not_duplicate_the_schema(self) -> None:
        messages = build_planner_messages(
            task(), 0, 2, "alpha",
            repair_feedback="the artifact decomposition is invalid: fix it",
        )
        combined = "\n".join(m.content for m in messages)
        self.assertEqual(combined.count('"title": "short deliverable name"'), 1)
        self.assertIn("PREVIOUS PLAN WAS REJECTED", combined)

    def test_bounded_decomposition_rules_are_present(self) -> None:
        combined = "\n".join(
            m.content for m in build_planner_messages(task(), 0, 2, "alpha")
        )
        for rule in ("BOUNDED execution units", "MUST set \"subtask_id\"",
                     "wholesale", "validation package runs AFTER",
                     "integration package runs AFTER"):
            self.assertIn(rule, combined)

    def test_recursive_plans_do_not_receive_the_root_artifact_schema(self) -> None:
        combined = "\n".join(
            m.content for m in build_planner_messages(task(), 1, 2, "alpha")
        )
        self.assertNotIn("ARTIFACT PLAN", combined)

    def test_scoped_recursive_prompt_declares_one_output_owner(self) -> None:
        scoped = Task(
            prompt="Build the application shell.", contract=IMPLEMENT,
            execution_scope=derive_execution_scope(react_work_plan(), "shell"),
        )
        combined = "\n".join(
            m.content for m in build_planner_messages(scoped, 1, 2, "alpha")
        )
        self.assertNotIn("ARTIFACT PLAN", combined)
        self.assertEqual(combined.count("ASSIGNED ARTIFACT SCOPE"), 1)
        self.assertIn('"produces_parent_artifacts": true', combined)
        self.assertIn("exactly ONE delegation", combined)

    def test_scoped_support_prompt_prohibits_output_ownership(self) -> None:
        scoped = Task(
            prompt="Research shell constraints.", contract=IMPLEMENT,
            execution_scope=derive_execution_scope(react_work_plan(), "shell"),
            execution_scope_output_required=False,
        )
        combined = "\n".join(
            m.content for m in build_planner_messages(scoped, 2, 3, "alpha")
        )
        self.assertIn('"produces_parent_artifacts": false', combined)
        self.assertIn("intermediate support plan", combined)

    def test_no_file_contents_are_requested_in_the_declaration(self) -> None:
        combined = "\n".join(
            m.content for m in build_planner_messages(task(), 0, 2, "alpha")
        )
        self.assertIn("NEVER include file contents", combined)
        self.assertIn("paths, names and short descriptions only", combined)
        self.assertNotRegex(
            combined, r'artifact_plan[^{]*\{[^}]*"(content|code|body)"'
        )

    def test_planner_prompt_stays_bounded(self) -> None:
        from dozen.prompts import PLANNER_ARTIFACT_PLAN_RULE
        # Bound raised once for the deliberate Phase 4F output-aware sizing
        # rule; the guard still catches accidental unbounded growth.
        self.assertLess(len(PLANNER_ARTIFACT_PLAN_RULE), 3000)


class TestRecursiveScopePlannerValidation(unittest.TestCase):
    def scoped_task(self) -> Task:
        return Task(
            prompt="Build the application shell.", contract=IMPLEMENT,
            execution_scope=derive_execution_scope(react_work_plan(), "shell"),
        )

    @staticmethod
    def payload(*producer_indexes: int) -> dict:
        delegations = [
            {"id": "r", "title": "Research", "instruction": "Research constraints.",
             "assigned_model": "alpha"},
            {"id": "i", "title": "Implement", "instruction": "Implement shell files.",
             "depends_on": ["r"], "assigned_model": "alpha"},
        ]
        for index in producer_indexes:
            delegations[index]["produces_parent_artifacts"] = True
        return {
            "analysis": "bounded recursive plan",
            "delegations": delegations,
            "synthesis_strategy": "return the implementation",
        }

    def test_missing_owner_uses_one_corrective_replan(self) -> None:
        client = QueuedPlannerClient(self.payload(), self.payload(1))
        plan = make_planner(client).plan(self.scoped_task(), 1, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("exactly one delegation", client.prompts[1])
        self.assertEqual(
            sum(st.produces_parent_artifacts for st in plan.subtasks), 1
        )

    def test_duplicate_owners_fail_after_one_correction(self) -> None:
        client = QueuedPlannerClient(self.payload(0, 1), self.payload(0, 1))
        with self.assertRaisesRegex(PlanError, "exactly one delegation"):
            make_planner(client).plan(self.scoped_task(), 1, 2)
        self.assertEqual(client.calls, 2)

    def test_direct_scoped_output_is_corrected_to_a_verified_worker(self) -> None:
        direct = {
            "analysis": "answer directly",
            "delegations": [],
            "direct_answer": "Here are some suggested shell files.",
            "synthesis_strategy": "",
        }
        client = QueuedPlannerClient(direct, self.payload(1))
        plan = make_planner(client).plan(self.scoped_task(), 1, 2)
        self.assertEqual(client.calls, 2)
        self.assertFalse(plan.is_direct())
        self.assertIn("bypass the scoped worker and verifier", client.prompts[1])


class ScriptedClient(QueuedPlannerClient):
    """Planner JSON for the manager; code for workers/synthesizer."""

    def __init__(self, *payloads):
        super().__init__(*payloads)
        self.worker_prompts = []

    def complete(self, *, provider, model, messages, **kwargs):
        from dozen.llm_client import LLMResponse
        system = messages[0].content
        if "MANAGER" in system:
            return super().complete(provider=provider, model=model,
                                    messages=messages, **kwargs)
        if "Worker" in system:
            self.worker_prompts.append(messages[1].content)
        return LLMResponse(
            text="```tsx\nexport default function App() { return <div/>; }\n```",
            provider=provider, model=model,
        )


class TestEndToEnd(unittest.TestCase):
    def test_bounded_decomposition_runs_end_to_end(self) -> None:
        from dozen import AgentPool, AgentSpec
        from dozen.config import OrchestratorConfig

        client = ScriptedClient(react_plan_json())
        pool = AgentPool([
            AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4),
        ])
        config = OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                    verify_outputs=False, use_llm_router=False)
        result = Orchestrator(client=client, pool=pool, config=config).run(
            "I want you to build me a React-based dashboard with tests."
        )
        self.assertTrue(result.error)
        self.assertEqual(result.artifact_assembly.status.value, "failed")
        self.assertEqual(client.calls, 1)  # no corrective re-plan needed
        self.assertIsNotNone(result.plan.artifact_plan)
        self.assertEqual(len(result.subtask_results), 6)

    def test_recursive_package_executes_one_scoped_output_worker(self) -> None:
        from dozen import AgentPool, AgentSpec
        from dozen.config import OrchestratorConfig

        root = react_plan_json()
        for delegation in root["delegations"]:
            if delegation["id"] == "s2":
                delegation["complex"] = True
        child = TestRecursiveScopePlannerValidation.payload(1)
        client = ScriptedClient(root, child)
        agent = AgentSpec(
            name="alpha", provider="openai", model="gpt",
            strengths={"reasoning": 0.9, "coding": 0.9}, tier=4,
        )
        result = Orchestrator(
            client=client, pool=AgentPool([agent]),
            config=OrchestratorConfig(
                max_parallelism=1, max_repair_attempts=0,
                verify_outputs=False, use_llm_router=False, max_depth=2,
            ),
        ).run("I want you to build me a React-based dashboard with tests.")
        self.assertTrue(result.error)
        self.assertEqual(result.artifact_assembly.status.value, "failed")
        self.assertEqual(client.calls, 2)
        research = next(p for p in client.worker_prompts
                        if "Research constraints" in p)
        implementation = next(p for p in client.worker_prompts
                              if "Implement shell files" in p)
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", research)
        self.assertEqual(implementation.count("ASSIGNED ARTIFACT PACKAGE"), 1)


if __name__ == "__main__":
    unittest.main()
