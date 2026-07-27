"""Phase 4B — executor scope threading, verifier compatibility, recursion."""

from __future__ import annotations

import json
import functools
import unittest
from unittest.mock import Mock, patch

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse
from dozen.config import OrchestratorConfig
from dozen.decomposition import ArtifactExecutionScope, derive_execution_scope
from dozen.executor import Executor
from dozen.intent import resolve_contract
from dozen.models import Plan, SubTask, Task
from dozen.router import Router
from dozen.verifier import Verdict, Verifier
from dozen.prompts import build_planner_messages

from .harness import react_plan, react_work_plan

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")

REAL_CODE = """```tsx
export default function App() {
  const [ready, setReady] = useState(true);
  return <div>{String(ready)}</div>;
}
```"""


class RecordingClient(LLMClient):
    """Records every worker prompt; returns real-looking code."""

    def __init__(self) -> None:
        super().__init__(mock=True)
        self.worker_prompts: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        if "Worker" in system:
            self.worker_prompts.append(messages[1].content)
            return LLMResponse(text=REAL_CODE, provider=provider, model=model)
        if "Verifier" in system:
            return LLMResponse(
                text=json.dumps({"passed": True, "score": 0.9, "feedback": ""}),
                provider=provider, model=model,
            )
        return LLMResponse(text=REAL_CODE, provider=provider, model=model)


def make_executor(client, verifier=None, recurse_fn=None, verify=False) -> Executor:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
    pool = AgentPool([agent])
    config = OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                                verify_outputs=verify, use_llm_router=False)
    return Executor(
        client=client, pool=pool,
        router=Router(client, pool, router_agent=agent, use_llm_router=False),
        verifier=verifier or Verifier(client, agent),
        config=config,
        recurse_fn=recurse_fn or (lambda *_a: REAL_CODE),
        log=lambda _m: None,
    )


class TestExecutorScopeThreading(unittest.TestCase):
    def prompt_for(self, prompts: list[str], subtask_id: str) -> str:
        """The worker prompt for one subtask, found by its unique instruction."""
        marker = f"Do the {subtask_id} work."
        matches = [prompt for prompt in prompts if marker in prompt]
        self.assertEqual(len(matches), 1, f"{subtask_id}: {len(matches)} prompts")
        return matches[0]

    def test_each_worker_receives_only_its_own_package_scope(self) -> None:
        client = RecordingClient()
        make_executor(client).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            react_plan(), depth=0,
        )
        prompts = client.worker_prompts
        self.assertEqual(len(prompts), 6)

        shell = self.prompt_for(prompts, "shell")
        self.assertIn("application-shell", shell)
        self.assertIn("src/app/App.tsx", shell)
        self.assertIn("src/app/router.tsx", shell)
        # Another package's artifacts never reach this worker.
        self.assertNotIn("src/components/ui/Card.tsx", shell)
        self.assertNotIn("tests/dashboard.test.tsx", shell)
        self.assertNotIn("src/features/dashboard/DashboardPage.tsx", shell)

        shared = self.prompt_for(prompts, "shared-ui")
        self.assertIn("src/components/ui/Card.tsx", shared)
        self.assertNotIn("src/app/router.tsx", shared)
        self.assertNotIn("tests/dashboard.test.tsx", shared)

        tests = self.prompt_for(prompts, "tests")
        self.assertIn("tests/dashboard.test.tsx", tests)
        self.assertNotIn("package.json", tests)

        # No prompt dumps the manifest, and every one keeps the 4A envelope.
        for prompt in prompts:
            self.assertNotIn("ARTIFACT MANIFEST", prompt)
            self.assertIn('"key_decisions"', prompt)

    def test_plan_without_artifact_plan_gets_no_scope_block(self) -> None:
        client = RecordingClient()
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Write the app", instruction="Write app.py.", id="s1"),
        ])
        make_executor(client).run(
            Task(prompt="Build an app.", contract=IMPLEMENT), plan, depth=0
        )
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", client.worker_prompts[0])

    def test_subtask_without_a_mapped_package_gets_no_scope_block(self) -> None:
        client = RecordingClient()
        plan = react_plan()
        plan.subtasks.append(
            SubTask(title="Write release notes",
                    instruction="Draft the release notes.", id="extra")
        )
        make_executor(client).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            plan, depth=0,
        )
        unmapped = [
            prompt for prompt in client.worker_prompts
            if "Draft the release notes." in prompt
        ]
        self.assertEqual(len(unmapped), 1)
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", unmapped[0])
        # Every mapped subtask still gets one.
        scoped = [
            prompt for prompt in client.worker_prompts
            if "ASSIGNED ARTIFACT PACKAGE" in prompt
        ]
        self.assertEqual(len(scoped), 6)


class ScopeCapturingVerifier:
    """A modern verifier: accepts contract AND scope."""

    def __init__(self) -> None:
        self.scopes: list = []

    def verify(self, subtask, output, dependency_outputs, contract=None,
               scope=None) -> Verdict:
        self.scopes.append(scope)
        return Verdict(True, 1.0, "")


class LegacyVerifier:
    """A pre-Phase-3 verifier: the original THREE-argument signature."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, subtask, output, dependency_outputs) -> Verdict:
        self.calls += 1
        return Verdict(True, 1.0, "")


class ContractOnlyVerifier:
    """A Phase 3 verifier: contract but no scope."""

    def __init__(self) -> None:
        self.contracts: list = []

    def verify(self, subtask, output, dependency_outputs, contract=None) -> Verdict:
        self.contracts.append(contract)
        return Verdict(True, 1.0, "")


class TestVerifierCompatibility(unittest.TestCase):
    def run_with(self, verifier) -> None:
        make_executor(RecordingClient(), verifier=verifier, verify=True).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            react_plan(), depth=0,
        )

    def test_scope_is_passed_when_supported(self) -> None:
        verifier = ScopeCapturingVerifier()
        self.run_with(verifier)
        self.assertEqual(len(verifier.scopes), 6)
        for scope in verifier.scopes:
            self.assertIsInstance(scope, ArtifactExecutionScope)

    def test_legacy_three_argument_verifier_remains_compatible(self) -> None:
        verifier = LegacyVerifier()
        self.run_with(verifier)
        self.assertEqual(verifier.calls, 6)

    def test_contract_only_verifier_remains_compatible(self) -> None:
        verifier = ContractOnlyVerifier()
        self.run_with(verifier)
        self.assertEqual(len(verifier.contracts), 6)
        self.assertTrue(all(c is IMPLEMENT for c in verifier.contracts))

    def test_concrete_verifier_receives_the_scope_in_its_prompt(self) -> None:
        prompts: list[str] = []

        class CapturingClient(RecordingClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "Verifier" in messages[0].content:
                    prompts.append(messages[1].content)
                return super().complete(provider=provider, model=model,
                                        messages=messages, **kwargs)

        make_executor(CapturingClient(), verify=True).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            react_plan(), depth=0,
        )
        self.assertTrue(prompts)
        self.assertTrue(
            any("ASSIGNED ARTIFACT PACKAGE" in prompt for prompt in prompts)
        )

    def test_var_keyword_verifier_receives_contract_and_scope(self) -> None:
        class VarKeywordVerifier:
            def __init__(self):
                self.kwargs = []

            def verify(self, *args, **kwargs):
                self.kwargs.append(kwargs)
                return Verdict(True, 1.0, "")

        verifier = VarKeywordVerifier()
        self.run_with(verifier)
        self.assertEqual(len(verifier.kwargs), 6)
        self.assertTrue(all(item["contract"] is IMPLEMENT
                            for item in verifier.kwargs))
        self.assertTrue(all(isinstance(item["scope"], ArtifactExecutionScope)
                            for item in verifier.kwargs))

    def test_var_positional_verifier_gets_no_unsupported_keywords(self) -> None:
        class VarPositionalVerifier:
            def __init__(self):
                self.calls = []

            def verify(self, *args):
                self.calls.append(args)
                return Verdict(True, 1.0, "")

        verifier = VarPositionalVerifier()
        self.run_with(verifier)
        self.assertEqual([len(args) for args in verifier.calls], [3] * 6)

    def test_mock_and_callable_verify_objects_are_supported(self) -> None:
        mock_verifier = Mock()
        mock_verifier.verify.return_value = Verdict(True, 1.0, "")
        self.run_with(mock_verifier)
        self.assertEqual(mock_verifier.verify.call_count, 6)
        self.assertIn("scope", mock_verifier.verify.call_args.kwargs)

        class VerifyCallable:
            def __init__(self):
                self.calls = []

            def __call__(self, subtask, output, dependencies, **kwargs):
                self.calls.append(kwargs)
                return Verdict(True, 1.0, "")

        class Holder:
            def __init__(self):
                self.verify = VerifyCallable()

        holder = Holder()
        self.run_with(holder)
        self.assertEqual(len(holder.verify.calls), 6)
        self.assertIn("contract", holder.verify.calls[0])
        self.assertIn("scope", holder.verify.calls[0])

    def test_decorated_verifier_preserves_supported_keywords(self) -> None:
        calls = []

        def target(subtask, output, dependencies, contract=None, scope=None):
            return Verdict(True, 1.0, "")

        @functools.wraps(target)
        def decorated(*args, **kwargs):
            calls.append(kwargs)
            return target(*args, **kwargs)

        class Holder:
            verify = staticmethod(decorated)

        self.run_with(Holder())
        self.assertEqual(len(calls), 6)
        self.assertIn("contract", calls[0])
        self.assertIn("scope", calls[0])

    def test_signature_inspection_failure_falls_back_to_legacy_call(self) -> None:
        verifier = LegacyVerifier()
        with patch("dozen.executor.inspect.signature", side_effect=ValueError):
            self.run_with(verifier)
        self.assertEqual(verifier.calls, 6)


class TestRecursiveScopeInheritance(unittest.TestCase):
    def recursive_plan(self) -> Plan:
        plan = react_plan()
        for subtask in plan.subtasks:
            if subtask.id == "shell":
                subtask.complex = True
        return plan

    def capture_child(self) -> Task:
        captured: list[Task] = []
        make_executor(
            RecordingClient(),
            recurse_fn=lambda _st, sub_task, _d: (
                captured.append(sub_task) or REAL_CODE
            ),
        ).run(
            Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
            self.recursive_plan(), depth=0,
        )
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_child_inherits_the_assigned_package_scope(self) -> None:
        child = self.capture_child()
        self.assertIsNotNone(child.execution_scope)
        self.assertEqual(child.execution_scope.package_ids, ("application-shell",))
        self.assertEqual(
            set(child.execution_scope.owned_artifact_ids),
            {"src/app/App.tsx", "src/app/router.tsx",
             "src/components/layout/DashboardLayout.tsx"},
        )

    def test_child_does_not_inherit_unrelated_packages(self) -> None:
        scope = self.capture_child().execution_scope
        self.assertNotIn("shared-ui", scope.package_ids)
        self.assertNotIn("dashboard-feature", scope.package_ids)
        self.assertNotIn("src/components/ui/Card.tsx", scope.owned_artifact_ids)
        self.assertNotIn("tests/dashboard.test.tsx", scope.owned_artifact_ids)

    def test_child_keeps_the_parent_contract(self) -> None:
        self.assertIs(self.capture_child().contract, IMPLEMENT)

    def test_child_planner_receives_the_bounded_scope(self) -> None:
        child = self.capture_child()
        text = "\n".join(
            m.content for m in build_planner_messages(child, 1, 2, "alpha")
        )
        self.assertIn("ASSIGNED ARTIFACT SCOPE", text)
        self.assertIn("src/app/App.tsx", text)
        self.assertNotIn("src/components/ui/Card.tsx", text)

    def test_child_cannot_redefine_the_root_manifest(self) -> None:
        child = self.capture_child()
        text = "\n".join(
            m.content for m in build_planner_messages(child, 0, 2, "alpha")
        )
        # Even at depth 0 (a scoped child re-entering the pipeline), the
        # artifact SCHEMA is withheld: it may not declare a new manifest.
        self.assertNotIn("ARTIFACT PLAN", text)
        self.assertNotIn("artifact_plan", text)
        self.assertIn("do not declare a new artifact manifest", text)

    def test_child_planner_permits_intermediate_prose(self) -> None:
        child = self.capture_child()
        text = "\n".join(
            m.content for m in build_planner_messages(child, 1, 2, "alpha")
        )
        self.assertIn("Intermediate research or design subtasks are allowed",
                      text)

    def test_final_recursive_result_stays_directed_to_owned_artifacts(self) -> None:
        child = self.capture_child()
        text = "\n".join(
            m.content for m in build_planner_messages(child, 1, 2, "alpha")
        )
        self.assertIn(
            "final deliverable must contain complete contents for the owned "
            "artifacts", text.replace("\n", " "),
        )

    def test_unscoped_recursive_child_is_unchanged(self) -> None:
        captured: list[Task] = []
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Big piece", instruction="Build the frontend.",
                    complex=True, id="s1"),
        ])
        make_executor(
            RecordingClient(),
            recurse_fn=lambda _st, sub_task, _d: (
                captured.append(sub_task) or REAL_CODE
            ),
        ).run(Task(prompt="Build an app.", contract=IMPLEMENT), plan, depth=0)
        self.assertIsNone(captured[0].execution_scope)

    def test_deeper_descendant_keeps_the_confined_scope(self) -> None:
        # A marked producer that itself recurses passes ITS scope down, never a
        # wider one.
        scope = derive_execution_scope(react_work_plan(), "shell")
        parent = Task(prompt="Build the shell.", contract=IMPLEMENT,
                      execution_scope=scope)
        captured: list[Task] = []
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Sub piece", instruction="Write the router.",
                    complex=True, id="s1", produces_parent_artifacts=True),
        ])
        make_executor(
            RecordingClient(),
            recurse_fn=lambda _st, sub_task, _d: (
                captured.append(sub_task) or REAL_CODE
            ),
        ).run(parent, plan, depth=0)
        self.assertIs(captured[0].execution_scope, scope)
        self.assertTrue(captured[0].execution_scope_output_required)

    def _run_scoped_children(self, subtasks):
        scope = derive_execution_scope(react_work_plan(), "shell")
        client = RecordingClient()
        make_executor(client).run(
            Task(prompt="Build the shell.", contract=IMPLEMENT,
                 execution_scope=scope),
            Plan(analysis="a", synthesis_strategy="s", subtasks=subtasks),
            depth=1,
        )
        return client.worker_prompts

    def test_research_and_implementation_only_scope_the_implementation(self) -> None:
        prompts = self._run_scoped_children([
            SubTask(title="Research", instruction="Research constraints.", id="r"),
            SubTask(title="Implement", instruction="Implement all shell files.",
                    depends_on=["r"], id="i", produces_parent_artifacts=True),
        ])
        research = next(p for p in prompts if "Research constraints" in p)
        implementation = next(p for p in prompts if "Implement all shell" in p)
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", research)
        self.assertEqual(implementation.count("ASSIGNED ARTIFACT PACKAGE"), 1)

    def test_design_implementation_review_do_not_duplicate_ownership(self) -> None:
        prompts = self._run_scoped_children([
            SubTask(title="Design", instruction="Design the shell.", id="d"),
            SubTask(title="Implement", instruction="Implement the shell.",
                    depends_on=["d"], id="i", produces_parent_artifacts=True),
            SubTask(title="Review", instruction="Review the implementation.",
                    depends_on=["i"], id="r"),
        ])
        scoped = [p for p in prompts if "ASSIGNED ARTIFACT PACKAGE" in p]
        self.assertEqual(len(scoped), 1)
        self.assertIn("Implement the shell", scoped[0])

    def test_two_implementation_children_have_one_output_owner(self) -> None:
        prompts = self._run_scoped_children([
            SubTask(title="Implementation support", instruction="Prototype routing.",
                    id="a"),
            SubTask(title="Final implementation", instruction="Implement owned files.",
                    depends_on=["a"], id="b", produces_parent_artifacts=True),
        ])
        self.assertEqual(
            sum("ASSIGNED ARTIFACT PACKAGE" in p for p in prompts), 1
        )

    def test_duplicate_recursive_output_owners_fail_closed(self) -> None:
        scope = derive_execution_scope(react_work_plan(), "shell")
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="One", instruction="Implement one.", id="a",
                    produces_parent_artifacts=True),
            SubTask(title="Two", instruction="Implement two.", id="b",
                    produces_parent_artifacts=True),
        ])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_executor(RecordingClient()).run(
                Task(prompt="Build shell", contract=IMPLEMENT,
                     execution_scope=scope), plan, depth=1,
            )

    def test_nested_support_recursion_keeps_boundary_without_ownership(self) -> None:
        scope = derive_execution_scope(react_work_plan(), "shell")
        captured = []
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Research deeply", instruction="Research constraints.",
                    complex=True, id="r"),
            SubTask(title="Implement", instruction="Implement shell.",
                    depends_on=["r"], id="i", produces_parent_artifacts=True),
        ])
        make_executor(
            RecordingClient(),
            recurse_fn=lambda _st, child, _depth: captured.append(child) or REAL_CODE,
        ).run(
            Task(prompt="Build shell", contract=IMPLEMENT,
                 execution_scope=scope), plan, depth=1,
        )
        self.assertIs(captured[0].execution_scope, scope)
        self.assertFalse(captured[0].execution_scope_output_required)


if __name__ == "__main__":
    unittest.main()
