"""Phase 3 — contract guards, prompt propagation, and end-to-end orchestration.

End-to-end tests drive the REAL Orchestrator with a scripted fake client, so
the whole path (boundary resolution → planner validation → worker/synth
prompts → final guard) is exercised without any provider.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from dozen import (
    AgentPool,
    AgentSpec,
    LLMClient,
    LLMMessage,
    LLMResponse,
    Orchestrator,
    Task,
)
from dozen.config import OrchestratorConfig
from dozen.intent import (
    RequestIntent,
    has_real_code,
    resolve_contract,
    validate_final_answer_against_contract,
    validate_plan_against_contract,
)
from dozen.models import Plan, SubTask
from dozen.planner import PlanError, Planner
from dozen.prompts import (
    build_planner_messages,
    build_synthesizer_messages,
    build_verifier_messages,
    build_worker_messages,
)

IMPLEMENT = resolve_contract("Build me a React dashboard.")
ARCHITECTURE = resolve_contract("Design the architecture. Do not write code.")
REVIEW = resolve_contract("Review this code, do not change anything.")
DEBUG = resolve_contract("Fix the crash in this endpoint.")

REAL_CODE = """Here is the dashboard.

```jsx
export default function Dashboard() {
  const [data, setData] = useState([]);
  useEffect(() => { fetch("/api/stats").then(r => r.json()).then(setData); }, []);
  return <div className="grid">{data.map(d => <Card key={d.id} {...d} />)}</div>;
}
```
"""

ARCHITECTURE_PROSE = """# Dashboard Architecture

## Components
The dashboard should use a component hierarchy with a shell, a sidebar and
widgets. State management is best handled by Redux Toolkit or Zustand.

## Recommended libraries
- React 18, Vite, TanStack Query

No application code is included.
"""


def plan_with(*titles_and_instructions) -> Plan:
    return Plan(
        analysis="a",
        subtasks=[SubTask(title=t, instruction=i)
                  for t, i in titles_and_instructions],
        synthesis_strategy="merge",
    )


# --------------------------------------------------------------------------- #
class TestHasRealCode(unittest.TestCase):
    def test_detects_substantial_fenced_code(self) -> None:
        self.assertTrue(has_real_code(REAL_CODE))

    def test_empty_fence_is_not_code(self) -> None:
        """A fence alone must never be taken as proof of a delivered artifact."""
        self.assertFalse(has_real_code("Here is the idea:\n\n```\n\n```\n"))
        self.assertFalse(has_real_code("```js\n// TODO\n```"))

    def test_detects_diffs(self) -> None:
        self.assertTrue(has_real_code("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y"))

    def test_prose_is_not_code(self) -> None:
        self.assertFalse(has_real_code(ARCHITECTURE_PROSE))

    def test_rejects_todo_pseudocode_and_header_only_diff(self) -> None:
        self.assertFalse(has_real_code(
            "```python\n# TODO: implement\n# TODO: validate\n# TODO: test\n```"
        ))
        self.assertFalse(has_real_code(
            "```python\ndef handler():\n    # TODO: implement\n    pass\n```"
        ))
        self.assertFalse(has_real_code(
            "```text\nIF ready THEN\n  LOAD data\nELSE retry\n```"
        ))
        self.assertFalse(has_real_code(
            "```python\nThis is prose placed in a Python fence.\n```"
        ))
        self.assertFalse(has_real_code("@@ -1 +1 @@"))

    def test_accepts_concise_typed_artifacts_and_raw_worker_source(self) -> None:
        for answer in (
            "```json\n{\"enabled\": true}\n```",
            "```sql\nSELECT * FROM users;\n```",
            "```sh\nset -eu\npytest\n```",
            "```html\n<div>ready</div>\n```",
            "def add(a, b):\n    return a + b",
            '{"service": "dashboard", "enabled": true}',
        ):
            with self.subTest(answer=answer):
                self.assertTrue(has_real_code(answer))


class TestPlanGuard(unittest.TestCase):
    def test_architecture_only_plan_fails_implementation_contract(self) -> None:
        plan = plan_with(
            ("Describe the architecture", "Outline the component hierarchy."),
            ("Recommend libraries", "Recommend state management libraries."),
        )
        check = validate_plan_against_contract(IMPLEMENT, plan)
        self.assertFalse(check.ok)
        self.assertIn("no concrete implementation work", check.feedback())

    def test_implementation_plan_passes(self) -> None:
        plan = plan_with(
            ("Design the layout", "Sketch the component tree."),
            ("Write the components", "Write the React component source files."),
            ("Add tests", "Write unit tests for the components."),
        )
        check = validate_plan_against_contract(IMPLEMENT, plan)
        self.assertTrue(check.ok)
        self.assertEqual(check.advisory, ())

    def test_missing_tests_is_advisory_not_fatal(self) -> None:
        plan = plan_with(("Write the code", "Write the dashboard component files."))
        check = validate_plan_against_contract(IMPLEMENT, plan)
        self.assertTrue(check.ok)                      # still delivers the artifact
        self.assertTrue(check.advisory)                # but tests were requested

    def test_architecture_contract_accepts_prose_plan(self) -> None:
        plan = plan_with(("Describe the architecture", "Outline the components."))
        self.assertTrue(validate_plan_against_contract(ARCHITECTURE, plan).ok)
        self.assertTrue(validate_plan_against_contract(REVIEW, plan).ok)

    def test_direct_answer_plan_is_left_to_the_final_guard(self) -> None:
        direct = Plan(analysis="a", subtasks=[], synthesis_strategy="",
                      direct_answer="here you go")
        self.assertTrue(validate_plan_against_contract(IMPLEMENT, direct).ok)

    def test_required_plan_matrix(self) -> None:
        terse = plan_with(*[(title, title) for title in (
            "Application shell", "Authentication", "Dashboard modules",
            "Integration", "Validation",
        )])
        self.assertTrue(validate_plan_against_contract(IMPLEMENT, terse).ok)

        prose = plan_with(
            ("Explain implementation options", "Explain implementation options"),
            ("Recommend a code structure", "Recommend a code structure"),
            ("Describe testing", "Describe how testing would work"),
        )
        self.assertFalse(validate_plan_against_contract(IMPLEMENT, prose).ok)

        architecture = plan_with(
            ("Analyze requirements", "Analyze requirements"),
            ("Define components", "Define components"),
            ("Document data flow", "Document data flow"),
        )
        self.assertTrue(validate_plan_against_contract(ARCHITECTURE, architecture).ok)

        debug = plan_with(
            ("Reproduce failure", "Reproduce the failure"),
            ("Identify root cause", "Identify the root cause"),
            ("Apply correction", "Apply the correction to the source"),
            ("Add regression test", "Add a regression test"),
        )
        self.assertTrue(validate_plan_against_contract(DEBUG, debug).ok)

    def test_prose_vocabulary_does_not_bypass_plan_guard(self) -> None:
        plan = plan_with((
            "Explain the approach",
            "Describe how to write source code files and tests without "
            "implementing them.",
        ))
        self.assertFalse(validate_plan_against_contract(IMPLEMENT, plan).ok)

    def test_debug_plan_requires_a_correction_not_only_a_test(self) -> None:
        plan = plan_with(
            ("Identify root cause", "Identify the root cause"),
            ("Add regression test", "Add a regression test"),
        )
        check = validate_plan_against_contract(DEBUG, plan)
        self.assertFalse(check.ok)
        self.assertIn("corrective fix", check.feedback())

    def test_existing_json_repair_behavior_is_preserved(self) -> None:
        class MalformedThenValidClient(LLMClient):
            def __init__(self):
                super().__init__(mock=True, max_retries=2, retry_backoff_s=0)
                self.calls = 0
                self.message_counts = []

            def complete(self, *, provider, model, messages, **kwargs):
                self.calls += 1
                self.message_counts.append(len(messages))
                text = "not valid json" if self.calls == 1 else json.dumps({
                    "analysis": "design",
                    "delegations": [{
                        "id": "s1",
                        "title": "Describe architecture",
                        "instruction": "Define components and data flow.",
                    }],
                    "synthesis_strategy": "present",
                })
                return LLMResponse(text=text, provider=provider, model=model)

        agent = AgentSpec(
            name="alpha", provider="openai", model="gpt",
            strengths={"reasoning": 1.0}, tier=4,
        )
        pool = AgentPool([agent])
        client = MalformedThenValidClient()
        plan = Planner(client, agent, pool).plan(
            Task(prompt="Design an architecture.", contract=ARCHITECTURE), 0, 2
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.message_counts, [2, 4])
        self.assertEqual(len(plan.subtasks), 1)


class TestFinalGuard(unittest.TestCase):
    def test_prose_only_fails_implementation(self) -> None:
        check = validate_final_answer_against_contract(IMPLEMENT, ARCHITECTURE_PROSE)
        self.assertFalse(check.ok)
        self.assertIn("declines to deliver code", check.feedback())

    def test_prose_without_disclaimer_still_fails_implementation(self) -> None:
        check = validate_final_answer_against_contract(
            IMPLEMENT, "You should use React with Vite and Redux Toolkit."
        )
        self.assertFalse(check.ok)
        self.assertIn("no code", check.feedback())

    def test_code_with_supporting_explanation_passes(self) -> None:
        answer = "Here's how it works, and why.\n\n" + REAL_CODE + "\nRun `npm start`."
        self.assertTrue(
            validate_final_answer_against_contract(IMPLEMENT, answer).ok,
            "supporting prose must not disqualify a real code deliverable",
        )

    def test_architecture_contract_accepts_prose(self) -> None:
        self.assertTrue(
            validate_final_answer_against_contract(ARCHITECTURE, ARCHITECTURE_PROSE).ok
        )

    def test_review_passes_without_code(self) -> None:
        review = "The module mixes I/O with domain logic; extract a port."
        self.assertTrue(validate_final_answer_against_contract(REVIEW, review).ok)

    def test_debug_explanation_without_fix_fails(self) -> None:
        answer = ("The crash happens because `user` is None when the session "
                  "expires, so `user.id` raises AttributeError.")
        check = validate_final_answer_against_contract(DEBUG, answer)
        self.assertFalse(check.ok)

    def test_debug_with_fix_passes(self) -> None:
        answer = "Root cause: `user` is None.\n\n" + REAL_CODE
        self.assertTrue(validate_final_answer_against_contract(DEBUG, answer).ok)

    def test_empty_answer_always_fails(self) -> None:
        self.assertFalse(validate_final_answer_against_contract(REVIEW, "").ok)

    def test_repository_change_summaries_are_indeterminate_not_false_failures(self) -> None:
        modify = resolve_contract("Add pagination to this endpoint.")
        answer = (
            "Changed api/routes.py and tests/test_routes.py. Pagination now "
            "validates cursors. Ran 24 tests; all passed."
        )
        check = validate_final_answer_against_contract(modify, answer)
        self.assertTrue(check.ok)
        self.assertIn("cannot independently attest", check.feedback())

        debug = resolve_contract("Find why this crashes and fix it.")
        answer = (
            "Root cause: cache.py reused an expired token. Fixed cache.py and "
            "tests/test_cache.py. The regression tests passed."
        )
        check = validate_final_answer_against_contract(debug, answer)
        self.assertTrue(check.ok)
        self.assertTrue(check.advisory)

    def test_debug_code_without_root_cause_fails(self) -> None:
        answer = "```python\ndef fix(value):\n    return value or 0\n```"
        check = validate_final_answer_against_contract(DEBUG, answer)
        self.assertFalse(check.ok)
        self.assertIn("root-cause diagnosis", check.feedback())

    def test_explicit_no_code_contract_rejects_substantive_code(self) -> None:
        contract = resolve_contract("Build a service, but do not write code.")
        answer = "```python\ndef service():\n    return 'ready'\n```"
        self.assertFalse(
            validate_final_answer_against_contract(contract, answer).ok
        )
        explicit = type(contract)(
            intent=RequestIntent.ARCHITECTURE,
            user_constraints=("Do not write code.",),
        )
        self.assertFalse(
            validate_final_answer_against_contract(explicit, answer).ok
        )

    def test_real_code_beats_a_negated_omission_phrase(self) -> None:
        answer = "No code was omitted. Full implementation follows.\n" + REAL_CODE
        self.assertTrue(
            validate_final_answer_against_contract(IMPLEMENT, answer).ok
        )

    def test_truncated_fence_is_not_accepted_as_a_complete_artifact(self) -> None:
        answer = "```python\ndef main():\n    print('unterminated')\n"
        self.assertFalse(
            validate_final_answer_against_contract(IMPLEMENT, answer).ok
        )

    def test_architecture_allows_incidental_inline_code(self) -> None:
        answer = (
            "Use a queue between the services. A producer can call "
            "`publish(event)`; this is illustrative, not an implementation."
        )
        self.assertTrue(
            validate_final_answer_against_contract(ARCHITECTURE, answer).ok
        )


class TestPromptPropagation(unittest.TestCase):
    def test_planner_receives_contract_and_validity_rule(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        user = build_planner_messages(task, 0, 2, "gpt (general)")[1].content
        self.assertIn("DELIVERABLE CONTRACT", user)
        self.assertIn("IMPLEMENT", user)
        self.assertIn("PLAN VALIDITY RULE", user)
        self.assertIn("tests or validation", user)

    def test_planner_repair_feedback_included(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        user = build_planner_messages(task, 0, 2, "gpt", repair_feedback="no code work")[1].content
        self.assertIn("PREVIOUS PLAN WAS REJECTED", user)
        self.assertIn("no code work", user)

    def test_worker_receives_concise_requirement(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        user = build_worker_messages(task, SubTask(title="t", instruction="i"), {})[1].content
        self.assertIn("DELIVERABLE REQUIREMENT", user)
        self.assertIn("IMPLEMENT", user)
        self.assertIn("not acceptable substitutes", user)

    def test_worker_receives_prose_contract_too(self) -> None:
        task = Task(prompt="Explain hooks.", contract=resolve_contract("Explain hooks."))
        user = build_worker_messages(task, SubTask(title="t", instruction="i"), {})[1].content
        self.assertIn("DELIVERABLE REQUIREMENT", user)
        self.assertIn("EXPLAIN", user)

    def test_worker_receives_bounded_explicit_task_constraints(self) -> None:
        task = Task(
            prompt="Build a parser.",
            constraints=["Use Python 3.12 only."],
            contract=resolve_contract(
                "Build a parser.", constraints=["Use Python 3.12 only."]
            ),
        )
        user = build_worker_messages(
            task, SubTask(title="Implement", instruction="Write parser.py"), {}
        )[1].content
        self.assertIn("TASK CONSTRAINTS", user)
        self.assertIn("Python 3.12", user)

    def test_verifier_receives_contract_both_ways(self) -> None:
        st = SubTask(title="t", instruction="i")
        code_user = build_verifier_messages(st, "out", {}, contract=IMPLEMENT)[1].content
        self.assertIn("DELIVERABLE CONTRACT", code_user)
        self.assertIn("FAILS", code_user)
        review_user = build_verifier_messages(st, "out", {}, contract=REVIEW)[1].content
        self.assertIn("Do NOT penalize", review_user)   # review needs no code

    def test_verifier_without_contract_still_works(self) -> None:
        st = SubTask(title="t", instruction="i")
        user = build_verifier_messages(st, "out", {})[1].content
        self.assertNotIn("DELIVERABLE CONTRACT", user)

    def test_synthesizer_receives_output_mode(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        user = build_synthesizer_messages(task, "merge", [("t", "out")])[1].content
        self.assertIn("OUTPUT MODE", user)
        self.assertIn("architecture essay", user)

    def test_synthesizer_mode_absent_for_architecture(self) -> None:
        task = Task(prompt="Design it.", contract=ARCHITECTURE)
        user = build_synthesizer_messages(task, "merge", [("t", "out")])[1].content
        self.assertNotIn("OUTPUT MODE", user)

    def test_contract_text_does_not_grow_without_bound(self) -> None:
        long_prompt = "Build a dashboard with tests. " + ("extra requirement. " * 200)
        contract = resolve_contract(long_prompt)
        self.assertLess(len(contract.to_brief()), 1200)

    def test_each_role_receives_one_contract_marker(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        subtask = SubTask(title="Build", instruction="Write dashboard.py")
        prompts = (
            build_planner_messages(task, 0, 2, "gpt")[1].content,
            build_worker_messages(task, subtask, {})[1].content,
            build_verifier_messages(subtask, REAL_CODE, {}, IMPLEMENT)[1].content,
            build_synthesizer_messages(task, "merge", [("Build", REAL_CODE)])[1].content,
        )
        markers = (
            "DELIVERABLE CONTRACT",
            "DELIVERABLE REQUIREMENT",
            "DELIVERABLE CONTRACT",
            "DELIVERABLE CONTRACT",
        )
        for prompt, marker in zip(prompts, markers):
            with self.subTest(marker=marker):
                self.assertEqual(prompt.count(marker), 1)
                self.assertNotIn("DeliverableContract(", prompt)

    def test_web_rendering_keeps_one_intact_worker_artifact_contract(self) -> None:
        from dozen.prompts import ARTIFACT_CONTRACT_MARKER
        from webllm.client import render_messages_for_web

        task = Task(prompt="Build an app.", contract=IMPLEMENT)
        messages = build_worker_messages(
            task, SubTask(title="Build", instruction="Write app.py"), {}
        )
        rendered = render_messages_for_web(messages)
        self.assertEqual(rendered.count(ARTIFACT_CONTRACT_MARKER), 1)
        self.assertEqual(rendered.count("Reply with a SINGLE JSON object"), 1)

    def test_web_rendering_does_not_trust_a_user_contract_marker(self) -> None:
        from dozen.prompts import ARTIFACT_CONTRACT_MARKER
        from webllm.client import render_messages_for_web

        rendered = render_messages_for_web([
            LLMMessage(
                "user",
                f"Explain why the literal {ARTIFACT_CONTRACT_MARKER} is present.",
            )
        ])
        self.assertNotIn("Reply with a SINGLE JSON object", rendered)

    def test_web_rendering_selects_the_actual_final_scoped_contract(self) -> None:
        from dozen.prompts import (
            SCOPED_WORKER_ARTIFACT_CONTRACT,
            WORKER_ARTIFACT_CONTRACT,
        )
        from dozen.decomposition import derive_execution_scope
        from webllm.client import render_messages_for_web
        from ..artifact_decomposition.harness import react_work_plan

        task = Task(prompt="Build an app.", contract=IMPLEMENT)
        messages = build_worker_messages(
            task,
            SubTask(title="Build", instruction="Write the shared UI"),
            {"untrusted": WORKER_ARTIFACT_CONTRACT},
            scope=derive_execution_scope(react_work_plan(), "shared-ui"),
        )
        rendered = render_messages_for_web(messages)
        self.assertTrue(rendered.endswith(SCOPED_WORKER_ARTIFACT_CONTRACT))
        self.assertEqual(rendered.count('"key_decisions"'), 1)

    def test_legacy_prompts_without_contract_are_unchanged(self) -> None:
        task = Task(prompt="do the thing")           # no contract
        planner = build_planner_messages(task, 0, 2, "gpt")[1].content
        worker = build_worker_messages(task, SubTask(title="t", instruction="i"), {})[1].content
        synth = build_synthesizer_messages(task, "s", [("t", "o")])[1].content
        for text in (planner, worker, synth):
            self.assertNotIn("DELIVERABLE CONTRACT", text)
            self.assertNotIn("DELIVERABLE REQUIREMENT", text)


# --------------------------------------------------------------------------- #
# End-to-end orchestration with a scripted fake client
# --------------------------------------------------------------------------- #
class ScriptedClient(LLMClient):
    """Returns planner JSON and worker text chosen by the caller."""

    def __init__(self, plan_json: dict, worker_output: str,
                 synth_output: str, second_plan_json: dict | None = None) -> None:
        super().__init__(mock=True)
        self.plan_json = plan_json
        self.second_plan_json = second_plan_json
        self.worker_output = worker_output
        self.synth_output = synth_output
        self.plan_calls = 0
        self.planner_prompts: list[str] = []
        self.worker_prompts: list[str] = []
        self.synth_prompts: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[1].content
        if "MANAGER" in system:
            self.plan_calls += 1
            self.planner_prompts.append(user)
            data = (self.second_plan_json
                    if self.plan_calls > 1 and self.second_plan_json is not None
                    else self.plan_json)
            text = json.dumps(data)
        elif "SYNTHESIZER" in system:
            self.synth_prompts.append(user)
            text = self.synth_output
        else:
            self.worker_prompts.append(user)
            text = self.worker_output
        return LLMResponse(text=text, provider=provider, model=model,
                           prompt_tokens=1, completion_tokens=1, latency_s=0.0)


def make_orchestrator(client, **overrides) -> Orchestrator:
    pool = AgentPool([
        AgentSpec(name="alpha", provider="openai", model="gpt",
                  strengths={"reasoning": 0.9, "coding": 0.9}, tier=4),
    ])
    cfg = OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                             verify_outputs=False, use_llm_router=False,
                             **overrides)
    return Orchestrator(client=client, pool=pool, config=cfg)


ARCH_PLAN = {
    "analysis": "design it",
    "delegations": [
        {"id": "s1", "title": "Describe the architecture",
         "instruction": "Outline the component hierarchy and recommend libraries.",
         "assigned_model": "alpha"},
    ],
    "synthesis_strategy": "present the design",
}

IMPL_PLAN = {
    "analysis": "build it",
    "delegations": [
        {"id": "s1", "title": "Write the dashboard components",
         "instruction": "Write the complete React component source files.",
         "assigned_model": "alpha"},
        {"id": "s2", "title": "Add tests",
         "instruction": "Write unit tests for the components.",
         "assigned_model": "alpha", "depends_on": ["s1"]},
    ],
    "synthesis_strategy": "merge code then tests",
}


class TestEndToEnd(unittest.TestCase):
    def test_boundary_resolves_missing_contract_exactly_once(self) -> None:
        import importlib

        module = importlib.import_module("dozen.orchestrator")
        client = ScriptedClient(ARCH_PLAN, "Some answer.", "Some final answer.")
        task = Task(prompt="Compare two databases and recommend one.")
        with mock.patch.object(
            module, "resolve_contract", wraps=module.resolve_contract
        ) as resolver:
            make_orchestrator(client).run(task)
        self.assertEqual(resolver.call_count, 1)

        client = ScriptedClient(ARCH_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        explicit = Task(prompt="Review this code.", contract=REVIEW)
        with mock.patch.object(
            module, "resolve_contract", wraps=module.resolve_contract
        ) as resolver:
            make_orchestrator(client).run(explicit)
        self.assertEqual(resolver.call_count, 0)

    def test_dashboard_cannot_succeed_with_architecture_only_work(self) -> None:
        """The headline regression: prose-only plan AND prose-only result."""
        client = ScriptedClient(ARCH_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        result = make_orchestrator(client).run("I want you to build me a React-based dashboard.")
        # The planner was re-asked once with corrective feedback…
        self.assertEqual(client.plan_calls, 2)
        # …and the run is NOT reported as a success.
        self.assertTrue(result.error)
        self.assertIn("contract", result.error.lower())

    def test_dashboard_succeeds_when_the_work_is_real(self) -> None:
        # Phase 4G: model synthesis is eliminated. The deliverable is assembled
        # deterministically, so this asserts the contract reaches the workers
        # and the real code survives into the final answer without any
        # synthesizer call.
        client = ScriptedClient(IMPL_PLAN, REAL_CODE, REAL_CODE)
        result = make_orchestrator(client).run(
            "I want you to build me a React-based dashboard."
        )
        self.assertEqual(client.plan_calls, 1)          # good plan accepted first time
        self.assertEqual(result.error, "")
        self.assertIn("export default function Dashboard", result.final_answer)
        # And the contract reached the workers.
        self.assertTrue(any("DELIVERABLE REQUIREMENT" in p for p in client.worker_prompts))
        self.assertFalse(result.model_synthesis_invoked)

    def test_prose_result_from_a_good_plan_still_fails_the_guard(self) -> None:
        """Plan was fine; the models returned an essay anyway."""
        client = ScriptedClient(IMPL_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        result = make_orchestrator(client).run("Build me a React dashboard.")
        self.assertTrue(result.error)
        self.assertIn("Deliverable contract violation", result.error)
        self.assertIn("Deliverable contract not met", result.final_answer)  # honest notice

    def test_planner_recovers_on_the_corrective_retry(self) -> None:
        client = ScriptedClient(ARCH_PLAN, REAL_CODE, REAL_CODE,
                                second_plan_json=IMPL_PLAN)
        result = make_orchestrator(client).run("Build me a React dashboard.")
        self.assertEqual(client.plan_calls, 2)
        self.assertEqual(result.error, "")              # retry produced a real plan
        self.assertIn("PREVIOUS PLAN WAS REJECTED", client.planner_prompts[1])

    def test_architecture_request_succeeds_with_prose(self) -> None:
        client = ScriptedClient(ARCH_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        result = make_orchestrator(client).run(
            "Design a scalable architecture for a React dashboard. Do not write code."
        )
        self.assertEqual(client.plan_calls, 1)          # prose plan is valid here
        self.assertEqual(result.error, "")
        self.assertNotIn("contract not met", result.final_answer.lower())

    def test_legacy_task_without_contract_still_runs(self) -> None:
        client = ScriptedClient(ARCH_PLAN, "Some answer.", "Some final answer.")
        task = Task(prompt="Compare two databases and recommend one.")
        self.assertIsNone(task.contract)                # constructed the old way
        result = make_orchestrator(client).run(task)
        self.assertEqual(result.error, "")
        self.assertIsNotNone(task.contract)             # resolved at the boundary
        self.assertIs(task.contract.intent, RequestIntent.RESEARCH)

    def test_explicit_contract_is_preserved(self) -> None:
        client = ScriptedClient(ARCH_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        # The caller says "architecture" even though the prompt says "build".
        task = Task(prompt="Build me a dashboard.", contract=ARCHITECTURE)
        result = make_orchestrator(client).run(task)
        self.assertIs(task.contract, ARCHITECTURE)      # untouched
        self.assertEqual(result.error, "")              # prose is valid under it

    def test_recursive_child_inherits_code_requirement(self) -> None:
        """A complex subtask's child task must not revert to explanation-only."""
        from dozen.executor import Executor
        from dozen.router import Router
        from dozen.verifier import Verifier

        captured: list[Task] = []

        def fake_recurse(subtask, sub_task, depth):
            captured.append(sub_task)
            return REAL_CODE

        client = ScriptedClient(IMPL_PLAN, REAL_CODE, REAL_CODE)
        orch = make_orchestrator(client)
        parent = Task(prompt="Build me a React dashboard.",
                      constraints=["Use React 18 only."],
                      contract=IMPLEMENT)
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Big piece", instruction="Build the whole frontend.",
                    complex=True),
        ])
        executor = Executor(
            client=client, pool=orch.pool, router=orch.router,
            verifier=orch.verifier, config=orch.config,
            recurse_fn=fake_recurse, log=lambda _m: None,
        )
        executor.run(parent, plan, depth=0)
        self.assertEqual(len(captured), 1)
        child = captured[0]
        self.assertIsNotNone(child.contract)
        self.assertIs(child.contract, parent.contract)  # same contract object
        self.assertTrue(child.contract.code_required)
        self.assertEqual(child.constraints, ["Use React 18 only."])
        self.assertIsNot(child.constraints, parent.constraints)

    def test_recursive_prose_plan_is_not_rejected_as_the_root_artifact(self) -> None:
        client = ScriptedClient(ARCH_PLAN, ARCHITECTURE_PROSE, ARCHITECTURE_PROSE)
        orch = make_orchestrator(client)
        child = Task(prompt="Research component options.", contract=IMPLEMENT)
        plan = orch.planner.plan(child, depth=1, max_depth=2)
        self.assertEqual(client.plan_calls, 1)
        self.assertEqual(plan.subtasks[0].title, "Describe the architecture")

    def test_executor_accepts_legacy_three_argument_verifier(self) -> None:
        from dozen.executor import Executor
        from dozen.verifier import Verdict

        class LegacyVerifier:
            def verify(self, subtask, output, dependency_outputs):
                return Verdict(True, 1.0, "")

        client = ScriptedClient(IMPL_PLAN, REAL_CODE, REAL_CODE)
        orch = make_orchestrator(client)
        orch.config.verify_outputs = True
        executor = Executor(
            client=client,
            pool=orch.pool,
            router=orch.router,
            verifier=LegacyVerifier(),  # type: ignore[arg-type]
            config=orch.config,
            recurse_fn=lambda *_args: REAL_CODE,
            log=lambda _message: None,
        )
        plan = plan_with(("Write app", "Write the app.py source file."))
        results = executor.run(
            Task(prompt="Build an app.", contract=IMPLEMENT), plan, depth=0
        )
        self.assertEqual(results[0].status.value, "completed")

    def test_compliant_worker_artifact_is_accepted_after_flattening(self) -> None:
        one_step_plan = {
            "analysis": "implement and validate",
            "delegations": [{
                "id": "s1",
                "title": "Implement and test the app",
                "instruction": "Write app.py source code and regression tests.",
                "assigned_model": "alpha",
            }],
            "synthesis_strategy": "return the artifact",
        }
        artifact = json.dumps({
            "summary": "Implemented the application and tests.",
            "key_decisions": ["Keep the entry point deterministic."],
            "artifacts": {
                "app.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass\nclass App:\n    name: str\n\n"
                    "def main():\n    app = App('ready')\n    print(app.name)\n\n"
                    "if __name__ == '__main__':\n    main()"
                )
            },
            "confidence": 0.95,
        })
        client = ScriptedClient(one_step_plan, artifact, "unused")
        result = make_orchestrator(client).run("Build a small Python application.")
        self.assertEqual(client.plan_calls, 1)
        self.assertEqual(result.error, "")
        self.assertIn("def main", result.final_answer)

    def test_direct_answers_receive_the_same_json_cleanup(self) -> None:
        artifact = json.dumps({
            "summary": "A clear explanation.",
            "key_decisions": [],
            "artifacts": {"answer.txt": "React batches state updates by lane."},
            "confidence": 1.0,
        })
        direct_plan = {
            "analysis": "direct",
            "delegations": [],
            "direct_answer": artifact,
            "synthesis_strategy": "",
        }
        client = ScriptedClient(direct_plan, "unused", "unused")
        result = make_orchestrator(client).run("Explain React batching.")
        self.assertEqual(result.error, "")
        self.assertEqual(result.final_answer, "React batches state updates by lane.")

        control_json = json.dumps({
            "analysis": "internal",
            "delegations": [{"id": "s1", "title": "hidden"}],
            "synthesis_strategy": "hidden",
        })
        direct_plan["direct_answer"] = control_json
        client = ScriptedClient(direct_plan, "unused", "unused")
        result = make_orchestrator(client).run("Explain React batching.")
        self.assertNotIn('"delegations"', result.final_answer)
        self.assertIn("internal planning data", result.final_answer)
        self.assertTrue(result.error)
        self.assertEqual(result.summary()["status"], "failed")

    def test_repository_evidence_advisory_reaches_result_and_summary(self) -> None:
        answer = (
            "Changed api/routes.py and tests/test_routes.py. Pagination now "
            "validates cursors. Ran 24 tests; all passed."
        )
        direct_plan = {
            "analysis": "already applied",
            "delegations": [],
            "direct_answer": answer,
            "synthesis_strategy": "",
        }
        client = ScriptedClient(direct_plan, "unused", "unused")
        result = make_orchestrator(client).run("Add pagination to this endpoint.")
        self.assertEqual(result.error, "")
        self.assertTrue(result.warnings)
        self.assertEqual(result.summary()["warnings"], result.warnings)


class TestTerminalResultConsistency(unittest.TestCase):
    def test_contract_failure_is_not_emitted_as_done(self) -> None:
        from dozen.models import OrchestrationResult
        from webllm.server import _terminal_result_event

        result = OrchestrationResult(
            task_id="task_1",
            final_answer="contract notice",
            error="Deliverable contract violation: no code",
        )
        event = _terminal_result_event(
            result, cancelled=False, conversation_id="conversation_1"
        )
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["result"]["summary"]["status"], "failed")
        self.assertEqual(event["result"]["error"], result.error)

    def test_cancelled_and_success_statuses_agree(self) -> None:
        from dozen.models import OrchestrationResult
        from webllm.server import _terminal_result_event

        cancelled = OrchestrationResult(
            task_id="task_1", final_answer="", error="Cancelled: stopped"
        )
        event = _terminal_result_event(
            cancelled, cancelled=True, conversation_id="conversation_1"
        )
        self.assertEqual(event["status"], "cancelled")
        self.assertEqual(event["result"]["summary"]["status"], "cancelled")

        success = OrchestrationResult(task_id="task_2", final_answer="done")
        event = _terminal_result_event(
            success, cancelled=False, conversation_id="conversation_1"
        )
        self.assertEqual(event["status"], "done")
        self.assertEqual(event["result"]["summary"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
