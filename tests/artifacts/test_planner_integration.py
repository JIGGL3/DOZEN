"""Phase 4A — planner artifact-plan integration, using fake clients only.

Covers: prompt gating (which requests are invited to declare an artifact plan),
deterministic parsing/validation of the planner's optional ``artifact_plan``
block, the single bounded corrective re-plan, recursion safety, and backward
compatibility of legacy ``Plan``/``Task``/``SubTask`` construction.
"""

from __future__ import annotations

import copy
import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator, Task
from dozen.artifacts import (
    ArtifactContractError,
    ArtifactKind,
    ValidationKind,
    WorkPackageKind,
    parse_planner_artifact_plan,
)
from dozen.config import OrchestratorConfig
from dozen.intent import resolve_contract
from dozen.models import Plan, SubTask
from dozen.planner import PlanError, Planner
from dozen.prompts import build_planner_messages

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")
ARCHITECTURE = resolve_contract("Design the architecture. Do not write code.")
EXPLAIN = resolve_contract("Explain how React hooks work.")

REAL_CODE = """```jsx
export default function Dashboard() {
  const [data, setData] = useState([]);
  return <div>{data.length}</div>;
}
```
"""

# A contract-satisfying delegation set (delivery + validation subtasks).
DELEGATIONS = [
    {"id": "s1", "title": "Write the dashboard components",
     "instruction": "Write the complete React component source files.",
     "assigned_model": "alpha"},
    {"id": "s2", "title": "Add tests",
     "instruction": "Write unit tests for the components.",
     "assigned_model": "alpha", "depends_on": ["s1"]},
]

VALID_ARTIFACT_BLOCK = {
    "title": "React dashboard",
    "artifacts": [
        {"id": "config", "path": "package.json", "kind": "config_file"},
        {"id": "app", "path": "src/App.tsx", "kind": "source_file"},
        {"id": "tests", "path": "tests/app.test.tsx", "kind": "test_file"},
    ],
    "packages": [
        {"id": "foundation", "title": "Project foundation",
         "kind": "implementation", "owns": ["config"], "subtask_id": "s1"},
        {"id": "shell", "title": "Application shell",
         "kind": "implementation", "owns": ["app"],
         "depends_on": ["foundation"], "subtask_id": "s1"},
        {"id": "testing", "title": "Tests", "kind": "validation",
         "owns": ["tests"], "depends_on": ["shell"], "subtask_id": "s2"},
    ],
    "validations": [
        {"id": "v1", "kind": "unit_tests", "targets": ["tests"],
         "criterion": "the dashboard test suite passes"},
    ],
}


def plan_json(artifact_block=None) -> dict:
    data = {
        "analysis": "build it",
        "delegations": copy.deepcopy(DELEGATIONS),
        "synthesis_strategy": "merge code then tests; repair defects",
    }
    if artifact_block is not None:
        data["artifact_plan"] = copy.deepcopy(artifact_block)
    return data


class QueuedPlannerClient(LLMClient):
    """Returns queued planner JSON payloads and records the prompts."""

    def __init__(self, *payloads: dict) -> None:
        super().__init__(mock=True)
        self.payloads = list(payloads)
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        self.prompts.append(messages[1].content)
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        return LLMResponse(text=json.dumps(payload), provider=provider, model=model)


def make_planner(client: LLMClient) -> Planner:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 1.0, "coding": 1.0}, tier=4)
    return Planner(client, agent, AgentPool([agent]))


# --------------------------------------------------------------------------- #
class TestPromptGating(unittest.TestCase):
    def test_implement_request_is_invited_to_declare_artifacts(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        messages = build_planner_messages(task, 0, 2, "alpha (coding)")
        combined = "\n".join(message.content for message in messages)
        self.assertIn("ARTIFACT PLAN", combined)
        self.assertIn("NEVER include file contents", combined)

    def test_the_planner_is_never_asked_for_file_contents(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        for message in build_planner_messages(task, 0, 2, "alpha"):
            self.assertNotRegex(
                message.content,
                r'artifact_plan[^{]*\{[^}]*"(content|code|body)"',
            )
        combined = "\n".join(
            message.content
            for message in build_planner_messages(task, 0, 2, "alpha")
        )
        self.assertIn("paths, names and short descriptions only", combined)

    def test_prose_contracts_do_not_get_the_artifact_block(self) -> None:
        for contract in (ARCHITECTURE, EXPLAIN):
            with self.subTest(intent=contract.intent):
                task = Task(prompt="whatever", contract=contract)
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertNotIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

    def test_required_activation_policy_matrix(self) -> None:
        cases = {
            "Build me a React dashboard.": True,
            "Add pagination to this endpoint.": True,
            "Fix this crash and add a regression test.": True,
            "Create a README.": True,
            "Write an architecture document.": True,
            "Produce a JSON configuration file.": True,
            "Generate a CSV report.": True,
            "Create a database migration.": True,
            "Write a SQL query.": True,
            "Explain React reconciliation.": False,
            "Review this repository without modifying it.": False,
            "Design the architecture, but do not implement it.": False,
            "Create a SADD.": True,
            "Create a SADD generator.": True,
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                task = Task(prompt=prompt, contract=resolve_contract(prompt))
                messages = build_planner_messages(task, 0, 2, "alpha")
                combined = "\n".join(message.content for message in messages)
                self.assertEqual("ARTIFACT PLAN" in combined, expected)

    def test_activation_requires_positive_action_or_explicit_output_slot(self) -> None:
        for prompt in (
            "Do not create a README.",
            "Do not ever create a README.",
            "Under no circumstances create a README.",
            "Do not create or write a README.",
            "Do not generate a CSV report.",
            "No README.",
            "README.md",
            "The data.csv file is attached.",
            "Do not revise the README.",
            "Never draft a README.",
            "Do not document the API.",
            "Explain how to create a README.",
            "Should we create a README?",
            "Do I need to create a README?",
            "Review whether we should generate a CSV report.",
            "Discuss how one might write an architecture document.",
            "Teach me how to produce a patch.",
            "Show me how to create a README.",
            "Tell me how to write a README.",
            "Walk me through how to create a README.",
            "How do I create a README?",
            "How can I write a README?",
            "Could we create a README?",
            "Would it make sense to create a README?",
            "What if we create a README?",
            "Do not, under any circumstances, create a README.",
            "Never, ever create a README.",
            "There is no need to create a README.",
            "I would rather not create a README.",
            "Create no README.",
            "Review the attached research report.",
            "Explain how to write a research report.",
            "Review whether to create a README.",
            "Assess whether to write a research report.",
            "Provide instructions to create a README.",
            "Can I create a README?",
            "May I create a README?",
            "Does this tool create a README?",
            "You must not create a README.",
            "Can I build a React dashboard?",
            "You must not build a React dashboard.",
            "I might build a dashboard.",
            "Would they add pagination?",
            "Remember not to create a README.",
            "Anything but create a README.",
            "Tell me whether the team should create a README.",
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                task = Task(prompt=prompt, contract=contract)
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertNotIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

        for prompt, desired_output in (
            ("Research competitors.", "results.csv"),
            ("Review this repository without modifying it.", "audit-report.md"),
            ("Explain the algorithm.", "README.md"),
            ("Compare two databases and recommend one.", "CSV file"),
            ("Research competitors.", "research report"),
            ("Research competitors.", "spreadsheet"),
            ("Research competitors.", "not a README but a research report"),
            ("Research competitors.", "Kubernetes YAML manifest"),
            ("Explain the API.", "OpenAPI spec"),
            ("Review dependencies.", "requirements.txt"),
            ("Research database options.", "SQL script"),
        ):
            with self.subTest(prompt=prompt, desired_output=desired_output):
                contract = resolve_contract(
                    prompt, desired_output=desired_output
                )
                task = Task(
                    prompt=prompt,
                    desired_output=desired_output,
                    contract=contract,
                )
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

        for prompt in (
            "Research competitors and export results to report.csv.",
            "Review repository and save findings to audit-report.md.",
            "Explain the algorithm and format the answer as README.md.",
            "Compare databases and give me a CSV file.",
            "Research competitors and supply results.csv.",
            "Do not create files; research competitors and return results.csv.",
            "Do not create files; review repo and output audit-report.md.",
            "Research competitors and send me results.csv.",
            "Review the repo and put findings in audit-report.md.",
            "Research competitors and present findings in report.csv.",
            "Review the repository and reply with audit-report.md.",
            "Research competitors and export, as the final deliverable, results.csv.",
            "Deliver a research report.",
            "Provide a document summarizing findings.",
            "Send me an Excel workbook.",
            "Could you create a README?",
            "Can you create a README?",
            "Could you build a React dashboard?",
            "Would you send results.csv?",
            "Produce a Kubernetes YAML manifest.",
            "Write an OpenAPI specification.",
            "Create requirements.txt.",
            "Generate a SQL file.",
            "Write a SQL script.",
            "The React dashboard crashes on startup.",
        ):
            with self.subTest(positive_action=prompt):
                contract = resolve_contract(prompt)
                task = Task(prompt=prompt, contract=contract)
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

        for desired_output in (
            "No README",
            "Plain text, no README.",
            "not a CSV file",
            "A prose answer, not a CSV file.",
            "Inline findings only; do not provide audit-report.md.",
            "anything but README.md",
            "Plain text instead of README.md",
            "Plain text rather than README.md",
            "README.md excluded",
            "Neither README.md nor report.csv",
            "No spreadsheet",
            "research report not required",
            "README.md is not desired",
            "README.md should be avoided",
            "README.md must not be produced",
            "README.md is prohibited",
            "README.md is disallowed",
            "anything other than README.md",
        ):
            with self.subTest(negative_desired_output=desired_output):
                prompt = "Research competitors."
                contract = resolve_contract(
                    prompt, desired_output=desired_output
                )
                task = Task(
                    prompt=prompt,
                    desired_output=desired_output,
                    contract=contract,
                )
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertNotIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

        for prompt, constraints in (
            (
                "Research competitors",
                ["Do not create source files", "Return results.csv"],
            ),
            (
                "Explain how to write documentation",
                ["Return README.md"],
            ),
            (
                "Can I build a dashboard?",
                ["Build the dashboard and return app.py"],
            ),
        ):
            with self.subTest(prompt=prompt, constraints=constraints):
                contract = resolve_contract(prompt, constraints=constraints)
                task = Task(
                    prompt=prompt,
                    constraints=constraints,
                    contract=contract,
                )
                messages = build_planner_messages(task, 0, 2, "alpha")
                self.assertIn(
                    "ARTIFACT PLAN",
                    "\n".join(message.content for message in messages),
                )

        prompt = "Produce a patch for this bug, but do not modify the repository."
        task = Task(prompt=prompt, contract=resolve_contract(prompt))
        messages = build_planner_messages(task, 0, 2, "alpha")
        self.assertIn(
            "ARTIFACT PLAN",
            "\n".join(message.content for message in messages),
        )

    def test_active_system_contract_allows_the_optional_top_level_key(self) -> None:
        active = build_planner_messages(
            Task(prompt="Build it", contract=IMPLEMENT), 0, 2, "alpha"
        )
        inactive = build_planner_messages(Task(prompt="legacy"), 0, 2, "alpha")
        active_combined = "\n".join(message.content for message in active)
        inactive_combined = "\n".join(message.content for message in inactive)
        self.assertIn("sole authorized exception", active[0].content)
        self.assertNotIn("sole authorized exception", inactive_combined)
        self.assertEqual(active_combined.count("ARTIFACT PLAN"), 1)
        self.assertEqual(active_combined.count('"title": "short deliverable name"'), 1)
        self.assertNotIn("ARTIFACT PLAN", active[1].content)

    def test_parser_limits_are_explicit_in_the_prompt(self) -> None:
        combined = "\n".join(
            message.content
            for message in build_planner_messages(
                Task(prompt="Build it", contract=IMPLEMENT), 0, 2, "alpha"
            )
        )
        for limit in ("80 artifacts", "24 packages", "40 validations",
                      "240 characters", "300 characters"):
            self.assertIn(limit, combined)

    def test_recursive_depth_does_not_get_the_artifact_block(self) -> None:
        task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
        messages = build_planner_messages(task, 1, 2, "alpha")
        self.assertNotIn(
            "ARTIFACT PLAN",
            "\n".join(message.content for message in messages),
        )

    def test_legacy_task_without_contract_prompt_is_unchanged(self) -> None:
        task = Task(prompt="do the thing")
        messages = build_planner_messages(task, 0, 2, "alpha")
        combined = "\n".join(message.content for message in messages)
        self.assertNotIn("ARTIFACT PLAN", combined)
        self.assertNotIn("artifact_plan", combined)

    def test_artifact_rule_keeps_planner_prompt_bounded(self) -> None:
        from dozen.prompts import PLANNER_ARTIFACT_PLAN_RULE
        # Bound raised once for the deliberate Phase 4F output-aware sizing
        # rule; the guard still catches accidental unbounded growth.
        self.assertLess(len(PLANNER_ARTIFACT_PLAN_RULE), 3000)


# --------------------------------------------------------------------------- #
class TestPlannerParsing(unittest.TestCase):
    def test_valid_artifact_plan_parses_and_attaches(self) -> None:
        client = QueuedPlannerClient(plan_json(VALID_ARTIFACT_BLOCK))
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(plan.artifact_plan)
        wp = plan.artifact_plan
        self.assertEqual(wp.manifest.title, "React dashboard")
        self.assertEqual(len(wp.manifest.artifacts), 3)
        self.assertEqual(wp.owner_of("app"), "shell")
        self.assertEqual(wp.packages[2].kind, WorkPackageKind.VALIDATION)
        # subtask_id references were remapped to INTERNAL SubTask ids.
        internal_ids = {s.id for s in plan.subtasks}
        for _, subtask_id in wp.subtask_map:
            self.assertIn(subtask_id, internal_ids)

    def test_plan_without_artifact_block_is_valid_for_implement(self) -> None:
        client = QueuedPlannerClient(plan_json())
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan.artifact_plan)

    def test_non_artifact_architecture_plan_remains_valid(self) -> None:
        payload = {
            "analysis": "design",
            "delegations": [
                {"id": "s1", "title": "Describe the architecture",
                 "instruction": "Define components and data flow.",
                 "assigned_model": "alpha"},
            ],
            "synthesis_strategy": "present the design",
        }
        client = QueuedPlannerClient(payload)
        task = Task(prompt="Design the architecture.", contract=ARCHITECTURE)
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan.artifact_plan)

    def test_activation_policy_also_governs_public_planner_parsing(self) -> None:
        for prompt, desired_output in (
            ("Under no circumstances create a README.", ""),
            ("Research competitors.", "No README"),
            ("Research competitors.", "README.md excluded"),
            ("Research competitors.", "Neither README.md nor report.csv"),
            ("Can I build a React dashboard?", ""),
            ("You must not build a React dashboard.", ""),
        ):
            with self.subTest(inactive=(prompt, desired_output)):
                contract = resolve_contract(
                    prompt, desired_output=desired_output
                )
                task = Task(
                    prompt=prompt,
                    desired_output=desired_output,
                    contract=contract,
                )
                client = QueuedPlannerClient(plan_json(VALID_ARTIFACT_BLOCK))
                plan = make_planner(client).plan(task, 0, 2)
                self.assertEqual(client.calls, 1)
                self.assertIsNone(plan.artifact_plan)

        for prompt, desired_output in (
            ("Research competitors and return results.csv.", ""),
            ("Review this repository without modifying it.", "audit-report.md"),
            ("Deliver a research report.", ""),
            ("Research competitors.", "spreadsheet"),
            ("Produce a Kubernetes YAML manifest.", ""),
            ("Explain the API.", "OpenAPI spec"),
        ):
            with self.subTest(active=(prompt, desired_output)):
                contract = resolve_contract(
                    prompt, desired_output=desired_output
                )
                task = Task(
                    prompt=prompt,
                    desired_output=desired_output,
                    contract=contract,
                )
                client = QueuedPlannerClient(plan_json(VALID_ARTIFACT_BLOCK))
                plan = make_planner(client).plan(task, 0, 2)
                self.assertEqual(client.calls, 1)
                self.assertIsNotNone(plan.artifact_plan)

    def test_recursive_plan_drops_artifact_block_and_keeps_parent_safe(self) -> None:
        client = QueuedPlannerClient(plan_json(VALID_ARTIFACT_BLOCK))
        parent_contract = IMPLEMENT
        child = Task(prompt="Build the frontend piece.", contract=parent_contract)
        plan = make_planner(client).plan(child, 1, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan.artifact_plan)         # never parsed at depth > 0
        self.assertIs(child.contract, parent_contract)  # parent contract untouched

    def test_legacy_task_without_contract_ignores_artifact_block(self) -> None:
        client = QueuedPlannerClient(plan_json(VALID_ARTIFACT_BLOCK))
        plan = make_planner(client).plan(Task(prompt="do the thing"), 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan.artifact_plan)

    def test_declared_falsy_artifact_blocks_receive_one_correction(self) -> None:
        for bad_block in ({}, [], "", 0, False):
            with self.subTest(block=bad_block):
                first = plan_json()
                first["artifact_plan"] = bad_block
                client = QueuedPlannerClient(first, plan_json(VALID_ARTIFACT_BLOCK))
                task = Task(prompt="Build a dashboard.", contract=IMPLEMENT)
                plan = make_planner(client).plan(task, 0, 2)
                self.assertEqual(client.calls, 2)
                self.assertIsNotNone(plan.artifact_plan)

    def test_direct_answer_still_parses_its_declared_artifact_plan(self) -> None:
        block = {
            "title": "README",
            "artifacts": [
                {"id": "readme", "path": "README.md", "kind": "document"},
            ],
            "packages": [
                {"id": "writing", "owns": ["readme"]},
            ],
        }
        payload = {
            "analysis": "write directly",
            "direct_answer": "# Project\n\nUsage details.",
            "delegations": [],
            "synthesis_strategy": "return the document",
            "artifact_plan": block,
        }
        client = QueuedPlannerClient(payload)
        prompt = "Create a README."
        plan = make_planner(client).plan(
            Task(prompt=prompt, contract=resolve_contract(prompt)), 0, 2
        )
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(plan.artifact_plan)

    def test_duplicate_planner_subtask_ids_fail_instead_of_remapping(self) -> None:
        payload = plan_json(VALID_ARTIFACT_BLOCK)
        payload["delegations"][1]["id"] = "s1"
        client = QueuedPlannerClient(payload)
        with self.assertRaises(PlanError) as ctx:
            make_planner(client).plan(
                Task(prompt="Build a dashboard.", contract=IMPLEMENT), 0, 2
            )
        self.assertEqual(client.calls, 1)
        self.assertIn("duplicate subtask id", str(ctx.exception))


def invalid_block(**overrides) -> dict:
    block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
    block.update(overrides)
    return block


class TestPlannerValidationFailures(unittest.TestCase):
    """Each structural defect fails, is retried ONCE, then fails the plan."""

    def assert_one_corrective_replan(self, bad_block: dict,
                                     expected_fragment: str) -> None:
        # First plan invalid, second plan clean -> recovered in exactly 2 calls.
        client = QueuedPlannerClient(plan_json(bad_block),
                                     plan_json(VALID_ARTIFACT_BLOCK))
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIsNotNone(plan.artifact_plan)
        self.assertIn("PREVIOUS PLAN WAS REJECTED", client.prompts[1])
        self.assertIn(expected_fragment, client.prompts[1])

        # Both plans invalid -> deterministic PlanError, still only 2 calls.
        client = QueuedPlannerClient(plan_json(bad_block), plan_json(bad_block))
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        with self.assertRaises(PlanError) as ctx:
            make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("invalid artifact plan", str(ctx.exception))

    def test_duplicate_artifact_path_fails(self) -> None:
        block = invalid_block()
        block["artifacts"].append(
            {"id": "dup", "path": "src\\App.tsx", "kind": "source_file",
             "external": True})
        self.assert_one_corrective_replan(block, "duplicate artifact path")

    def test_duplicate_ownership_fails(self) -> None:
        block = invalid_block()
        block["packages"][1]["owns"] = ["app", "config"]
        self.assert_one_corrective_replan(block, "owned by more than one package")

    def test_unknown_package_dependency_fails(self) -> None:
        block = invalid_block()
        block["packages"][1]["depends_on"] = ["ghost"]
        self.assert_one_corrective_replan(block, "unknown package")

    def test_package_cycle_fails(self) -> None:
        block = invalid_block()
        block["packages"][0]["depends_on"] = ["testing"]
        self.assert_one_corrective_replan(block, "cycle")

    @staticmethod
    def drop_shell_package(block: dict) -> None:
        block["packages"] = [p for p in block["packages"] if p["id"] != "shell"]
        for package in block["packages"]:
            package["depends_on"] = [
                d for d in package.get("depends_on", []) if d != "shell"
            ]

    def test_required_artifact_without_producer_fails(self) -> None:
        block = invalid_block()
        self.drop_shell_package(block)
        self.assert_one_corrective_replan(block, "no owning work package")

    def test_external_required_artifact_needs_no_producer(self) -> None:
        block = invalid_block()
        self.drop_shell_package(block)
        for artifact in block["artifacts"]:
            if artifact["id"] == "app":
                artifact["external"] = True
        client = QueuedPlannerClient(plan_json(block))
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        plan = make_planner(client).plan(task, 0, 2)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(plan.artifact_plan)

    def test_unknown_subtask_reference_fails(self) -> None:
        block = invalid_block()
        block["packages"][0]["subtask_id"] = "s99"
        self.assert_one_corrective_replan(block, "unknown subtask")

    def test_file_contents_in_the_block_fail(self) -> None:
        block = invalid_block()
        block["artifacts"][1]["content"] = "export default function App() {}"
        self.assert_one_corrective_replan(block, "must not carry file contents")

    def test_unsafe_path_fails(self) -> None:
        block = invalid_block()
        block["artifacts"][0]["path"] = "../../etc/passwd"
        self.assert_one_corrective_replan(block, "parent traversal")


class TestParseFunctionDirectly(unittest.TestCase):
    def test_unknown_enum_values_do_not_fail_the_block(self) -> None:
        block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
        block["artifacts"][0]["kind"] = "mystery"
        block["packages"][0]["kind"] = "mystery"
        block["validations"][0]["kind"] = "mystery"
        wp = parse_planner_artifact_plan(block)
        self.assertEqual(wp.manifest.artifacts[0].kind, ArtifactKind.OTHER)
        self.assertEqual(wp.packages[0].kind, WorkPackageKind.COORDINATION)
        self.assertEqual(wp.manifest.validations[0].kind, ValidationKind.CUSTOM)

    def test_missing_or_null_package_kind_keeps_implementation_default(self) -> None:
        for marker in ("missing", None):
            with self.subTest(marker=marker):
                block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
                if marker == "missing":
                    block["packages"][0].pop("kind")
                else:
                    block["packages"][0]["kind"] = None
                wp = parse_planner_artifact_plan(block)
                self.assertEqual(
                    wp.packages[0].kind,
                    WorkPackageKind.IMPLEMENTATION,
                )

    def test_missing_artifacts_list_fails(self) -> None:
        with self.assertRaises(ArtifactContractError):
            parse_planner_artifact_plan({"title": "x", "packages": []})

    def test_non_object_block_fails(self) -> None:
        with self.assertRaises(ArtifactContractError):
            parse_planner_artifact_plan("not an object")

    def test_bounded_counts_are_enforced(self) -> None:
        block = {
            "title": "big",
            "artifacts": [{"id": f"a{i}", "path": f"f{i}.py"}
                          for i in range(200)],
        }
        with self.assertRaises(ArtifactContractError) as ctx:
            parse_planner_artifact_plan(block)
        self.assertIn("limit", str(ctx.exception))

    def test_nested_reference_counts_are_bounded(self) -> None:
        cases = (
            ("artifacts", "depends_on", ["config"] * 81),
            ("packages", "inputs", ["config"] * 81),
            ("packages", "depends_on", ["foundation"] * 25),
            ("packages", "validations", ["v1"] * 41),
        )
        for collection, field, oversized in cases:
            with self.subTest(collection=collection, field=field):
                block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
                block[collection][0][field] = oversized
                with self.assertRaises(ArtifactContractError) as ctx:
                    parse_planner_artifact_plan(block)
                self.assertIn("limit", str(ctx.exception))

    def test_scalar_nested_reference_arrays_are_rejected(self) -> None:
        for collection, field in (
            ("artifacts", "depends_on"),
            ("validations", "targets"),
            ("packages", "owns"),
            ("packages", "inputs"),
            ("packages", "validations"),
        ):
            with self.subTest(collection=collection, field=field):
                block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
                block[collection][0][field] = 42
                with self.assertRaises(ArtifactContractError):
                    parse_planner_artifact_plan(block)

    def test_non_string_subtask_reference_is_rejected(self) -> None:
        for value in ({"bad": "id"}, 42):
            with self.subTest(value=value):
                block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
                block["packages"][0]["subtask_id"] = value
                with self.assertRaises(ArtifactContractError):
                    parse_planner_artifact_plan(block)

    def test_artifact_id_defaults_to_canonical_path(self) -> None:
        block = {
            "title": "x",
            "artifacts": [{"path": "src\\main.py", "external": True}],
        }
        wp = parse_planner_artifact_plan(block)
        self.assertEqual(wp.manifest.artifacts[0].id, "src/main.py")

    def test_subtask_map_kept_verbatim_without_id_map(self) -> None:
        wp = parse_planner_artifact_plan(copy.deepcopy(VALID_ARTIFACT_BLOCK))
        self.assertEqual(wp.subtask_for("foundation"), "s1")

    def test_content_key_aliases_and_nested_content_are_rejected(self) -> None:
        for key in ("content", "code", "body", "source", "text",
                    "file_content", "fileContent", "contents"):
            with self.subTest(key=key):
                block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
                block["artifacts"][0][key] = "print('transported file')"
                with self.assertRaises(ArtifactContractError):
                    parse_planner_artifact_plan(block)
        block = copy.deepcopy(VALID_ARTIFACT_BLOCK)
        block["artifacts"][0]["metadata"] = {
            "nested": {"content": "print('transported file')"},
        }
        with self.assertRaises(ArtifactContractError):
            parse_planner_artifact_plan(block)


# --------------------------------------------------------------------------- #
# End-to-end compatibility through the real Orchestrator (fake client only)
# --------------------------------------------------------------------------- #
class ScriptedClient(LLMClient):
    def __init__(self, plan_data: dict, worker_output: str, synth_output: str):
        super().__init__(mock=True)
        self.plan_data = plan_data
        self.worker_output = worker_output
        self.synth_output = synth_output
        self.plan_calls = 0

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        if "MANAGER" in system:
            self.plan_calls += 1
            text = json.dumps(self.plan_data)
        elif "SYNTHESIZER" in system:
            text = self.synth_output
        else:
            text = self.worker_output
        return LLMResponse(text=text, provider=provider, model=model)


def make_orchestrator(client) -> Orchestrator:
    pool = AgentPool([
        AgentSpec(name="alpha", provider="openai", model="gpt",
                  strengths={"reasoning": 0.9, "coding": 0.9}, tier=4),
    ])
    cfg = OrchestratorConfig(max_parallelism=1, max_repair_attempts=0,
                             verify_outputs=False, use_llm_router=False)
    return Orchestrator(client=client, pool=pool, config=cfg)


class TestEndToEndCompatibility(unittest.TestCase):
    def test_artifact_plan_requires_typed_worker_results(self) -> None:
        client = ScriptedClient(plan_json(VALID_ARTIFACT_BLOCK),
                                REAL_CODE, REAL_CODE)
        result = make_orchestrator(client).run(
            "I want you to build me a React-based dashboard with tests."
        )
        self.assertTrue(result.error)
        self.assertEqual(result.artifact_assembly.status.value, "failed")
        self.assertEqual(client.plan_calls, 1)
        self.assertIsNotNone(result.plan.artifact_plan)
        self.assertEqual(len(result.subtask_results), 2)  # subtasks unaffected

    def test_run_without_artifact_plan_is_unchanged(self) -> None:
        client = ScriptedClient(plan_json(), REAL_CODE, REAL_CODE)
        result = make_orchestrator(client).run(
            "I want you to build me a React-based dashboard with tests."
        )
        self.assertEqual(result.error, "")
        self.assertIsNone(result.plan.artifact_plan)

    def test_legacy_plan_construction_remains_valid(self) -> None:
        plan = Plan(analysis="a",
                    subtasks=[SubTask(title="t", instruction="i")],
                    synthesis_strategy="s")
        self.assertIsNone(plan.artifact_plan)
        self.assertFalse(plan.is_direct())
        direct = Plan(analysis="a", subtasks=[], synthesis_strategy="",
                      direct_answer="done")
        self.assertTrue(direct.is_direct())

    def test_result_summary_still_serializes(self) -> None:
        client = ScriptedClient(plan_json(VALID_ARTIFACT_BLOCK),
                                REAL_CODE, REAL_CODE)
        result = make_orchestrator(client).run("Build me a dashboard with tests.")
        json.dumps(result.summary())  # must not raise


if __name__ == "__main__":
    unittest.main()
