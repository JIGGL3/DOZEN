"""Shared Phase 4B fixtures: the React-dashboard reference decomposition.

Provides the conceptual decomposition the phase specification requires to be
valid, both as directly constructed dataclasses (policy / mapping / DAG /
scope tests) and as planner JSON payloads (planner-integration tests with
fake clients). No real provider is ever contacted.
"""

from __future__ import annotations

import copy
import json

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse
from dozen.artifacts import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactSpec,
    ArtifactValidation,
    ArtifactWorkPlan,
    ValidationKind,
    WorkPackage,
    WorkPackageKind,
)
from dozen.models import Plan, SubTask
from dozen.planner import Planner


def artifact(path: str, **kwargs) -> ArtifactSpec:
    return ArtifactSpec(id=kwargs.pop("id", path), path=path, **kwargs)


def manifest_of(artifacts, validations=(), **kwargs) -> ArtifactManifest:
    return ArtifactManifest(
        id=kwargs.pop("id", "m1"),
        title=kwargs.pop("title", "deliverable"),
        artifacts=tuple(artifacts),
        validations=tuple(validations),
        **kwargs,
    )


def work_plan_of(manifest, packages, subtask_map=(), **kwargs) -> ArtifactWorkPlan:
    return ArtifactWorkPlan(
        id=kwargs.pop("id", "wp1"),
        manifest=manifest,
        packages=tuple(packages),
        subtask_map=tuple(subtask_map),
        **kwargs,
    )


def plan_of(subtask_edges, work_plan=None) -> Plan:
    """Build a Plan whose SubTasks carry FIXED ids: [(id, [deps]), ...]."""
    subtasks = [
        SubTask(title=sid, instruction=f"Do the {sid} work.",
                depends_on=list(deps), id=sid)
        for sid, deps in subtask_edges
    ]
    return Plan(analysis="a", subtasks=subtasks, synthesis_strategy="s",
                artifact_plan=work_plan)


# --------------------------------------------------------------------------- #
# The required React-dashboard example
# --------------------------------------------------------------------------- #
FOUNDATION_PATHS = ("package.json", "tsconfig.json", "src/main.tsx")
SHELL_PATHS = (
    "src/app/App.tsx",
    "src/app/router.tsx",
    "src/components/layout/DashboardLayout.tsx",
)
SHARED_UI_PATHS = ("src/components/ui/Card.tsx",)
FEATURE_PATHS = ("src/features/dashboard/DashboardPage.tsx",)
TEST_PATHS = ("tests/dashboard.test.tsx",)
RESULT_PATHS = ("build-result", "test-result")

# package id -> (subtask id, owned paths, package depends_on)
REACT_PACKAGES = {
    "project-foundation": ("foundation", FOUNDATION_PATHS, ()),
    "application-shell": ("shell", SHELL_PATHS, ("project-foundation",)),
    "shared-ui": ("shared-ui", SHARED_UI_PATHS, ("project-foundation",)),
    "dashboard-feature": (
        "dashboard", FEATURE_PATHS, ("application-shell", "shared-ui"),
    ),
    "dashboard-tests": ("tests", TEST_PATHS, ("dashboard-feature",)),
    "integration-validation": (
        "validation", RESULT_PATHS,
        ("project-foundation", "application-shell", "shared-ui",
         "dashboard-feature", "dashboard-tests"),
    ),
}
REACT_SUBTASK_EDGES = [
    ("foundation", []),
    ("shell", ["foundation"]),
    ("shared-ui", ["foundation"]),
    ("dashboard", ["shell", "shared-ui"]),
    ("tests", ["dashboard"]),
    ("validation", ["foundation", "shell", "shared-ui", "dashboard", "tests"]),
]


def react_manifest() -> ArtifactManifest:
    specs = [
        artifact("package.json", kind=ArtifactKind.CONFIG_FILE),
        artifact("tsconfig.json", kind=ArtifactKind.CONFIG_FILE),
        artifact("src/main.tsx", kind=ArtifactKind.SOURCE_FILE),
        artifact("src/app/App.tsx", kind=ArtifactKind.SOURCE_FILE),
        artifact("src/app/router.tsx", kind=ArtifactKind.SOURCE_FILE),
        artifact("src/components/layout/DashboardLayout.tsx",
                 kind=ArtifactKind.SOURCE_FILE),
        artifact("src/components/ui/Card.tsx", kind=ArtifactKind.SOURCE_FILE),
        artifact("src/features/dashboard/DashboardPage.tsx",
                 kind=ArtifactKind.SOURCE_FILE),
        artifact("tests/dashboard.test.tsx", kind=ArtifactKind.TEST_FILE),
        artifact("build-result", kind=ArtifactKind.BUILD_RESULT),
        artifact("test-result", kind=ArtifactKind.TEST_RESULT),
    ]
    validations = [
        ArtifactValidation(
            id="v-build", kind=ValidationKind.BUILD,
            target_artifact_ids=("src/app/App.tsx",),
            produces_artifact_id="build-result",
            success_criterion="the project builds cleanly",
        ),
        ArtifactValidation(
            id="v-test", kind=ValidationKind.UNIT_TESTS,
            target_artifact_ids=("tests/dashboard.test.tsx",),
            produces_artifact_id="test-result",
            success_criterion="the dashboard test suite passes",
        ),
    ]
    return manifest_of(specs, validations, title="React dashboard")


def react_packages() -> list[WorkPackage]:
    packages = []
    for pid, (_subtask, owned, deps) in REACT_PACKAGES.items():
        extra = {}
        if pid == "application-shell":
            extra["input_artifact_ids"] = ("package.json", "tsconfig.json")
        if pid == "dashboard-feature":
            extra["input_artifact_ids"] = (
                "src/app/App.tsx", "src/components/ui/Card.tsx",
            )
        if pid == "integration-validation":
            extra["kind"] = WorkPackageKind.VALIDATION
            extra["validation_ids"] = ("v-build", "v-test")
            extra["completion_criteria"] = ("build and tests pass",)
        packages.append(WorkPackage(
            id=pid, title=pid.replace("-", " "),
            objective=f"Produce the {pid} artifacts.",
            owns=owned, depends_on=deps, **extra,
        ))
    return packages


def react_work_plan(subtask_map=None) -> ArtifactWorkPlan:
    if subtask_map is None:
        subtask_map = tuple(
            (pid, subtask) for pid, (subtask, _o, _d) in REACT_PACKAGES.items()
        )
    return work_plan_of(react_manifest(), react_packages(), subtask_map)


def react_plan(subtask_edges=None, work_plan=None) -> Plan:
    return plan_of(
        subtask_edges if subtask_edges is not None else REACT_SUBTASK_EDGES,
        work_plan if work_plan is not None else react_work_plan(),
    )


# --------------------------------------------------------------------------- #
# Planner JSON payloads (fake clients only)
# --------------------------------------------------------------------------- #
REACT_DELEGATIONS = [
    {"id": "s1", "title": "Project foundation",
     "instruction": "Write the complete project configuration and entry files.",
     "assigned_model": "alpha"},
    {"id": "s2", "title": "Application shell",
     "instruction": "Write the complete application shell source files.",
     "assigned_model": "alpha", "depends_on": ["s1"]},
    {"id": "s3", "title": "Shared UI",
     "instruction": "Write the complete shared UI component source files.",
     "assigned_model": "alpha", "depends_on": ["s1"]},
    {"id": "s4", "title": "Dashboard feature",
     "instruction": "Write the complete dashboard page source files.",
     "assigned_model": "alpha", "depends_on": ["s2", "s3"]},
    {"id": "s5", "title": "Tests",
     "instruction": "Write complete unit tests for the dashboard.",
     "assigned_model": "alpha", "depends_on": ["s4"]},
    {"id": "s6", "title": "Integration validation",
     "instruction": "Write a complete validation summary of build and tests.",
     "assigned_model": "alpha",
     "depends_on": ["s1", "s2", "s3", "s4", "s5"]},
]

_PACKAGE_SUBTASK_JSON = {
    "project-foundation": "s1",
    "application-shell": "s2",
    "shared-ui": "s3",
    "dashboard-feature": "s4",
    "dashboard-tests": "s5",
    "integration-validation": "s6",
}

REACT_ARTIFACT_BLOCK = {
    "title": "React dashboard",
    "artifacts": (
        [{"path": p, "kind": "config_file"} for p in FOUNDATION_PATHS[:2]]
        + [{"path": FOUNDATION_PATHS[2], "kind": "source_file"}]
        + [{"path": p, "kind": "source_file"}
           for p in SHELL_PATHS + SHARED_UI_PATHS + FEATURE_PATHS]
        + [{"path": TEST_PATHS[0], "kind": "test_file"}]
        + [{"path": "build-result", "kind": "build_result"},
           {"path": "test-result", "kind": "test_result"}]
    ),
    "packages": [
        {
            "id": pid,
            "title": pid,
            "objective": f"Produce the {pid} artifacts.",
            "kind": "validation" if pid == "integration-validation"
            else "implementation",
            "owns": list(owned),
            "depends_on": list(deps),
            "subtask_id": _PACKAGE_SUBTASK_JSON[pid],
            **(
                {"validations": ["v-build", "v-test"],
                 "completion": ["build and tests pass"]}
                if pid == "integration-validation" else {}
            ),
        }
        for pid, (_subtask, owned, deps) in REACT_PACKAGES.items()
    ],
    "validations": [
        {"id": "v-build", "kind": "build", "targets": ["src/app/App.tsx"],
         "produces": "build-result", "criterion": "the project builds cleanly"},
        {"id": "v-test", "kind": "unit_tests",
         "targets": ["tests/dashboard.test.tsx"], "produces": "test-result",
         "criterion": "the dashboard test suite passes"},
    ],
}


def react_plan_json(artifact_block=REACT_ARTIFACT_BLOCK) -> dict:
    data = {
        "analysis": "decompose into bounded artifact packages",
        "delegations": copy.deepcopy(REACT_DELEGATIONS),
        "synthesis_strategy": "merge the package outputs and repair defects",
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
        return LLMResponse(text=json.dumps(payload), provider=provider,
                           model=model)


def make_planner(client: LLMClient) -> Planner:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 1.0, "coding": 1.0}, tier=4)
    return Planner(client, agent, AgentPool([agent]))
