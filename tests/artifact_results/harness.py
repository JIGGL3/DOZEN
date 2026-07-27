"""Shared Phase 4C fixtures: the six-package React-dashboard result set.

Reuses the APPROVED Phase 4B decomposition harness (same manifest, same work
packages, same subtask map) and adds what workers are simulated to RETURN. No
real provider is ever contacted and nothing is ever written to disk.
"""

from __future__ import annotations

from typing import Any, Optional

from dozen.artifact_results import (
    assemble_deliverable,
    collect_subtask_artifacts,
)
from dozen.decomposition import derive_execution_scope

from ..artifact_decomposition.harness import react_work_plan  # noqa: F401

# --------------------------------------------------------------------------- #
# What each worker is simulated to produce (artifact id == path in the manifest)
# --------------------------------------------------------------------------- #
CONTENT: dict[str, str] = {
    "package.json": '{\n  "name": "dashboard",\n  "version": "1.0.0"\n}\n',
    "tsconfig.json": '{\n  "compilerOptions": {"jsx": "react-jsx"}\n}\n',
    "src/main.tsx": (
        'import { createRoot } from "react-dom/client";\n'
        'import App from "./app/App";\n\n'
        'createRoot(document.getElementById("root")!).render(<App />);\n'
    ),
    "src/app/App.tsx": (
        'import { Router } from "./router";\n\n'
        "export default function App() {\n  return <Router />;\n}\n"
    ),
    "src/app/router.tsx": (
        "export function Router() {\n  return <DashboardPage />;\n}\n"
    ),
    "src/components/layout/DashboardLayout.tsx": (
        "export function DashboardLayout({ children }) {\n"
        "  return <main>{children}</main>;\n}\n"
    ),
    "src/components/ui/Card.tsx": (
        "export function Card({ title }) {\n  return <section>{title}</section>;\n}\n"
    ),
    "src/features/dashboard/DashboardPage.tsx": (
        'import { Card } from "../../components/ui/Card";\n\n'
        "export function DashboardPage() {\n  return <Card title=\"Revenue\" />;\n}\n"
    ),
    "tests/dashboard.test.tsx": (
        'import { render } from "@testing-library/react";\n\n'
        'test("renders", () => { render(<DashboardPage />); });\n'
    ),
    "build-result": "vite build: 0 errors, 0 warnings. Bundle written in 3.1s.",
    "test-result": "vitest: 4 passed, 0 failed, 0 skipped.",
}

LANGUAGE = {
    "package.json": "json",
    "tsconfig.json": "json",
    "build-result": "",
    "test-result": "",
}

# subtask id -> the artifact ids that subtask's package owns.
SUBTASK_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "foundation": ("package.json", "tsconfig.json", "src/main.tsx"),
    "shell": (
        "src/app/App.tsx",
        "src/app/router.tsx",
        "src/components/layout/DashboardLayout.tsx",
    ),
    "shared-ui": ("src/components/ui/Card.tsx",),
    "dashboard": ("src/features/dashboard/DashboardPage.tsx",),
    "tests": ("tests/dashboard.test.tsx",),
    "validation": ("build-result", "test-result"),
}

ALL_SUBTASKS = tuple(SUBTASK_ARTIFACTS)


# --------------------------------------------------------------------------- #
# Envelope builders
# --------------------------------------------------------------------------- #
def entry(
    artifact_id: str,
    *,
    path: Optional[str] = None,
    content: Optional[str] = None,
    **extra: Any,
) -> dict[str, Any]:
    """One typed produced-artifact entry of the Phase 4C worker envelope."""
    item: dict[str, Any] = {
        "artifact_id": artifact_id,
        "path": artifact_id if path is None else path,
        "content": CONTENT.get(artifact_id, "placeholder\n")
        if content is None else content,
        "language": LANGUAGE.get(artifact_id, "tsx"),
        "complete": True,
    }
    item.update(extra)
    return item


def envelope(*entries: Any, summary: str = "Implemented the package.") -> dict[str, Any]:
    return {
        "summary": summary,
        "key_decisions": [],
        "artifacts": list(entries),
        "confidence": 0.91,
    }


def legacy_envelope(**files: str) -> dict[str, Any]:
    """The pre-4C envelope: ``artifacts`` as a filename -> content mapping."""
    return {
        "summary": "Did the work.",
        "key_decisions": [],
        "artifacts": dict(files),
        "confidence": 0.9,
    }


def worker_envelope_for(subtask_id: str) -> dict[str, Any]:
    """The compliant envelope for one package: every owned artifact, in full."""
    return envelope(*(entry(aid) for aid in SUBTASK_ARTIFACTS[subtask_id]))


# --------------------------------------------------------------------------- #
# Collection helpers
# --------------------------------------------------------------------------- #
def scope_for(subtask_id: str, work_plan=None):
    return derive_execution_scope(work_plan or react_work_plan(), subtask_id)


def collect_for(subtask_id: str, payload: Any, work_plan=None, **kwargs):
    plan = work_plan or react_work_plan()
    return collect_subtask_artifacts(
        payload, scope_for(subtask_id, plan), work_plan=plan, **kwargs
    )


def collect_dashboard(overrides: Optional[dict[str, Any]] = None, work_plan=None):
    """Collect all six workers, optionally overriding one worker's envelope.

    ``overrides`` maps subtask id -> payload (or a list of payloads, simulating
    several producing branches). ``None`` as the payload omits that worker.
    """
    plan = work_plan or react_work_plan()
    overrides = overrides or {}
    collections = []
    for subtask_id in ALL_SUBTASKS:
        if subtask_id in overrides:
            payload = overrides[subtask_id]
            if payload is None:
                continue
        else:
            payload = worker_envelope_for(subtask_id)
        collections.append(collect_for(subtask_id, payload, work_plan=plan))
    return collections


def assemble_dashboard(overrides: Optional[dict[str, Any]] = None, work_plan=None):
    plan = work_plan or react_work_plan()
    return assemble_deliverable(
        plan, collect_dashboard(overrides, work_plan=plan)
    )
