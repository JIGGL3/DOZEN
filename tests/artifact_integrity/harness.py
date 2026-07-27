"""Shared Phase 4D fixtures: the APPROVED React dashboard, cut off mid-reply.

Reuses the Phase 4B decomposition harness and the Phase 4C result harness (same
manifest, same packages, same subtask map, same worker envelopes) and adds only
what a TRUNCATED worker returns. Nothing is written to disk and no provider is
ever contacted.
"""

from __future__ import annotations

from typing import Any

from dozen.artifact_integrity import (
    DEFAULT_INTEGRITY_POLICY,
    IntegrityPolicy,
    IntegrityStatus,
    evaluate_artifact_integrity,
)
from dozen.artifacts import ArtifactKind, ArtifactOperation

from ..artifact_decomposition.harness import react_work_plan  # noqa: F401
from ..artifact_results.harness import (  # noqa: F401
    ALL_SUBTASKS,
    CONTENT,
    assemble_dashboard,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    scope_for,
    worker_envelope_for,
)

# --------------------------------------------------------------------------- #
# Truncated variants of the approved dashboard bodies
# --------------------------------------------------------------------------- #
# Ends inside a JSX attribute string: the classic provider cut-off.
APP_CUT_IN_ATTRIBUTE = (
    'import { Router } from "./router";\n\n'
    "export default function App() {\n"
    "  return (\n"
    '    <button className="'
)

# Ends after an unfinished import: balanced, parseable-looking, but nothing else.
ROUTER_CUT_AFTER_IMPORT = (
    'import { DashboardPage } from "../features/dashboard/DashboardPage";\n'
    "import { Suspense } from"
)

# A layout whose JSX element is never closed.
LAYOUT_UNCLOSED_TAG = (
    "export function DashboardLayout({ children }) {\n"
    "  return (\n"
    "    <main>\n"
    "      <section>{children}\n"
    "    </main>\n"
    "  );\n"
    "}\n"
)

# Braces, quotes and an ellipsis INSIDE strings: entirely valid, must be accepted.
CARD_BRACES_IN_STRINGS = (
    "export function Card({ title }) {\n"
    '  const cls = "rounded { } p-4";\n'
    "  const hint = 'It\\'s loading...';\n"
    "  const tpl = `w-${title.length} { }`;\n"
    "  return <section className={cls} title={hint} data-tpl={tpl}>{title}</section>;\n"
    "}\n"
)

# A test file cut off inside a template literal.
TEST_UNCLOSED_TEMPLATE = (
    'import { render } from "@testing-library/react";\n\n'
    'test("renders", () => {\n'
    "  const markup = `<div class=\"card\">\n"
)

PACKAGE_JSON_MISSING_BRACE = '{\n  "name": "dashboard",\n  "version": "1.0.0"\n'

BUILD_RESULT_TRUNCATED = (
    "vite build\n"
    "transforming modules...\n"
    "[... output truncated ...]"
)

# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #
def broken_entry(artifact_id: str, content: str, **extra: Any) -> dict[str, Any]:
    """One entry a worker SWEARS is complete while the body says otherwise."""
    return entry(artifact_id, content=content, complete=True, **extra)


def envelope_with(subtask_id: str, **overrides: str) -> dict[str, Any]:
    """The compliant envelope for one package with some bodies replaced."""
    from ..artifact_results.harness import SUBTASK_ARTIFACTS

    return envelope(*(
        entry(aid, content=overrides[aid]) if aid in overrides else entry(aid)
        for aid in SUBTASK_ARTIFACTS[subtask_id]
    ))


# --------------------------------------------------------------------------- #
# Direct integrity helpers
# --------------------------------------------------------------------------- #
def check(
    content: str,
    path: str = "src/app/App.tsx",
    *,
    kind: ArtifactKind = ArtifactKind.SOURCE_FILE,
    operation: ArtifactOperation = ArtifactOperation.CREATE,
    language: str = "",
    media_type: str = "",
    policy: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY,
):
    return evaluate_artifact_integrity(
        content, artifact_id=path, path=path, kind=kind, operation=operation,
        language=language, media_type=media_type, policy=policy,
    )


def assert_valid(case, content: str, path: str, **kwargs) -> None:
    result = check(content, path, **kwargs)
    case.assertIn(
        result.status,
        (IntegrityStatus.VALID, IntegrityStatus.NOT_APPLICABLE),
        msg=f"{path} was expected to pass but reported: {result.problem_summary()}",
    )
    case.assertTrue(result.structurally_complete)


def assert_invalid(case, content: str, path: str, code=None, **kwargs):
    result = check(content, path, **kwargs)
    case.assertEqual(
        result.status, IntegrityStatus.INVALID,
        msg=f"{path} was expected to be INVALID; issues={result.issues}",
    )
    case.assertFalse(result.structurally_complete)
    if code is not None:
        case.assertIn(code, [issue.code for issue in result.issues])
    return result


def assert_suspicious(case, content: str, path: str, code=None, **kwargs):
    result = check(content, path, **kwargs)
    case.assertEqual(
        result.status, IntegrityStatus.SUSPICIOUS,
        msg=f"{path} was expected to be SUSPICIOUS; issues={result.issues}",
    )
    if code is not None:
        case.assertIn(code, [issue.code for issue in result.issues])
    return result
