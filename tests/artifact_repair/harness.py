"""Shared Phase 4E fixtures: the APPROVED React dashboard, repaired.

Reuses the Phase 4B decomposition harness, the Phase 4C result harness and the
Phase 4D truncated bodies unchanged. Nothing is written to disk and no provider
is ever contacted.
"""

from __future__ import annotations

from typing import Any, Optional

from dozen.artifact_repair import (
    DEFAULT_REPAIR_POLICY,
    ArtifactRepairPolicy,
    ArtifactRepairState,
    RepairDecision,
    begin_repair_state,
    collect_repair_artifacts,
    derive_repair_scope,
    merge_repair_collection,
    plan_repair,
    submitted_content_hashes,
)

from ..artifact_decomposition.harness import react_work_plan  # noqa: F401
from ..artifact_integrity.harness import (  # noqa: F401
    APP_CUT_IN_ATTRIBUTE,
    LAYOUT_UNCLOSED_TAG,
    PACKAGE_JSON_MISSING_BRACE,
    ROUTER_CUT_AFTER_IMPORT,
    envelope_with,
)
from ..artifact_results.harness import (  # noqa: F401
    ALL_SUBTASKS,
    CONTENT,
    SUBTASK_ARTIFACTS,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    scope_for,
    worker_envelope_for,
)

APP = "src/app/App.tsx"
ROUTER = "src/app/router.tsx"
LAYOUT = "src/components/layout/DashboardLayout.tsx"
CARD = "src/components/ui/Card.tsx"
PAGE = "src/features/dashboard/DashboardPage.tsx"
PKG = "package.json"

SHELL = ("shell", (APP, ROUTER, LAYOUT))


def broken_shell(**overrides: str) -> dict[str, Any]:
    """The shell package with ``DashboardLayout.tsx`` cut off, unless overridden."""
    return envelope_with("shell", **(overrides or {LAYOUT: "export function Dash"}))


def first_attempt(
    payload: dict[str, Any],
    subtask_id: str = "shell",
    *,
    attempt: int = 1,
) -> ArtifactRepairState:
    """Collect attempt 1 exactly as the executor does, and seed preserved state."""
    collection = collect_for(subtask_id, payload, attempt=attempt)
    return begin_repair_state(
        collection, submitted_hashes=submitted_content_hashes(payload)
    )


def decide(
    state: ArtifactRepairState,
    subtask_id: str = "shell",
    *,
    attempt: int = 1,
    max_attempts: int = 3,
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
) -> RepairDecision:
    return plan_repair(
        scope_for(subtask_id), state,
        attempt=attempt, max_attempts=max_attempts, policy=policy,
    )


def repair_attempt(
    state: ArtifactRepairState,
    decision: RepairDecision,
    payload: dict[str, Any],
    subtask_id: str = "shell",
    *,
    attempt: int = 2,
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
    producer_subtask_id: str = "",
    recursion_depth: int = 0,
) -> ArtifactRepairState:
    """Parse ONE repair reply against the narrow scope and merge it in."""
    scope = scope_for(subtask_id)
    request = decision.request
    assert request is not None, "the decision carries no repair request"
    repair_scope = derive_repair_scope(scope, request)
    collection = collect_repair_artifacts(
        payload, repair_scope, request,
        policy=policy, attempt=attempt,
        producer_subtask_id=producer_subtask_id or subtask_id,
        recursion_depth=recursion_depth,
    )
    return merge_repair_collection(
        scope, state, collection, request,
        attempt=attempt,
        submitted_hashes=submitted_content_hashes(payload),
        policy=policy,
    )


def cycle(
    payloads: list[dict[str, Any]],
    subtask_id: str = "shell",
    *,
    max_attempts: Optional[int] = None,
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
) -> tuple[ArtifactRepairState, RepairDecision, list[Optional[RepairDecision]]]:
    """Drive the whole bounded lifecycle: attempt 1, then targeted repairs.

    Mirrors the executor exactly — one initial attempt plus the existing repair
    attempts, never more — and returns the final state, the final decision and
    every decision along the way.
    """
    budget = len(payloads) if max_attempts is None else max_attempts
    state = first_attempt(payloads[0], subtask_id)
    decision = plan_repair(
        scope_for(subtask_id), state,
        attempt=1, max_attempts=budget, policy=policy,
    )
    history: list[Optional[RepairDecision]] = [decision]
    attempt = 1
    while decision.should_retry and attempt < budget and attempt < len(payloads):
        attempt += 1
        state = repair_attempt(
            state, decision, payloads[attempt - 1], subtask_id,
            attempt=attempt, policy=policy,
        )
        decision = plan_repair(
            scope_for(subtask_id), state,
            attempt=attempt, max_attempts=budget, policy=policy,
        )
        history.append(decision)
    return state, decision, history
