"""Core data structures shared across the orchestrator.

These are deliberately plain dataclasses (no third-party deps) so the control
flow is easy to read and serialize for logging / debugging.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from .artifact_repair import ArtifactRepairReport
from .artifact_results import AssembledDeliverable, SubtaskCollectionResult
from .artifacts import ArtifactWorkPlan
from .decomposition import ArtifactExecutionScope
from .intent import DeliverableContract
from .presentation import FinalPresentation


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_REPAIR = "needs_repair"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Task:
    """A top-level request handed to the orchestrator."""

    prompt: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    desired_output: str = ""
    id: str = field(default_factory=lambda: _new_id("task"))
    # What kind of deliverable this request owes the user (Phase 3). Optional
    # for backward compatibility: library consumers may keep constructing
    # Task(prompt=...) and the orchestrator resolves the contract at its
    # entry boundary. An explicitly supplied contract is always preserved.
    contract: Optional[DeliverableContract] = None
    # Phase 4B: the artifact scope this task is confined to. Set by the
    # executor when it recursively re-orchestrates a scoped subtask so the
    # child inherits exactly its assigned packages and never the whole root
    # manifest. ``None`` (the default, and for every legacy caller) leaves
    # planning and prompts byte-identical to pre-4B behavior.
    execution_scope: Optional[ArtifactExecutionScope] = None
    # A scoped recursive task normally owes the parent-owned artifacts.  A
    # nested research/design/review recursion keeps the boundary but sets this
    # false so none of its workers inherits artifact ownership.
    execution_scope_output_required: bool = True

    def to_brief(self) -> str:
        parts = [f"OBJECTIVE:\n{self.prompt.strip()}"]
        if self.context.strip():
            parts.append(f"\nCONTEXT:\n{self.context.strip()}")
        if self.constraints:
            joined = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"\nCONSTRAINTS:\n{joined}")
        if self.desired_output.strip():
            parts.append(f"\nDESIRED FINAL OUTPUT:\n{self.desired_output.strip()}")
        return "\n".join(parts)


@dataclass
class SubTask:
    """A single node in the execution DAG produced by the planner."""

    title: str
    instruction: str
    # Capability tags the router uses to pick a model (e.g. "coding", "reasoning").
    required_capabilities: list[str] = field(default_factory=list)
    # IDs of subtasks whose outputs are needed before this one can run.
    depends_on: list[str] = field(default_factory=list)
    # What a good answer looks like; used by the verifier.
    success_criteria: str = ""
    expected_output: str = ""
    # If true, the executor may recursively re-orchestrate this subtask
    # (Dozen calling itself) instead of doing a single model call.
    complex: bool = False
    # Rough difficulty 1-5; nudges the router toward stronger models.
    difficulty: int = 3
    # Intelligent routing: the Manager LLM's chosen model (an AgentSpec.name)
    # and its justification. Empty => fall back to the deterministic router.
    assigned_model: str = ""
    model_selection_reasoning: str = ""
    id: str = field(default_factory=lambda: _new_id("sub"))
    # For a recursive plan inside an inherited artifact scope, exactly one
    # delegation is the output producer. Support/research/review children stay
    # false and therefore never receive the parent's complete-file obligation.
    produces_parent_artifacts: bool = False

    def __post_init__(self) -> None:
        self.difficulty = max(1, min(5, int(self.difficulty)))


@dataclass
class Plan:
    """The planner's output: how the task is broken down and stitched together."""

    analysis: str
    subtasks: list[SubTask]
    synthesis_strategy: str
    # If the planner decides the task is trivial, it can mark it direct.
    direct_answer: Optional[str] = None
    # Phase 4A: optional artifact-oriented contract (manifest + work packages)
    # for requests whose deliverable is a set of concrete artifacts. ``None``
    # for every other plan — legacy construction remains byte-identical.
    artifact_plan: Optional[ArtifactWorkPlan] = None

    def is_direct(self) -> bool:
        return self.direct_answer is not None or not self.subtasks

    def by_id(self) -> dict[str, SubTask]:
        return {s.id: s for s in self.subtasks}


@dataclass
class SubTaskResult:
    subtask_id: str
    title: str
    status: TaskStatus
    output: str = ""
    agent_name: str = ""
    # Why this agent was chosen (from the Manager's intelligent routing, or a
    # note when the deterministic router was used as a fallback).
    model_selection_reasoning: str = ""
    attempts: int = 0
    verifier_score: float = 0.0
    verifier_feedback: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # Phase 4C: what this subtask actually produced against its assigned artifact
    # scope. ``None`` for every unscoped subtask (and every legacy caller), which
    # keeps non-artifact runs byte-identical. Deliberately independent of
    # ``status``: a semantically good answer can still miss an owned artifact.
    artifact_collection: Optional[SubtaskCollectionResult] = None
    # Phase 4E: the compact targeted-repair trace (what was preserved, what was
    # re-requested, what was actually repaired, why repair stopped). ``None`` for
    # every unscoped subtask; ``attempted`` stays false when the first attempt
    # already delivered the package, so a healthy run reports nothing new.
    artifact_repair: Optional[ArtifactRepairReport] = None
    # Phase 6: bounded key decisions captured from the worker's envelope before
    # it is flattened, so synthesis capsules can carry them without ever seeing
    # the envelope itself. Empty for non-envelope replies and legacy callers.
    key_decisions: list[str] = field(default_factory=list)
    # Phase 4F (Part E): bounded warnings a typed prose envelope declared.
    # Metadata for ordered sections — never rendered raw.
    worker_warnings: list[str] = field(default_factory=list)
    # User-facing evidence references declared by a typed prose envelope.
    worker_evidence_refs: list[str] = field(default_factory=list)
    # Runtime identity/cleanup diagnostics are observability metadata only.
    worker_internal_notes: list[str] = field(default_factory=list)
    # Phase 4F (Part H): typed delivery classification for a scoped artifact
    # worker that did not fully deliver ("" = delivered / unscoped / legacy).
    delivery_code: str = ""

    @property
    def artifacts_satisfied(self) -> bool:
        """True unless a scoped subtask failed its artifact obligations."""
        return (
            self.artifact_collection is None
            or self.artifact_collection.satisfied
        )

    @property
    def duration_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 3)


@dataclass
class OrchestrationResult:
    task_id: str
    final_answer: str
    plan: Optional[Plan] = None
    subtask_results: list[SubTaskResult] = field(default_factory=list)
    depth: int = 0
    direct: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    # Phase 4C (both optional; ``None`` keeps every legacy consumer working):
    # the assembled in-memory deliverable of an artifact-bearing root run...
    artifact_assembly: Optional[AssembledDeliverable] = None
    # ...and, for a scoped RECURSIVE run, the typed candidates this child owes
    # its parent (the parent re-collects them; nothing else consumes this).
    artifact_collection: Optional[SubtaskCollectionResult] = None
    # Phase 6: the typed presentation the final rendering boundary produced for
    # ``final_answer``. ``None`` for legacy construction paths; when present,
    # ``final_answer`` is exactly ``presentation.text``.
    presentation: Optional[FinalPresentation] = None
    # Phase 4F: which deterministic finalization mode produced ``final_answer``
    # ("" for direct answers and legacy construction paths).
    finalization_mode: str = ""
    # Live-acceptance correction: content-free delivery telemetry for the root
    # boundary. These are safe enum strings / bools / counts — never prompts or
    # provider output — so they can travel into logs and the terminal event.
    intent: str = ""                       # safe request-intent kind
    code_only: bool = False                # the resolved code-only obligation
    artifact_plan_present: bool = False    # the validated plan owed artifacts
    model_synthesis_invoked: bool = False  # a synthesizer call actually happened

    def summary(self) -> dict[str, Any]:
        """A compact, serializable trace useful for logging / debugging."""
        base = {
            "task_id": self.task_id,
            "status": (
                "cancelled" if self.error.startswith("Cancelled:")
                else "failed" if self.error
                else "completed"
            ),
            "direct": self.direct,
            "depth": self.depth,
            "num_subtasks": len(self.subtask_results),
            "subtasks": [
                {
                    "title": r.title,
                    "agent": r.agent_name,
                    "reasoning": r.model_selection_reasoning,
                    "status": r.status.value,
                    "attempts": r.attempts,
                    "score": r.verifier_score,
                    "duration_s": r.duration_s,
                }
                for r in self.subtask_results
            ],
            "error": self.error,
            "warnings": list(self.warnings),
        }
        # Additive (Phase 6): the presentation kind appears only when the final
        # rendering boundary ran, so legacy consumers see the exact prior shape.
        if self.presentation is not None:
            base["answer_kind"] = self.presentation.kind.value
        # Additive (Phase 4F): the deterministic finalization mode, only when a
        # non-direct run selected one — legacy consumers see the prior shape.
        if self.finalization_mode:
            base["finalization_mode"] = self.finalization_mode
        # Additive (live-acceptance correction): content-free delivery
        # telemetry so a caller (and the terminal SSE event) can verify HOW the
        # answer was produced — the finalization mode, the safe intent kind,
        # whether the run owed code-only delivery, whether the validated plan
        # carried an artifact manifest, the assembly status, and whether any
        # model synthesis was invoked. No prompt or provider text appears here.
        base["delivery"] = {
            "finalization_mode": self.finalization_mode or "",
            "intent": self.intent,
            "code_only": self.code_only,
            "artifact_plan_present": self.artifact_plan_present,
            "assembly_status": (
                self.artifact_assembly.status.value
                if self.artifact_assembly is not None else "none"
            ),
            "model_synthesis_invoked": self.model_synthesis_invoked,
        }
        # Additive: the artifact block appears ONLY for artifact-bearing runs, so
        # consumers that predate Phase 4C see exactly the summary they saw before.
        if self.artifact_assembly is not None:
            base["artifacts"] = self.artifact_assembly.summary()
        # Additive (Phase 4E): counts and ids only — never artifact content — and
        # only when targeted repair actually ran.
        repairs = [
            r.artifact_repair for r in self.subtask_results
            if r.artifact_repair is not None and r.artifact_repair.attempted
        ]
        if repairs:
            base["artifact_repair"] = {
                "repair_attempted": True,
                "subtasks_repaired": len(repairs),
                "repair_attempts_used": sum(r.attempts_used for r in repairs),
                "artifacts_preserved": sum(
                    len(r.preserved_artifact_ids) for r in repairs
                ),
                "artifacts_repaired": sum(
                    len(r.repaired_artifact_ids) for r in repairs
                ),
                "remaining_repair_targets": sum(
                    len(r.remaining_target_ids) for r in repairs
                ),
                "no_progress_terminations": sum(
                    1 for r in repairs
                    if r.termination.value == "no_progress"
                ),
            }
        return base


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
