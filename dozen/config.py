"""Tunable configuration for the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrchestratorConfig:
    # Recursion: how deep "Dozen calling itself" can go for `complex` subtasks.
    max_depth: int = 2

    # Parallelism: how many subtasks may run concurrently (LLM calls are IO-bound).
    max_parallelism: int = 4

    # Repair loop: how many times a worker may retry after a failed verification.
    max_repair_attempts: int = 2

    # Verifier gate. Set verify_outputs=False to skip verification entirely.
    verify_outputs: bool = True
    pass_threshold: float = 0.7

    # If a subtask keeps failing verification, escalate to the highest-tier agent
    # for a final attempt before giving up.
    escalate_on_failure: bool = True

    # Use an LLM call for routing (more nuanced) vs. the deterministic heuristic.
    use_llm_router: bool = False

    # Role assignment: names of agents to use for planner/router/verifier/
    # synthesizer roles. If None, the orchestrator picks the strongest reasoner.
    planner_agent: str | None = None
    router_agent: str | None = None
    verifier_agent: str | None = None
    synthesizer_agent: str | None = None

    # Input budgets (chars). Worker outputs are stitched back into later
    # prompts; unbounded, a big run produces synthesis prompts that web chat
    # composers reject outright ("message too long"). Clipping keeps head+tail.
    # 0 disables a budget.
    max_synthesis_input_chars: int = 24000   # total across all outputs stitched
    max_dep_output_chars: int = 8000         # per prerequisite output in worker prompts

    # Phase 4F: the TRUSTED configuration gate for optional model polish.
    # Deterministic ordered-section finalization is the default; only this
    # flag (or an explicit user request for a polished unified narrative)
    # permits the optional model-synthesis pass — never required delivery.
    enable_model_polish: bool = False

    # Emit a step-by-step trace via the logger.
    verbose: bool = True
