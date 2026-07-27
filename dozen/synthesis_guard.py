"""Phase 4G — the authoritative elimination of model-based synthesis.

WHY THIS EXISTS
---------------
Live testing (Phase 4G) showed model synthesis was still REACHABLE even though
root finalization had been made deterministic in Phase 4F. The Manager kept
emitting worker subtasks whose whole job was to merge OTHER workers' outputs —
delegations such as "Final integration review", "Final project assembly",
"Merge multiple worker sections" and "Integrate runtime components". Each was
routed to a provider (GPT), fed every prerequisite worker output, and asked to
merge them. The provider then refused ("too large", "truncated", "missing
modules", "cannot fit into one output"), and that refusal — plus raw internal
JSON envelopes — became the user's final answer.

The architectural decision for this phase is absolute: **DOZEN must never use an
LLM to merge worker results.** This module is the ONE authoritative policy that
makes that structural rather than advisory:

* ``MODEL_BASED_SYNTHESIS_ENABLED`` — the invariant (always ``False``).
* ``detect_synthesis_role`` — deterministically classify ONE subtask as a
  forbidden synthesizer / merger / integrator / final-assembly / final-review
  role from its trusted plan text.
* ``validate_no_synthesis_tasks`` — the planner-flow validator, applied at the
  root AND to every recursive plan.
* ``forbidden_synthesis_reason`` — the executor dispatch guard.

Everything here is pure, deterministic, and duck-typed over the plan/subtask
shape (it never imports ``models``). It sees only TRUSTED plan text — never
worker or provider output — so no response can talk a synthesis subtask back
into existence, and a user writing "synthesise the results" cannot re-enable a
model-based synthesis step.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .decomposition import DecompositionCheck

# --------------------------------------------------------------------------- #
# THE authoritative invariant.
#
# This is deliberately a module constant, not a config field: no orchestrator,
# planner, router, executor or finalization path may re-enable model-based
# synthesis by passing a flag. Finalization enforces it (never selecting the
# optional-synthesis mode while it is False), the planner rejects synthesis
# subtasks, and the executor refuses to dispatch one.
# --------------------------------------------------------------------------- #
MODEL_BASED_SYNTHESIS_ENABLED = False


# --------------------------------------------------------------------------- #
# Detection vocabulary
# --------------------------------------------------------------------------- #
# Verbs that denote combining separate work products into one. Deliberately the
# VERB forms only ("synthesize/synthesise", not the noun "synthesis"): a request
# to "write the synthesis module" produces a code file and is NOT a merge role.
_MERGE_VERB = (
    r"synthesi[sz]e|synthesi[sz]ing|merg(?:e|ing)|combin(?:e|ing)|"
    r"consolidat(?:e|ing)|integrat(?:e|ing)|assembl(?:e|ing)|stitch(?:ing)?|"
    r"unif(?:y|ying)|aggregat(?:e|ing)|collat(?:e|ing)|reconcil(?:e|ing)|"
    r"weav(?:e|ing)|fus(?:e|ing)"
)

# Nouns that unambiguously name OTHER workers' work products (a fan-in target).
# Generic engineering nouns (modules, components, files) are intentionally
# EXCLUDED here so legitimate code such as "write main.py that wires the modules
# together" is never mis-flagged; those are handled only under the fan-in gate.
_WORK_PRODUCT_NOUN = (
    r"outputs?|results?|responses?|replies?|sections?|answers?|"
    r"contributions?|submissions?|deliverables?|subtasks?|sub-?agents?|"
    r"workers?|findings?"
)

# Tier 1a — an explicit instruction to merge OTHER work products into one
# result. The verb and the work-product noun may sit up to a few words apart.
_MERGE_OF_WORK_RE = re.compile(
    r"\b(?:" + _MERGE_VERB + r")\b"
    r"(?:\W+\w+){0,6}?\W+"
    r"\b(?:" + _WORK_PRODUCT_NOUN + r")\b",
    re.IGNORECASE,
)

# Tier 1b — explicit final-synthesis / assembly / integration-review role
# phrases, and the "return part N of M" / "complete merged part" language that
# is itself the size-refusal failure mode. These are forbidden regardless of
# fan-in because their wording alone establishes a synthesizer role.
_ROLE_PHRASE_RE = re.compile(
    r"\bfinal\s+(?:project\s+)?"
    r"(?:integration(?:\s+review)?|assembly|synthesis|consolidation|merge|"
    r"stitch(?:ing)?)\b"
    r"|\bfinal\s+project\s+assembly\b"
    r"|\bproject\s+assembly\b"
    r"|\bcomplete\s+merged\b"
    r"|\bmerged\s+(?:final|complete)\s+"
    r"(?:answer|result|deliverable|project|codebase|response|part)\b"
    r"|\bconsolidat(?:e|ed|ing)\s+the\s+(?:complete|entire|full|whole)\s+"
    r"(?:codebase|project|code\s*base|application|system|answer)\b"
    r"|\bassembl(?:e|ing|y\s+of)\s+the\s+"
    r"(?:complete|entire|full|final|whole)\s+"
    r"(?:project|codebase|application|system|answer|deliverable)\b"
    r"|\bpart\s+\d+\s+of\s+\d+\b"
    r"|\breturn\s+part\s+\d+\b"
    r"|\btrust\s+each\s+(?:output|result|section|worker)\b"
    r"|\bfinal\s+(?:qa|quality)\s+(?:review|pass|check)\b"
    r"|\b(?:synthesi[sz]er|integrator|merger|final\s+reviewer)\s+role\b",
    re.IGNORECASE,
)

# Tier 2 — softer integration wording applied to generic engineering nouns.
# Alone this is legitimate ("wire the modules together in main.py"); it is a
# forbidden synthesis role ONLY when the subtask is a genuine fan-in node that
# consumes two or more upstream producers, which is the shape that turns it into
# "hand every prior output to one provider and ask it to merge".
_SOFT_INTEGRATION_RE = re.compile(
    r"\b(?:integrat(?:e|ing)|assembl(?:e|ing)|combin(?:e|ing)|merg(?:e|ing)|"
    r"wir(?:e|ing)|bring(?:ing)?)\b"
    r"(?:\W+\w+){0,6}?\W+"
    r"\b(?:components?|modules?|pieces?|parts?|files?|everything|together|"
    r"into\s+one)\b",
    re.IGNORECASE,
)

# Structural paraphrases that avoid an explicit merge verb but still assign a
# broad fan-in finalization role: a final/unified product is derived from prior
# tasks, upstream work, or every dependency.  Both halves are required, and the
# rule is gated by real plan fan-in below, so ordinary final deliverables and
# ordinary dependency-consuming implementation tasks remain valid.
_UPSTREAM_SOURCE_RE = re.compile(
    r"\b(?:previous|prior|upstream|earlier|dependent)\s+"
    r"(?:tasks?|subtasks?|work|outputs?|results?|modules?|sections?)\b"
    r"|\b(?:every|all)\s+(?:of\s+the\s+)?dependenc(?:y|ies)\b",
    re.IGNORECASE,
)
_FINAL_FAN_IN_PRODUCT_RE = re.compile(
    r"\b(?:one\s+)?unified\s+deliverable\b"
    r"|\bdefinitive\s+(?:answer|response|deliverable)\b"
    r"|\brelease\s+candidate\b"
    r"|\bconsolidated\s+(?:implementation|deliverable|project|response)\b"
    r"|\bfinal\s+(?:answer|response|deliverable|project|implementation)\b",
    re.IGNORECASE,
)

# A concrete data transformation owns a distinct engineering output; it is not
# a model role that merges worker prose.  This exclusion is deliberately narrow
# and applies only to the soft fan-in heuristic (never to explicit worker-output
# merge instructions or final-synthesis role phrases).
_CONCRETE_DATA_MERGE_RE = re.compile(
    r"\b(?:merge|combine|consolidate)\b[^.\n]{0,80}\b"
    r"(?:csv|tsv|jsonl|parquet|spreadsheet|data(?:set)?s?)\b",
    re.IGNORECASE,
)


def detect_synthesis_role(
    *,
    title: str = "",
    instruction: str = "",
    expected_output: str = "",
    success_criteria: str = "",
    dependency_count: int = 0,
) -> Optional[str]:
    """Classify ONE subtask as a forbidden model-synthesis role, or ``None``.

    Returns a concise, content-free reason string when the subtask's trusted
    plan text establishes a synthesizer / merger / integrator / final-assembly
    / final-review role, else ``None``. ``dependency_count`` is the subtask's
    number of in-plan prerequisites; it gates the softer Tier-2 wording so a
    standalone integration file is never mistaken for a fan-in merge.
    """
    surfaces = [
        str(title or ""),
        str(instruction or ""),
        str(expected_output or ""),
        str(success_criteria or ""),
    ]
    haystack = "  ".join(surface for surface in surfaces if surface.strip())
    if not haystack.strip():
        return None
    if _ROLE_PHRASE_RE.search(haystack):
        return (
            "assigns a final synthesis / integration-review / project-assembly "
            "role"
        )
    if _MERGE_OF_WORK_RE.search(haystack):
        return "instructs a provider to merge other workers' outputs/results"
    if (
        dependency_count >= 2
        and _UPSTREAM_SOURCE_RE.search(haystack)
        and _FINAL_FAN_IN_PRODUCT_RE.search(haystack)
    ):
        return (
            "turns broad upstream work into a final deliverable "
            "(a fan-in synthesis role)"
        )
    if (
        dependency_count >= 2
        and _SOFT_INTEGRATION_RE.search(haystack)
        and not _CONCRETE_DATA_MERGE_RE.search(haystack)
    ):
        return (
            "integrates multiple upstream results into one deliverable "
            "(a fan-in synthesis role)"
        )
    return None


def forbidden_synthesis_reason(
    subtask: Any, *, dependency_count: Optional[int] = None
) -> Optional[str]:
    """Executor dispatch guard: reason a subtask is a forbidden synthesis role.

    Duck-typed over the ``SubTask`` shape. ``dependency_count`` defaults to the
    subtask's own ``depends_on`` length so the executor can call it with just
    the subtask.
    """
    if dependency_count is None:
        dependency_count = len(tuple(getattr(subtask, "depends_on", ()) or ()))
    return detect_synthesis_role(
        title=getattr(subtask, "title", "") or "",
        instruction=getattr(subtask, "instruction", "") or "",
        expected_output=getattr(subtask, "expected_output", "") or "",
        success_criteria=getattr(subtask, "success_criteria", "") or "",
        dependency_count=dependency_count,
    )


def validate_no_synthesis_tasks(plan: Any) -> DecompositionCheck:
    """Reject any plan (root or recursive) that delegates a synthesis role.

    Pure and deterministic. A direct-answer plan has no delegations to judge.
    Returns a :class:`DecompositionCheck`; its fatal problems flow through the
    planner's ONE shared corrective re-plan exactly like the artifact
    decomposition and package-sizing validators. Advisory output is never
    produced — a synthesis delegation is always fatal.
    """
    is_direct = getattr(plan, "is_direct", None)
    if callable(is_direct) and is_direct():
        return DecompositionCheck()
    subtasks = list(getattr(plan, "subtasks", None) or [])
    plan_ids = {str(getattr(subtask, "id", "") or "") for subtask in subtasks}
    fatal: list[str] = []
    for subtask in subtasks:
        deps = [
            dep
            for dep in (getattr(subtask, "depends_on", ()) or [])
            if str(dep) in plan_ids
        ]
        reason = detect_synthesis_role(
            title=getattr(subtask, "title", "") or "",
            instruction=getattr(subtask, "instruction", "") or "",
            expected_output=getattr(subtask, "expected_output", "") or "",
            success_criteria=getattr(subtask, "success_criteria", "") or "",
            dependency_count=len(deps),
        )
        if reason:
            title = " ".join(str(getattr(subtask, "title", "") or "").split())[:60]
            sid = str(getattr(subtask, "id", "") or "?")
            fatal.append(
                f"subtask {sid!r} ({title}) {reason}; DOZEN combines validated "
                "results deterministically — do not delegate a merge, "
                "integration, final-assembly or final-review step to a model"
            )
    return DecompositionCheck(fatal=tuple(fatal))


__all__ = [
    "MODEL_BASED_SYNTHESIS_ENABLED",
    "detect_synthesis_role",
    "forbidden_synthesis_reason",
    "validate_no_synthesis_tasks",
]
