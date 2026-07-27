"""Synthesis scalability primitives: budget policy, capsules, grouping, fallback.

The synthesizer must never depend on ONE model call receiving every complete
raw worker result. This module owns the deterministic pieces of the scalable
pipeline:

* :class:`SynthesisBudgetPolicy` — the single immutable authority on how large
  any synthesis-related provider call may be.
* :class:`SynthesisCapsule` — the bounded, typed representation of one subtask
  result that synthesis consumes (never the raw envelope).
* :func:`bound_text` — deterministic, EXPLICIT bounding (never silent).
* :func:`plan_groups` — stable, deterministic packing of capsules into
  budget-safe groups.
* :func:`render_capsule_fallback` — the deterministic final answer used when
  model-based synthesis cannot run (provider limits/failures, depth cap,
  cancellation cleanup paths).

Everything here is pure stdlib, deterministic and safely serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

SYNTHESIS_POLICY_SCHEMA_VERSION = 1
CAPSULE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Part E — the ONE authoritative synthesis-input policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SynthesisBudgetPolicy:
    """Immutable input-budget policy for every synthesis provider call.

    ``max_call_input_chars`` bounds the WHOLE prompt (frame included), so no
    call may knowingly exceed what a provider composer accepts. The remaining
    fields bound each stage of the hierarchical pipeline so total work — and
    the number of provider calls — is capped deterministically.
    """

    # Hard cap for one provider call's total input (all messages, chars).
    max_call_input_chars: int = 24000
    # Margin reserved for the prompt frame (system text, task brief, strategy,
    # section headers) inside ``max_call_input_chars``.
    reserved_frame_chars: int = 4000
    # No single capsule may carry more content than this.
    max_capsule_chars: int = 6000
    # At most this many capsules/sections are merged in one group call.
    max_group_size: int = 6
    # At most this many groups exist at any reduction level.
    max_groups: int = 12
    # At most this many reduction levels before the deterministic fallback.
    max_reduction_depth: int = 3
    # An intermediate (group) synthesis result is bounded to this size.
    max_intermediate_chars: int = 8000
    # An oversized single result is split into at most this many chunks.
    max_chunks_per_result: int = 12
    # Strict aggregate ceiling across chunk, merge and final calls in one run.
    max_calls_per_run: int = 64
    # Deterministic non-artifact fallback presentation bound.
    max_fallback_chars: int = 24000
    # Any diagnostic derived from synthesis is bounded to this size.
    max_diagnostic_chars: int = 500
    schema_version: int = SYNTHESIS_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "max_call_input_chars", "reserved_frame_chars", "max_capsule_chars",
            "max_group_size", "max_groups", "max_reduction_depth",
            "max_intermediate_chars", "max_chunks_per_result",
            "max_calls_per_run", "max_fallback_chars", "max_diagnostic_chars",
            "schema_version",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.max_call_input_chars <= 0:
            raise ValueError("max_call_input_chars must be positive")
        if not 0 <= self.reserved_frame_chars < self.max_call_input_chars:
            raise ValueError(
                "reserved_frame_chars must be non-negative and smaller than "
                "max_call_input_chars"
            )
        if self.max_capsule_chars <= 0 or self.max_capsule_chars > self.content_budget:
            raise ValueError(
                "max_capsule_chars must be positive and fit the content budget"
            )
        if self.max_intermediate_chars <= 0 or (
            self.max_intermediate_chars > self.content_budget
        ):
            raise ValueError(
                "max_intermediate_chars must be positive and fit the content budget"
            )
        for name in ("max_group_size", "max_groups", "max_reduction_depth",
                     "max_chunks_per_result", "max_calls_per_run",
                     "max_fallback_chars", "max_diagnostic_chars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def content_budget(self) -> int:
        """Characters available for SECTION CONTENT inside one call."""
        return self.max_call_input_chars - self.reserved_frame_chars

    def call_within_budget(self, prompt_chars: int) -> bool:
        return prompt_chars <= self.max_call_input_chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_call_input_chars": self.max_call_input_chars,
            "reserved_frame_chars": self.reserved_frame_chars,
            "max_capsule_chars": self.max_capsule_chars,
            "max_group_size": self.max_group_size,
            "max_groups": self.max_groups,
            "max_reduction_depth": self.max_reduction_depth,
            "max_intermediate_chars": self.max_intermediate_chars,
            "max_chunks_per_result": self.max_chunks_per_result,
            "max_calls_per_run": self.max_calls_per_run,
            "max_fallback_chars": self.max_fallback_chars,
            "max_diagnostic_chars": self.max_diagnostic_chars,
            "schema_version": self.schema_version,
        }


DEFAULT_SYNTHESIS_POLICY = SynthesisBudgetPolicy()


def policy_for_budget(max_input_chars: int) -> SynthesisBudgetPolicy:
    """Derive a conservative policy from the configured per-call input budget.

    Every derived bound scales with the configured cap so a small budget (as
    tests use) still yields a valid, internally consistent policy. The derived
    values are clamped to the module defaults — never above them.
    """
    cap = int(max_input_chars)
    if cap <= 0:
        raise ValueError("policy_for_budget requires a positive budget")
    reserved = min(DEFAULT_SYNTHESIS_POLICY.reserved_frame_chars, cap // 4)
    content = cap - reserved
    return SynthesisBudgetPolicy(
        max_call_input_chars=cap,
        reserved_frame_chars=reserved,
        max_capsule_chars=min(
            DEFAULT_SYNTHESIS_POLICY.max_capsule_chars, max(1, content // 3)
        ),
        max_intermediate_chars=min(
            DEFAULT_SYNTHESIS_POLICY.max_intermediate_chars,
            max(1, (content * 2) // 5),
        ),
    )


# --------------------------------------------------------------------------- #
# Deterministic, explicit text bounding — never a silent truncation
# --------------------------------------------------------------------------- #
def bound_text(text: str, limit: int) -> str:
    """Bound ``text`` to ``limit`` chars, keeping head and tail EXPLICITLY.

    The elision marker states exactly how much was omitted, so a bounded
    excerpt can never masquerade as the complete content and the tail is never
    silently dropped. ``limit <= 0`` disables bounding.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    marker = f"\n\n[... {len(text) - limit} characters omitted of {len(text)} total ...]\n\n"
    keep = limit - len(marker)
    if keep <= 40:
        # Degenerate limit: keep the head and still declare the omission.
        head = text[: max(1, limit - 24)]
        return head + f"[+{len(text) - len(head)} chars omitted]"
    head = int(keep * 0.6)
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def split_chunks(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    """Deterministically split ``text`` into ordered chunks of ``chunk_chars``.

    If the text needs more than ``max_chunks`` chunks, the chunk size grows so
    exactly ``max_chunks`` ordered chunks cover the WHOLE text — the tail is
    never dropped.
    """
    if chunk_chars <= 0 or not text:
        return [text] if text else []
    total = len(text)
    needed = -(-total // chunk_chars)  # ceil division
    if needed > max_chunks:
        chunk_chars = -(-total // max_chunks)
    return [text[i: i + chunk_chars] for i in range(0, total, chunk_chars)]


# --------------------------------------------------------------------------- #
# Part F — bounded synthesis capsules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SynthesisCapsule:
    """The bounded, typed unit a non-artifact subtask contributes to synthesis.

    Carries only what meaning-level composition needs: identity, objective,
    status, the bounded answer, decisions, dependency references, warnings and
    provenance. Never a raw envelope, never complete artifact bodies, never
    unbounded metadata.
    """

    subtask_id: str
    title: str
    status: str
    content: str
    key_decisions: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    agent_name: str = ""
    truncated: bool = False
    original_chars: int = 0
    schema_version: int = CAPSULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Direct construction is public as well as ``build_capsule``.  Keep
        # every stored string bounded using the authoritative default policy;
        # custom-policy production paths apply their (possibly tighter) bound
        # before constructing the capsule.
        diagnostic_cap = DEFAULT_SYNTHESIS_POLICY.max_diagnostic_chars
        intermediate_cap = DEFAULT_SYNTHESIS_POLICY.max_intermediate_chars
        clean = lambda value: "" if value is None else str(value)
        object.__setattr__(
            self, "subtask_id", bound_text(clean(self.subtask_id), diagnostic_cap)
        )
        object.__setattr__(
            self, "title", bound_text(clean(self.title), diagnostic_cap)
        )
        object.__setattr__(
            self, "status", bound_text(clean(self.status), diagnostic_cap)
        )
        object.__setattr__(
            self, "content", bound_text(clean(self.content), intermediate_cap)
        )
        for name in ("key_decisions", "depends_on", "warnings"):
            value = getattr(self, name) or ()
            object.__setattr__(
                self, name,
                tuple(bound_text(clean(item), diagnostic_cap) for item in value),
            )
        object.__setattr__(
            self, "agent_name", bound_text(clean(self.agent_name), diagnostic_cap)
        )
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "original_chars", int(self.original_chars or 0))
        object.__setattr__(
            self, "schema_version", int(self.schema_version or CAPSULE_SCHEMA_VERSION)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "title": self.title,
            "status": self.status,
            "content": self.content,
            "key_decisions": list(self.key_decisions),
            "depends_on": list(self.depends_on),
            "warnings": list(self.warnings),
            "agent_name": self.agent_name,
            "truncated": self.truncated,
            "original_chars": self.original_chars,
            "schema_version": self.schema_version,
        }

    def render(self) -> str:
        """Deterministic text block for prompts and the fallback answer."""
        lines: list[str] = [self.content]
        if self.key_decisions:
            lines.append("")
            lines.append("Key decisions:")
            lines.extend(f"- {item}" for item in self.key_decisions)
        if self.warnings:
            lines.append("")
            lines.append("Warnings / open contradictions:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines)


def build_capsule(
    result: object,
    policy: SynthesisBudgetPolicy,
    *,
    key_decisions: Sequence[str] = (),
    content: Optional[str] = None,
) -> SynthesisCapsule:
    """Build one bounded capsule from a completed :class:`SubTaskResult`.

    The content is bounded EXPLICITLY (marker, never a silent cut) to the
    policy's capsule limit; identity, ordering inputs and provenance are
    preserved so hierarchical synthesis can keep stable references. Pass
    ``content`` to substitute an already-reduced representation of an
    oversized output (Part G) — it is still bounded here.
    """
    raw = str(getattr(result, "output", "") or "")
    source = content if content is not None else raw
    bounded = bound_text(source, policy.max_capsule_chars)
    status = getattr(result, "status", "")
    # Only a FLAGGED result contributes warnings (its error, plus the verifier
    # feedback explaining it). Clean passing results keep their content pure so
    # legacy synthesis prompts remain byte-identical.
    warnings: list[str] = []
    error = str(getattr(result, "error", "") or "")
    if error:
        warnings.append(bound_text(error, policy.max_diagnostic_chars))
        feedback = str(getattr(result, "verifier_feedback", "") or "")
        if feedback and feedback != error:
            warnings.append(bound_text(feedback, policy.max_diagnostic_chars))
    return SynthesisCapsule(
        subtask_id=str(getattr(result, "subtask_id", "") or ""),
        title=str(getattr(result, "title", "") or ""),
        status=str(getattr(status, "value", status) or ""),
        content=bounded,
        key_decisions=tuple(
            bound_text(str(item), policy.max_diagnostic_chars)
            for item in (key_decisions or ())
        ),
        warnings=tuple(warnings),
        agent_name=str(getattr(result, "agent_name", "") or ""),
        truncated=bounded != raw,
        original_chars=len(raw),
    )


# --------------------------------------------------------------------------- #
# Part H helper — deterministic, stable grouping
# --------------------------------------------------------------------------- #
def plan_groups(
    sizes: Sequence[int], policy: SynthesisBudgetPolicy
) -> list[list[int]]:
    """Pack ordered section indices into budget-safe groups, deterministically.

    Greedy in-order packing under ``content_budget`` and ``max_group_size``;
    stable input order is preserved (a section never moves across a section
    that precedes it). If greedy packing needs more than ``max_groups``
    groups, construction fails closed so the caller can choose a bounded
    fallback without violating any declared limit.
    """
    indices = list(range(len(sizes)))
    if not indices:
        return []
    groups: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for index in indices:
        size = int(sizes[index])
        over_budget = current and current_size + size > policy.content_budget
        over_count = len(current) >= policy.max_group_size
        if over_budget or over_count:
            groups.append(current)
            current, current_size = [], 0
        current.append(index)
        current_size += size
    if current:
        groups.append(current)
    if len(groups) > policy.max_groups:
        # It is mathematically impossible to keep every section while also
        # honoring both the group-count and per-group constraints. Fail closed;
        # the synthesizer will use its explicit, bounded original-capsule
        # fallback instead of constructing knowingly invalid groups.
        raise ValueError(
            f"{len(groups)} groups required; policy allows {policy.max_groups}"
        )
    return groups


# --------------------------------------------------------------------------- #
# Part I — deterministic fallback
# --------------------------------------------------------------------------- #
FALLBACK_HEADLINE = (
    "*(Assembled from the individual subtask results — a final synthesis pass "
    "could not be completed.)*"
)


def render_capsule_fallback(
    capsules: Sequence[SynthesisCapsule],
    *,
    reason: str = "",
    max_chars: int = 0,
) -> str:
    """Readable, deterministic answer built ONLY from accepted capsules.

    Stable order, subtask headings, contradictions/warnings preserved, no
    envelope JSON, and an explicit statement that this is grouped material —
    it never claims to be a full synthesis.
    """
    lines: list[str] = [FALLBACK_HEADLINE]
    if reason:
        lines.append(f"*(Reason: {reason})*")
    lines.append("")
    contradictions: list[str] = []
    for capsule in capsules:
        if len(capsules) > 1:
            lines.append(f"## {capsule.title or capsule.subtask_id}")
            lines.append("")
        lines.append(capsule.content)
        if capsule.key_decisions:
            lines.append("")
            lines.append("Key decisions:")
            lines.extend(f"- {item}" for item in capsule.key_decisions)
        lines.append("")
        contradictions.extend(
            f"{capsule.title or capsule.subtask_id}: {item}"
            for item in capsule.warnings
        )
    if contradictions:
        lines.append("### Unresolved warnings / contradictions")
        lines.append("")
        lines.extend(f"- {item}" for item in contradictions)
        lines.append("")
    rendered = "\n".join(lines).strip()
    return bound_text(rendered, max_chars) if max_chars > 0 else rendered
