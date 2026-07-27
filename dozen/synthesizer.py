"""Synthesizer: composes subtask results into one final answer.

Artifact-bearing runs bypass model stitching entirely: the Phase 4 assembled
deliverable is rendered deterministically (no model may rewrite a finished
file). Non-artifact runs are composed hierarchically under ONE immutable
budget policy (:class:`~dozen.synthesis_scaling.SynthesisBudgetPolicy`):

    subtask results
      -> bounded capsules (oversized single outputs chunk-reduced first)
      -> budget-safe ordered groups (one bounded merge call per group)
      -> intermediate nodes (repeat, bounded depth)
      -> final synthesis call

Every provider call is measured against the policy BEFORE it is issued — no
call may knowingly exceed the configured input budget. When model synthesis
cannot run (provider limits, provider failure, malformed response, depth cap),
the answer degrades to a deterministic, readable capsule fallback that keeps
stable ordering and preserves warnings/contradictions — never raw protocol
JSON, and never a silent truncation.
"""

from __future__ import annotations

from typing import Callable, Optional

from .agent_pool import AgentSpec
from .artifact_results import AssembledDeliverable, render_assembled_deliverable
from .cancellation import CancelledError
from .llm_client import LLMClient, LLMMessage
from .models import Plan, SubTaskResult, Task, TaskStatus
from .prompts import build_synthesizer_messages
from .presentation import _requested_json_data, user_requested_json
from .synthesis_scaling import (
    SynthesisBudgetPolicy,
    SynthesisCapsule,
    bound_text,
    build_capsule,
    plan_groups,
    policy_for_budget,
    render_capsule_fallback,
    split_chunks,
)
from .validation import (
    decode_nested_envelope,
    is_malformed_protocol_envelope,
    looks_like_protocol_envelope,
    looks_like_orchestration_json,
    parse_worker_artifact,
    render_artifact,
)

# Compact role prompts for the bounded reduction calls. Deliberately short so
# corrective/intermediate prompts never duplicate large schemas (Part J), and
# prose-only so no reduction step reintroduces a JSON envelope.
_GROUP_SYSTEM = (
    "You are combining consecutive sections of ONE larger answer produced by "
    "a team. Merge the given sections into one coherent, condensed part. "
    "Preserve every key fact, decision, number, code behavior and warning; "
    "state genuine contradictions explicitly instead of erasing them; remove "
    "only duplication and filler. Output plain prose/markdown only — no JSON, "
    "no preamble, no meta-commentary."
)

_CHUNK_SYSTEM = (
    "You are condensing one part of a single long result so a later editor "
    "can compose the final answer. Write a faithful, compact digest of the "
    "given part: keep every key fact, decision, number, warning and "
    "conclusion; drop repetition and filler. Output plain prose/markdown "
    "only — no JSON, no preamble."
)


def _measure(messages: list[LLMMessage]) -> int:
    """Legacy content-only measurement helper retained for public compatibility."""
    return sum(len(message.content) for message in messages)


class SynthesisCallBudgetExceeded(RuntimeError):
    """No further provider call may start in the current synthesis run."""


class Synthesizer:
    def __init__(
        self,
        client: LLMClient,
        synthesizer_agent: AgentSpec,
        max_input_chars: int = 24000,
        policy: Optional[SynthesisBudgetPolicy] = None,
    ) -> None:
        self.client = client
        self.agent = synthesizer_agent
        # Per-call input budget. Web chat composers reject overlong prompts, so
        # no single call may carry more than this. ``<= 0`` disables budgeting
        # (legacy library behavior: one unbounded call).
        self.max_input_chars = max_input_chars
        # THE authoritative synthesis-input policy (Part E). Derived once from
        # the configured budget; immutable for the synthesizer's lifetime.
        self.policy: Optional[SynthesisBudgetPolicy] = (
            policy if policy is not None
            else (policy_for_budget(max_input_chars) if max_input_chars > 0 else None)
        )
        # Measured input size of every provider call in the LAST synthesize()
        # run, in call order — for tests, logs and prompt-size reporting.
        self.last_call_input_chars: list[int] = []

    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        task: Task,
        plan: Plan,
        results: list[SubTaskResult],
        assembly: Optional[AssembledDeliverable] = None,
    ) -> str:
        self.last_call_input_chars = []

        # Phase 4C: an artifact-bearing run is PRESENTED, never re-assembled or
        # model-stitched, here. The assembly layer already decided which
        # artifacts are complete, which conflict and which are missing;
        # rendering them deterministically is both lossless (no model may
        # rewrite a finished file) and honest (a partial or conflicted
        # deliverable says so). Complete file bodies never enter a prompt.
        if assembly is not None:
            return render_assembled_deliverable(assembly)

        usable_results = [
            r for r in results
            if r.status == TaskStatus.COMPLETED and r.output.strip()
        ]

        # Shortcut: a single subtask needs no real synthesis.
        if len(usable_results) == 1 and len(results) == 1:
            return usable_results[0].output

        if not usable_results:
            failed = "; ".join(
                f"{r.title}: {r.error or r.status.value}" for r in results
            )
            return f"[No subtasks produced a usable result] {failed}"

        json_requested = user_requested_json(
            prompt=task.prompt,
            desired_output=task.desired_output,
            constraints=task.constraints,
            contract=task.contract,
        )

        # Legacy unbounded mode: budgeting disabled by configuration. One call
        # with the complete outputs, exceptions propagating exactly as before
        # (the orchestrator owns that fallback) — but the RESPONSE still passes
        # the Part J fail-closed cleaning, with a lossless deterministic join
        # as its fallback.
        if self.policy is None:
            sections = [(r.title, r.output) for r in usable_results]
            messages = build_synthesizer_messages(
                task, plan.synthesis_strategy, sections
            )
            text = self._complete(messages)
            return self._clean_response(
                text,
                lambda _reason: "\n\n".join(
                    f"## {t}\n\n{o}" for t, o in sections
                ),
                json_requested=json_requested,
            )

        policy = self.policy

        # Part F/G: bounded capsules, chunk-reducing any single oversized
        # output through bounded calls first.
        capsules: list[SynthesisCapsule] = []
        for r in usable_results:
            content: Optional[str] = None
            if len(r.output) > policy.max_capsule_chars:
                content = self._reduce_oversized_output(r, policy)
            capsules.append(
                build_capsule(
                    r, policy,
                    key_decisions=getattr(r, "key_decisions", ()),
                    content=content,
                )
            )
        capsules = _suppress_duplicates(capsules)

        # Part H: hierarchical reduction over bounded sections.
        sections: list[SynthesisCapsule] = list(capsules)
        depth = 0
        while True:
            messages = build_synthesizer_messages(
                task,
                plan.synthesis_strategy,
                [(c.title, c.render()) for c in sections],
            )
            if policy.call_within_budget(self._measure(messages)):
                return self._final_call(messages, capsules, json_requested)
            if depth >= policy.max_reduction_depth:
                return render_capsule_fallback(
                    capsules, reason="reduction depth limit reached",
                    max_chars=policy.max_fallback_chars,
                )
            try:
                groups = plan_groups(
                    [len(c.render()) for c in sections], policy
                )
            except ValueError:
                return render_capsule_fallback(
                    capsules,
                    reason="group-count limit reached",
                    max_chars=policy.max_fallback_chars,
                )
            if len(groups) >= len(sections):
                # Merging cannot shrink this any further within the policy.
                return render_capsule_fallback(
                    capsules,
                    reason="inputs could not be reduced within the input budget",
                    max_chars=policy.max_fallback_chars,
                )
            sections = [
                self._merge_group([sections[i] for i in group], idx, len(groups),
                                  policy)
                for idx, group in enumerate(groups, 1)
            ]
            depth += 1

    # ------------------------------------------------------------------ #
    # Bounded provider calls
    # ------------------------------------------------------------------ #
    def _complete(self, messages: list[LLMMessage]) -> str:
        """One measured provider call. The caller has already budget-checked."""
        self.client.cancel_token.check()
        if (
            self.policy is not None
            and len(self.last_call_input_chars) >= self.policy.max_calls_per_run
        ):
            raise SynthesisCallBudgetExceeded(
                "aggregate synthesis call budget reached"
            )
        self.last_call_input_chars.append(self._measure(messages))
        resp = self.client.complete(
            provider=self.agent.provider,
            model=self.agent.model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.agent.max_tokens,
        )
        return resp.text.strip()

    def _final_call(
        self,
        messages: list[LLMMessage],
        capsules: list[SynthesisCapsule],
        json_requested: bool = False,
    ) -> str:
        """The final synthesis call, with fail-closed response handling."""
        fallback = lambda reason: render_capsule_fallback(  # noqa: E731
            capsules,
            reason=reason,
            max_chars=self.policy.max_fallback_chars if self.policy else 0,
        )
        try:
            text = self._complete(messages)
        except CancelledError:
            raise
        except SynthesisCallBudgetExceeded:
            return fallback("the aggregate synthesis call budget was reached")
        except Exception:  # noqa: BLE001 - deterministic fallback (Part I)
            return fallback("the synthesis provider call failed")
        return self._clean_response(text, fallback, json_requested=json_requested)

    def _clean_response(
        self,
        text: str,
        fallback: Callable[[str], str],
        *,
        json_requested: bool = False,
    ) -> str:
        """Part J: never return the synthesizer's raw protocol envelope.

        A well-formed envelope is decoded to its content through the bounded
        explicit rules; a malformed envelope or control-JSON echo falls back
        deterministically. Plain prose passes through untouched.
        """
        if json_requested and _requested_json_data(text):
            return text
        envelope = (
            parse_worker_artifact(text)
            if looks_like_protocol_envelope(text)
            else None
        )
        if envelope is not None:
            flattened = decode_nested_envelope(render_artifact(envelope))
            if flattened.strip():
                text = flattened
        if json_requested and _requested_json_data(text):
            return text
        if (
            not text.strip()
            or is_malformed_protocol_envelope(text)
            or (
                looks_like_protocol_envelope(text)
                and parse_worker_artifact(text) is not None
            )
            or looks_like_orchestration_json(text)
        ):
            return fallback("the synthesis response was not a usable answer")
        return text

    # ------------------------------------------------------------------ #
    # Part G — oversized single result
    # ------------------------------------------------------------------ #
    def _reduce_oversized_output(
        self, result: SubTaskResult, policy: SynthesisBudgetPolicy
    ) -> str:
        """Reduce ONE oversized output via ordered, bounded chunk calls.

        Deterministic chunking preserves order and covers the whole text (the
        tail is never dropped). Each chunk is condensed by one bounded call;
        the compact digests are combined in order. If any call fails or a
        chunk cannot fit the budget, the deterministic explicitly-marked
        excerpt of the ORIGINAL output is used instead.
        """
        chunks = split_chunks(
            result.output, policy.content_budget, policy.max_chunks_per_result
        )
        deterministic = bound_text(result.output, policy.max_capsule_chars)
        digests: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            user = (
                f"This is part {index} of {len(chunks)} of the result for "
                f"'{result.title}'. Condense this part faithfully now.\n\n"
                f"{chunk}"
            )
            messages = [
                LLMMessage("system", _CHUNK_SYSTEM),
                LLMMessage("user", user),
            ]
            if not policy.call_within_budget(self._measure(messages)):
                # Never submit one over-budget chunk; degrade deterministically.
                return deterministic
            try:
                digest = self._complete(messages)
            except CancelledError:
                raise
            except Exception:  # noqa: BLE001 - deterministic per-result fallback
                return deterministic
            digest = digest.strip()
            if not digest or is_malformed_protocol_envelope(digest):
                return deterministic
            digests.append(
                bound_text(
                    digest,
                    max(1, policy.max_capsule_chars // max(1, len(chunks))),
                )
            )
        combined = "\n\n".join(
            f"(part {i} of {len(digests)}) {d}" for i, d in enumerate(digests, 1)
        )
        return combined if combined.strip() else deterministic

    # ------------------------------------------------------------------ #
    # Part H — one bounded group merge
    # ------------------------------------------------------------------ #
    def _merge_group(
        self,
        members: list[SynthesisCapsule],
        part_idx: int,
        part_total: int,
        policy: SynthesisBudgetPolicy,
    ) -> SynthesisCapsule:
        """Merge one ordered group into a bounded intermediate node.

        A single-member group passes through untouched (no call). A group the
        budget cannot accommodate, or whose call fails, is joined
        deterministically and bounded EXPLICITLY — content loss is declared,
        never silent, and the full capsules remain available to the fallback.
        """
        if len(members) == 1:
            return members[0]
        title = f"{members[0].title} — {members[-1].title}"
        merged_ids = tuple(m.subtask_id for m in members)
        merged_warnings = tuple(
            w for m in members for w in m.warnings
        )

        def deterministic() -> SynthesisCapsule:
            joined = "\n\n".join(
                f"## {m.title}\n\n{m.render()}" for m in members
            )
            return SynthesisCapsule(
                subtask_id="+".join(merged_ids),
                title=title,
                status="merged",
                content=bound_text(joined, policy.max_intermediate_chars),
                depends_on=merged_ids,
                warnings=merged_warnings,
            )

        body = "\n\n".join(
            f"--- Section [{m.title}] ---\n{m.render()}" for m in members
        )
        user = (
            f"These are the sections for part {part_idx} of {part_total} of a "
            f"larger answer.\n\n{body}\n\n"
            "Produce the merged, condensed part now."
        )
        messages = [
            LLMMessage("system", _GROUP_SYSTEM),
            LLMMessage("user", user),
        ]
        if not policy.call_within_budget(self._measure(messages)):
            return deterministic()
        try:
            text = self._complete(messages)
        except CancelledError:
            raise
        except Exception:  # noqa: BLE001 - lossless-enough local fallback
            return deterministic()
        text = text.strip()
        if not text or is_malformed_protocol_envelope(text):
            return deterministic()
        return SynthesisCapsule(
            subtask_id="+".join(merged_ids),
            title=title,
            status="merged",
            content=bound_text(text, policy.max_intermediate_chars),
            depends_on=merged_ids,
            warnings=merged_warnings,
        )

    def _measure(self, messages: list[LLMMessage]) -> int:
        """Measure the exact provider input known to the active client layer."""
        return self.client.measure_input_chars(messages)


def _suppress_duplicates(
    capsules: list[SynthesisCapsule],
) -> list[SynthesisCapsule]:
    """Replace EXACT duplicate contents with a short reference (safe by identity).

    Only byte-identical content is suppressed — near-duplicates are left alone
    so nothing subtly different is erased.
    """
    seen: dict[str, str] = {}
    out: list[SynthesisCapsule] = []
    for capsule in capsules:
        key = capsule.content
        if key.strip() and key in seen:
            out.append(
                SynthesisCapsule(
                    subtask_id=capsule.subtask_id,
                    title=capsule.title,
                    status=capsule.status,
                    content=(
                        f"(identical to the output of '{seen[key]}' — "
                        "content not repeated)"
                    ),
                    key_decisions=capsule.key_decisions,
                    depends_on=capsule.depends_on,
                    warnings=capsule.warnings,
                    agent_name=capsule.agent_name,
                    truncated=capsule.truncated,
                    original_chars=capsule.original_chars,
                )
            )
            continue
        if key.strip():
            seen[key] = capsule.title or capsule.subtask_id
        out.append(capsule)
    return out
