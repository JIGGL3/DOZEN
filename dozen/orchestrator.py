"""The Orchestrator: the single entry point that behaves like one model.

Pipeline per request:

    plan -> (direct answer?) -> execute DAG -> verify/repair
         -> deterministic finalization (artifact assembly / ordered sections,
            with model synthesis only as an OPTIONAL polish pass)

`complex` subtasks are recursively re-orchestrated up to ``config.max_depth``
(Dozen calling itself).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Union

from .agent_pool import AgentPool, AgentSpec, default_pool
from .artifact_results import (
    DEFAULT_RESULT_POLICY,
    AssembledDeliverable,
    AssemblyStatus,
    RecursiveArtifactOutput,
    SubtaskCollectionResult,
    assemble_deliverable,
    merge_subtask_collections,
)
from .cancellation import NULL_TOKEN, CancelToken, CancelledError
from .config import OrchestratorConfig
from .executor import Executor
from .finalization import (
    CODE_ONLY_CONTRACT_LOST_DIAGNOSTIC,
    CODE_ONLY_NO_ARTIFACTS_DIAGNOSTIC,
    FinalizationDecision,
    FinalizationMode,
    FinalizationPolicy,
    NO_USABLE_RESULT_SENTINEL,
    OrderedSection,
    assert_code_only_decision,
    assert_no_model_synthesis,
    build_ordered_sections,
    code_only_contract_intact,
    decide_finalization,
    finalize_artifact_delivery,
    render_ordered_metadata,
    render_ordered_sections,
    user_requested_polish,
    user_requested_protocol_content,
)
from .synthesis_guard import MODEL_BASED_SYNTHESIS_ENABLED
from .intent import (
    RequestIntent,
    demands_code_only,
    resolve_contract,
    validate_final_answer_against_contract,
)
from .llm_client import LLMClient
from .models import OrchestrationResult, Plan, SubTask, SubTaskResult, Task, TaskStatus
from .planner import Planner
from .presentation import (
    FinalPresentation,
    PresentationKind,
    render_final_presentation,
    render_results_fallback,
    sanitize_diagnostic,
    user_requested_json,
)
from .router import Router
from .synthesis_scaling import FALLBACK_HEADLINE
from .synthesizer import Synthesizer
from .validation import validate_model_response
from .verifier import Verifier


class Orchestrator:
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        pool: Optional[AgentPool] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.client = client or LLMClient(mock=True)
        self.pool = pool or default_pool()
        self.config = config or OrchestratorConfig()
        self._cancel: CancelToken = NULL_TOKEN
        # Structured progress sink (Live Execution Log). No-op until set per run.
        self._on_event: Callable[[dict], None] = lambda _e: None
        # Content-free per-run delivery telemetry: whether a synthesizer call
        # actually happened. A code-only run must finish with this false.
        self._model_synthesis_invoked: bool = False
        self._delivery_finalization_mode: str = ""
        self._delivery_artifact_plan_present: bool = False
        self._delivery_artifact_assembly: Optional[AssembledDeliverable] = None

        if len(self.pool) == 0:
            raise ValueError("Agent pool is empty; add at least one AgentSpec.")

        # Resolve which agents play the planner/router/verifier/synthesizer roles.
        self._planner_agent = self._resolve_role(self.config.planner_agent, prefer="reasoning")
        self._router_agent = self._resolve_role(self.config.router_agent, prefer="reasoning")
        self._verifier_agent = self._resolve_role(self.config.verifier_agent, prefer="reasoning")
        self._synth_agent = self._resolve_role(self.config.synthesizer_agent, prefer="writing")

        self.planner = Planner(self.client, self._planner_agent, self.pool)
        self.router = Router(
            self.client, self.pool,
            router_agent=self._router_agent,
            use_llm_router=self.config.use_llm_router,
        )
        self.verifier = Verifier(
            self.client, self._verifier_agent, pass_threshold=self.config.pass_threshold
        )
        self.synthesizer = Synthesizer(
            self.client, self._synth_agent,
            max_input_chars=self.config.max_synthesis_input_chars,
        )
        # Phase 4F: THE authoritative finalization policy. Deterministic by
        # default; only the trusted config flag can enable optional polish.
        self.finalization_policy = FinalizationPolicy(
            allow_model_polish=bool(
                getattr(self.config, "enable_model_polish", False)
            ),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        task: Union[str, Task],
        cancel: Optional[CancelToken] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> OrchestrationResult:
        """Solve a task end-to-end. Accepts a raw prompt string or a Task.

        Pass a :class:`CancelToken` to make the run abortable: tripping it from
        another thread (e.g. the Stop button) unwinds planning, execution and
        synthesis at the next checkpoint.

        Pass ``on_event`` (a callable taking a dict) to receive STRUCTURED,
        real-time progress events — plan started, subtask dispatched/received,
        synthesizing — for a Live Execution Log. It is invoked from worker
        threads, so the sink must be thread-safe.
        """
        if isinstance(task, str):
            task = Task(prompt=task)

        self._cancel = cancel or NULL_TOKEN
        self._on_event = on_event or (lambda _e: None)
        # Reset per-run delivery telemetry (content-free). A synthesizer call
        # sets this true; a code-only run must leave it false.
        self._model_synthesis_invoked = False
        self._delivery_finalization_mode = ""
        self._delivery_artifact_plan_present = False
        self._delivery_artifact_assembly = None
        # Every model call flows through the client, so giving it the token is
        # enough to make planner/router/verifier/synthesizer/workers cancelable.
        self.client.cancel_token = self._cancel
        # Surface JSON re-asks in the live log. A model answering a structured
        # request conversationally is otherwise invisible: the run just looks
        # slow, then fails. Content-free — only the reply's shape is reported.
        try:
            self.client.on_json_format_retry = self._on_json_format_retry
        except Exception:  # noqa: BLE001 - a client without the hook is fine
            pass

        # Phase 3: THE orchestration boundary. Every entry path — library
        # callers, the FastAPI server, tests — passes through here, so the
        # deliverable contract is resolved exactly once, for everyone. An
        # explicitly supplied contract is preserved as-is (never re-resolved).
        if task.contract is None:
            task.contract = resolve_contract(
                prompt=task.prompt,
                desired_output=task.desired_output,
                constraints=task.constraints,
            )
        self._log(f"[intent] {task.contract.intent.value} "
                  f"(code_required={task.contract.code_required}, "
                  f"code_only={task.contract.code_only})")
        self._emit(phase="intent", status="log", icon="🎯",
                   message=f"Request intent: {task.contract.intent.value} — "
                           f"{task.contract.deliverable}")

        try:
            # Cancellation owns terminal semantics even when the integrity
            # guard would otherwise fail closed. Keeping both checks inside
            # this try/finally also guarantees cleanup on the early return.
            self._cancel.check()
            guarded = self._root_code_only_guard(task)
            if guarded is not None:
                return guarded
            return self._orchestrate(task, depth=0)
        except CancelledError as exc:
            reason = sanitize_diagnostic(exc)
            self._emit(phase="cancelled", status="cancelled", icon="🛑",
                       message="Run cancelled by user.")
            return OrchestrationResult(
                task_id=task.id, final_answer="", depth=0,
                error=f"Cancelled: {reason}",
                artifact_assembly=self._delivery_artifact_assembly,
                finalization_mode=self._delivery_finalization_mode,
                intent=task.contract.intent.value if task.contract else "",
                code_only=bool(getattr(task.contract, "code_only", False)),
                artifact_plan_present=self._delivery_artifact_plan_present,
                model_synthesis_invoked=self._model_synthesis_invoked,
            )
        finally:
            self.client.cancel_token = NULL_TOKEN
            try:
                self.client.on_json_format_retry = None
            except Exception:  # noqa: BLE001
                pass
            self._on_event = lambda _e: None

    def _on_json_format_retry(self, attempt: int, total: int, shape: str) -> None:
        """Report one JSON re-ask to the live log (content-free)."""
        self._log(f"[json] non-JSON reply ({shape}); re-asking {attempt + 1}/{total}")
        self._emit(
            phase="format-retry", status="log", icon="🔁",
            message=f"A model replied without valid JSON ({shape}); "
                    f"asking again ({attempt + 1} of {total}).",
        )

    def _emit(self, **event) -> None:
        """Send one structured progress event to the sink (never raises)."""
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass

    def _root_code_only_guard(
        self, task: Task
    ) -> Optional[OrchestrationResult]:
        """Fail closed when the user's strong code-only signal was dropped."""
        if not (
            demands_code_only(task.prompt, task.desired_output, task.constraints)
            and not code_only_contract_intact(task.contract)
        ):
            return None
        self._log("[contract] code-only obligation lost before finalization "
                  "— failing closed")
        self._emit(phase="contract", status="error", icon="❌",
                   message="Code-only delivery contract not preserved.")
        loss_decision = FinalizationDecision(
            mode=FinalizationMode.ARTIFACT_ASSEMBLY,
            reason=CODE_ONLY_CONTRACT_LOST_DIAGNOSTIC,
            artifacts_required=True,
            complete_assembly_required=True,
            prose_allowed=False,
            model_synthesis_allowed=False,
            code_only=True,
        )
        assert_code_only_decision(loss_decision)
        self._delivery_finalization_mode = loss_decision.mode.value
        self._log(f"[finalize] (depth 0) mode={loss_decision.mode.value} "
                  "(code-only contract loss)")
        return OrchestrationResult(
            task_id=task.id,
            final_answer=finalize_artifact_delivery(
                None, loss_decision, policy=self.finalization_policy
            ),
            depth=0,
            error=CODE_ONLY_CONTRACT_LOST_DIAGNOSTIC,
            finalization_mode=loss_decision.mode.value,
            intent=RequestIntent.IMPLEMENT.value,
            code_only=True,
        )

    # ------------------------------------------------------------------ #
    # Core recursive pipeline
    # ------------------------------------------------------------------ #
    def _orchestrate(self, task: Task, depth: int) -> OrchestrationResult:
        self._cancel.check()
        self._log(f"[plan] (depth {depth}) {task.prompt[:80]}")
        self._emit(
            phase="plan", status="start", icon="⏳", depth=depth,
            message=(
                "Analyzing user prompt to generate orchestration plan..."
                if depth == 0
                else "Analyzing sub-problem to plan a sub-decomposition..."
            ),
        )

        try:
            plan = self.planner.plan(task, depth, self.config.max_depth)
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            reason = sanitize_diagnostic(exc)
            self._emit(phase="plan", status="error", icon="❌",
                       message=f"Planning failed: {reason}")
            return OrchestrationResult(
                task_id=task.id, final_answer="", depth=depth,
                error=f"Planning failed: {reason}",
                intent=task.contract.intent.value if task.contract else "",
                code_only=bool(getattr(task.contract, "code_only", False)),
                model_synthesis_invoked=self._model_synthesis_invoked,
            )

        if depth == 0:
            self._delivery_artifact_plan_present = plan.artifact_plan is not None

        # Direct answer: planner decided no decomposition is needed.
        if plan.is_direct():
            # Unbypassable code-only guard at the finalization boundary: a
            # code-only (artifact-owed) contract can NEVER be satisfied by a
            # direct prose answer. A direct answer means the planner produced no
            # artifact project, so there is nothing to assemble — deliver the ONE
            # clean deterministic non-delivery result here, never the prose, never
            # a refusal, never a synthesizer. This closes the exact live bypass
            # where a "code only" request returned an architecture essay with a
            # contract-violation banner instead of a clean artifact failure.
            direct_decision = (
                decide_finalization(
                    contract=task.contract,
                    artifact_plan=None,
                    assembly=None,
                    execution_scope=task.execution_scope,
                    scope_output_required=task.execution_scope_output_required,
                    policy=self.finalization_policy,
                )
                if depth == 0 and task.contract is not None
                else None
            )
            if (
                direct_decision is not None
                and direct_decision.mode is FinalizationMode.ARTIFACT_ASSEMBLY
            ):
                assert_code_only_decision(direct_decision)
                self._delivery_finalization_mode = direct_decision.mode.value
                self._log(
                    f"[finalize] (depth {depth}) mode={direct_decision.mode.value} "
                    "(direct answer rejected for an artifact-owed contract)"
                )
                self._emit(
                    phase="finalize", status="start", icon="🧩", depth=depth,
                    message="Rendering the assembled deliverable deterministically "
                            "(no model synthesis).",
                )
                final = finalize_artifact_delivery(
                    None, direct_decision, policy=self.finalization_policy
                )
                final, cleanup_error, presentation = self._normalize_final(
                    task, final, [], assembly=None
                )
                presentation = _with_text(presentation, final)
                return OrchestrationResult(
                    task_id=task.id, final_answer=final, plan=plan,
                    depth=depth, direct=True,
                    error=CODE_ONLY_NO_ARTIFACTS_DIAGNOSTIC,
                    presentation=presentation,
                    finalization_mode=direct_decision.mode.value,
                    intent=task.contract.intent.value,
                    code_only=direct_decision.code_only,
                    artifact_plan_present=False,
                    model_synthesis_invoked=self._model_synthesis_invoked,
                )
            answer = plan.direct_answer or ""
            self._log(f"[direct] (depth {depth}) answered without decomposition")
            self._emit(phase="plan", status="done", icon="🤖",
                       message="Plan complete — answering directly (no decomposition needed).")
            answer, cleanup_error, presentation = self._normalize_final(
                task, answer, []
            )
            answer, violation, warnings = self._enforce_contract(task, answer, depth)
            presentation = _with_text(presentation, answer)
            return OrchestrationResult(
                task_id=task.id, final_answer=answer, plan=plan,
                depth=depth, direct=True, error=violation or cleanup_error,
                warnings=warnings, presentation=presentation,
                intent=task.contract.intent.value if task.contract else "",
                code_only=bool(getattr(task.contract, "code_only", False)),
                model_synthesis_invoked=self._model_synthesis_invoked,
            )

        self._emit(
            phase="plan", status="done", icon="🤖", depth=depth,
            subtasks=len(plan.subtasks),
            message=f"Plan generated. Task divided into {len(plan.subtasks)} subtask(s).",
        )

        self._log(
            f"[plan] {len(plan.subtasks)} subtask(s); "
            f"strategy: {plan.synthesis_strategy[:80]}"
        )

        executor = Executor(
            client=self.client,
            pool=self.pool,
            router=self.router,
            verifier=self.verifier,
            config=self.config,
            recurse_fn=self._recurse,
            log=self._log,
            cancel=self._cancel,
            on_event=self._on_event,
        )
        results = executor.run(task, plan, depth)

        # Phase 4C: combine what the workers actually produced BEFORE synthesis,
        # so the final answer presents an assembled deliverable instead of a
        # concatenation of possibly conflicting file blobs. Runs with no artifact
        # plan gets None here and behaves exactly as before. Once a plan owes
        # artifacts, assembly is mandatory even when every scoped worker ignored
        # the typed envelope: absence of evidence is an incomplete deliverable,
        # not permission to fall back to legacy success.
        assembly = self._assemble(plan, results)
        collection = self._collect_for_parent(task, plan, results)
        if depth == 0:
            self._delivery_artifact_assembly = assembly

        # Phase 4F: deterministic ordered finalization is THE default delivery
        # mechanism. The mode is selected from trusted runtime information only
        # (contract, validated manifest, scope, explicit preference) — workers
        # and providers can never choose it. Model synthesis survives solely as
        # the optional polish pass with a mandatory deterministic fallback.
        self._cancel.check()
        decision = decide_finalization(
            contract=task.contract,
            artifact_plan=plan.artifact_plan,
            assembly=assembly,
            execution_scope=task.execution_scope,
            scope_output_required=task.execution_scope_output_required,
            polish_requested=user_requested_polish(
                prompt=task.prompt,
                desired_output=task.desired_output,
                constraints=task.constraints,
            ),
            policy=self.finalization_policy,
        )
        # Fail closed if a code-only decision is anything but pure artifact
        # assembly (no ordered sections, no synthesis, no prose). This is a
        # structural invariant, enforced at the boundary rather than trusted.
        assert_code_only_decision(decision)
        # Phase 4G: fail closed if any decision would permit model-based
        # synthesis while the authoritative invariant disables it. This is the
        # root-finalization enforcement of MODEL_BASED_SYNTHESIS_ENABLED.
        assert_no_model_synthesis(decision)
        if depth == 0:
            self._delivery_finalization_mode = decision.mode.value
        self._log(
            f"[finalize] (depth {depth}) mode={decision.mode.value} "
            f"({decision.reason}); {len(results)} result(s)"
        )
        if depth == 0:
            # Content-free delivery telemetry (Part N): prove the run finalized
            # deterministically with no model synthesis. Carries no prompt or
            # response text — only the mode and the invariant outcome.
            self._log(
                f"[delivery] finalization_mode={decision.mode.value} "
                f"model_based_synthesis_enabled={MODEL_BASED_SYNTHESIS_ENABLED} "
                f"model_synthesis_invoked={self._model_synthesis_invoked} "
                f"synthesis_task_planned=false"
            )
            self._emit(
                phase="delivery", status="log", icon="🔒",
                message=(
                    f"Deterministic delivery: mode={decision.mode.value}, "
                    "no model synthesis."
                ),
            )

        section_warnings: list[str] = []
        if decision.mode is FinalizationMode.ARTIFACT_ASSEMBLY:
            self._emit(
                phase="finalize", status="start", icon="🧩", depth=depth,
                message="Rendering the assembled deliverable deterministically "
                        "(no model synthesis).",
            )
            final = finalize_artifact_delivery(
                assembly, decision, policy=self.finalization_policy
            )
        else:
            sections = build_ordered_sections(
                plan, results,
                policy=self.finalization_policy,
                json_requested=user_requested_json(
                    prompt=task.prompt,
                    desired_output=task.desired_output,
                    constraints=task.constraints,
                    contract=task.contract,
                ),
                protocol_content_requested=user_requested_protocol_content(
                    prompt=task.prompt,
                    desired_output=task.desired_output,
                    constraints=task.constraints,
                ),
            )
            for section in sections:
                for note in section.internal_notes:
                    section_warnings.append(
                        f"section '{section.title or section.subtask_id}': {note}"
                    )
            self._emit(
                phase="finalize", status="start", icon="🧩", depth=depth,
                message=f"Stitching {len(sections)} section(s) in plan order "
                        "(deterministic).",
            )
            final = render_ordered_sections(
                sections, policy=self.finalization_policy
            )
            if decision.mode is FinalizationMode.OPTIONAL_MODEL_SYNTHESIS:
                final = self._polish(task, plan, results, sections, final, depth)

        final, cleanup_error, presentation = self._normalize_final(
            task, final, results, assembly=assembly
        )

        # A code-only project that could not be completed already IS the clean
        # deterministic failure result; judging that diagnostic against the
        # code contract would only bury it under a second, redundant notice.
        skip_contract = (
            decision.code_only and assembly is not None and not assembly.complete
        )
        if skip_contract:
            violation, warnings = "", []
        else:
            final, violation, warnings = self._enforce_contract(task, final, depth)

        assembly_error, assembly_warnings = self._enforce_assembly(assembly, depth)
        warnings.extend(assembly_warnings)
        warnings.extend(section_warnings)
        presentation = _with_text(presentation, final)

        return OrchestrationResult(
            task_id=task.id,
            final_answer=final,
            plan=plan,
            subtask_results=results,
            depth=depth,
            error=violation or cleanup_error or assembly_error,
            warnings=warnings,
            artifact_assembly=assembly,
            artifact_collection=collection,
            presentation=presentation,
            finalization_mode=decision.mode.value,
            intent=task.contract.intent.value if task.contract else "",
            code_only=decision.code_only,
            artifact_plan_present=plan.artifact_plan is not None,
            model_synthesis_invoked=self._model_synthesis_invoked,
        )

    # ------------------------------------------------------------------ #
    # Phase 4C: artifact assembly
    # ------------------------------------------------------------------ #
    def _assemble(
        self, plan: Plan, results
    ) -> Optional[AssembledDeliverable]:
        """Deterministically assemble an artifact-bearing run's deliverable.

        Returns ``None`` only when the plan owes no artifacts. A scoped worker
        that ignores the typed contract is represented by an empty collection,
        so assembly reports the missing deliverable after bounded repair.
        """
        if plan.artifact_plan is None:
            return None
        collections = [
            result.artifact_collection
            for result in results
            if result.artifact_collection is not None
        ]
        assembly = assemble_deliverable(
            plan.artifact_plan, collections, policy=DEFAULT_RESULT_POLICY
        )
        summary = assembly.summary()
        self._log(
            f"[artifacts] assembly {summary['status']}: "
            f"{summary['assembled']}/{len(assembly.expected_required_artifact_ids)} "
            f"required, {summary['conflicts']} conflict(s)"
        )
        self._emit(
            phase="artifacts", status="log",
            icon="📦" if assembly.complete else "⚠️",
            message=(
                f"Artifact assembly: {summary['status']} — "
                f"{summary['assembled']} artifact(s) assembled, "
                f"{summary['missing_required']} required missing, "
                f"{summary['conflicts']} conflict(s)."
            ),
        )
        return assembly

    def _collect_for_parent(
        self, task: Task, plan: Plan, results
    ) -> Optional[SubtaskCollectionResult]:
        """Typed candidates a scoped RECURSIVE run owes its parent.

        Only the designated output-producing delegation carries a collection
        (the executor scopes no one else), so intermediate research/design/review
        children can never leak artifacts into the parent's assembly.
        """
        if plan.artifact_plan is not None or task.execution_scope is None:
            return None
        if not task.execution_scope_output_required:
            return None
        collections = [
            result.artifact_collection
            for result in results
            if result.artifact_collection is not None
        ]
        if not collections:
            return None
        return merge_subtask_collections(
            collections, task.execution_scope, policy=DEFAULT_RESULT_POLICY
        )

    def _enforce_assembly(
        self, assembly: Optional[AssembledDeliverable], depth: int
    ) -> tuple[str, list[str]]:
        """Never report an incomplete or conflicted deliverable as a success.

        Judged once, at depth 0, like the Phase 3 contract guard. Conflicts are
        reported, never resolved: no merge, no winner, no model adjudication.
        """
        if assembly is None or depth > 0:
            return "", []
        warnings: list[str] = []
        for duplicate in assembly.duplicates:
            warnings.append(
                f"artifact {duplicate.artifact_id!r} was submitted more than once "
                "with identical content (ownership should be unique)"
            )
        for artifact_id in assembly.missing_optional_artifact_ids:
            warnings.append(f"optional artifact {artifact_id!r} was not produced")
        for warning in assembly.warnings:
            warnings.append(warning)
        if assembly.status is AssemblyStatus.COMPLETE:
            return "", warnings

        reason = assembly.problem_summary()
        label = assembly.status.value
        self._log(f"[artifacts] deliverable {label}: {reason}")
        self._emit(phase="artifacts", status="error", icon="❌",
                   message=f"Artifact deliverable {label}: {reason}")
        return f"Artifact assembly {label}: {reason}", warnings

    # ------------------------------------------------------------------ #
    # Phase 4F Part J: optional model polish (never a required stage)
    # ------------------------------------------------------------------ #
    def _polish(
        self,
        task: Task,
        plan: Plan,
        results: list[SubTaskResult],
        sections: tuple[OrderedSection, ...],
        deterministic: str,
        depth: int,
    ) -> str:
        """Optionally polish the ordered sections through the hierarchical
        synthesizer. The deterministic ordered result already exists and is
        the source of truth: any refusal, malformed reply, provider error,
        queue rejection or budget failure returns it UNCHANGED. Cancellation
        remains cancellation. The model receives the CLEANED section bodies —
        never raw internal envelopes and never complete artifact bodies
        (polish mode is unreachable for artifact runs).
        """
        # Phase 4G: hard stop. Model-based synthesis is eliminated from every
        # production path; while the authoritative invariant is False this
        # method never contacts the synthesizer and always returns the
        # deterministic ordered result. ``decide_finalization`` already never
        # selects the optional-synthesis mode, so this branch is unreachable in
        # production — the guard keeps it structurally impossible regardless.
        if not MODEL_BASED_SYNTHESIS_ENABLED:
            return deterministic
        usable = [section for section in sections if section.usable]
        if not usable or deterministic.startswith(NO_USABLE_RESULT_SENTINEL):
            return deterministic
        decisions_by_id = {
            r.subtask_id: list(getattr(r, "key_decisions", []) or [])
            for r in results
        }
        shim = [
            SubTaskResult(
                subtask_id=section.subtask_id,
                title=section.title,
                status=TaskStatus.COMPLETED,
                output=section.content,
                key_decisions=decisions_by_id.get(section.subtask_id, []),
            )
            for section in usable
        ]
        self._log(f"[synthesize] (depth {depth}) optional polish over "
                  f"{len(shim)} ordered section(s)")
        self._emit(phase="synthesize", status="start", icon="🔄", depth=depth,
                   message="Optional model polish over the ordered sections...")
        # A synthesizer call is about to happen: record it for delivery
        # telemetry. Only OPTIONAL_MODEL_SYNTHESIS runs ever reach this point,
        # so a code-only or artifact run leaves the flag false.
        self._model_synthesis_invoked = True
        try:
            polished = self.synthesizer.synthesize(task, plan, shim)
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - mandatory deterministic fallback
            self._log(
                f"[synthesize] polish failed ({sanitize_diagnostic(exc)}); "
                "kept the deterministic ordered sections"
            )
            return deterministic
        text = (polished or "").strip()
        if (
            not text
            or text.startswith(FALLBACK_HEADLINE)
            or text.startswith(NO_USABLE_RESULT_SENTINEL)
            # A refusal is not a polished answer; the ordered result survives.
            or not validate_model_response(text).ok
        ):
            # The synthesizer degraded internally; OUR ordered result is the
            # authoritative deterministic fallback.
            return deterministic
        metadata = render_ordered_metadata(
            usable, policy=self.finalization_policy
        )
        return text if not metadata else f"{text}\n\n{metadata}"

    def _normalize_final(
        self, task: Task, final: str, results, assembly=None
    ) -> tuple[str, str, FinalPresentation]:
        """THE final rendering boundary for every final-answer path (Phase 6).

        Delegates the decisions to :func:`render_final_presentation` — envelope
        flattening, bounded nested decode, fail-closed handling of malformed
        protocol payloads, control-JSON replacement and the JSON-deliverable
        contract — and maps a DIAGNOSTIC presentation back onto the historical
        ``(final, error)`` semantics every caller and test already relies on.
        """
        presentation = render_final_presentation(
            final,
            results,
            assembly=assembly,
            json_requested=user_requested_json(
                prompt=task.prompt,
                desired_output=task.desired_output,
                constraints=task.constraints,
                contract=task.contract,
            ),
            protocol_content_requested=user_requested_protocol_content(
                prompt=task.prompt,
                desired_output=task.desired_output,
                constraints=task.constraints,
            ),
        )
        for note in presentation.diagnostics:
            self._log(f"[final] {note}")
        error = ""
        if presentation.kind is PresentationKind.DIAGNOSTIC:
            error = (
                presentation.diagnostics[-1]
                if presentation.diagnostics
                else "Final answer validation failed."
            )
            self._emit(phase="result_validation", status="error", icon="❌",
                       message=error)
        return presentation.text, error, presentation

    def _enforce_contract(
        self, task: Task, final: str, depth: int
    ) -> tuple[str, str, list[str]]:
        """Phase 3 minimal final guard: never present an obvious mode violation
        as a successful completion.

        Applies at depth 0 ONLY. A recursive child's answer is an intermediate
        feeding synthesis — a child that legitimately researches or outlines
        part of an implementation must not be failed for containing no code.
        The user-facing deliverable is judged once, at the top.

        Returns ``(answer, error, warnings)``. The produced content is NOT
        discarded — it may still be useful — but a fatal result carries an
        explicit error and prefixed notice, while indeterminate evidence is
        retained as a structured advisory.
        """
        if task.contract is None or depth > 0:
            return final, "", []
        check = validate_final_answer_against_contract(task.contract, final)
        warnings = list(check.advisory)
        for warning in warnings:
            self._log(f"[contract] advisory: {warning}")
            self._emit(phase="contract", status="warning", icon="⚠️",
                       message=f"Deliverable contract advisory: {warning}")
        if check.ok:
            return final, "", warnings

        reason = "; ".join(check.fatal)
        self._log(f"[contract] final answer violates the deliverable contract: {reason}")
        self._emit(phase="contract", status="error", icon="⚠️",
                   message=f"Deliverable contract not met: {reason}")
        notice = (
            "> ⚠️ **Deliverable contract not met.** This was a "
            f"{task.contract.intent.value.upper()} request "
            f"({task.contract.deliverable}), but the run did not produce it: "
            f"{reason}. The material below is what the models returned — treat "
            "it as supporting analysis, not the requested deliverable, and "
            "re-run the task.\n\n---\n\n"
        )
        return (
            notice + final,
            f"Deliverable contract violation: {reason}",
            warnings,
        )

    def _recurse(self, subtask: SubTask, sub_task: Task, depth: int) -> object:
        """Callback handed to the executor for `complex` subtasks.

        A scoped child hands back its TYPED artifact candidates alongside the
        text, so the parent assembles what the designated producer actually
        emitted — synthesis prose can never invent an artifact.
        """
        result = self._orchestrate(sub_task, depth)
        if result.error:
            raise RuntimeError(result.error)
        if result.artifact_collection is not None:
            return RecursiveArtifactOutput(
                text=result.final_answer, collection=result.artifact_collection
            )
        return result.final_answer

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_role(self, name: Optional[str], prefer: str) -> AgentSpec:
        if name is not None:
            if name not in self.pool:
                raise ValueError(f"Role agent {name!r} not found in pool.")
            return self.pool.get(name)
        # Pick the agent that best combines the preferred capability and tier.
        agents = self.pool.enabled_agents() or self.pool.all_agents()
        return max(
            agents,
            key=lambda a: a.capability_score(prefer) + a.tier / 10.0,
        )

    @staticmethod
    def _fallback_stitch(results) -> str:
        """Deterministic, envelope-free stitch used when synthesis fails."""
        return render_results_fallback(results)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, file=sys.stderr)


def _with_text(presentation: FinalPresentation, text: str) -> FinalPresentation:
    """Rebind the presentation to the final (possibly contract-annotated) text."""
    if presentation.text == text:
        return presentation
    return FinalPresentation(
        kind=presentation.kind, text=text, diagnostics=presentation.diagnostics
    )
