"""Executor: runs the subtask DAG.

Responsibilities:
* Schedule subtasks in dependency order, running independent ones in parallel.
* For each subtask: route -> run worker -> verify -> repair (with feedback) ->
  escalate to a stronger agent if it keeps failing.
* Delegate `complex` subtasks back to the orchestrator for recursive
  decomposition (Dozen calling itself).
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Optional

from .agent_pool import AgentPool, AgentSpec
from .artifact_repair import (
    DEFAULT_REPAIR_POLICY,
    ArtifactRepairPolicy,
    ArtifactRepairRequest,
    ArtifactRepairState,
    RepairDecision,
    begin_repair_state,
    collect_repair_artifacts,
    derive_repair_scope,
    merge_repair_collection,
    plan_repair,
    submitted_content_hashes,
)
from .artifact_results import (
    DEFAULT_RESULT_POLICY,
    RecursiveArtifactOutput,
    ResultPolicy,
    SubtaskCollectionResult,
    collect_subtask_artifacts,
)
from .artifacts import ArtifactWorkPlan
from .cancellation import NULL_TOKEN, CancelToken, CancelledError
from .config import OrchestratorConfig
from .decomposition import (
    ArtifactExecutionScope,
    derive_execution_scope,
    estimate_scope_output_chars,
    provider_output_budget_chars,
    validate_recursive_scope_assignment,
)
from .finalization import (
    FATAL_DELIVERY_CODES,
    WorkerDeliveryCode,
    classify_scoped_delivery,
    is_malformed_prose_envelope,
    parse_prose_envelope,
)
from .llm_client import LLMClient
from .models import Plan, SubTask, SubTaskResult, Task, TaskStatus
from .presentation import sanitize_diagnostic
from .prompts import build_worker_messages
from .router import Router
from .synthesis_guard import forbidden_synthesis_reason
from .validation import (
    ResponseValidation,
    artifact_confidence,
    clip_text,
    decode_nested_envelope,
    is_malformed_protocol_envelope,
    looks_like_orchestration_json,
    parse_worker_artifact,
    render_artifact,
    validate_model_response,
)
from .verifier import Verifier

# A callback the orchestrator supplies so the executor can recurse on a
# `complex` subtask: (subtask, parent_task, depth) -> final_output_text, or a
# RecursiveArtifactOutput carrying that text PLUS the child's typed artifact
# candidates (Phase 4C). A legacy callback returning a plain string still works.
RecurseFn = Callable[[SubTask, Task, int], object]

# Optional logging callback: (message) -> None
LogFn = Callable[[str], None]

# The router already estimates input tokens as characters // 4.  Use the same
# conservative conversion for the dispatch preflight, reserving the agent's
# configured output-token allowance from its total context window.
_INPUT_CHARS_PER_TOKEN = 4


class Executor:
    def __init__(
        self,
        client: LLMClient,
        pool: AgentPool,
        router: Router,
        verifier: Verifier,
        config: OrchestratorConfig,
        recurse_fn: Optional[RecurseFn] = None,
        log: Optional[LogFn] = None,
        cancel: Optional[CancelToken] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        result_policy: Optional[ResultPolicy] = None,
        repair_policy: Optional[ArtifactRepairPolicy] = None,
    ) -> None:
        self.client = client
        self.pool = pool
        self.router = router
        self.verifier = verifier
        self.config = config
        self.recurse_fn = recurse_fn
        self.log = log or (lambda _msg: None)
        self.cancel = cancel or NULL_TOKEN
        self.on_event = on_event or (lambda _e: None)
        # The ONE authoritative source of produced-artifact bounds (Phase 4C).
        self.result_policy = result_policy or DEFAULT_RESULT_POLICY
        # ...and of targeted-repair bounds (Phase 4E). The number of worker CALLS
        # stays governed solely by ``config.max_repair_attempts``.
        self.repair_policy = repair_policy or DEFAULT_REPAIR_POLICY

    def _emit(self, **event) -> None:
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass

    # ------------------------------------------------------------------ #
    def run(self, task: Task, plan: Plan, depth: int) -> list[SubTaskResult]:
        subtasks = plan.by_id()
        results: dict[str, SubTaskResult] = {}
        results_lock = threading.Lock()

        # Phase 4B: derive each subtask's bounded artifact scope ONCE, up
        # front, from the validated artifact plan. Subtasks with no assigned
        # packages (and every plan without an artifact plan) get None and run
        # exactly as before. The DAG scheduler below is untouched.
        scopes: dict[str, ArtifactExecutionScope] = {}
        if plan.artifact_plan is not None:
            for sid in subtasks:
                derived = derive_execution_scope(plan.artifact_plan, sid)
                if derived is not None:
                    scopes[sid] = derived
        elif task.execution_scope is not None:
            assignment = validate_recursive_scope_assignment(
                plan,
                task.execution_scope,
                output_required=task.execution_scope_output_required,
            )
            if not assignment.ok:
                raise ValueError(assignment.feedback())
            for sid, subtask in subtasks.items():
                if subtask.produces_parent_artifacts:
                    scopes[sid] = task.execution_scope

        # Remaining unsatisfied dependencies per subtask.
        remaining: dict[str, set[str]] = {
            sid: set(st.depends_on) for sid, st in subtasks.items()
        }
        scheduled: set[str] = set()
        done: set[str] = set()

        with ThreadPoolExecutor(max_workers=self.config.max_parallelism) as pool_exec:
            futures: dict = {}  # Future -> subtask id

            def schedule_ready() -> None:
                if self.cancel.cancelled:
                    return  # stop launching new subtasks once Stop is pressed
                for sid, deps in remaining.items():
                    if sid in scheduled or sid in done:
                        continue
                    if not deps:
                        scheduled.add(sid)
                        fut = pool_exec.submit(
                            self._run_subtask, task, subtasks[sid],
                            results, results_lock, depth, scopes.get(sid),
                            plan.artifact_plan,
                        )
                        futures[fut] = sid

            schedule_ready()

            while futures:
                completed, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in completed:
                    sid = futures.pop(fut)
                    result = fut.result()
                    with results_lock:
                        results[result.subtask_id] = result
                    done.add(sid)
                    # Unblock dependents.
                    for deps in remaining.values():
                        deps.discard(sid)
                schedule_ready()

        # Return results in plan order for stable synthesis.
        return [results[sid] for sid in subtasks if sid in results]

    # ------------------------------------------------------------------ #
    def _run_subtask(
        self,
        task: Task,
        subtask: SubTask,
        results: dict[str, SubTaskResult],
        results_lock: threading.Lock,
        depth: int,
        scope: Optional[ArtifactExecutionScope] = None,
        work_plan: Optional[ArtifactWorkPlan] = None,
    ) -> SubTaskResult:
        self.cancel.check()
        result = SubTaskResult(
            subtask_id=subtask.id, title=subtask.title, status=TaskStatus.RUNNING
        )

        # Gather prerequisite outputs (title -> output) for context. Each one is
        # clipped: unbounded dependency outputs balloon worker prompts past what
        # web chat composers will accept.
        with results_lock:
            dep_outputs: dict[str, str] = {}
            for dependency_id in subtask.depends_on:
                dependency = results.get(dependency_id)
                if dependency is None or dependency.status != TaskStatus.COMPLETED:
                    continue
                label = dependency.title
                if label in dep_outputs:
                    label = f"{label} [{dependency_id}]"
                dep_outputs[label] = clip_text(
                    self._dependency_content(dependency, scope),
                    self.config.max_dep_output_chars,
                )
        # The legacy aggregate stitch budget now also bounds recursive input
        # context.  This happens before the recursion branch, so a complex
        # fan-in task cannot smuggle an unbounded collection of individually
        # clipped dependency bodies into its child planner prompt.
        dep_outputs = self._bound_dependency_bodies(
            dep_outputs, self.config.max_synthesis_input_chars
        )

        # If a prerequisite failed, mark this skipped.
        with results_lock:
            failed_deps = [
                d for d in subtask.depends_on
                if d in results and results[d].status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            ]
        if failed_deps:
            result.status = TaskStatus.SKIPPED
            result.error = "Skipped because a prerequisite subtask did not complete."
            result.finished_at = _now()
            self.log(f"  [skip] {subtask.title} (prereq failed)")
            self._emit(phase="subtask", status="skipped", icon="⏭️",
                       title=subtask.title,
                       message=f"Skipped “{subtask.title}” (a prerequisite did not complete).")
            return result

        # Phase 4G executor dispatch guard (defense in depth). The planner
        # already rejects synthesis/merge/integration/final-assembly
        # delegations, but a forbidden synthesis subtask must NEVER reach a
        # provider even if one slipped through: no model may be handed every
        # worker output and asked to merge it. Fail the subtask deterministically
        # with a typed reason instead of routing or recursing it.
        synthesis_reason = forbidden_synthesis_reason(subtask)
        if synthesis_reason is not None:
            result.status = TaskStatus.FAILED
            result.verifier_score = 0.0
            result.error = (
                f"Prohibited model-based synthesis delegation: {synthesis_reason}."
            )
            result.finished_at = _now()
            self.log(
                f"  [reject] {subtask.title}: forbidden synthesis role "
                f"({synthesis_reason})"
            )
            self._emit(
                phase="subtask", status="error", icon="🚫",
                title=subtask.title,
                message=(
                    f"Refused “{subtask.title}”: DOZEN combines results "
                    "deterministically and never delegates a merge step to a "
                    "model."
                ),
            )
            return result

        # Recurse for complex subtasks (Dozen calling itself).
        if subtask.complex and self.recurse_fn is not None and depth < self.config.max_depth:
            self.log(f"  [recurse] {subtask.title} (depth {depth + 1})")
            self._emit(phase="subtask", status="recurse", icon="🧩",
                       title=subtask.title,
                       message=f"“{subtask.title}” is complex — re-orchestrating it as a sub-plan.")
            try:
                sub_prompt = subtask.instruction
                if dep_outputs:
                    ctx = "\n\n".join(f"[{t}]\n{o}" for t, o in dep_outputs.items())
                    sub_prompt = f"{subtask.instruction}\n\nInputs:\n{ctx}"
                # Phase 3: the child inherits the parent's deliverable contract.
                # Without this a recursive child of an IMPLEMENT request would
                # re-resolve from its own narrow instruction (e.g. "outline the
                # component tree") and silently revert to explanation-only.
                # Phase 4B: it also inherits ONLY this subtask's assigned
                # artifact scope (never unrelated root packages); a deeper
                # descendant keeps the scope already confined at its root.
                sub_task = Task(
                    prompt=sub_prompt,
                    context=task.prompt,
                    constraints=list(task.constraints),
                    desired_output=subtask.expected_output,
                    contract=task.contract,
                    # Every nested recursion retains the package boundary, but
                    # only an explicitly marked producer inherits ownership.
                    execution_scope=(
                        scope if scope is not None else task.execution_scope
                    ),
                    execution_scope_output_required=(scope is not None),
                )
                output = self.recurse_fn(subtask, sub_task, depth + 1)
                # Phase 4C: a scoped child hands back the typed artifact
                # candidates its designated producer emitted, not just prose —
                # so the parent collects real artifacts instead of re-parsing a
                # synthesized narrative. A legacy string result stays a string.
                if isinstance(output, RecursiveArtifactOutput):
                    result.output = output.text
                    if scope is not None:
                        result.artifact_collection = output.collection
                else:
                    result.output = output
                result.agent_name = f"<orchestrator depth {depth + 1}>"
                result.status = TaskStatus.COMPLETED
                result.attempts = 1
                result.finished_at = _now()
                self._emit(phase="subtask", status="done", icon="✅",
                           title=subtask.title, duration_s=result.duration_s,
                           message=f"Completed sub-plan for “{subtask.title}” "
                                   f"({result.duration_s:.0f}s).")
                return result
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                result.status = TaskStatus.FAILED
                result.error = (
                    f"Recursive orchestration failed: {sanitize_diagnostic(exc)}"
                )
                result.finished_at = _now()
                return result

        # Standard path: route, run, verify, repair.
        context_chars = sum(len(o) for o in dep_outputs.values()) + len(subtask.instruction)
        agent = self._select_agent(subtask, context_chars)
        result.agent_name = agent.name
        result.model_selection_reasoning = subtask.model_selection_reasoning
        self.log(
            f"  [run] {subtask.title} -> {agent.name}"
            + (f" :: {subtask.model_selection_reasoning}"
               if subtask.model_selection_reasoning else "")
        )
        snippet = " ".join(subtask.instruction.split())
        if len(snippet) > 80:
            snippet = snippet[:80].rstrip() + "…"
        self._emit(phase="subtask", status="start", icon="🚀",
                   agent=agent.name, title=subtask.title,
                   message=f"Sending “{subtask.title}” to {agent.name}: \"{snippet}\"")
        self._emit(phase="subtask", status="waiting", icon="⏳",
                   agent=agent.name, title=subtask.title,
                   message=f"Waiting on {agent.name} to finish...")

        feedback = ""
        # The last output that passed semantic validation (a genuine answer).
        # Stays None if every attempt refused / was empty -> the subtask FAILS.
        accepted_output: Optional[str] = None
        # Phase 4E: the preserved-candidate state and the NEXT targeted repair
        # request. Both stay None for an unscoped subtask and for a scoped worker
        # that delivered everything first time, which keeps those paths identical
        # to Phase 4D. ``repair_scope`` is the narrow scope of the outstanding
        # artifacts; the original ``scope`` is never mutated.
        repair_state: Optional[ArtifactRepairState] = None
        repair_request: Optional[ArtifactRepairRequest] = None
        repair_scope: Optional[ArtifactExecutionScope] = None
        max_attempts = 1 + self.config.max_repair_attempts
        for attempt in range(1, max_attempts + 1):
            self.cancel.check()
            result.attempts = attempt
            active_scope = repair_scope if repair_request is not None else scope

            # Escalate to strongest agent on the final attempt if enabled.
            active_agent = agent
            if (
                attempt == max_attempts
                and self.config.escalate_on_failure
                and attempt > 1
            ):
                active_agent = self._strongest_agent() or agent
                result.agent_name = active_agent.name
                self.log(f"    [escalate] {subtask.title} -> {active_agent.name}")

            if active_scope is not None and active_scope.output_artifact_ids:
                estimated_chars = estimate_scope_output_chars(active_scope)
                provider_budget = provider_output_budget_chars(
                    active_agent.max_tokens
                )
                if estimated_chars > provider_budget:
                    result.error = (
                        f"Selected provider {active_agent.name!r} has a safe "
                        f"output budget of {provider_budget} characters, below "
                        f"the scope estimate of {estimated_chars}; redesign the "
                        "artifact into smaller logical modules or select a "
                        "provider with a larger output budget."
                    )
                    feedback = result.error
                    self.log(
                        f"    [budget] {subtask.title}: ~{estimated_chars} > "
                        f"{provider_budget} chars for {active_agent.name}"
                    )
                    continue

            try:
                output = self._call_worker(
                    task, subtask, dep_outputs, active_agent, feedback,
                    active_scope, repair_request,
                )
            except CancelledError:
                raise  # propagate Stop; never retry a cancelled call
            except Exception as exc:  # noqa: BLE001
                result.error = f"Worker call failed: {sanitize_diagnostic(exc)}"
                continue

            # --- Artifact parsing ----------------------------------------- #
            # Workers reply with a strict JSON artifact. Parse it and flatten the
            # `artifacts` payload into the plain content downstream synthesis
            # consumes; capture the model's self-reported confidence. If the reply
            # isn't artifact-shaped (a model ignored the contract), fall back to
            # treating the raw text as the content so we stay robust.
            artifact = parse_worker_artifact(output)
            if artifact is not None:
                # Bounded nested decode: a double-encoded envelope (an envelope
                # whose artifact content is ITSELF a serialized envelope) is
                # flattened at most twice; code that merely contains JSON is
                # never touched.
                content = decode_nested_envelope(render_artifact(artifact))
                confidence = artifact_confidence(artifact)
                decisions = artifact.get("key_decisions")
                if isinstance(decisions, (list, tuple)):
                    # Bounded capture for synthesis capsules; the envelope
                    # itself never travels further.
                    result.key_decisions = [
                        " ".join(str(item).split())[:300]
                        for item in decisions[:8]
                        if str(item).strip()
                    ]
            elif is_malformed_protocol_envelope(output):
                # Fail closed (Phase 6): the reply IS protocol traffic but the
                # JSON is broken. It must never become user-visible content —
                # reject it into the existing bounded retry loop with a clean,
                # bounded diagnostic instead of falling back to the raw payload.
                result.verifier_score = 0.0
                result.verifier_feedback = "malformed protocol envelope"
                # The payload head goes to the LOG only — the stored error can
                # reach user-visible fallback text and must stay payload-free.
                result.error = (
                    "Rejected malformed worker envelope: the reply was protocol "
                    f"JSON that could not be parsed ({len(output)} chars)."
                )
                feedback = (
                    "Your previous reply was a JSON envelope that could not be "
                    "parsed (it was truncated or syntactically invalid). Resend "
                    "the COMPLETE, valid JSON object exactly as the output "
                    "contract requires — escape internal double quotes and do "
                    "not truncate the content."
                )
                self.log(
                    f"    [reject] {subtask.title}: malformed worker envelope "
                    f"({len(output)} chars)"
                )
                self._emit(phase="subtask", status="retry", icon="⚠️",
                           agent=result.agent_name, title=subtask.title,
                           message=f"{result.agent_name} returned a malformed "
                                   "result envelope; retrying...")
                continue
            elif (prose_envelope := parse_prose_envelope(output)) is not None:
                # Phase 4F (Part E): the typed prose envelope. ``content`` is
                # the only visible body; task_id is CHECKED but never trusted
                # as ownership (the runtime scope owns identity); warnings are
                # bounded metadata for ordered sections, never raw output.
                content, confidence = prose_envelope.content, None
                result.worker_warnings = list(prose_envelope.warnings[:8])
                result.worker_evidence_refs = list(prose_envelope.evidence_refs[:8])
                if prose_envelope.task_id and prose_envelope.task_id not in (
                    subtask.id, subtask.title
                ):
                    result.worker_internal_notes.append(
                        "the worker-declared task id was ignored; the runtime "
                        "scope identity is authoritative"
                    )
            elif is_malformed_prose_envelope(output):
                # Fail closed exactly like a malformed artifact envelope: the
                # broken payload never becomes content; bounded retry instead.
                result.verifier_score = 0.0
                result.verifier_feedback = "malformed prose result envelope"
                result.error = (
                    "Rejected malformed worker envelope: the reply was a typed "
                    f"result envelope that could not be parsed ({len(output)} "
                    "chars)."
                )
                feedback = (
                    "Your previous reply was a JSON result envelope that could "
                    "not be parsed (truncated or syntactically invalid). Resend "
                    "the COMPLETE, valid response — escape internal double "
                    "quotes and do not truncate the content."
                )
                self.log(
                    f"    [reject] {subtask.title}: malformed prose envelope "
                    f"({len(output)} chars)"
                )
                self._emit(phase="subtask", status="retry", icon="⚠️",
                           agent=result.agent_name, title=subtask.title,
                           message=f"{result.agent_name} returned a malformed "
                                   "result envelope; retrying...")
                continue
            else:
                content, confidence = output, None

            # --- Semantic validation -------------------------------------- #
            # A successful DOM scrape is NOT a successful answer. Reject model
            # refusals ("I cannot fulfill this request.") and empty/too-short
            # replies BEFORE anything downstream (verifier OR synthesizer) is
            # allowed to trust them — this is what fixes the score-1.00 false
            # positive. Rejected attempts are retried with corrective feedback.
            validation = validate_model_response(content)
            # A scrape can also return the system's OWN control JSON (e.g. the
            # planner's plan left on screen in the same chat window). That is
            # never an answer — reject it so the retry loop re-asks cleanly.
            if validation.ok and looks_like_orchestration_json(content):
                validation = ResponseValidation(
                    False,
                    "reply was internal orchestration JSON (a plan/verdict echo), "
                    "not an answer to the subtask",
                    "echo",
                )
            if not validation.ok:
                result.verifier_score = 0.0
                result.verifier_feedback = validation.reason
                if (
                    validation.kind == "refusal"
                    and active_scope is not None
                    and active_scope.output_artifact_ids
                ):
                    classification = classify_scoped_delivery(
                        active_scope, result.artifact_collection, content
                    )
                    result.delivery_code = classification.code.value
                result.error = f"Rejected non-answer: {validation.reason}"
                feedback = (
                    "Your previous reply was rejected because it was not a real "
                    f"answer ({validation.reason}). Do NOT refuse, apologize, or "
                    "explain limitations — actually complete the subtask and "
                    "return only the requested output."
                )
                self.log(f"    [reject] {subtask.title}: {validation.reason}")
                self._emit(phase="subtask", status="retry", icon="⚠️",
                           agent=result.agent_name, title=subtask.title,
                           message=f"{result.agent_name} returned a non-answer "
                                   f"({validation.reason}); retrying...")
                continue

            # Genuine answer: safe to keep and consider downstream. We store the
            # flattened artifact content (not the JSON wrapper) so synthesis gets
            # clean text/code.
            accepted_output = content
            result.output = content

            # --- Artifact collection (Phase 4C) --------------------------- #
            # A scoped worker owes concrete artifacts. Collect them from the
            # SAME envelope the text came from — typed, owner-checked, exact
            # content — and attach the result. This never changes the worker's
            # semantic status: a real answer that misses an owned file is a
            # genuine answer AND an unmet artifact obligation, and the two stay
            # separately visible. Unmet obligations are folded into the EXISTING
            # repair feedback; no new retry loop and no extra provider call.
            # Phase 4E: only the MISSING and REJECTED artifacts are re-requested;
            # everything already accepted is preserved byte-identically and never
            # asked for again. Still no new loop, no new provider role, and no
            # attempt-budget reset.
            repair_state, decision = self._collect_artifacts(
                result, subtask, artifact, scope, work_plan, attempt, depth,
                max_attempts, repair_state, repair_request, repair_scope,
            )
            repair_request = decision.request if decision is not None else None
            repair_scope = (
                derive_repair_scope(scope, repair_request)
                if repair_request is not None and scope is not None
                else None
            )
            artifact_feedback = (
                result.artifact_collection.feedback(self.result_policy)
                if result.artifact_collection is not None else ""
            )
            retry_for_artifacts = (
                repair_request is not None and attempt < max_attempts
            )
            structurally_unusable = bool(
                scope is not None
                and result.artifact_collection is not None
                and result.artifact_collection.engaged
                and not result.artifact_collection.accepted
                and not result.artifact_collection.satisfied
            )

            if not self.config.verify_outputs or structurally_unusable:
                if retry_for_artifacts:
                    feedback = artifact_feedback
                    continue
                if self._fail_scoped_non_delivery(result, subtask, scope, content):
                    return result
                result.status = TaskStatus.COMPLETED
                # Trust the worker's self-reported confidence as the score when
                # verification is off; default to 1.0 if it wasn't provided.
                result.verifier_score = confidence if confidence is not None else 1.0
                result.finished_at = _now()
                self.log(f"    [ok] {subtask.title} (validated, attempt {attempt})")
                self._emit(phase="subtask", status="done", icon="✅",
                           agent=result.agent_name, title=subtask.title,
                           duration_s=result.duration_s,
                           message=f"Received result from {result.agent_name} "
                                   f"({result.duration_s:.0f}s).")
                return result

            # The Phase 3 ``contract`` and Phase 4B ``scope`` arguments are
            # optional on the concrete Verifier, and duck-typed legacy
            # verifiers with the original three-argument signature remain
            # usable by public Executor callers: each optional argument is
            # passed only when the implementation's signature accepts it.
            try:
                verify_params = inspect.signature(
                    self.verifier.verify
                ).parameters
            except (TypeError, ValueError):
                verify_params = None
            verify_kwargs = {}
            accepts_kwargs = bool(
                verify_params is not None
                and any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in verify_params.values()
                )
            )
            if verify_params is not None and (
                accepts_kwargs or "contract" in verify_params
            ):
                verify_kwargs["contract"] = task.contract
            # Phase 4E: on a repair attempt the verifier receives the NARROW
            # repair scope, so it judges the repaired artifacts and their
            # criteria — and never demands the preserved files back. It is the
            # same verifier instance; no repair-specific verifier exists.
            if active_scope is not None and (
                verify_params is not None
                and (accepts_kwargs or "scope" in verify_params)
            ):
                verify_kwargs["scope"] = active_scope
            verdict = self.verifier.verify(
                subtask, content, dep_outputs, **verify_kwargs
            )
            result.verifier_score = verdict.score
            result.verifier_feedback = verdict.feedback

            if verdict.passed:
                if retry_for_artifacts:
                    # Verified prose, unmet artifact obligation: reuse the same
                    # bounded repair attempt rather than accepting a deliverable
                    # that is missing files.
                    feedback = artifact_feedback
                    continue
                if self._fail_scoped_non_delivery(result, subtask, scope, content):
                    return result
                result.status = TaskStatus.COMPLETED
                result.finished_at = _now()
                self.log(f"    [ok] {subtask.title} (score {verdict.score:.2f}, "
                         f"attempt {attempt})")
                self._emit(phase="subtask", status="done", icon="✅",
                           agent=result.agent_name, title=subtask.title,
                           duration_s=result.duration_s, score=verdict.score,
                           message=f"Received result from {result.agent_name} "
                                   f"({result.duration_s:.0f}s, score {verdict.score:.2f}).")
                return result

            feedback = verdict.feedback
            if artifact_feedback:
                feedback = f"{feedback}\n{artifact_feedback}".strip()
            self.log(f"    [repair] {subtask.title} (score {verdict.score:.2f}): "
                     f"{verdict.feedback[:80]}")

        # Exhausted attempts.
        if accepted_output is not None:
            # We have a genuine answer that only failed the verifier gate; keep
            # it as usable-but-flagged so synthesis still has real content.
            if self._fail_scoped_non_delivery(
                result, subtask, scope, accepted_output
            ):
                return result
            result.output = accepted_output
            result.status = TaskStatus.COMPLETED
            result.finished_at = _now()
            self._emit(phase="subtask", status="done", icon="✅",
                       agent=result.agent_name, title=subtask.title,
                       duration_s=result.duration_s,
                       message=f"Received result from {result.agent_name} "
                               f"({result.duration_s:.0f}s, accepted with low confidence).")
            return result

        # Every attempt refused / was empty / errored: this subtask FAILED.
        # Crucially, do NOT pass a refusal downstream to synthesis.
        result.output = ""
        result.status = TaskStatus.FAILED
        result.verifier_score = 0.0
        if not result.error:
            result.error = "All attempts were refusals or empty/non-answers."
        self.log(f"  [fail] {subtask.title}: {result.error}")
        result.finished_at = _now()
        self._emit(phase="subtask", status="error", icon="❌",
                   agent=result.agent_name, title=subtask.title,
                   message=f"“{subtask.title}” failed: {result.error}")
        return result

    # ------------------------------------------------------------------ #
    def _fail_scoped_non_delivery(
        self,
        result: SubTaskResult,
        subtask: SubTask,
        scope: Optional[ArtifactExecutionScope],
        content: str,
    ) -> bool:
        """Phase 4F (Part H): typed non-delivery classification at completion.

        Primary evidence is the artifact-scope obligation, never phrase
        matching. A scoped worker that delivered ZERO owned artifacts while
        refusing on response size (or returning an empty typed envelope) has
        NOT completed: it fails with a typed code, and its refusal text is
        quarantined so it can never enter a final answer. A zero-delivery
        free-text reply keeps the approved Phase 4C completion semantics
        (visible unmet obligation) but carries the classification, so a
        refusal can never be mistaken for a successful prose subtask.
        """
        if scope is None or not scope.output_artifact_ids:
            return False
        classification = classify_scoped_delivery(
            scope, result.artifact_collection, content
        )
        if classification.code is WorkerDeliveryCode.DELIVERED:
            return False
        result.delivery_code = classification.code.value
        if classification.code not in FATAL_DELIVERY_CODES:
            return False
        result.output = ""  # refusal text must never travel further
        result.status = TaskStatus.FAILED
        result.verifier_score = 0.0
        result.error = (
            f"Scoped artifact non-delivery [{classification.code.value}]: "
            f"{classification.reason}."
        )
        result.finished_at = _now()
        self.log(
            f"  [fail] {subtask.title}: {classification.code.value} — "
            f"{classification.reason}"
        )
        self._emit(
            phase="subtask", status="error", icon="❌",
            agent=result.agent_name, title=subtask.title,
            message=(
                f"“{subtask.title}” did not deliver its assigned artifacts "
                f"({classification.code.value})."
            ),
        )
        return True

    # ------------------------------------------------------------------ #
    def _collect_artifacts(
        self,
        result: SubTaskResult,
        subtask: SubTask,
        envelope: Optional[dict],
        scope: Optional[ArtifactExecutionScope],
        work_plan: Optional[ArtifactWorkPlan],
        attempt: int,
        depth: int,
        max_attempts: int,
        state: Optional[ArtifactRepairState],
        request: Optional[ArtifactRepairRequest],
        repair_scope: Optional[ArtifactExecutionScope],
    ) -> tuple[Optional[ArtifactRepairState], Optional[RepairDecision]]:
        """Collect ONE attempt, merge it into the preserved state, and decide.

        Attempt 1 (``request is None``) is the untouched Phase 4C/4D collection of
        the FULL assigned package. A later attempt is a targeted repair reply: it
        is parsed against the narrow repair scope, merged into the preserved
        candidates (which are never mutated, replaced or re-hashed) and re-judged.

        Returns the new preserved state and the repair decision. ``(None, None)``
        for an unscoped subtask keeps pre-4C behavior byte-identical.
        """
        if scope is None:
            return None, None

        payload = envelope if envelope is not None else {}
        if request is None or state is None or repair_scope is None:
            collection: SubtaskCollectionResult = collect_subtask_artifacts(
                payload,
                scope,
                policy=self.result_policy,
                attempt=attempt,
                # The scope's subtask id is the PLAN-mapped owner; the delegation
                # actually emitting the file may be a nested one, and both are kept.
                producer_subtask_id=subtask.id,
                recursion_depth=depth,
                work_plan=work_plan,
            )
            state = begin_repair_state(
                collection, submitted_hashes=submitted_content_hashes(payload)
            )
        else:
            repair_collection = collect_repair_artifacts(
                payload,
                repair_scope,
                request,
                policy=self.repair_policy,
                result_policy=self.result_policy,
                attempt=attempt,
                producer_subtask_id=subtask.id,
                recursion_depth=depth,
            )
            state = merge_repair_collection(
                scope, state, repair_collection, request,
                attempt=attempt,
                submitted_hashes=submitted_content_hashes(payload),
                policy=self.repair_policy,
                result_policy=self.result_policy,
            )

        result.artifact_collection = state.collection
        decision = plan_repair(
            scope, state,
            attempt=attempt,
            max_attempts=max_attempts,
            policy=self.repair_policy,
        )
        result.artifact_repair = state.report(decision)

        if state.collection.satisfied:
            return state, decision

        summary = state.collection.problem_summary()
        if decision.should_retry and decision.request is not None:
            targets = ", ".join(decision.request.target_paths[:4])
            self.log(
                f"    [repair] {subtask.title}: requesting only {targets} "
                f"({len(state.collection.accepted)} artifact(s) preserved)"
            )
            self._emit(
                phase="subtask", status="artifacts", icon="🩹",
                agent=result.agent_name, title=subtask.title,
                message=(
                    f"“{subtask.title}”: preserving "
                    f"{len(state.collection.accepted)} valid artifact(s) and "
                    f"re-requesting only {targets}."
                ),
            )
        else:
            self.log(f"    [artifacts] {subtask.title}: {summary}")
            self._emit(
                phase="subtask", status="artifacts", icon="📦",
                agent=result.agent_name, title=subtask.title,
                message=(
                    f"“{subtask.title}” did not deliver its assigned artifacts: "
                    f"{summary}"
                    + (f" ({decision.reason})" if decision.reason else "")
                ),
            )
        return state, decision

    # ------------------------------------------------------------------ #
    def _call_worker(
        self,
        task: Task,
        subtask: SubTask,
        dep_outputs: dict[str, str],
        agent: AgentSpec,
        feedback: str,
        scope: Optional[ArtifactExecutionScope] = None,
        repair: Optional[ArtifactRepairRequest] = None,
    ) -> str:
        messages = self._bounded_worker_messages(
            task,
            subtask,
            dep_outputs,
            agent,
            feedback,
            scope=scope,
            repair=repair,
        )
        resp = self.client.complete(
            provider=agent.provider,
            model=agent.model,
            messages=messages,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        return resp.text.strip()

    @staticmethod
    def _dependency_content(
        dependency: SubTaskResult,
        scope: Optional[ArtifactExecutionScope],
    ) -> str:
        """Return only dependency material relevant to this worker's scope.

        Unscoped prose keeps the established direct-dependency behavior.  For a
        typed artifact flow, only artifacts explicitly named as this scope's
        inputs are carried, with their trusted id/path/hash reference; complete
        unrelated artifact bodies are never copied into the next worker prompt.
        """
        collection = dependency.artifact_collection
        if scope is None or collection is None:
            return dependency.output
        allowed = set(scope.input_artifact_ids)
        relevant = [
            candidate
            for candidate in collection.accepted
            if candidate.artifact_id in allowed
        ]
        if not relevant:
            return (
                "[No artifact body forwarded: this producer has no artifact "
                "listed in the receiving scope's typed input references.]"
            )
        blocks = []
        for candidate in relevant:
            blocks.append(
                "ARTIFACT INPUT REFERENCE "
                f"id={candidate.artifact_id} path={candidate.path} "
                f"sha256={candidate.content_hash}\n{candidate.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _worker_input_budget_chars(agent: AgentSpec) -> int:
        context_tokens = max(0, int(agent.max_context_tokens))
        output_tokens = max(0, int(agent.max_tokens))
        return max(0, context_tokens - output_tokens) * _INPUT_CHARS_PER_TOKEN

    @staticmethod
    def _clip_to_exact_limit(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        return clip_text(text, limit)

    @classmethod
    def _bound_dependency_bodies(
        cls, outputs: dict[str, str], max_chars: int
    ) -> dict[str, str]:
        """Fair, deterministic aggregate cap over dependency body characters."""
        if max_chars <= 0 or sum(len(value) for value in outputs.values()) <= max_chars:
            return dict(outputs)
        remaining_chars = max_chars
        remaining_items = len(outputs)
        bounded: dict[str, str] = {}
        for title, output in outputs.items():
            fair_limit = (
                remaining_chars // remaining_items if remaining_items else 0
            )
            body = cls._clip_to_exact_limit(output, fair_limit)
            bounded[title] = body
            remaining_chars -= len(body)
            remaining_items -= 1
        return bounded

    def _bounded_worker_messages(
        self,
        task: Task,
        subtask: SubTask,
        dependency_outputs: dict[str, str],
        agent: AgentSpec,
        feedback: str,
        *,
        scope: Optional[ArtifactExecutionScope],
        repair: Optional[ArtifactRepairRequest],
    ):
        """Build one worker prompt that fits the selected agent's input budget.

        The client's own serializer is the measurement authority (important for
        browser wrappers).  Dependency bodies share one aggregate allowance;
        titles/references remain in plan order.  If even the dependency-free
        frame cannot fit, dispatch is refused before any provider call.
        """
        budget = self._worker_input_budget_chars(agent)

        def build(outputs: dict[str, str]):
            return build_worker_messages(
                task,
                subtask,
                outputs,
                feedback,
                scope=scope,
                repair=repair,
                repair_policy=self.repair_policy,
            )

        messages = build(dependency_outputs)
        if self.client.measure_input_chars(messages) <= budget:
            return messages

        references = {title: "" for title in dependency_outputs}
        reference_messages = build(references)
        reference_size = self.client.measure_input_chars(reference_messages)
        if reference_size > budget:
            messages = build({})
            measured = self.client.measure_input_chars(messages)
            if measured > budget:
                raise ValueError(
                    f"Worker prompt frame requires {measured} input characters, "
                    f"above selected provider {agent.name!r}'s safe budget of "
                    f"{budget}; reduce the task/scope or choose a larger-context "
                    "provider."
                )
            return messages

        remaining_chars = budget - reference_size
        remaining_items = len(dependency_outputs)
        bounded: dict[str, str] = {}
        for title, output in dependency_outputs.items():
            fair_limit = (
                remaining_chars // remaining_items if remaining_items else 0
            )
            body = self._clip_to_exact_limit(output, fair_limit)
            bounded[title] = body
            remaining_chars -= len(body)
            remaining_items -= 1
        messages = build(bounded)
        measured = self.client.measure_input_chars(messages)
        if measured > budget:  # defensive against a non-additive serializer
            raise ValueError(
                f"Worker prompt requires {measured} input characters after "
                f"dependency bounding, above selected provider {agent.name!r}'s "
                f"safe budget of {budget}."
            )
        return messages

    def _select_agent(self, subtask: SubTask, context_chars: int) -> AgentSpec:
        """Honor the Manager's intelligent routing, else fall back to the router.

        The Manager LLM's per-subtask ``assigned_model`` takes precedence (that
        is the whole point of intelligent delegation). Only when it is absent or
        no longer resolves to an enabled agent do we defer to the deterministic
        capability router — and we annotate the reasoning so the trace explains
        what happened.
        """
        if subtask.assigned_model:
            agent = self.pool.resolve(subtask.assigned_model)
            if agent is not None and agent.enabled:
                return agent
            note = (
                f"(Assigned model {subtask.assigned_model!r} is unavailable; "
                "fell back to automatic routing.)"
            )
            subtask.model_selection_reasoning = (
                f"{subtask.model_selection_reasoning} {note}".strip()
                if subtask.model_selection_reasoning else note
            )

        agent = self.router.route(subtask, context_chars)
        if not subtask.model_selection_reasoning:
            subtask.model_selection_reasoning = (
                f"Auto-routed by capability match to '{agent.name}'."
            )
        return agent

    def _strongest_agent(self) -> Optional[AgentSpec]:
        agents = self.pool.enabled_agents()
        return max(agents, key=lambda a: a.tier) if agents else None


def _now() -> float:
    import time

    return time.time()
