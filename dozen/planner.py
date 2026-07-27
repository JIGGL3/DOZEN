"""Planner: turns a Task into a validated subtask DAG (or a direct answer)."""

from __future__ import annotations

from typing import Optional

from .agent_pool import AgentPool, AgentSpec
from .artifacts import ArtifactContractError, parse_planner_artifact_plan
from .decomposition import (
    provider_output_budget_chars,
    validate_artifact_decomposition,
    validate_package_sizing,
    validate_recursive_scope_assignment,
)
from .intent import validate_plan_against_contract
from .llm_client import LLMClient
from .models import Plan, SubTask, Task
from .prompts import artifact_planning_enabled, build_planner_messages
from .synthesis_guard import validate_no_synthesis_tasks
from .validation import sanitize_prompt


class PlanError(RuntimeError):
    pass


class Planner:
    def __init__(
        self,
        client: LLMClient,
        planner_agent: AgentSpec,
        pool: Optional[AgentPool] = None,
    ) -> None:
        self.client = client
        self.agent = planner_agent
        # The pool powers the "Available Models" catalog the Manager routes over,
        # and lets us validate the names it returns in `assigned_model`.
        self.pool = pool

    def _provider_output_budgets(self, plan: Plan) -> dict[str, int]:
        """Safe per-subtask budgets from selected (or eligible) providers."""
        if self.pool is None:
            return {}
        enabled = self.pool.enabled_agents()
        fallback_tokens = min(
            (agent.max_tokens for agent in enabled), default=0
        )
        budgets: dict[str, int] = {}
        for subtask in plan.subtasks:
            max_tokens = fallback_tokens
            if subtask.assigned_model:
                try:
                    max_tokens = self.pool.get(subtask.assigned_model).max_tokens
                except KeyError:
                    # Unknown model validation is handled by the normal planner
                    # parsing path; do not invent a budget for it here.
                    continue
            if max_tokens > 0:
                budgets[subtask.id] = provider_output_budget_chars(max_tokens)
        return budgets

    def plan(self, task: Task, depth: int, max_depth: int) -> Plan:
        catalog = self.pool.describe_for_planner() if self.pool is not None else ""
        plan, artifact_error = self._plan_once(
            task, depth, max_depth, catalog, feedback=""
        )

        # Phase 3: contract-aware plan validation. A code-bearing request whose
        # plan is entirely "describe / recommend / explain" work cannot succeed,
        # so re-plan ONCE with the specific problem as corrective feedback
        # (reusing the existing repair-then-retry shape). A second fatal
        # violation raises PlanError, which the orchestrator already surfaces.
        # Phase 4A folds artifact-plan problems into the SAME single corrective
        # re-plan: a malformed artifact_plan block gets exactly one retry, and a
        # second invalid one fails deterministically. No extra retries exist.
        # Recursive plans are intermediate work products. They inherit the
        # authoritative contract for prompts, but a legitimate research/design
        # child of an implementation must not itself be forced to contain the
        # root artifact. The depth-0 plan and final guard remain authoritative.
        # Phase 4B folds bounded-decomposition problems (unmapped executable
        # packages, oversized package scopes, package/subtask DAG mismatches)
        # into the SAME single corrective re-plan. Advisory decomposition
        # warnings never trigger a planner call.
        root_contract_check = task.contract is not None and depth == 0
        check = (
            validate_plan_against_contract(task.contract, plan)
            if root_contract_check else None
        )
        decomposition = validate_artifact_decomposition(plan)
        sizing = validate_package_sizing(
            plan, subtask_output_budgets=self._provider_output_budgets(plan)
        )
        recursive_scope = validate_recursive_scope_assignment(
            plan,
            task.execution_scope,
            output_required=task.execution_scope_output_required,
        )
        # Phase 4G: NO plan — root or recursive — may delegate a model-based
        # synthesis / merge / integration-review / final-assembly step. This is
        # what let broken "final integration review" delegations survive Phase
        # 4F. Folded into the SAME single corrective re-plan as the checks above.
        synthesis = validate_no_synthesis_tasks(plan)
        code_only_error = self._code_only_plan_error(task, depth, plan)
        if (
            (check is None or (check.ok and not check.advisory))
            and not artifact_error
            and decomposition.ok
            and sizing.ok
            and recursive_scope.ok
            and synthesis.ok
            and not code_only_error
        ):
            return plan

        decomposition_error = ""
        if decomposition.fatal:
            decomposition_error = (
                f"the artifact decomposition is invalid: {decomposition.feedback()}"
            )
        sizing_error = ""
        if sizing.fatal:
            sizing_error = (
                f"the artifact packages are oversized: {sizing.feedback()}"
            )
        recursive_scope_error = ""
        if recursive_scope.fatal:
            recursive_scope_error = (
                "the recursive artifact-scope assignment is invalid: "
                f"{recursive_scope.feedback()}"
            )
        synthesis_error = ""
        if synthesis.fatal:
            synthesis_error = (
                "the plan delegates model-based synthesis, which is prohibited: "
                f"{synthesis.feedback()}"
            )
        problems = [
            msg
            for msg in (
                check.feedback() if check is not None else "",
                artifact_error,
                decomposition_error,
                sizing_error,
                recursive_scope_error,
                synthesis_error,
                code_only_error,
            )
            if msg
        ]
        retry, retry_artifact_error = self._plan_once(
            task, depth, max_depth, catalog, feedback="; ".join(problems)
        )
        fatal = []
        if root_contract_check:
            fatal.extend(validate_plan_against_contract(task.contract, retry).fatal)
        if retry_artifact_error:
            fatal.append(f"invalid artifact plan: {retry_artifact_error}")
        fatal.extend(
            f"invalid artifact decomposition: {msg}"
            for msg in validate_artifact_decomposition(retry).fatal
        )
        fatal.extend(
            f"oversized artifact package: {msg}"
            for msg in validate_package_sizing(
                retry,
                subtask_output_budgets=self._provider_output_budgets(retry),
            ).fatal
        )
        fatal.extend(
            f"invalid recursive artifact-scope assignment: {msg}"
            for msg in validate_recursive_scope_assignment(
                retry,
                task.execution_scope,
                output_required=task.execution_scope_output_required,
            ).fatal
        )
        fatal.extend(
            f"prohibited model-based synthesis delegation: {msg}"
            for msg in validate_no_synthesis_tasks(retry).fatal
        )
        retry_code_only_error = self._code_only_plan_error(task, depth, retry)
        if retry_code_only_error:
            fatal.append(retry_code_only_error)
        if fatal:
            contract_name = (
                f" ({task.contract.intent.value})"
                if root_contract_check else ""
            )
            raise PlanError(
                "The plan does not satisfy the request's execution contract"
                f"{contract_name}: {'; '.join(fatal)}"
            )
        # Advisory-only problems (e.g. no explicit test step) never fail the
        # run: the plan still delivers the artifact, and the executor's
        # verify/repair loop remains the validation backstop.
        return retry

    @staticmethod
    def _code_only_plan_error(task: Task, depth: int, plan: Plan) -> str:
        """Phase 4F (Part G): a code-only request's artifact plan is MANDATORY.

        Without the typed manifest there are no artifact-producing scopes, no
        assembly, and finalization would have nothing to deliver — the exact
        live failure. A direct answer stays permitted (the Phase 3 final guard
        judges it), and a scoped recursive child never declares a new manifest.
        """
        if (
            depth != 0
            or task.contract is None
            or not getattr(task.contract, "code_only", False)
            or task.execution_scope is not None
            or plan.is_direct()
            or plan.artifact_plan is not None
        ):
            return ""
        if not artifact_planning_enabled(task, depth):
            return ""
        return (
            "this code-only request requires the artifact_plan block: declare "
            "every required file and the bounded work package that owns it"
        )

    def _plan_once(
        self, task: Task, depth: int, max_depth: int, catalog: str, feedback: str
    ) -> tuple[Plan, str]:
        messages = build_planner_messages(
            task, depth, max_depth, catalog, repair_feedback=feedback
        )
        data = self.client.complete_json(
            provider=self.agent.provider,
            model=self.agent.model,
            messages=messages,
            temperature=0.1,
            max_tokens=self.agent.max_tokens,
        )
        return self._parse(
            data,
            depth,
            max_depth,
            contract=task.contract,
            artifact_enabled=artifact_planning_enabled(task, depth),
        )

    # ------------------------------------------------------------------ #
    def _parse(
        self,
        data: dict,
        depth: int,
        max_depth: int,
        contract=None,
        artifact_enabled: Optional[bool] = None,
    ) -> tuple[Plan, str]:
        analysis = str(data.get("analysis", "")).strip()
        direct = data.get("direct_answer")
        # Accept the new "delegations" key (intelligent routing) and fall back to
        # the legacy "subtasks" key for older Manager outputs.
        raw_subtasks = data.get("delegations") or data.get("subtasks") or []
        strategy = str(data.get("synthesis_strategy", "")).strip()

        is_direct = bool(direct is not None and str(direct).strip()) and not raw_subtasks

        subtasks: list[SubTask] = []
        id_map: dict[str, str] = {}  # planner id -> internal SubTask id

        for raw in raw_subtasks:
            # Sanitize every free-text field: even if the Manager's JSON parsed,
            # a value can still carry leaked schema fragments (the "prompt
            # mangling" bug). ``sanitize_prompt`` strips those so nothing but the
            # real instruction/context is ever sent to a worker model.
            instruction = sanitize_prompt(str(raw.get("instruction", "")))
            assigned, reasoning = self._resolve_assignment(raw)
            st = SubTask(
                title=sanitize_prompt(str(raw.get("title", "untitled"))) or "untitled",
                instruction=instruction,
                required_capabilities=[str(c) for c in raw.get("required_capabilities", [])],
                depends_on=[str(d) for d in raw.get("depends_on", [])],
                success_criteria=sanitize_prompt(str(raw.get("success_criteria", ""))),
                expected_output=sanitize_prompt(str(raw.get("expected_output", ""))),
                complex=bool(raw.get("complex", False)) and depth < max_depth,
                difficulty=int(raw.get("difficulty", 3) or 3),
                assigned_model=assigned,
                model_selection_reasoning=reasoning,
                produces_parent_artifacts=(
                    raw.get("produces_parent_artifacts") is True
                ),
            )
            raw_planner_id = raw.get("id")
            planner_id = (
                str(raw_planner_id).strip()
                if raw_planner_id is not None and str(raw_planner_id).strip()
                else st.id
            )
            if planner_id in id_map:
                raise PlanError(f"Planner produced duplicate subtask id {planner_id!r}.")
            id_map[planner_id] = st.id
            subtasks.append(st)

        # Remap depends_on from planner ids to internal ids; drop unknown refs.
        for st in subtasks:
            st.depends_on = [id_map[d] for d in st.depends_on if d in id_map]

        self._validate_dag(subtasks)

        if is_direct:
            plan = Plan(
                analysis=analysis,
                subtasks=[],
                synthesis_strategy=strategy,
                direct_answer=str(direct).strip(),
            )
        elif not subtasks:
            raise PlanError("Planner returned neither a direct answer nor any subtasks.")
        else:
            plan = Plan(analysis=analysis, subtasks=subtasks, synthesis_strategy=strategy)

        # Phase 4A: optional artifact-oriented plan block. Parsed only at the
        # authoritative depth-0 plan of a contract-bearing task — a recursive
        # child's block is dropped so it can never corrupt the parent contract.
        # A structural problem is returned (not raised) so the caller can spend
        # the ONE existing corrective re-plan on it.
        artifact_error = ""
        raw_artifact_plan = data.get("artifact_plan")
        if artifact_enabled is None:
            artifact_enabled = bool(
                depth == 0
                and contract is not None
                and (contract.code_required or contract.repo_changes_required)
            )
        if (
            "artifact_plan" in data
            and raw_artifact_plan is not None
            and artifact_enabled
        ):
            try:
                plan.artifact_plan = parse_planner_artifact_plan(
                    raw_artifact_plan, subtask_id_map=id_map
                )
            except ArtifactContractError as exc:
                artifact_error = f"the artifact_plan block is invalid: {exc}"

        return plan, artifact_error

    def _resolve_assignment(self, raw: dict) -> tuple[str, str]:
        """Extract + validate the Manager's model choice for one subtask.

        Returns ``(assigned_model_name, reasoning)``. The name is normalized to
        the pool's exact agent name when it resolves; if it doesn't (unknown or
        hallucinated), it is cleared so the executor falls back to the
        deterministic router rather than crashing on a bad name.
        """
        raw_name = str(raw.get("assigned_model", "")).strip()
        reasoning = sanitize_prompt(str(raw.get("model_selection_reasoning", "")))

        if not raw_name:
            return "", reasoning
        if self.pool is None:
            # No catalog to validate against; trust the name as-is.
            return raw_name, reasoning

        agent = self.pool.resolve(raw_name)
        if agent is not None:
            return agent.name, reasoning
        # Unknown model name: drop the assignment, note it in the reasoning.
        note = f"(Manager named unknown model {raw_name!r}; using automatic routing.)"
        reasoning = (reasoning + " " + note).strip() if reasoning else note
        return "", reasoning

    @staticmethod
    def _validate_dag(subtasks: list[SubTask]) -> None:
        """Detect cycles and self-dependencies via topological sort."""
        ids = {s.id for s in subtasks}
        for s in subtasks:
            s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]

        indegree = {s.id: 0 for s in subtasks}
        adj: dict[str, list[str]] = {s.id: [] for s in subtasks}
        for s in subtasks:
            for dep in s.depends_on:
                adj[dep].append(s.id)
                indegree[s.id] += 1

        queue = [sid for sid, d in indegree.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for nxt in adj[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(subtasks):
            raise PlanError("Planner produced a cyclic dependency graph.")
