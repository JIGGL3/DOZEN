"""Bounded artifact-aware decomposition (production-hardening Phase 4B).

WHY THIS EXISTS
---------------
Phase 4A gave DOZEN a typed vocabulary for artifact deliverables
(``ArtifactManifest`` + ``WorkPackage`` on ``Plan.artifact_plan``), but nothing
consumed it: a planner could still hand an entire multi-file project to one
unbounded worker response. This module makes artifact plans EXECUTABLE in a
bounded way, and ONLY that:

    DecompositionPolicy        → the single authoritative boundedness limits
    DecompositionCheck         → fatal/advisory validation result (planner flow)
    validate_artifact_decomposition
                               → package↔subtask mapping + DAG consistency
    ArtifactExecutionScope     → the immutable artifact scope of ONE subtask
    derive_execution_scope     → Plan.artifact_plan + SubTask.id → scope

Everything here is pure, deterministic and immutable. Nothing touches the
filesystem, calls a provider, assembles artifacts, or contains file contents.
Assembly, conflict resolution, truncation detection and repair are later
phases.

Like ``intent.validate_plan_against_contract``, the plan-level validator is
duck-typed over the ``Plan`` shape (``subtasks`` / ``artifact_plan`` /
``is_direct``) so this module never imports ``models`` — ``models`` imports
the scope type from here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .artifacts import (
    ArtifactKind,
    ArtifactOperation,
    ArtifactSpec,
    ArtifactWorkPlan,
    WorkPackage,
    WorkPackageKind,
    _clip,  # shared field sanitizer: collapses whitespace (kills newline injection)
    _str_tuple,
)

# --------------------------------------------------------------------------- #
# Part A — boundedness policy (the ONE authoritative home for these limits)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecompositionPolicy:
    """When is one work package (and one subtask's combined scope) bounded?

    Deterministic and immutable; requires no provider call and no filesystem
    access. Defaults are conservative fractions of the Phase 4A plan-wide
    limits (80 artifacts / 24 packages / 40 validations) chosen so ordinary
    small projects always fit in one implementation package.
    """

    # Per-package limits (also applied to the COMBINED packages of a subtask).
    max_owned_artifacts_per_package: int = 20
    max_input_artifacts_per_package: int = 16
    max_validations_per_package: int = 10
    # How many packages may share one supposedly bounded subtask.
    max_packages_per_subtask: int = 4
    # Hard cap on any rendered package-scope prompt block.
    max_scope_render_chars: int = 4000
    # A manifest with more ownable artifacts than this is "large": it must be
    # split, so one package owning >= ``wholesale_ownership_percent`` of it is
    # a fatal wholesale assignment even below the per-package owns limit.
    # Deliberately well below ``max_owned_artifacts_per_package``: the owns
    # limit alone cannot catch "one package took the whole 11-file project",
    # while a small project (<= this many artifacts) may still legitimately
    # use a single implementation package.
    single_package_manifest_threshold: int = 10
    wholesale_ownership_percent: int = 80
    # Advisory nearness: a package at >= this percent of a hard limit warns.
    near_limit_percent: int = 80


DEFAULT_DECOMPOSITION_POLICY = DecompositionPolicy()


# --------------------------------------------------------------------------- #
# Part I — validation result (compatible with the planner correction flow)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecompositionCheck:
    """Result of decomposition validation.

    ``fatal`` problems consume the ONE shared corrective re-plan (Phase 3 /
    4A / 4B share it); ``advisory`` warnings never trigger a planner call.
    """

    fatal: tuple[str, ...] = ()
    advisory: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.fatal

    def feedback(self) -> str:
        """Corrective feedback: FATAL problems only, never advisories."""
        return "; ".join(self.fatal)


# --------------------------------------------------------------------------- #
# Part B — which packages owe an executable subtask
# --------------------------------------------------------------------------- #
def package_requires_subtask(package: WorkPackage) -> bool:
    """Whether this work package must map to an executable subtask.

    Any package that produces artifacts (implementation / modification /
    integration / validation evidence / a review deliverable it owns) or that
    claims responsibility for validations is executable. Only a purely
    declarative package — owning nothing, producing nothing, validating
    nothing (in practice a COORDINATION package) — may stay unmapped.
    """
    return bool(
        package.owns or package.output_artifact_ids or package.validation_ids
    )


def _dedup(items: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _relevant_validation_ids(
    manifest: Any,
    packages: Any,
    owned_artifact_ids: Any,
) -> tuple[str, ...]:
    """Validations a package scope must know about, in deterministic order.

    Claimed validations come first.  A validation that directly targets an
    assigned package or one of its owned artifacts is equally relevant even
    when the package did not repeat its id in ``validation_ids``.  Manifest-
    wide validations apply to every producing scope.
    """
    assigned_ids = {package.id for package in packages}
    owned_ids = set(owned_artifact_ids)
    claimed = (
        validation_id
        for package in packages
        for validation_id in package.validation_ids
    )
    targeted = (
        validation.id
        for validation in manifest.validations
        if (
            validation.target_package_id in assigned_ids
            or bool(owned_ids.intersection(validation.target_artifact_ids))
            or (
                not validation.target_package_id
                and not validation.target_artifact_ids
            )
        )
    )
    return _dedup((*claimed, *targeted))


def _subtask_upstream_closure(subtasks: Any) -> dict[str, set[str]]:
    """Transitive ``depends_on`` closure per subtask id (DAG assumed valid)."""
    direct = {s.id: tuple(s.depends_on) for s in subtasks}
    closure: dict[str, set[str]] = {}

    def resolve(sid: str, trail: set[str]) -> set[str]:
        if sid in closure:
            return closure[sid]
        reachable: set[str] = set()
        for dep in direct.get(sid, ()):
            if dep not in direct or dep in trail:
                continue  # unknown refs / defensive cycle guard
            reachable.add(dep)
            reachable |= resolve(dep, trail | {sid})
        closure[sid] = reachable
        return reachable

    for sid in direct:
        resolve(sid, set())
    return closure


# --------------------------------------------------------------------------- #
# Parts A+B+C — deterministic plan-level validation
# --------------------------------------------------------------------------- #
def validate_artifact_decomposition(
    plan: Any,
    policy: DecompositionPolicy = DEFAULT_DECOMPOSITION_POLICY,
) -> DecompositionCheck:
    """Validate that ``plan.artifact_plan`` decomposes into bounded, mapped,
    correctly ordered execution units.

    Pure and deterministic. Plans without an artifact plan are trivially
    valid (Phase 1–4A behavior is preserved bit-for-bit), and a direct-answer
    plan has no execution units to bound, so it is not judged here — the
    Phase 3 final-answer guard remains its backstop.
    """
    work_plan = getattr(plan, "artifact_plan", None)
    if work_plan is None or not isinstance(work_plan, ArtifactWorkPlan):
        return DecompositionCheck()
    if plan.is_direct():
        return DecompositionCheck()

    fatal: list[str] = []
    advisory: list[str] = []
    manifest = work_plan.manifest
    specs_by_id = manifest.by_id()
    mapping = {pid: sid for pid, sid in work_plan.subtask_map}
    subtask_ids = {s.id for s in plan.subtasks}

    # ---- Part A: per-package boundedness -------------------------------- #
    ownable = [spec for spec in manifest.artifacts if not spec.external]
    manifest_has_signal = bool(
        manifest.completion_criteria or manifest.validations
    )
    validated_targets: set[str] = set()
    validated_packages: set[str] = set()
    for validation in manifest.validations:
        validated_targets.update(validation.target_artifact_ids)
        if validation.target_package_id:
            validated_packages.add(validation.target_package_id)

    for package in work_plan.packages:
        owns = len(package.owns)
        if owns > policy.max_owned_artifacts_per_package:
            fatal.append(
                f"package {package.id!r} owns {owns} artifacts; the bounded "
                f"limit is {policy.max_owned_artifacts_per_package} — split it "
                "into smaller packages"
            )
        elif owns * 100 >= (
            policy.max_owned_artifacts_per_package * policy.near_limit_percent
        ):
            advisory.append(
                f"package {package.id!r} owns {owns} artifacts, near the "
                f"limit of {policy.max_owned_artifacts_per_package}"
            )
        inputs = len(package.input_artifact_ids)
        if inputs > policy.max_input_artifacts_per_package:
            fatal.append(
                f"package {package.id!r} consumes {inputs} input artifacts; "
                f"the bounded limit is {policy.max_input_artifacts_per_package}"
            )
        relevant_validations = _relevant_validation_ids(
            manifest, (package,), package.owns
        )
        validations = len(relevant_validations)
        if validations > policy.max_validations_per_package:
            fatal.append(
                f"package {package.id!r} has {validations} relevant validations; "
                f"the bounded limit is {policy.max_validations_per_package}"
            )
        if package.owns and not (package.objective or package.title):
            fatal.append(
                f"package {package.id!r} produces artifacts but declares no "
                "objective"
            )
        # Wholesale assignment: a large manifest may not be ~entirely owned by
        # one package even when the raw owns-count limit is respected.
        if (
            len(ownable) > policy.single_package_manifest_threshold
            and owns * 100 >= len(ownable) * policy.wholesale_ownership_percent
        ):
            fatal.append(
                f"package {package.id!r} owns {owns} of the manifest's "
                f"{len(ownable)} artifacts — a large manifest must be split "
                "into multiple bounded packages, not assigned wholesale"
            )
        # Observable completion: a package producing required artifacts must
        # be checkable somewhere — its own criteria, a claimed validation, or
        # a manifest validation aimed at it / its artifacts. When only the
        # manifest-wide signal covers it, that is sparse but not absent.
        produces_required = any(
            specs_by_id[aid].required
            for aid in package.owns
            if aid in specs_by_id
        )
        if produces_required and package_requires_subtask(package):
            has_own_criterion = bool(
                package.completion_criteria
                or package.validation_ids
                or package.id in validated_packages
                or any(aid in validated_targets for aid in package.owns)
                or any(
                    specs_by_id[aid].completion_criteria
                    for aid in package.owns
                    if aid in specs_by_id
                )
            )
            if not has_own_criterion and not manifest_has_signal:
                fatal.append(
                    f"package {package.id!r} produces required artifacts but "
                    "has no observable completion criterion (add completion "
                    "criteria or a validation)"
                )
            elif not has_own_criterion:
                advisory.append(
                    f"package {package.id!r} has no package-level completion "
                    "criterion; only manifest-wide validation covers it"
                )

    # Unassigned OPTIONAL artifacts are legal (Phase 4A already fails required
    # orphans), but worth surfacing.
    owner_of = {
        aid: package.id
        for package in work_plan.packages
        for aid in package.owns
    }
    for spec in manifest.artifacts:
        if not spec.required and not spec.external and spec.id not in owner_of:
            advisory.append(
                f"optional artifact {spec.id!r} is not assigned to any package"
            )

    # ---- Part B: executable-package mapping ------------------------------ #
    for package_id, subtask_id in work_plan.subtask_map:
        if subtask_id not in subtask_ids:
            fatal.append(
                f"package {package_id!r} maps to unknown subtask {subtask_id!r}"
            )
    for package in work_plan.packages:
        if package_requires_subtask(package):
            if package.id not in mapping:
                fatal.append(
                    f"executable package {package.id!r} "
                    f"({package.kind.value}) has no mapped subtask — every "
                    "package that produces artifacts or claims validations "
                    "must name the delegation that performs it"
                )
        elif package.id not in mapping:
            advisory.append(
                f"package {package.id!r} ({package.kind.value}) remains "
                "declarative (no owned artifacts, no validations, no subtask)"
            )

    # Combined scope of packages sharing one subtask must remain bounded.
    grouped: dict[str, list[WorkPackage]] = {}
    for package in work_plan.packages:
        subtask_id = mapping.get(package.id, "")
        if subtask_id:
            grouped.setdefault(subtask_id, []).append(package)
    for subtask_id, group in grouped.items():
        if len(group) > policy.max_packages_per_subtask:
            fatal.append(
                f"{len(group)} packages map to subtask {subtask_id!r}; at "
                f"most {policy.max_packages_per_subtask} bounded packages may "
                "share one subtask"
            )
        combined_owns = _dedup(
            aid for package in group for aid in package.owns
        )
        if len(combined_owns) > policy.max_owned_artifacts_per_package:
            fatal.append(
                f"the packages mapped to subtask {subtask_id!r} own "
                f"{len(combined_owns)} artifacts combined; the bounded limit "
                f"per subtask is {policy.max_owned_artifacts_per_package}"
            )
        if (
            len(ownable) > policy.single_package_manifest_threshold
            and len(combined_owns) * 100
            >= len(ownable) * policy.wholesale_ownership_percent
        ):
            fatal.append(
                f"the packages mapped to subtask {subtask_id!r} collectively "
                f"own {len(combined_owns)} of the manifest's "
                f"{len(ownable)} artifacts — a large manifest must not be "
                "assigned wholesale to one delegation"
            )
        combined_inputs = _dedup(
            aid
            for package in group
            for aid in package.input_artifact_ids
            if aid not in combined_owns
        )
        if len(combined_inputs) > policy.max_input_artifacts_per_package:
            fatal.append(
                f"the packages mapped to subtask {subtask_id!r} consume "
                f"{len(combined_inputs)} input artifacts combined; the "
                f"bounded limit is {policy.max_input_artifacts_per_package}"
            )
        combined_validations = _relevant_validation_ids(
            manifest, group, combined_owns
        )
        if len(combined_validations) > policy.max_validations_per_package:
            fatal.append(
                f"the packages mapped to subtask {subtask_id!r} have "
                f"{len(combined_validations)} validations combined; the "
                f"bounded limit is {policy.max_validations_per_package}"
            )

    # ---- Part C: package graph ↔ subtask DAG consistency ----------------- #
    upstream = _subtask_upstream_closure(plan.subtasks)
    for package in work_plan.packages:
        package_subtask = mapping.get(package.id, "")
        if not package_subtask or package_subtask not in subtask_ids:
            continue
        for dependency_id in package.depends_on:
            dependency_subtask = mapping.get(dependency_id, "")
            if not dependency_subtask or dependency_subtask not in subtask_ids:
                continue  # unmapped executables already failed above
            if dependency_subtask == package_subtask:
                continue  # same subtask: ordering is internal
            if dependency_subtask not in upstream.get(package_subtask, set()):
                if package.kind is WorkPackageKind.VALIDATION:
                    what = (
                        f"validation package {package.id!r} would execute "
                        f"before producer package {dependency_id!r}"
                    )
                elif package.kind is WorkPackageKind.INTEGRATION:
                    what = (
                        f"integration package {package.id!r} would execute "
                        f"before its input package {dependency_id!r}"
                    )
                else:
                    what = (
                        f"package {package.id!r} depends on package "
                        f"{dependency_id!r}"
                    )
                fatal.append(
                    f"{what}, but its subtask {package_subtask!r} does not "
                    f"depend (directly or transitively) on subtask "
                    f"{dependency_subtask!r} — mirror package dependencies in "
                    "the delegation depends_on graph"
                )

    return DecompositionCheck(fatal=tuple(fatal), advisory=tuple(advisory))


# --------------------------------------------------------------------------- #
# Part D — the immutable artifact scope of ONE executable subtask
# --------------------------------------------------------------------------- #
_SCOPE_LINE_CHARS = 200
_SCOPE_PATH_CHARS = 120
_SCOPE_MAX_CRITERIA = 8

_WORKER_SCOPE_RULES = (
    "Required output:\n"
    "- Complete contents for every owned artifact listed above\n"
    "- No placeholders and no architecture-only substitution\n"
    "- No unrelated files, and no files owned by other packages"
)
_VERIFIER_SCOPE_RULES = (
    "Judge ONLY the response text against this scope: it should address "
    "every owned artifact and the package objective, and satisfy the listed "
    "validation expectations. Flag output that claims artifacts outside this "
    "scope. Never assume any file was actually written to disk."
)
_PLANNER_SCOPE_RULES = (
    "Plan strictly within this scope. The final deliverable must contain "
    "complete contents for the owned artifacts above — do not add unrelated "
    "files, do not redefine artifact ownership, and do not declare a new "
    "artifact manifest. Intermediate research or design subtasks are allowed "
    "when they feed the owned artifacts."
)
_SCOPED_OUTPUT_RULES = (
    'Set "produces_parent_artifacts": true on exactly ONE delegation — the '
    "one responsible for returning the complete owned artifacts. Set it to "
    "false (or omit it) on research, design, review, and support delegations."
)
_SCOPED_SUPPORT_RULES = (
    'Set "produces_parent_artifacts": false (or omit it) on EVERY delegation. '
    "This is an intermediate support plan: return only the requested support "
    "result and do not produce or redefine the parent-owned artifacts."
)
_SYNTHESIZER_OUTPUT_RULES = (
    "Synthesize strictly within this assigned scope. Preserve the complete "
    "contents of every owned artifact from the output-producing delegation; "
    "do not add unrelated files or files owned by another package."
)
_SYNTHESIZER_SUPPORT_RULES = (
    "Synthesize only the requested intermediate support result within this "
    "boundary. Do not produce, redefine, or claim completion of the owned "
    "artifacts."
)


def _sanitize_block(text: str, limit: int) -> str:
    """Per-line whitespace sanitation for a multi-line brief, then a hard cap."""
    lines = [_clip(line, _SCOPE_LINE_CHARS) for line in str(text or "").splitlines()]
    block = "\n".join(line for line in lines if line)
    if len(block) > limit:
        block = block[: max(0, limit - 1)].rstrip() + "…"
    return block


@dataclass(frozen=True)
class ArtifactExecutionScope:
    """The artifact scope assigned to one executable subtask. Declarations
    only — never file contents, filesystem handles, or provider names.

    Derivable deterministically from ``Plan.artifact_plan + SubTask.id`` via
    :func:`derive_execution_scope`; carried by recursion so a child never
    sees unrelated root packages.
    """

    manifest_id: str
    subtask_id: str
    package_ids: tuple[str, ...]
    owned_artifact_ids: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    # Assigned packages' dependencies OUTSIDE this subtask (context only).
    package_dependencies: tuple[str, ...] = ()
    # Pre-rendered, bounded, newline-sanitized package brief (ids AND paths).
    brief: str = ""
    # The authoritative policy cap used by every rendered scope block.
    max_render_chars: int = DEFAULT_DECOMPOSITION_POLICY.max_scope_render_chars
    # ---- Phase 4C: the contract this subtask's produced artifacts are checked
    # against. Declarations only (the Phase 4A specs of the artifacts it owns,
    # their owning package, and the manifest's id space) — never file contents,
    # and never rendered into any prompt, so the bounded worker/verifier/planner
    # blocks above stay byte-identical. Carried across recursion so a nested
    # child can validate its own submissions without the root work plan.
    owned_specs: tuple[ArtifactSpec, ...] = ()
    artifact_owners: tuple[tuple[str, str], ...] = ()
    known_artifact_ids: tuple[str, ...] = ()
    external_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        manifest_id = _clip(self.manifest_id, 300)
        subtask_id = _clip(self.subtask_id, 300)
        if not manifest_id:
            raise ValueError("execution scope requires a manifest id")
        if not subtask_id:
            raise ValueError("execution scope requires a subtask id")
        object.__setattr__(self, "manifest_id", manifest_id)
        object.__setattr__(self, "subtask_id", subtask_id)
        for name in (
            "package_ids",
            "owned_artifact_ids",
            "input_artifact_ids",
            "output_artifact_ids",
            "validation_ids",
            "package_dependencies",
        ):
            object.__setattr__(
                self, name, _str_tuple(getattr(self, name), clip=300)
            )
        if not self.package_ids:
            raise ValueError("execution scope requires at least one package id")
        object.__setattr__(
            self,
            "completion_criteria",
            _str_tuple(self.completion_criteria, clip=_SCOPE_LINE_CHARS)[
                :_SCOPE_MAX_CRITERIA
            ],
        )
        try:
            max_render_chars = max(1, int(self.max_render_chars))
        except (TypeError, ValueError):
            max_render_chars = DEFAULT_DECOMPOSITION_POLICY.max_scope_render_chars
        object.__setattr__(self, "max_render_chars", max_render_chars)
        object.__setattr__(
            self, "brief", _sanitize_block(self.brief, max_render_chars)
        )
        specs = () if self.owned_specs is None else self.owned_specs
        if not isinstance(specs, (list, tuple)):
            raise ValueError("execution scope owned_specs must be a sequence")
        for spec in specs:
            if not isinstance(spec, ArtifactSpec):
                raise ValueError(
                    "execution scope owned_specs must contain ArtifactSpec items"
                )
        object.__setattr__(self, "owned_specs", tuple(specs))
        owners = () if self.artifact_owners is None else self.artifact_owners
        if not isinstance(owners, (list, tuple)):
            raise ValueError("execution scope artifact_owners must be a sequence")
        pairs: list[tuple[str, str]] = []
        for item in owners:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("malformed execution scope artifact_owners entry")
            pairs.append((_clip(item[0], 300), _clip(item[1], 300)))
        object.__setattr__(self, "artifact_owners", tuple(pairs))
        for name in ("known_artifact_ids", "external_artifact_ids"):
            object.__setattr__(
                self, name, _str_tuple(getattr(self, name), clip=300)
            )

    def _render_block(self, header: str, rules: str) -> str:
        """Render declarations plus fixed rules within the exact hard cap."""
        fixed_chars = len(header) + len(rules) + 2
        brief_limit = max(0, self.max_render_chars - fixed_chars)
        brief = _sanitize_block(self.brief, brief_limit) if brief_limit else ""
        block = f"{header}\n{brief}\n{rules}" if brief else f"{header}\n{rules}"
        if len(block) > self.max_render_chars:
            return block[: max(0, self.max_render_chars - 1)].rstrip() + "…"
        return block

    # ------------------------------- prompts --------------------------- #
    def to_worker_block(self) -> str:
        """Bounded worker-prompt block: the subtask's OWN scope, nothing else."""
        return self._render_block("ASSIGNED ARTIFACT PACKAGE", _WORKER_SCOPE_RULES)

    def to_verifier_block(self) -> str:
        return self._render_block(
            "ASSIGNED ARTIFACT PACKAGE (for this subtask)",
            _VERIFIER_SCOPE_RULES,
        )

    def to_planner_block(self, *, output_required: bool = True) -> str:
        """Bounded block for a recursive child's planner."""
        assignment_rules = (
            _SCOPED_OUTPUT_RULES if output_required else _SCOPED_SUPPORT_RULES
        )
        return self._render_block(
            "ASSIGNED ARTIFACT SCOPE (inherited from the parent plan)",
            f"{_PLANNER_SCOPE_RULES} {assignment_rules}",
        )

    def to_synthesizer_block(self, *, output_required: bool = True) -> str:
        """Bounded final-synthesis rule for a recursive scoped task."""
        rules = (
            _SYNTHESIZER_OUTPUT_RULES
            if output_required else _SYNTHESIZER_SUPPORT_RULES
        )
        return self._render_block("ASSIGNED ARTIFACT SCOPE", rules)


def validate_recursive_scope_assignment(
    plan: Any,
    scope: Optional[ArtifactExecutionScope],
    *,
    output_required: bool = True,
) -> DecompositionCheck:
    """Ensure one recursive child — never every child — inherits ownership."""
    if scope is None:
        return DecompositionCheck()
    if plan.is_direct():
        if not output_required:
            return DecompositionCheck()
        return DecompositionCheck(fatal=(
            "a scoped recursive output plan must delegate exactly one "
            "output producer; a direct answer would bypass the scoped worker "
            "and verifier",
        ))
    producers = tuple(
        subtask.id
        for subtask in plan.subtasks
        if getattr(subtask, "produces_parent_artifacts", False)
    )
    expected = 1 if output_required else 0
    if len(producers) == expected:
        return DecompositionCheck()
    if output_required:
        return DecompositionCheck(fatal=(
            "a scoped recursive plan must mark exactly one delegation with "
            f"produces_parent_artifacts=true; found {len(producers)}",
        ))
    return DecompositionCheck(fatal=(
        "an intermediate scoped support plan must not mark any delegation "
        f"with produces_parent_artifacts=true; found {len(producers)}",
    ))


def derive_execution_scope(
    work_plan: ArtifactWorkPlan,
    subtask_id: str,
    policy: DecompositionPolicy = DEFAULT_DECOMPOSITION_POLICY,
) -> Optional[ArtifactExecutionScope]:
    """Deterministically derive one subtask's artifact scope from the plan.

    Returns ``None`` when no package maps to ``subtask_id`` (that subtask
    executes exactly as before Phase 4B). Pure: no provider, no filesystem.
    """
    assigned = [
        package
        for package in work_plan.packages
        if work_plan.subtask_for(package.id) == subtask_id
    ]
    if not assigned:
        return None

    manifest = work_plan.manifest
    specs_by_id = manifest.by_id()
    validations_by_id = {v.id: v for v in manifest.validations}
    assigned_ids = {package.id for package in assigned}

    owned = _dedup(aid for package in assigned for aid in package.owns)
    outputs = _dedup(
        aid
        for package in assigned
        for aid in (package.output_artifact_ids or package.owns)
    )
    inputs = _dedup(
        aid
        for package in assigned
        for aid in package.input_artifact_ids
        if aid not in owned
    )
    validation_ids = _relevant_validation_ids(
        manifest, assigned, owned
    )
    criteria = _dedup(
        criterion
        for package in assigned
        for criterion in package.completion_criteria
    )
    criteria = _dedup(
        (
            *criteria,
            *(
                criterion
                for artifact_id in owned
                for criterion in (
                    specs_by_id[artifact_id].completion_criteria
                    if artifact_id in specs_by_id else ()
                )
            ),
            *manifest.completion_criteria,
        )
    )
    external_deps = _dedup(
        dep
        for package in assigned
        for dep in package.depends_on
        if dep not in assigned_ids
    )

    # ---- bounded brief (ids AND paths; only THIS subtask's packages) ----- #
    lines: list[str] = ["Packages:"]
    for package in assigned[: policy.max_packages_per_subtask]:
        summary = package.objective or package.title
        suffix = f" — {_clip(summary, _SCOPE_LINE_CHARS)}" if summary else ""
        lines.append(f"- {package.id} [{package.kind.value}]{suffix}")
    if owned:
        lines.append("You own (complete contents required for each):")
        shown = owned[: policy.max_owned_artifacts_per_package]
        for aid in shown:
            spec = specs_by_id.get(aid)
            path = spec.path if spec else aid
            # Phase 4C: produced artifacts are collected BY ID, so a worker must
            # be able to name the id. It is shown only when it differs from the
            # path — with the usual path-derived ids the brief is unchanged.
            label = path if spec is None or spec.id == path else f"{aid} → {path}"
            lines.append(f"- {_clip(label, _SCOPE_PATH_CHARS)}")
        if len(owned) > len(shown):
            lines.append(f"- … (+{len(owned) - len(shown)} more owned artifacts)")
    if inputs:
        lines.append("Inputs (produced outside this subtask; do not rewrite them):")
        shown = inputs[: policy.max_input_artifacts_per_package]
        for aid in shown:
            spec = specs_by_id.get(aid)
            lines.append(f"- {_clip(spec.path if spec else aid, _SCOPE_PATH_CHARS)}")
        if len(inputs) > len(shown):
            lines.append(f"- … (+{len(inputs) - len(shown)} more inputs)")
    if validation_ids:
        lines.append("Validation:")
        for vid in validation_ids[: policy.max_validations_per_package]:
            validation = validations_by_id.get(vid)
            if validation is None:
                lines.append(f"- {_clip(vid, _SCOPE_LINE_CHARS)}")
                continue
            detail = validation.success_criterion or validation.command
            suffix = f": {_clip(detail, _SCOPE_LINE_CHARS)}" if detail else ""
            lines.append(f"- {validation.kind.value}{suffix}")
    if criteria:
        lines.append("Done when:")
        for criterion in criteria[:_SCOPE_MAX_CRITERIA]:
            lines.append(f"- {_clip(criterion, _SCOPE_LINE_CHARS)}")
    if external_deps:
        rendered = ", ".join(_clip(dep, 60) for dep in external_deps[:6])
        if len(external_deps) > 6:
            rendered += f" (+{len(external_deps) - 6} more)"
        lines.append(
            f"Upstream packages (their outputs arrive as inputs): {rendered}"
        )
    brief = _sanitize_block("\n".join(lines), policy.max_scope_render_chars)

    # Phase 4C: the artifact contract this subtask's submissions are checked
    # against, carried on the scope so a recursive child validates its own
    # produced artifacts without ever seeing the root work plan.
    owned_specs = tuple(
        specs_by_id[aid] for aid in owned if aid in specs_by_id
    )
    artifact_owners = tuple(
        (aid, package.id)
        for package in assigned
        for aid in package.owns
        if aid in specs_by_id
    )

    return ArtifactExecutionScope(
        manifest_id=manifest.id,
        subtask_id=subtask_id,
        package_ids=tuple(package.id for package in assigned),
        owned_artifact_ids=owned,
        input_artifact_ids=inputs,
        output_artifact_ids=outputs,
        validation_ids=validation_ids,
        completion_criteria=criteria,
        package_dependencies=external_deps,
        brief=brief,
        max_render_chars=policy.max_scope_render_chars,
        owned_specs=owned_specs,
        artifact_owners=artifact_owners,
        known_artifact_ids=manifest.artifact_ids(),
        external_artifact_ids=tuple(
            spec.id for spec in manifest.artifacts if spec.external
        ),
    )


# --------------------------------------------------------------------------- #
# Phase 4F Part I — output-aware package sizing
#
# The live failure showed workers refusing or truncating because a package
# demanded too many large source files in ONE response. These estimates never
# predict exact code length; they reliably prevent OBVIOUSLY oversized
# packages, deterministically and without any provider call.
# --------------------------------------------------------------------------- #
PACKAGE_SIZING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PackageSizingPolicy:
    """THE authoritative, immutable output-sizing limits for work packages."""

    # Maximum estimated response output (characters) one package may demand.
    max_package_output_chars: int = 20_000
    # At most this many substantial (non-trivial) files in one package.
    max_substantial_files_per_package: int = 3
    # At most this many small config-sized files in one package.
    max_small_files_per_package: int = 10
    # An artifact at or above this estimate counts as substantial.
    substantial_file_chars: int = 3_000
    # An artifact at or above this estimate is LARGE: it deserves its own
    # package and must never share one with another produced artifact.
    large_artifact_chars: int = 9_000
    # Safety margin applied to a package's summed estimate (percent).
    safety_margin_percent: int = 15
    # Provider max_tokens is an output-token ceiling. Convert it to a
    # conservative character budget before dispatch and keep a reserve for
    # tokenizer/content variance.
    # Five is the repository's established package-policy calibration: the
    # default 4,096-token AgentSpec can carry the approved three-standard-file
    # package after the reserve, while smaller provider caps scale down.
    estimated_chars_per_output_token: int = 5
    provider_budget_utilization_percent: int = 90
    # Per-artifact default estimates by kind (characters of produced output).
    source_file_chars: int = 5_000
    test_file_chars: int = 4_500
    config_file_chars: int = 1_200
    document_chars: int = 3_500
    data_file_chars: int = 1_500
    patch_chars: int = 2_500
    result_chars: int = 600
    other_chars: int = 2_000
    # Deterministic complexity bonuses.
    criterion_bonus_chars: int = 400
    description_bonus_unit_chars: int = 60
    description_bonus_chars: int = 200
    schema_version: int = PACKAGE_SIZING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "max_package_output_chars", "max_substantial_files_per_package",
            "max_small_files_per_package", "substantial_file_chars",
            "large_artifact_chars", "source_file_chars", "test_file_chars",
            "config_file_chars", "document_chars", "data_file_chars",
            "patch_chars", "result_chars", "other_chars",
            "criterion_bonus_chars", "description_bonus_unit_chars",
            "description_bonus_chars", "estimated_chars_per_output_token",
            "schema_version",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        margin = int(self.safety_margin_percent)
        if margin < 0:
            raise ValueError("safety_margin_percent must be non-negative")
        object.__setattr__(self, "safety_margin_percent", margin)
        utilization = int(self.provider_budget_utilization_percent)
        if not 1 <= utilization <= 100:
            raise ValueError(
                "provider_budget_utilization_percent must be between 1 and 100"
            )
        object.__setattr__(
            self, "provider_budget_utilization_percent", utilization
        )

    def with_margin(self, total: int) -> int:
        return total + (total * self.safety_margin_percent) // 100


DEFAULT_PACKAGE_SIZING_POLICY = PackageSizingPolicy()

# Extensions whose produced output is config-sized regardless of declared kind.
_CONFIG_EXTENSIONS = frozenset({
    "json", "yaml", "yml", "toml", "ini", "cfg", "env", "properties", "lock",
})
_DOCUMENT_EXTENSIONS = frozenset({"md", "rst", "txt", "adoc"})
_DATA_EXTENSIONS = frozenset({"csv", "tsv", "jsonl"})

_KIND_BASE_FIELD = {
    ArtifactKind.SOURCE_FILE: "source_file_chars",
    ArtifactKind.TEST_FILE: "test_file_chars",
    ArtifactKind.CONFIG_FILE: "config_file_chars",
    ArtifactKind.DOCUMENT: "document_chars",
    ArtifactKind.DATA_FILE: "data_file_chars",
    ArtifactKind.PATCH: "patch_chars",
    ArtifactKind.COMMAND_RESULT: "result_chars",
    ArtifactKind.BUILD_RESULT: "result_chars",
    ArtifactKind.TEST_RESULT: "result_chars",
}


def _path_extension(path: str) -> str:
    name = str(path or "").rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def estimate_artifact_output_chars(
    spec: ArtifactSpec,
    policy: PackageSizingPolicy = DEFAULT_PACKAGE_SIZING_POLICY,
) -> int:
    """Deterministic per-artifact output estimate (characters).

    Kind decides the base; a config/document/data file extension refines a
    generic or source declaration downward; completion criteria and a complex
    description add bounded deterministic bonuses. Artifacts that produce no
    file body (external inputs, deletes, directories) estimate zero.
    """
    if spec.external:
        return 0
    if spec.operation is ArtifactOperation.DELETE:
        return 0
    if spec.kind is ArtifactKind.DIRECTORY:
        return 0
    if spec.operation is ArtifactOperation.INSPECT:
        return policy.result_chars

    extension = _path_extension(spec.path)
    if spec.kind in (ArtifactKind.SOURCE_FILE, ArtifactKind.OTHER):
        if extension in _CONFIG_EXTENSIONS:
            base = policy.config_file_chars
        elif extension in _DOCUMENT_EXTENSIONS:
            base = policy.document_chars
        elif extension in _DATA_EXTENSIONS:
            base = policy.data_file_chars
        elif spec.kind is ArtifactKind.SOURCE_FILE:
            base = policy.source_file_chars
        else:
            base = policy.other_chars
    else:
        base = getattr(policy, _KIND_BASE_FIELD.get(spec.kind, "other_chars"))

    bonus = len(spec.completion_criteria) * policy.criterion_bonus_chars
    description_units = min(len(spec.description), 300) // max(
        1, policy.description_bonus_unit_chars
    )
    bonus += description_units * policy.description_bonus_chars
    return base + bonus


def estimate_package_output_chars(
    package: WorkPackage,
    manifest: Any,
    policy: PackageSizingPolicy = DEFAULT_PACKAGE_SIZING_POLICY,
) -> int:
    """Margin-adjusted estimate of one package's demanded response output."""
    specs_by_id = manifest.by_id()
    total = sum(
        estimate_artifact_output_chars(specs_by_id[artifact_id], policy)
        if artifact_id in specs_by_id else policy.other_chars
        for artifact_id in package.owns
    )
    return policy.with_margin(total)


def provider_output_budget_chars(
    max_tokens: int,
    policy: PackageSizingPolicy = DEFAULT_PACKAGE_SIZING_POLICY,
) -> int:
    """Safe usable output characters for one provider call."""
    tokens = max(1, int(max_tokens))
    provider_chars = (
        tokens
        * policy.estimated_chars_per_output_token
        * policy.provider_budget_utilization_percent
    ) // 100
    return min(policy.max_package_output_chars, max(1, provider_chars))


def estimate_scope_output_chars(
    scope: ArtifactExecutionScope,
    policy: PackageSizingPolicy = DEFAULT_PACKAGE_SIZING_POLICY,
) -> int:
    """Margin-adjusted output estimate for a runtime artifact scope."""
    output_ids = set(scope.output_artifact_ids)
    specs_by_id = {spec.id: spec for spec in scope.owned_specs}
    total = sum(
        estimate_artifact_output_chars(specs_by_id[artifact_id], policy)
        if artifact_id in specs_by_id else policy.other_chars
        for artifact_id in output_ids
    )
    return policy.with_margin(total)


def _package_estimates(
    owns: tuple[str, ...],
    specs_by_id: dict[str, ArtifactSpec],
    policy: PackageSizingPolicy,
) -> dict[str, int]:
    return {
        artifact_id: (
            estimate_artifact_output_chars(specs_by_id[artifact_id], policy)
            if artifact_id in specs_by_id else policy.other_chars
        )
        for artifact_id in owns
    }


def _sizing_problems(
    label: str,
    owns: tuple[str, ...],
    specs_by_id: dict[str, ArtifactSpec],
    policy: PackageSizingPolicy,
    *,
    max_output_chars: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Fatal/advisory sizing findings for one package (or combined subtask)."""
    fatal: list[str] = []
    advisory: list[str] = []
    estimates = _package_estimates(owns, specs_by_id, policy)
    budget = (
        policy.max_package_output_chars
        if max_output_chars is None else max(1, int(max_output_chars))
    )
    produced = [aid for aid, estimate in estimates.items() if estimate > 0]
    if not produced:
        return fatal, advisory
    total = policy.with_margin(sum(estimates.values()))
    substantial = [
        aid for aid in produced
        if estimates[aid] >= policy.substantial_file_chars
    ]
    small = [aid for aid in produced if aid not in substantial]
    large = [
        aid for aid in produced if estimates[aid] >= policy.large_artifact_chars
    ]

    if len(produced) == 1:
        # Never fragment one logical file into incoherent response chunks. If
        # it cannot fit, redesign it into smaller logical modules or fail the
        # plan before a provider is asked to truncate/refuse.
        if total > budget:
            fatal.append(
                f"{label} produces one artifact estimated at ~{total} output "
                f"characters (budget {budget}); redesign the artifact into "
                "smaller logical modules or fail planning; never fragment one "
                "file across replies"
            )
        return fatal, advisory

    if total > budget:
        fatal.append(
            f"{label} demands ~{total} estimated output characters "
            f"(budget {budget}) across "
            f"{len(produced)} artifacts — split it into smaller "
            "dependency-aware packages"
        )
    if len(substantial) > policy.max_substantial_files_per_package:
        fatal.append(
            f"{label} bundles {len(substantial)} substantial files; at most "
            f"{policy.max_substantial_files_per_package} may share one "
            "package — split it"
        )
    if len(small) > policy.max_small_files_per_package:
        fatal.append(
            f"{label} bundles {len(small)} small files; at most "
            f"{policy.max_small_files_per_package} may share one package"
        )
    if large:
        fatal.append(
            f"{label} pairs the large artifact(s) "
            f"{', '.join(sorted(large)[:4])} with other produced artifacts — "
            "a large implementation file needs its own package"
        )
    if (
        not fatal
        and total * 100
        >= budget * DEFAULT_DECOMPOSITION_POLICY.near_limit_percent
    ):
        advisory.append(
            f"{label} is near the output budget (~{total} of "
            f"{budget} estimated characters)"
        )
    return fatal, advisory


def validate_package_sizing(
    plan: Any,
    policy: PackageSizingPolicy = DEFAULT_PACKAGE_SIZING_POLICY,
    *,
    subtask_output_budgets: Optional[Mapping[str, int]] = None,
) -> DecompositionCheck:
    """Reject artifact plans whose packages demand obviously oversized output.

    Pure and deterministic. Applied in the planner's ONE corrective re-plan
    flow (like contract/decomposition validation): a plan that cannot fit the
    limits fails planning cleanly instead of creating a package a provider
    will refuse or truncate. Plans without an artifact plan are trivially
    valid; direct answers are not judged here.
    """
    work_plan = getattr(plan, "artifact_plan", None)
    if work_plan is None or not isinstance(work_plan, ArtifactWorkPlan):
        return DecompositionCheck()
    if plan.is_direct():
        return DecompositionCheck()

    fatal: list[str] = []
    advisory: list[str] = []
    specs_by_id = work_plan.manifest.by_id()

    for package in work_plan.packages:
        if not package.owns:
            continue
        package_fatal, package_advisory = _sizing_problems(
            f"package {package.id!r}", package.owns, specs_by_id, policy
        )
        fatal.extend(package_fatal)
        advisory.extend(package_advisory)

    # Packages sharing one delegation answer in ONE response together, so the
    # combined obligation is judged against the same budget.
    grouped: dict[str, list[WorkPackage]] = {}
    for package in work_plan.packages:
        subtask_id = work_plan.subtask_for(package.id) or ""
        if subtask_id:
            grouped.setdefault(subtask_id, []).append(package)
    for subtask_id, group in grouped.items():
        if len(group) < 2:
            continue
        combined_owns = _dedup(
            artifact_id for package in group for artifact_id in package.owns
        )
        group_fatal, group_advisory = _sizing_problems(
            f"the packages mapped to subtask {subtask_id!r}",
            combined_owns, specs_by_id, policy,
        )
        fatal.extend(group_fatal)
        advisory.extend(group_advisory)

    # Provider-specific output ceilings are checked over the complete response
    # obligation of each mapped subtask. The planner supplies budgets from the
    # actual selected AgentSpec; runtime repeats the check after routing.
    for subtask_id, raw_budget in (subtask_output_budgets or {}).items():
        group = grouped.get(str(subtask_id), [])
        if not group:
            continue
        budget = min(policy.max_package_output_chars, max(1, int(raw_budget)))
        if budget >= policy.max_package_output_chars:
            continue
        combined_owns = _dedup(
            artifact_id for package in group for artifact_id in package.owns
        )
        provider_fatal, provider_advisory = _sizing_problems(
            f"subtask {subtask_id!r} for its selected provider",
            combined_owns,
            specs_by_id,
            policy,
            max_output_chars=budget,
        )
        fatal.extend(provider_fatal)
        advisory.extend(provider_advisory)

    return DecompositionCheck(
        fatal=tuple(dict.fromkeys(fatal)),
        advisory=tuple(dict.fromkeys(advisory)),
    )
