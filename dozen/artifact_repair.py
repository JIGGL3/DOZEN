"""Bounded targeted artifact repair (production-hardening Phase 4E).

WHY THIS EXISTS
---------------
Phases 4C/4D let DOZEN collect typed artifacts, trust their provenance, and
detect the ones that are missing, malformed or structurally truncated. Those
failures already feed the EXISTING bounded worker-attempt loop — but the retry
asked the worker to resubmit the COMPLETE assigned package. A package with two
good files and one truncated file therefore re-generated all three, which risks
regressing the good ones, spends the whole output budget again, and makes the
next reply just as likely to be cut off.

This module is the missing seam, and ONLY that seam:

    ArtifactRepairPolicy   → the ONE authoritative home for every repair bound
    RepairReason           → why one artifact must come back again
    ArtifactRepairRequest  → the immutable, serializable targeted request
    ArtifactRepairState    → preserved candidates + per-artifact repair history
    plan_repair            → SubtaskCollectionResult -> next bounded request
    derive_repair_scope    → a NARROW ArtifactExecutionScope over the targets
    collect_repair_artifacts / merge_repair_collection
                           → parse a repair reply, then merge it deterministically

Everything here is pure, deterministic, immutable and serializable. It adds NO
retry loop (the executor's existing attempt budget is the only budget), NO
provider call, NO planner call, no filesystem write, no compilation, no syntax
repair, and no conflict resolution. Accepted candidates are never mutated,
never re-hashed and never overwritten: a repaired artifact may only FILL a
target that is missing or was rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .artifact_results import (
    DEFAULT_RESULT_POLICY,
    ArtifactCandidate,
    CandidateRejection,
    CollectionErrorCode,
    ResultPolicy,
    SubtaskCollectionResult,
    collect_subtask_artifacts,
    content_hash,
)
from .decomposition import ArtifactExecutionScope

SCHEMA_VERSION = 1


class ArtifactRepairError(ValueError):
    """A structural violation of the targeted-repair contracts."""


# --------------------------------------------------------------------------- #
# Part A — the repair vocabulary
# --------------------------------------------------------------------------- #
class RepairReason(str, Enum):
    """Why ONE artifact is being asked for again. Deterministic, typed, bounded."""

    MISSING_REQUIRED = "missing_required"
    INTEGRITY_INVALID = "integrity_invalid"
    TRUNCATION_SUSPECTED = "truncation_suspected"
    INCOMPLETE_DECLARATION = "incomplete_declaration"
    MISSING_CONTENT = "missing_content"
    INVALID_CONTENT = "invalid_content"
    PATH_MISMATCH = "path_mismatch"
    RESPONSE_FORMAT = "response_format"
    NON_REPAIRABLE = "non_repairable"


class RepairStatus(str, Enum):
    """The verdict of one repair derivation."""

    SATISFIED = "satisfied"            # nothing left to ask for
    REPAIRABLE = "repairable"          # a bounded request was produced
    NON_REPAIRABLE = "non_repairable"  # the reply crossed a trust boundary
    EXHAUSTED = "exhausted"            # the EXISTING attempt budget is spent
    NO_PROGRESS = "no_progress"        # the provider is not improving the artifact


# Rejections a targeted regeneration can legitimately fix: the artifact is
# owned, addressed to the right package, and merely came back wrong.
_REPAIRABLE_CODES: dict[CollectionErrorCode, RepairReason] = {
    CollectionErrorCode.STRUCTURALLY_INVALID: RepairReason.INTEGRITY_INVALID,
    CollectionErrorCode.TRUNCATION_SUSPECTED: RepairReason.TRUNCATION_SUSPECTED,
    CollectionErrorCode.INCOMPLETE_DECLARATION: RepairReason.INCOMPLETE_DECLARATION,
    CollectionErrorCode.MISSING_CONTENT: RepairReason.MISSING_CONTENT,
    CollectionErrorCode.INVALID_CONTENT_TYPE: RepairReason.INVALID_CONTENT,
    CollectionErrorCode.PATH_MISMATCH: RepairReason.PATH_MISMATCH,
    CollectionErrorCode.MALFORMED_ENTRY: RepairReason.RESPONSE_FORMAT,
    CollectionErrorCode.MISSING_PATH: RepairReason.RESPONSE_FORMAT,
    # Targeted repair is the direct remedy for an over-large package reply: the
    # next attempt carries only the outstanding files.
    CollectionErrorCode.AGGREGATE_TOO_LARGE: RepairReason.INVALID_CONTENT,
}

# Rejections that prove the reply crossed the manifest's trust boundary, spoofed
# ownership, or contradicted itself. Targeted regeneration must NOT be attempted:
# these fail safely and keep the deliverable's failure visible.
_UNSAFE_CODES = frozenset({
    CollectionErrorCode.UNKNOWN_ARTIFACT_ID,
    CollectionErrorCode.OUT_OF_SCOPE,
    CollectionErrorCode.WRONG_PACKAGE,
    CollectionErrorCode.WRONG_SUBTASK,
    CollectionErrorCode.EXTERNAL_ARTIFACT,
    CollectionErrorCode.INVALID_PATH,
    CollectionErrorCode.INVALID_OPERATION,
    CollectionErrorCode.INVALID_PROVENANCE,
    CollectionErrorCode.INVALID_METADATA,
    CollectionErrorCode.DIRECTORY_CONTENT,
    CollectionErrorCode.DELETE_CONTENT,
    CollectionErrorCode.DUPLICATE_IN_RESPONSE,
    CollectionErrorCode.CONTENT_TOO_LARGE,
    CollectionErrorCode.TOO_MANY_CANDIDATES,
    CollectionErrorCode.PRESERVED_RESUBMISSION,
})

_REASON_TEXT = {
    RepairReason.MISSING_REQUIRED: "it was not returned at all",
    RepairReason.INTEGRITY_INVALID: "the returned body is structurally incomplete",
    RepairReason.TRUNCATION_SUSPECTED: "the returned body looks truncated",
    RepairReason.INCOMPLETE_DECLARATION: 'it was returned with "complete": false',
    RepairReason.MISSING_CONTENT: "it was returned with no content",
    RepairReason.INVALID_CONTENT: "its content was not usable as a complete file",
    RepairReason.PATH_MISMATCH: "it was returned under the wrong declared path",
    RepairReason.RESPONSE_FORMAT: "its typed artifact entry was malformed",
    RepairReason.NON_REPAIRABLE: "it cannot be repaired by regeneration",
}


# --------------------------------------------------------------------------- #
# Part B — the ONE authoritative repair policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactRepairPolicy:
    """Every bound and switch targeted repair obeys. Nothing is hard-coded elsewhere.

    The number of worker CALLS is deliberately not a field here: it is the
    executor's existing ``1 + config.max_repair_attempts`` budget. A second,
    independent round counter could exceed it, so none exists.
    """

    # Bounded request.
    max_targets_per_attempt: int = 12
    max_preserved_per_package: int = 24
    max_prompt_chars: int = 4000
    max_diagnostics: int = 8
    max_diagnostic_chars: int = 240
    # Bounded preserved-hash disclosure (never file contents).
    max_hash_chars: int = 23
    # How many worker attempts one artifact may consume before it is abandoned.
    # Sized to the default executor budget (1 initial + 2 repairs).
    max_attempts_per_artifact: int = 3
    # Optional-artifact behavior (Scenarios 5 and 6).
    repair_missing_optional: bool = False   # never chase an optional file nobody sent
    repair_rejected_optional: bool = True   # but a BROKEN optional file may be re-asked
    # No-progress termination (Part K).
    stop_on_no_progress: bool = True
    max_no_progress_attempts: int = 1
    stop_on_repeated_content_hash: bool = True
    # Preserved artifacts (Part G / Scenario 3).
    allow_preserved_in_response: bool = False
    # ...but a BYTE-IDENTICAL echo of a preserved artifact changes nothing and is
    # recorded as an unnecessary resubmission, not as a corrupted package. Only a
    # DIFFERING body is an overwrite attempt, and that is a fatal response
    # violation: the preserved candidate always wins and the obligation fails.
    tolerate_identical_preserved_echo: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        positive = (
            "max_targets_per_attempt",
            "max_prompt_chars",
            "max_diagnostic_chars",
            "max_hash_chars",
            "max_attempts_per_artifact",
            "max_no_progress_attempts",
            "schema_version",
        )
        non_negative = ("max_preserved_per_package", "max_diagnostics")
        for name in positive + non_negative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ArtifactRepairError(f"repair policy {name} must be an integer")
            minimum = 1 if name in positive else 0
            if value < minimum:
                raise ArtifactRepairError(
                    f"repair policy {name} must be at least {minimum}"
                )


DEFAULT_REPAIR_POLICY = ArtifactRepairPolicy()


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _clip(value: Any, limit: int) -> str:
    """Collapse whitespace (kills newline injection) and clip one field."""
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _coerce_enum(value: Any, enum_cls: type, fallback: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return fallback
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return fallback


def _pairs(value: Any) -> tuple[tuple[str, int], ...]:
    items: Sequence[Any]
    if value is None:
        items = ()
    elif isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise ArtifactRepairError("repair counters must be a mapping or pairs")
    out: dict[str, int] = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ArtifactRepairError("malformed repair counter entry")
        try:
            out[_clip(item[0], 300)] = max(0, int(item[1]))
        except (TypeError, ValueError):
            raise ArtifactRepairError("repair counter values must be integers")
    return tuple(sorted(out.items()))


def _str_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    items: Sequence[Any]
    if value is None:
        items = ()
    elif isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise ArtifactRepairError("repair hashes must be a mapping or pairs")
    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ArtifactRepairError("malformed repair hash entry")
        out[_clip(item[0], 300)] = _clip(item[1], 100)
    return tuple(sorted(out.items()))


def _ids(value: Any, what: str) -> tuple[str, ...]:
    items = () if value is None else value
    if isinstance(items, str) or not isinstance(items, (list, tuple)):
        raise ArtifactRepairError(f"{what} must be a JSON array")
    return tuple(_clip(item, 300) for item in items if _clip(item, 300))


def short_hash(value: str, policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY) -> str:
    """A bounded, non-reversible identifier for a preserved body. Never content."""
    return _clip(value, policy.max_hash_chars)


# --------------------------------------------------------------------------- #
# Part C — the immutable targeted repair request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairTarget:
    """ONE artifact the next attempt must return, and why."""

    artifact_id: str
    path: str
    reason: RepairReason = RepairReason.MISSING_REQUIRED
    evidence: str = ""
    required: bool = True
    package_id: str = ""
    attempts: int = 0            # worker attempts already spent on this artifact
    last_content_hash: str = ""  # hash of the last REJECTED body ("" if never sent)

    def __post_init__(self) -> None:
        policy = DEFAULT_REPAIR_POLICY
        artifact_id = _clip(self.artifact_id, 300)
        if not artifact_id:
            raise ArtifactRepairError("repair target requires an artifact id")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "path", _clip(self.path, 300))
        object.__setattr__(
            self, "reason",
            _coerce_enum(self.reason, RepairReason, RepairReason.MISSING_REQUIRED),
        )
        object.__setattr__(
            self, "evidence", _clip(self.evidence, policy.max_diagnostic_chars)
        )
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "package_id", _clip(self.package_id, 300))
        try:
            object.__setattr__(self, "attempts", max(0, int(self.attempts)))
        except (TypeError, ValueError):
            raise ArtifactRepairError("repair target attempts must be an integer")
        object.__setattr__(
            self, "last_content_hash", _clip(self.last_content_hash, 100)
        )

    def describe(self) -> str:
        """One deterministic clause naming the defect."""
        return _REASON_TEXT.get(self.reason, _REASON_TEXT[RepairReason.NON_REPAIRABLE])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "reason": self.reason.value,
            "evidence": self.evidence,
            "required": self.required,
            "package_id": self.package_id,
            "attempts": self.attempts,
            "last_content_hash": self.last_content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepairTarget":
        if not isinstance(data, Mapping):
            raise ArtifactRepairError("repair target must be a JSON object")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            path=data.get("path") or "",
            reason=_coerce_enum(
                data.get("reason"), RepairReason, RepairReason.MISSING_REQUIRED
            ),
            evidence=data.get("evidence") or "",
            required=bool(data.get("required", True)),
            package_id=data.get("package_id") or "",
            attempts=data.get("attempts") or 0,
            last_content_hash=data.get("last_content_hash") or "",
        )


@dataclass(frozen=True)
class PreservedArtifact:
    """One already-accepted artifact. Identity and hash ONLY — never content.

    The trusted body lives in the preserved ``ArtifactCandidate``; re-sending it
    to the provider would spend the output budget on work already done, and give
    the model a chance to change a file that is already correct.
    """

    artifact_id: str
    path: str
    content_hash: str
    attempt: int = 0

    def __post_init__(self) -> None:
        artifact_id = _clip(self.artifact_id, 300)
        if not artifact_id:
            raise ArtifactRepairError("preserved artifact requires an artifact id")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "path", _clip(self.path, 300))
        content_hash_value = _clip(self.content_hash, 100)
        if not content_hash_value:
            raise ArtifactRepairError(
                f"preserved artifact {artifact_id!r} requires a content hash"
            )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", content_hash_value) is None:
            raise ArtifactRepairError(
                f"preserved artifact {artifact_id!r} requires a canonical "
                "sha256 content hash"
            )
        object.__setattr__(self, "content_hash", content_hash_value)
        try:
            object.__setattr__(self, "attempt", max(0, int(self.attempt)))
        except (TypeError, ValueError):
            raise ArtifactRepairError("preserved artifact attempt must be an integer")

    @classmethod
    def of(cls, candidate: ArtifactCandidate) -> "PreservedArtifact":
        return cls(
            artifact_id=candidate.artifact_id,
            path=candidate.path,
            content_hash=candidate.content_hash,
            attempt=candidate.attempt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "content_hash": self.content_hash,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreservedArtifact":
        if not isinstance(data, Mapping):
            raise ArtifactRepairError("preserved artifact must be a JSON object")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            path=data.get("path") or "",
            content_hash=str(data.get("content_hash") or ""),
            attempt=data.get("attempt") or 0,
        )


@dataclass(frozen=True)
class ArtifactRepairRequest:
    """The bounded, immutable, serializable request for ONE repair attempt.

    Carries a SAFE SUBSET of the original execution scope (manifest id, subtask
    id, package ids, the scope's full output set) — never the scope object's
    prompt text, and never any preserved file content.
    """

    manifest_id: str
    subtask_id: str
    package_ids: tuple[str, ...] = ()
    attempt: int = 0                       # the attempt this request will be SENT on
    targets: tuple[RepairTarget, ...] = ()
    preserved: tuple[PreservedArtifact, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    origin_output_artifact_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        policy = DEFAULT_REPAIR_POLICY
        manifest_id = _clip(self.manifest_id, 300)
        subtask_id = _clip(self.subtask_id, 300)
        if not manifest_id:
            raise ArtifactRepairError("repair request requires a manifest id")
        if not subtask_id:
            raise ArtifactRepairError("repair request requires a subtask id")
        object.__setattr__(self, "manifest_id", manifest_id)
        object.__setattr__(self, "subtask_id", subtask_id)
        object.__setattr__(
            self, "package_ids", _ids(self.package_ids, "package_ids")
        )
        try:
            object.__setattr__(self, "attempt", max(0, int(self.attempt)))
        except (TypeError, ValueError):
            raise ArtifactRepairError("repair request attempt must be an integer")

        targets = () if self.targets is None else self.targets
        if not isinstance(targets, (list, tuple)):
            raise ArtifactRepairError("repair targets must be a JSON array")
        seen: set[str] = set()
        bounded: list[RepairTarget] = []
        for target in targets:
            if not isinstance(target, RepairTarget):
                raise ArtifactRepairError("repair targets must contain RepairTarget items")
            if target.artifact_id in seen:
                raise ArtifactRepairError(
                    f"repair target {target.artifact_id!r} is listed twice"
                )
            seen.add(target.artifact_id)
            bounded.append(target)
        if not bounded:
            raise ArtifactRepairError("a repair request must target at least one artifact")
        object.__setattr__(
            self, "targets", tuple(bounded[: policy.max_targets_per_attempt])
        )

        preserved = () if self.preserved is None else self.preserved
        if not isinstance(preserved, (list, tuple)):
            raise ArtifactRepairError("preserved artifacts must be a JSON array")
        kept: list[PreservedArtifact] = []
        for item in preserved:
            if not isinstance(item, PreservedArtifact):
                raise ArtifactRepairError(
                    "preserved artifacts must contain PreservedArtifact items"
                )
            if item.artifact_id in seen:
                raise ArtifactRepairError(
                    f"artifact {item.artifact_id!r} cannot be both preserved and a "
                    "repair target"
                )
            kept.append(item)
        object.__setattr__(
            self, "preserved", tuple(kept[: policy.max_preserved_per_package])
        )

        object.__setattr__(
            self, "completion_criteria",
            _ids(self.completion_criteria, "completion_criteria")[: policy.max_diagnostics],
        )
        object.__setattr__(
            self, "validation_ids", _ids(self.validation_ids, "validation_ids")
        )
        object.__setattr__(
            self, "origin_output_artifact_ids",
            _ids(self.origin_output_artifact_ids, "origin_output_artifact_ids"),
        )
        object.__setattr__(
            self, "diagnostics",
            tuple(
                _clip(item, policy.max_diagnostic_chars)
                for item in (self.diagnostics or ())
                if _clip(item, policy.max_diagnostic_chars)
            )[: policy.max_diagnostics],
        )
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

    # ------------------------------- queries --------------------------- #
    @property
    def target_artifact_ids(self) -> tuple[str, ...]:
        return tuple(target.artifact_id for target in self.targets)

    @property
    def target_paths(self) -> tuple[str, ...]:
        return tuple(target.path for target in self.targets)

    @property
    def preserved_artifact_ids(self) -> tuple[str, ...]:
        return tuple(item.artifact_id for item in self.preserved)

    def preserved_hashes(self) -> dict[str, str]:
        return {item.artifact_id: item.content_hash for item in self.preserved}

    def target_for(self, artifact_id: str) -> Optional[RepairTarget]:
        for target in self.targets:
            if target.artifact_id == artifact_id:
                return target
        return None

    # ------------------------------- prompt ---------------------------- #
    def to_worker_block(
        self, policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY
    ) -> str:
        """The bounded TARGETED ARTIFACT REPAIR block (Part F).

        Lists preserved artifacts by path and BOUNDED hash — never their bodies —
        and asks for exactly the repair targets, in full. Every field is
        whitespace-collapsed, so no diagnostic can inject a new prompt line.
        """
        limit = policy.max_prompt_chars
        closing = (
            "Return one typed artifact envelope containing ONLY the requested "
            "repair artifacts, each with its complete raw file contents. Do not "
            "return preserved artifacts, unrelated files, fragments, diffs, "
            "continuations, or Markdown fences around the content."
        )
        preserved_intro = (
            f"{len(self.preserved)} artifact(s) were already accepted and are "
            "preserved unchanged. Do NOT return them or modify them."
        ) if self.preserved else ""

        # Targets and the final envelope rule are obligations, not diagnostics.
        # Reserve room for every one before adding hashes, evidence or criteria;
        # a raw prefix slice could otherwise leave a prompt naming no target.
        def mandatory(label_limit: int, *, reasons: bool = True) -> list[str]:
            out = ["TARGETED ARTIFACT REPAIR"]
            if preserved_intro:
                out.append(preserved_intro)
            out.append("Return ONLY these repair artifacts, each complete:")
            for target in self.targets:
                out.append(f"- {_clip(target.path or target.artifact_id, label_limit)}")
                if reasons:
                    out.append(f"  Reason: {target.describe()}")
            return out

        label_limit = 300
        core = mandatory(label_limit)
        while len("\n".join(core + [closing])) > limit and label_limit > 24:
            label_limit -= 1
            core = mandatory(label_limit)
        if len("\n".join(core + [closing])) > limit:
            core = mandatory(label_limit, reasons=False)
        while len("\n".join(core + [closing])) > limit and label_limit > 8:
            label_limit -= 1
            core = mandatory(label_limit, reasons=False)

        # A deliberately tiny custom prompt budget cannot carry even the target
        # identities. Fail closed with a bounded instruction instead of emitting
        # a misleading fragment of a larger request.
        if len("\n".join(core + [closing])) > limit:
            minimal = (
                "TARGETED ARTIFACT REPAIR\nReturn only the repair targets listed "
                "in the assigned package, in one typed artifact envelope."
            )
            return _clip(minimal, limit)

        before_targets: list[str] = []
        after_targets: list[str] = []
        target_heading_index = 2 if preserved_intro else 1

        def render_with_optional(
            before: Sequence[str], after: Sequence[str]
        ) -> str:
            return "\n".join(
                core[:target_heading_index]
                + list(before)
                + core[target_heading_index:]
                + list(after)
                + [closing]
            )

        def add_if_fits(destination: list[str], *new_lines: str) -> bool:
            before = before_targets if destination is before_targets else before_targets
            after = after_targets if destination is after_targets else after_targets
            candidate_before = before + list(new_lines) if destination is before_targets else before
            candidate_after = after + list(new_lines) if destination is after_targets else after
            if len(render_with_optional(candidate_before, candidate_after)) > limit:
                return False
            destination.extend(new_lines)
            return True

        if self.preserved:
            add_if_fits(before_targets, "Preserved read-only references (hashes only):")
            shown = 0
            for item in self.preserved:
                rendered = (
                    before_targets,
                    f"- {item.path or item.artifact_id} "
                    f"[{short_hash(item.content_hash, policy)}]",
                )
                if not add_if_fits(*rendered):
                    if not add_if_fits(
                        before_targets,
                        f"- {item.artifact_id} "
                        f"[{short_hash(item.content_hash, policy)}]",
                    ):
                        break
                shown += 1
            if shown < len(self.preserved):
                add_if_fits(
                    before_targets,
                    f"- … (+{len(self.preserved) - shown} more preserved)",
                )

        evidence = [
            f"- {target.artifact_id}: {target.evidence}"
            for target in self.targets if target.evidence
        ]
        if evidence and add_if_fits(after_targets, "Bounded rejection diagnostics:"):
            for line in evidence:
                if not add_if_fits(after_targets, f"  Evidence: {line[2:]}"):
                    break
        if self.completion_criteria and add_if_fits(after_targets, "Done when:"):
            for criterion in self.completion_criteria:
                if not add_if_fits(after_targets, f"- {criterion}"):
                    break

        return render_with_optional(before_targets, after_targets)

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "subtask_id": self.subtask_id,
            "package_ids": list(self.package_ids),
            "attempt": self.attempt,
            "targets": [target.to_dict() for target in self.targets],
            "preserved": [item.to_dict() for item in self.preserved],
            "completion_criteria": list(self.completion_criteria),
            "validation_ids": list(self.validation_ids),
            "origin_output_artifact_ids": list(self.origin_output_artifact_ids),
            "diagnostics": list(self.diagnostics),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRepairRequest":
        if not isinstance(data, Mapping):
            raise ArtifactRepairError("repair request must be a JSON object")
        return cls(
            manifest_id=str(data.get("manifest_id") or ""),
            subtask_id=str(data.get("subtask_id") or ""),
            package_ids=tuple(data.get("package_ids") or ()),
            attempt=data.get("attempt") or 0,
            targets=tuple(
                RepairTarget.from_dict(item) for item in (data.get("targets") or ())
            ),
            preserved=tuple(
                PreservedArtifact.from_dict(item)
                for item in (data.get("preserved") or ())
            ),
            completion_criteria=tuple(data.get("completion_criteria") or ()),
            validation_ids=tuple(data.get("validation_ids") or ()),
            origin_output_artifact_ids=tuple(
                data.get("origin_output_artifact_ids") or ()
            ),
            diagnostics=tuple(data.get("diagnostics") or ()),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class RepairDecision:
    """What targeted repair concluded after ONE attempt. Never mutates anything."""

    status: RepairStatus = RepairStatus.SATISFIED
    request: Optional[ArtifactRepairRequest] = None
    reason: str = ""
    unrepairable_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status",
            _coerce_enum(self.status, RepairStatus, RepairStatus.NON_REPAIRABLE),
        )
        if self.request is not None and not isinstance(
            self.request, ArtifactRepairRequest
        ):
            raise ArtifactRepairError(
                "repair decision request must be an ArtifactRepairRequest"
            )
        if self.status is RepairStatus.REPAIRABLE and self.request is None:
            raise ArtifactRepairError("a repairable decision requires a request")
        if self.status is not RepairStatus.REPAIRABLE and self.request is not None:
            raise ArtifactRepairError(
                "only a repairable decision may carry a repair request"
            )
        object.__setattr__(
            self, "reason", _clip(self.reason, DEFAULT_REPAIR_POLICY.max_diagnostic_chars)
        )
        object.__setattr__(
            self, "unrepairable_artifact_ids",
            _ids(self.unrepairable_artifact_ids, "unrepairable_artifact_ids"),
        )

    @property
    def should_retry(self) -> bool:
        return self.status is RepairStatus.REPAIRABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "request": None if self.request is None else self.request.to_dict(),
            "reason": self.reason,
            "unrepairable_artifact_ids": list(self.unrepairable_artifact_ids),
        }


@dataclass(frozen=True)
class ArtifactRepairReport:
    """The compact, serializable repair trace for one subtask (Part N).

    Counts and ids only: no artifact body ever enters a summary.
    """

    subtask_id: str
    attempted: bool = False
    attempts_used: int = 0
    preserved_artifact_ids: tuple[str, ...] = ()
    repaired_artifact_ids: tuple[str, ...] = ()
    remaining_target_ids: tuple[str, ...] = ()
    no_progress_attempts: int = 0
    termination: RepairStatus = RepairStatus.SATISFIED
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        policy = DEFAULT_REPAIR_POLICY
        subtask_id = _clip(self.subtask_id, 300)
        if not subtask_id:
            raise ArtifactRepairError("repair report requires a subtask id")
        object.__setattr__(self, "subtask_id", subtask_id)
        object.__setattr__(self, "attempted", bool(self.attempted))
        try:
            object.__setattr__(self, "attempts_used", max(0, int(self.attempts_used)))
        except (TypeError, ValueError):
            object.__setattr__(self, "attempts_used", 0)
        try:
            object.__setattr__(
                self, "no_progress_attempts", max(0, int(self.no_progress_attempts))
            )
        except (TypeError, ValueError):
            object.__setattr__(self, "no_progress_attempts", 0)
        for name in (
            "preserved_artifact_ids",
            "repaired_artifact_ids",
            "remaining_target_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "termination",
            _coerce_enum(self.termination, RepairStatus, RepairStatus.SATISFIED),
        )
        object.__setattr__(
            self, "diagnostics",
            tuple(
                _clip(item, policy.max_diagnostic_chars)
                for item in (self.diagnostics or ())
                if _clip(item, policy.max_diagnostic_chars)
            )[: policy.max_diagnostics],
        )

    def summary(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "repair_attempted": self.attempted,
            "repair_attempts_used": self.attempts_used,
            "artifacts_preserved": len(self.preserved_artifact_ids),
            "artifacts_repaired": len(self.repaired_artifact_ids),
            "remaining_repair_targets": len(self.remaining_target_ids),
            "no_progress_terminations": self.no_progress_attempts,
            "termination": self.termination.value,
        }

    def to_dict(self) -> dict[str, Any]:
        data = {
            "subtask_id": self.subtask_id,
            "attempted": self.attempted,
            "attempts_used": self.attempts_used,
            "preserved_artifact_ids": list(self.preserved_artifact_ids),
            "repaired_artifact_ids": list(self.repaired_artifact_ids),
            "remaining_target_ids": list(self.remaining_target_ids),
            "no_progress_attempts": self.no_progress_attempts,
            "termination": self.termination.value,
            "diagnostics": list(self.diagnostics),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRepairReport":
        if not isinstance(data, Mapping):
            raise ArtifactRepairError("repair report must be a JSON object")
        return cls(
            subtask_id=str(data.get("subtask_id") or ""),
            attempted=bool(data.get("attempted", False)),
            attempts_used=data.get("attempts_used") or 0,
            preserved_artifact_ids=tuple(data.get("preserved_artifact_ids") or ()),
            repaired_artifact_ids=tuple(data.get("repaired_artifact_ids") or ()),
            remaining_target_ids=tuple(data.get("remaining_target_ids") or ()),
            no_progress_attempts=data.get("no_progress_attempts") or 0,
            termination=_coerce_enum(
                data.get("termination"), RepairStatus, RepairStatus.SATISFIED
            ),
            diagnostics=tuple(data.get("diagnostics") or ()),
        )


# --------------------------------------------------------------------------- #
# Part H — the immutable preserved-candidate state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactRepairState:
    """Preserved candidates plus the per-artifact repair history of ONE subtask.

    ``collection`` is ALWAYS the merged view over the full original scope: the
    accepted candidates it carries are the exact, byte-identical objects the
    collector produced on the attempt that first accepted them, with their
    original hashes, attempt numbers and nested provenance. Nothing in this
    module rewrites one.
    """

    collection: SubtaskCollectionResult
    attempts_used: int = 1
    repair_attempts: int = 0
    attempt_counts: tuple[tuple[str, int], ...] = ()
    invalid_hashes: tuple[tuple[str, str], ...] = ()
    repaired_artifact_ids: tuple[str, ...] = ()
    no_progress_attempts: int = 0
    repeated_hash_artifact_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    termination: RepairStatus = RepairStatus.SATISFIED

    def __post_init__(self) -> None:
        if not isinstance(self.collection, SubtaskCollectionResult):
            raise ArtifactRepairError(
                "repair state requires a SubtaskCollectionResult"
            )
        for name in ("attempts_used", "repair_attempts", "no_progress_attempts"):
            try:
                object.__setattr__(self, name, max(0, int(getattr(self, name))))
            except (TypeError, ValueError):
                object.__setattr__(self, name, 0)
        object.__setattr__(self, "attempt_counts", _pairs(self.attempt_counts))
        object.__setattr__(self, "invalid_hashes", _str_pairs(self.invalid_hashes))
        object.__setattr__(
            self, "repaired_artifact_ids",
            _ids(self.repaired_artifact_ids, "repaired_artifact_ids"),
        )
        object.__setattr__(
            self, "repeated_hash_artifact_ids",
            _ids(self.repeated_hash_artifact_ids, "repeated_hash_artifact_ids"),
        )
        object.__setattr__(
            self, "diagnostics",
            tuple(
                _clip(item, DEFAULT_REPAIR_POLICY.max_diagnostic_chars)
                for item in (self.diagnostics or ())
                if _clip(item, DEFAULT_REPAIR_POLICY.max_diagnostic_chars)
            )[: DEFAULT_REPAIR_POLICY.max_diagnostics],
        )
        object.__setattr__(
            self, "termination",
            _coerce_enum(self.termination, RepairStatus, RepairStatus.SATISFIED),
        )

    @property
    def preserved(self) -> tuple[ArtifactCandidate, ...]:
        return self.collection.accepted

    def attempts_for(self, artifact_id: str) -> int:
        for aid, count in self.attempt_counts:
            if aid == artifact_id:
                return count
        return 0

    def invalid_hash_for(self, artifact_id: str) -> str:
        for aid, digest in self.invalid_hashes:
            if aid == artifact_id:
                return digest
        return ""

    def report(self, decision: RepairDecision) -> ArtifactRepairReport:
        remaining = (
            decision.request.target_artifact_ids
            if decision.request is not None
            else _outstanding_ids(self.collection)
        )
        repaired = set(self.repaired_artifact_ids)
        return ArtifactRepairReport(
            subtask_id=self.collection.subtask_id,
            attempted=self.repair_attempts > 0,
            attempts_used=self.repair_attempts,
            # "preserved" is what was carried over UNCHANGED — the freshly
            # repaired candidates are counted separately.
            preserved_artifact_ids=tuple(
                candidate.artifact_id for candidate in self.preserved
                if candidate.artifact_id not in repaired
            ),
            repaired_artifact_ids=self.repaired_artifact_ids,
            remaining_target_ids=remaining,
            no_progress_attempts=self.no_progress_attempts,
            termination=decision.status,
            diagnostics=self.diagnostics,
        )


def begin_repair_state(
    collection: SubtaskCollectionResult,
    *,
    submitted_hashes: Sequence[tuple[str, str]] = (),
) -> ArtifactRepairState:
    """Seed the preserved state from the FIRST attempt's collection.

    ``submitted_hashes`` (from :func:`submitted_content_hashes`) lets a REJECTED
    first body be compared against the next attempt's body without keeping the
    broken content anywhere.
    """
    collection = _canonicalize_collection_rejections(collection)
    accepted = {candidate.artifact_id for candidate in collection.accepted}
    return ArtifactRepairState(
        collection=collection,
        attempts_used=1,
        repair_attempts=0,
        attempt_counts=tuple(
            (artifact_id, 1) for artifact_id in _touched_ids(collection)
        ),
        invalid_hashes=tuple(sorted(
            (artifact_id, digest)
            for artifact_id, digest in submitted_hashes
            if artifact_id not in accepted
        )),
    )


def _canonicalize_collection_rejections(
    collection: SubtaskCollectionResult,
) -> SubtaskCollectionResult:
    """Remove response-position variance from Phase 4E's retained diagnostics."""
    normalized = tuple(sorted(
        (replace(rejection, index=-1) for rejection in collection.rejected),
        key=lambda rejection: (
            rejection.code.value,
            rejection.artifact_id,
            rejection.path,
            rejection.message,
            rejection.subtask_id,
        ),
    ))
    if normalized == collection.rejected:
        return collection
    return replace(collection, rejected=normalized)


def _touched_ids(collection: SubtaskCollectionResult) -> tuple[str, ...]:
    """Every artifact id this collection owed or heard about, deterministically."""
    ids = {candidate.artifact_id for candidate in collection.accepted}
    ids.update(collection.missing_required_artifact_ids)
    ids.update(collection.missing_optional_artifact_ids)
    ids.update(
        rejection.artifact_id
        for rejection in collection.rejected
        if rejection.artifact_id
    )
    return tuple(sorted(ids))


def _outstanding_ids(collection: SubtaskCollectionResult) -> tuple[str, ...]:
    """Required artifacts this collection still does not hold, in order."""
    accepted = {candidate.artifact_id for candidate in collection.accepted}
    return tuple(
        aid for aid in collection.missing_required_artifact_ids
        if aid not in accepted
    )


# --------------------------------------------------------------------------- #
# Part D — deterministic repair derivation
# --------------------------------------------------------------------------- #
def _rejection_index(
    collection: SubtaskCollectionResult,
) -> dict[str, CandidateRejection]:
    """The FIRST rejection recorded per artifact id (deterministic, message-bearing)."""
    index: dict[str, CandidateRejection] = {}
    for rejection in collection.rejected:
        if rejection.artifact_id and rejection.artifact_id not in index:
            index[rejection.artifact_id] = rejection
    return index


def _unsafe_rejections(
    collection: SubtaskCollectionResult,
) -> tuple[CandidateRejection, ...]:
    return tuple(
        rejection for rejection in collection.rejected
        if rejection.code in _UNSAFE_CODES
    )


def classify_rejection(rejection: CandidateRejection) -> RepairReason:
    """Map ONE typed collection rejection to its repair semantics.

    Typed codes are authoritative: no free-form message parsing happens here.
    """
    if rejection.code in _UNSAFE_CODES:
        return RepairReason.NON_REPAIRABLE
    reason = _REPAIRABLE_CODES.get(rejection.code)
    if reason is None:
        # MISSING_ARTIFACT_ID and any future code: the entry names no artifact we
        # could target. The owed artifact simply stays missing and is picked up
        # as MISSING_REQUIRED below.
        return RepairReason.NON_REPAIRABLE
    return reason


def is_repairable(rejection: CandidateRejection) -> bool:
    return classify_rejection(rejection) is not RepairReason.NON_REPAIRABLE


def plan_repair(
    scope: ArtifactExecutionScope,
    state: ArtifactRepairState,
    *,
    attempt: int,
    max_attempts: int,
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
) -> RepairDecision:
    """Derive the NEXT bounded repair request, or say why there is none.

    The ONE authoritative derivation. Inputs are the original scope, the merged
    collection (preserved accepted candidates + outstanding failures) and the
    repair policy; nothing else. ``attempt`` is the attempt just completed and
    ``max_attempts`` is the executor's EXISTING budget — this function can never
    ask for a call beyond it.
    """
    collection = state.collection
    accepted_ids = {candidate.artifact_id for candidate in collection.accepted}
    rejections = _rejection_index(collection)

    # (10) A reply that crossed the trust boundary is never "repaired".
    unsafe = _unsafe_rejections(collection)
    if unsafe:
        return RepairDecision(
            status=RepairStatus.NON_REPAIRABLE,
            reason=(
                "targeted repair is unsafe after a fatal artifact violation: "
                + unsafe[0].message
            ),
            unrepairable_artifact_ids=tuple(sorted({
                rejection.artifact_id for rejection in unsafe if rejection.artifact_id
            })),
        )

    # (9) Obligations already satisfied: never ask for anything.
    if collection.satisfied:
        return RepairDecision(status=RepairStatus.SATISFIED)

    specs = {spec.id: spec for spec in scope.owned_specs}
    owned_order = [
        aid for aid in scope.owned_artifact_ids if aid in scope.output_artifact_ids
    ]

    required_targets: list[RepairTarget] = []
    optional_targets: list[RepairTarget] = []
    abandoned: list[str] = []
    for artifact_id in owned_order:  # (6)(7) manifest order, package boundaries
        if artifact_id in accepted_ids:
            continue  # (1) a valid accepted candidate is NEVER re-requested
        spec = specs.get(artifact_id)
        if spec is None:
            continue
        rejection = rejections.get(artifact_id)
        if rejection is not None:
            reason = classify_rejection(rejection)
            if reason is RepairReason.NON_REPAIRABLE:
                abandoned.append(artifact_id)
                continue
            evidence = rejection.message
        elif artifact_id in collection.missing_required_artifact_ids:
            reason = RepairReason.MISSING_REQUIRED
            evidence = ""
        elif artifact_id in collection.missing_optional_artifact_ids:
            # (3) a missing OPTIONAL artifact never triggers repair by default.
            if not policy.repair_missing_optional:
                continue
            reason = RepairReason.MISSING_REQUIRED
            evidence = ""
        else:
            continue

        # (4) an optional artifact enters repair only when it was SUBMITTED and
        # rejected, and only when policy permits it.
        if not spec.required:
            submitted_and_rejected = rejection is not None
            if submitted_and_rejected and not policy.repair_rejected_optional:
                continue
            if not submitted_and_rejected and not policy.repair_missing_optional:
                continue

        spent = state.attempts_for(artifact_id)
        if spent >= policy.max_attempts_per_artifact:
            abandoned.append(artifact_id)
            continue

        target = RepairTarget(
            artifact_id=artifact_id,
            path=spec.path,
            reason=reason,
            evidence=_clip(evidence, policy.max_diagnostic_chars),
            required=spec.required,
            package_id=_owner_of(scope, artifact_id),
            attempts=spent,
            last_content_hash=state.invalid_hash_for(artifact_id),
        )
        (required_targets if spec.required else optional_targets).append(target)

    # Required obligations always consume the bounded request first. Manifest
    # order remains stable within required and optional classes.
    targets = required_targets + optional_targets

    if not targets:
        return RepairDecision(
            status=RepairStatus.NON_REPAIRABLE,
            reason=(
                "no artifact in this package can be recovered by targeted "
                "regeneration"
            ),
            unrepairable_artifact_ids=tuple(abandoned),
        )

    # (K) The provider is not improving the artifact: stop rather than spend the
    # remaining attempt on the same broken body.
    if policy.stop_on_no_progress and (
        state.no_progress_attempts >= policy.max_no_progress_attempts
    ):
        return RepairDecision(
            status=RepairStatus.NO_PROGRESS,
            reason=(
                "the repair attempt produced no accepted artifact; further "
                "attempts would repeat it"
            ),
            unrepairable_artifact_ids=tuple(
                target.artifact_id for target in targets
            ),
        )
    if policy.stop_on_repeated_content_hash and state.repeated_hash_artifact_ids:
        return RepairDecision(
            status=RepairStatus.NO_PROGRESS,
            reason=(
                "the provider returned identical invalid content for "
                + ", ".join(state.repeated_hash_artifact_ids[:4])
            ),
            unrepairable_artifact_ids=state.repeated_hash_artifact_ids,
        )

    # The EXISTING attempt budget is the ONLY budget.
    if attempt >= max_attempts:
        return RepairDecision(
            status=RepairStatus.EXHAUSTED,
            reason="the configured worker attempt budget is exhausted",
            unrepairable_artifact_ids=tuple(
                target.artifact_id for target in targets
            ),
        )

    preserved = tuple(
        PreservedArtifact.of(candidate)
        for candidate in collection.accepted
    )
    diagnostics = tuple(
        _clip(
            f"{target.artifact_id}: {target.describe()}",
            policy.max_diagnostic_chars,
        )
        for target in targets[: policy.max_diagnostics]
    )
    request = ArtifactRepairRequest(
        manifest_id=scope.manifest_id,
        subtask_id=scope.subtask_id,
        package_ids=tuple(sorted({
            target.package_id for target in targets if target.package_id
        })) or scope.package_ids,
        attempt=attempt + 1,
        targets=tuple(targets[: policy.max_targets_per_attempt]),
        preserved=preserved[: policy.max_preserved_per_package],
        completion_criteria=scope.completion_criteria[: policy.max_diagnostics],
        validation_ids=scope.validation_ids,
        origin_output_artifact_ids=scope.output_artifact_ids,
        diagnostics=diagnostics,
        schema_version=policy.schema_version,
    )
    return RepairDecision(
        status=RepairStatus.REPAIRABLE,
        request=request,
        reason=f"{len(request.targets)} artifact(s) require targeted repair",
        unrepairable_artifact_ids=tuple(abandoned),
    )


def _owner_of(scope: ArtifactExecutionScope, artifact_id: str) -> str:
    for owned_id, package_id in scope.artifact_owners:
        if owned_id == artifact_id:
            return package_id
    return scope.package_ids[0] if scope.package_ids else ""


# --------------------------------------------------------------------------- #
# Part E — the narrow repair execution scope
# --------------------------------------------------------------------------- #
def derive_repair_scope(
    scope: ArtifactExecutionScope,
    request: ArtifactRepairRequest,
) -> ArtifactExecutionScope:
    """A NEW scope that owns ONLY the repair targets. Never mutates ``scope``.

    It keeps the original manifest and root-subtask identity, the trusted specs
    of the targets, their owning packages, the relevant inputs, criteria and
    validation expectations — and nothing else. The unrelated owned artifacts of
    the package are NOT outputs of this scope, so the Phase 4C/4D collector will
    refuse them exactly as it refuses any other out-of-scope submission, and a
    preserved artifact can never be overwritten through it.
    """
    target_ids = set(request.target_artifact_ids)
    owned_specs = tuple(
        spec for spec in scope.owned_specs if spec.id in target_ids
    )
    owned_ids = tuple(spec.id for spec in owned_specs)
    if not owned_ids:
        raise ArtifactRepairError(
            "a repair scope must own at least one target artifact"
        )
    preserved_ids = tuple(request.preserved_artifact_ids)
    # Preserved artifacts stay visible as INPUTS: the worker may need to keep the
    # repaired file consistent with them, but it owns none of them any more.
    inputs = tuple(dict.fromkeys(
        tuple(scope.input_artifact_ids) + preserved_ids
    ))
    packages = tuple(
        pid for pid in scope.package_ids
        if pid in {
            package_id for artifact_id, package_id in scope.artifact_owners
            if artifact_id in target_ids
        }
    ) or scope.package_ids

    specs_by_id = {spec.id: spec for spec in scope.owned_specs}
    lines: list[str] = ["Repair packages:"]
    for pid in packages:
        lines.append(f"- {pid}")
    lines.append("You must return ONLY these artifacts (complete contents):")
    for spec in owned_specs:
        lines.append(f"- {spec.path}")
    if preserved_ids:
        lines.append(
            "Already accepted and preserved (do not return, do not modify):"
        )
        for artifact_id in preserved_ids:
            spec = specs_by_id.get(artifact_id)
            lines.append(f"- {spec.path if spec else artifact_id}")
    if scope.completion_criteria:
        lines.append("Done when:")
        for criterion in scope.completion_criteria:
            lines.append(f"- {criterion}")

    return ArtifactExecutionScope(
        manifest_id=scope.manifest_id,
        subtask_id=scope.subtask_id,
        package_ids=packages,
        owned_artifact_ids=owned_ids,
        input_artifact_ids=inputs,
        output_artifact_ids=owned_ids,
        validation_ids=scope.validation_ids,
        completion_criteria=scope.completion_criteria,
        package_dependencies=scope.package_dependencies,
        brief="\n".join(lines),
        max_render_chars=scope.max_render_chars,
        owned_specs=owned_specs,
        artifact_owners=tuple(
            (artifact_id, package_id)
            for artifact_id, package_id in scope.artifact_owners
            if artifact_id in target_ids
        ),
        known_artifact_ids=scope.known_artifact_ids,
        external_artifact_ids=scope.external_artifact_ids,
    )


# --------------------------------------------------------------------------- #
# Part G — repair-response parsing
# --------------------------------------------------------------------------- #
def collect_repair_artifacts(
    payload: Any,
    repair_scope: ArtifactExecutionScope,
    request: ArtifactRepairRequest,
    *,
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
    result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    attempt: int = 0,
    producer_subtask_id: str = "",
    recursion_depth: int = 0,
) -> SubtaskCollectionResult:
    """Collect ONE repair reply against the NARROW repair scope.

    Phase 4C/4D validation is reused verbatim — path, ownership, operation,
    bounds, duplicates and structural integrity all run exactly as on a first
    attempt, and worker-supplied attempt/provenance fields remain ignored. The
    ONLY thing added here is the preserved-artifact gate: a preserved id may not
    come back, because that is the one submission the narrow scope alone could
    not name precisely.
    """
    entries, engaged = _typed_entries(payload)
    preserved_hashes = request.preserved_hashes()
    preserved_by_id = {item.artifact_id: item for item in request.preserved}
    preserved_by_hash: dict[str, tuple[str, ...]] = {}
    for item in request.preserved:
        preserved_by_hash[item.content_hash] = tuple(sorted(
            (*preserved_by_hash.get(item.content_hash, ()), item.artifact_id)
        ))

    kept: list[Any] = []
    extra_rejections: list[CandidateRejection] = []
    warnings: list[str] = []
    for index, entry in enumerate(entries or ()):
        artifact_id = ""
        if isinstance(entry, Mapping):
            artifact_id = _clip(entry.get("artifact_id") or entry.get("id"), 300)
        if artifact_id and artifact_id in preserved_hashes:
            submitted = entry.get("content") if isinstance(entry, Mapping) else None
            preserved_item = preserved_by_id[artifact_id]
            identical = (
                isinstance(submitted, str)
                and content_hash(submitted) == preserved_hashes[artifact_id]
                and entry.get("path") == preserved_item.path
                and entry.get("operation") is None
                and entry.get("complete", True) is not False
            )
            if identical and policy.tolerate_identical_preserved_echo:
                # A byte-identical echo changes nothing: the preserved candidate
                # (with its ORIGINAL attempt and provenance) is kept as-is and the
                # echo is dropped. It is wasted output, not a corrupted package.
                warnings.append(
                    f"artifact {artifact_id!r} was resubmitted unchanged although "
                    "it was already accepted and preserved"
                )
                continue
            if policy.allow_preserved_in_response:
                warnings.append(
                    f"preserved artifact {artifact_id!r} was resubmitted and ignored"
                )
                continue
            extra_rejections.append(CandidateRejection(
                code=CollectionErrorCode.PRESERVED_RESUBMISSION,
                message=(
                    f"artifact {artifact_id!r} was already accepted and is "
                    "preserved; a repair reply must return only the requested "
                    "repair artifacts and must never overwrite a preserved file"
                ),
                artifact_id=artifact_id,
                subtask_id=repair_scope.subtask_id,
                index=index,
            ))
            continue
        if artifact_id and isinstance(entry, Mapping):
            submitted = entry.get("content")
            digest = content_hash(submitted) if isinstance(submitted, str) else ""
            echoed_ids = preserved_by_hash.get(digest, ())
            if echoed_ids and artifact_id not in echoed_ids:
                extra_rejections.append(CandidateRejection(
                    code=CollectionErrorCode.PRESERVED_RESUBMISSION,
                    message=(
                        f"artifact {artifact_id!r} resubmitted the preserved body "
                        f"of {echoed_ids[0]!r} under a different artifact identity"
                    ),
                    artifact_id=artifact_id,
                    subtask_id=repair_scope.subtask_id,
                    index=index,
                ))
                continue
        kept.append(entry)

    filtered: dict[str, Any] = {"artifacts": kept}
    if isinstance(payload, Mapping):
        summary = payload.get("summary")
        if summary is not None:
            filtered["summary"] = summary
    if not engaged:
        filtered = payload if isinstance(payload, Mapping) else {}

    collection = collect_subtask_artifacts(
        filtered,
        repair_scope,
        policy=result_policy,
        attempt=attempt,
        producer_subtask_id=producer_subtask_id,
        recursion_depth=recursion_depth,
    )
    if not extra_rejections and not warnings:
        return collection
    return replace(
        collection,
        rejected=tuple(
            (tuple(collection.rejected) + tuple(extra_rejections))
            [: result_policy.max_rejections]
        ),
        warnings=tuple(
            (tuple(collection.warnings) + tuple(warnings))
            [: result_policy.max_warnings]
        ),
        engaged=collection.engaged or bool(extra_rejections),
    )


def _typed_entries(payload: Any) -> tuple[Optional[Sequence[Any]], bool]:
    if not isinstance(payload, Mapping):
        return None, False
    raw = payload.get("artifacts")
    if isinstance(raw, (list, tuple)):
        return list(raw), True
    return None, False


# --------------------------------------------------------------------------- #
# Part H — the deterministic preserved-candidate merge
# --------------------------------------------------------------------------- #
def merge_repair_collection(
    scope: ArtifactExecutionScope,
    state: ArtifactRepairState,
    repair: SubtaskCollectionResult,
    request: ArtifactRepairRequest,
    *,
    attempt: int,
    submitted_hashes: Sequence[tuple[str, str]] = (),
    policy: ArtifactRepairPolicy = DEFAULT_REPAIR_POLICY,
    result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
) -> ArtifactRepairState:
    """Fold ONE repair reply into the preserved state. No latest-wins, ever.

    Rules, in force regardless of response ordering:
      * a preserved valid candidate is carried through byte-identically, with its
        original hash, attempt number and nested provenance;
      * a repair candidate may only FILL a target that is missing or was
        rejected — it can never replace an accepted artifact;
      * a rejected repair never deletes a preserved candidate;
      * a rejection that a later attempt actually fixed is retired from the
        collection (the artifact is no longer missing) but stays in the repair
        diagnostics, so the history is not lost;
      * repeated identical invalid content is recorded as no progress.
    """
    preserved = state.collection
    preserved_ids = {candidate.artifact_id for candidate in preserved.accepted}
    target_ids = set(request.target_artifact_ids)

    accepted_repairs = tuple(
        candidate for candidate in repair.accepted
        # (4) belt and braces: the narrow scope already forbids this.
        if candidate.artifact_id in target_ids
        and candidate.artifact_id not in preserved_ids
    )
    repaired_ids = {candidate.artifact_id for candidate in accepted_repairs}

    # ---- deterministic accepted set, in the scope's manifest order ---- #
    order = {aid: index for index, aid in enumerate(scope.owned_artifact_ids)}
    fallback = len(order)
    merged_accepted = sorted(
        tuple(preserved.accepted) + accepted_repairs,
        key=lambda candidate: (
            order.get(candidate.artifact_id, fallback),
            candidate.artifact_id,
            candidate.provenance,
            candidate.content_hash,
        ),
    )

    # ---- rejections: keep every unresolved one, retire the repaired ---- #
    seen_rejections: set[tuple] = set()
    merged_rejected: list[CandidateRejection] = []
    for rejection in tuple(preserved.rejected) + tuple(repair.rejected):
        if rejection.artifact_id and rejection.artifact_id in repaired_ids:
            continue  # this defect was actually fixed
        key = (
            rejection.code.value, rejection.artifact_id, rejection.path,
            rejection.message,
        )
        if key in seen_rejections:
            continue
        seen_rejections.add(key)
        merged_rejected.append(replace(rejection, index=-1))
    merged_rejected.sort(key=lambda rejection: (
        rejection.code.value,
        rejection.artifact_id,
        rejection.path,
        rejection.message,
        rejection.subtask_id,
    ))

    # ---- no-progress accounting (Part K) ------------------------------ #
    # Only the bodies that were REJECTED are tracked: an accepted repair is
    # progress by definition, and its hash belongs to the candidate, not here.
    invalid_hashes = dict(state.invalid_hashes)
    repeated: list[str] = list(state.repeated_hash_artifact_ids)
    diagnostics: list[str] = list(state.diagnostics)
    for artifact_id, digest in submitted_hashes:
        if artifact_id in repaired_ids or request.target_for(artifact_id) is None:
            continue
        previous = invalid_hashes.get(artifact_id, "")
        if previous and previous == digest and artifact_id not in repeated:
            repeated.append(artifact_id)
            diagnostics.append(
                f"{artifact_id}: identical invalid content returned twice "
                f"({short_hash(digest, policy)})"
            )
        invalid_hashes[artifact_id] = digest

    no_progress = state.no_progress_attempts
    if not accepted_repairs:
        no_progress += 1
        diagnostics.append(
            f"repair attempt {attempt} accepted no artifact for "
            + ", ".join(sorted(target_ids)[:4])
        )
    for warning in repair.warnings:
        diagnostics.append(warning)

    attempt_counts = dict(state.attempt_counts)
    for artifact_id in target_ids:
        attempt_counts[artifact_id] = attempt_counts.get(artifact_id, 0) + 1

    accepted_now = {candidate.artifact_id for candidate in merged_accepted}
    owed_required = tuple(
        spec.id for spec in scope.owned_specs
        if spec.required and spec.id in scope.output_artifact_ids
    )
    owed_optional = tuple(
        spec.id for spec in scope.owned_specs
        if not spec.required and spec.id in scope.output_artifact_ids
    )
    merged = SubtaskCollectionResult(
        subtask_id=scope.subtask_id,
        package_ids=scope.package_ids,
        accepted=tuple(merged_accepted),
        rejected=tuple(merged_rejected[: result_policy.max_rejections]),
        missing_required_artifact_ids=tuple(
            aid for aid in owed_required if aid not in accepted_now
        ),
        missing_optional_artifact_ids=tuple(
            aid for aid in owed_optional if aid not in accepted_now
        ),
        unexpected_artifact_ids=tuple(sorted(set(
            tuple(preserved.unexpected_artifact_ids)
            + tuple(repair.unexpected_artifact_ids)
        ))),
        duplicate_artifact_ids=tuple(sorted(set(
            tuple(preserved.duplicate_artifact_ids)
            + tuple(repair.duplicate_artifact_ids)
        ))),
        warnings=tuple(dict.fromkeys(
            tuple(preserved.warnings) + tuple(repair.warnings)
        ))[: result_policy.max_warnings],
        engaged=preserved.engaged or repair.engaged,
    )

    return ArtifactRepairState(
        collection=merged,
        attempts_used=state.attempts_used + 1,
        repair_attempts=state.repair_attempts + 1,
        attempt_counts=tuple(sorted(attempt_counts.items())),
        invalid_hashes=tuple(sorted(invalid_hashes.items())),
        repaired_artifact_ids=tuple(dict.fromkeys(
            tuple(state.repaired_artifact_ids)
            + tuple(candidate.artifact_id for candidate in accepted_repairs)
        )),
        no_progress_attempts=no_progress,
        repeated_hash_artifact_ids=tuple(repeated),
        diagnostics=tuple(
            _clip(item, policy.max_diagnostic_chars) for item in diagnostics
        )[: policy.max_diagnostics],
        termination=state.termination,
    )


def submitted_content_hashes(payload: Any) -> tuple[tuple[str, str], ...]:
    """``(artifact_id, content_hash)`` for every typed entry in ONE worker reply.

    This is what makes "the provider returned the same broken file again"
    detectable without ever retaining the broken body: only the digest survives.
    Deterministic, bounded, and independent of whether the entry was accepted.
    """
    entries, engaged = _typed_entries(payload)
    if not engaged:
        return ()
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        artifact_id = _clip(entry.get("artifact_id") or entry.get("id"), 300)
        content = entry.get("content")
        if not artifact_id or artifact_id in seen or not isinstance(content, str):
            continue
        seen.add(artifact_id)
        out.append((artifact_id, content_hash(content)))
    return tuple(out)
