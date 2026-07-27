"""Produced-artifact collection and assembly (production-hardening Phase 4C).

WHY THIS EXISTS
---------------
Phases 4A/4B told DOZEN WHICH deliverables a request owes (``ArtifactManifest``
+ ``WorkPackage``) and WHICH of them one subtask owns
(``ArtifactExecutionScope``). Workers now know exactly what to produce — but a
worker reply was still just rendered text. Nothing collected the produced
artifacts, associated them with their declared owner, combined several workers'
results, or noticed that two workers returned different contents for the same
file.

This module is that missing layer, and ONLY that layer:

    ArtifactCandidate         → one artifact a worker claims to have produced
    SubtaskCollectionResult   → what ONE subtask actually delivered vs. owed
    AssembledDeliverable      → the deterministic in-memory combination

Everything here is pure, deterministic, immutable and serializable. Nothing
writes to the filesystem, executes a build, calls a provider, resolves a
conflict with model judgment, merges competing contents, or picks a "latest"
winner. Truncation detection, syntax validation, repair and materialization are
later phases.

The expected-deliverable contracts (Phase 4A) stay immutable and untouched:
this module imports them and never mutates them, so produced-result state never
mixes into the contract vocabulary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from .artifact_integrity import (
    DEFAULT_INTEGRITY_POLICY,
    ArtifactIntegrityResult,
    IntegrityPolicy,
    IntegrityStatus,
    evaluate_artifact_integrity,
    integrity_feedback,
)
from .artifacts import (
    ArtifactContractError,
    ArtifactKind,
    ArtifactOperation,
    ArtifactSpec,
    ArtifactWorkPlan,
    canonical_artifact_path,
)
from .decomposition import ArtifactExecutionScope

SCHEMA_VERSION = 1


class ArtifactResultError(ValueError):
    """A structural violation of the produced-artifact contracts."""


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class CollectionErrorCode(str, Enum):
    """Why one candidate entry was rejected. Deterministic and message-bearing."""

    MALFORMED_ENTRY = "malformed_entry"
    MISSING_ARTIFACT_ID = "missing_artifact_id"
    UNKNOWN_ARTIFACT_ID = "unknown_artifact_id"
    MISSING_PATH = "missing_path"
    PATH_MISMATCH = "path_mismatch"
    INVALID_PATH = "invalid_path"
    OUT_OF_SCOPE = "out_of_scope"
    WRONG_PACKAGE = "wrong_package"
    WRONG_SUBTASK = "wrong_subtask"
    EXTERNAL_ARTIFACT = "external_artifact"
    DUPLICATE_IN_RESPONSE = "duplicate_in_response"
    MISSING_CONTENT = "missing_content"
    INVALID_CONTENT_TYPE = "invalid_content_type"
    DIRECTORY_CONTENT = "directory_content"
    DELETE_CONTENT = "delete_content"
    CONTENT_TOO_LARGE = "content_too_large"
    AGGREGATE_TOO_LARGE = "aggregate_too_large"
    TOO_MANY_CANDIDATES = "too_many_candidates"
    INVALID_OPERATION = "invalid_operation"
    INVALID_PROVENANCE = "invalid_provenance"
    INVALID_METADATA = "invalid_metadata"
    # Phase 4D: the submitted body itself does not hold together.
    STRUCTURALLY_INVALID = "structurally_invalid"
    TRUNCATION_SUSPECTED = "truncation_suspected"
    INCOMPLETE_DECLARATION = "incomplete_declaration"
    # Phase 4E: a targeted repair reply returned an artifact that was ALREADY
    # accepted and preserved. Recorded by the repair layer, never by the
    # first-attempt collector.
    PRESERVED_RESUBMISSION = "preserved_resubmission"


class ConflictKind(str, Enum):
    CONTENT = "content"              # same artifact, different contents
    OPERATION = "operation"          # same artifact, incompatible operations
    PATH_COLLISION = "path_collision"  # different artifacts, same canonical path
    OWNERSHIP = "ownership"          # a candidate credited to a non-owning package


class AssemblyStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    CONFLICTED = "conflicted"
    FAILED = "failed"


# Conflict kinds that make the produced structure fundamentally uninterpretable
# rather than merely contested. These fail the assembly outright.
_FATAL_CONFLICTS = frozenset({ConflictKind.PATH_COLLISION, ConflictKind.OWNERSHIP})

# Rejections that prove the submitted identity/path/provenance crossed the
# manifest trust boundary. A valid candidate elsewhere must not turn one of
# these into a successful root deliverable.
_FATAL_REJECTIONS = frozenset({
    CollectionErrorCode.UNKNOWN_ARTIFACT_ID,
    CollectionErrorCode.PATH_MISMATCH,
    CollectionErrorCode.INVALID_PATH,
    CollectionErrorCode.OUT_OF_SCOPE,
    CollectionErrorCode.WRONG_PACKAGE,
    CollectionErrorCode.WRONG_SUBTASK,
    CollectionErrorCode.EXTERNAL_ARTIFACT,
    CollectionErrorCode.INVALID_OPERATION,
    CollectionErrorCode.INVALID_PROVENANCE,
    CollectionErrorCode.PRESERVED_RESUBMISSION,
})

# Phase 4D rejections: the body was owned, well-addressed and in-bounds, but it
# is not structurally whole. These are REPAIRABLE (the existing worker repair
# loop re-asks for the complete artifact), so they must NOT be fatal — a
# truncated file leaves the deliverable PARTIAL, not FAILED.
_INTEGRITY_REJECTIONS = frozenset({
    CollectionErrorCode.STRUCTURALLY_INVALID,
    CollectionErrorCode.TRUNCATION_SUSPECTED,
    CollectionErrorCode.INCOMPLETE_DECLARATION,
})

# Artifacts whose "content" is evidence text (a build/test log), not source.
_RESULT_KINDS = frozenset({
    ArtifactKind.COMMAND_RESULT,
    ArtifactKind.BUILD_RESULT,
    ArtifactKind.TEST_RESULT,
})


# --------------------------------------------------------------------------- #
# Part L — the ONE authoritative home for result bounds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResultPolicy:
    """Bounds for produced-artifact collection and assembly.

    Conservative but usable: a per-file cap that comfortably holds an ordinary
    source file, and aggregate caps that stop a pathological (or adversarial)
    model response from consuming unbounded memory. Every limit lives here —
    parser, collector, assembler and tests all read these, never their own.
    """

    # Per worker response.
    max_candidates_per_response: int = 24
    # Per artifact content (characters). ~256 KiB: far above a normal file.
    max_content_chars: int = 262_144
    # Aggregate content one subtask may submit.
    max_subtask_content_chars: int = 1_048_576
    # Aggregate content one assembled deliverable may hold.
    max_total_content_chars: int = 4_194_304
    max_total_candidates: int = 200
    # Bounded provenance / metadata fields.
    max_summary_chars: int = 400
    max_metadata_items: int = 8
    max_metadata_key_chars: int = 60
    max_metadata_value_chars: int = 200
    max_language_chars: int = 40
    max_media_type_chars: int = 80
    # Bounded diagnostics (the model always retains counts; only lists clip).
    max_warnings: int = 16
    max_rejections: int = 40
    max_duplicates: int = 40
    max_conflicts: int = 40
    # Bounded repair feedback handed back to the existing worker repair loop.
    max_feedback_chars: int = 1200
    max_feedback_items: int = 8
    # Phase 4D: structural-integrity bounds live in their OWN authoritative
    # policy; this is only the seam that carries it through collection.
    integrity: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY


DEFAULT_RESULT_POLICY = ResultPolicy()


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _clip_line(value: Any, limit: int) -> str:
    """Collapse whitespace (kills newline injection) and clip one field."""
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _str_pairs(value: Any, policy: ResultPolicy) -> tuple[tuple[str, str], ...]:
    """Bounded, sorted, immutable metadata pairs."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        try:
            items = [(k, v) for k, v in value]
        except (TypeError, ValueError):
            raise ArtifactResultError(
                "candidate metadata must be a mapping or key/value pairs"
            )
    pairs: dict[str, str] = {}
    for key, val in items:
        k = _clip_line(key, policy.max_metadata_key_chars)
        if not k:
            continue
        v = _clip_line(val, policy.max_metadata_value_chars)
        if k in pairs and pairs[k] != v:
            raise ArtifactResultError(
                f"candidate metadata keys collide after canonicalization: {k!r}"
            )
        pairs[k] = v
    return tuple(sorted(pairs.items()))[: policy.max_metadata_items]


def content_hash(content: str) -> str:
    """Deterministic content hash. Exact bytes in, stable digest out."""
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _tuple_of(value: Any, kind: type, what: str) -> tuple:
    items = () if value is None else value
    if not isinstance(items, (list, tuple)):
        raise ArtifactResultError(f"{what} must be a JSON array")
    frozen = tuple(items)
    for item in frozen:
        if not isinstance(item, kind):
            raise ArtifactResultError(f"{what} must contain {kind.__name__} items")
    return frozen


def _coerce_enum(value: Any, enum_cls: type, fallback: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return fallback
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return fallback


# --------------------------------------------------------------------------- #
# Part A — produced artifact candidate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactCandidate:
    """ONE artifact a worker claims to have produced.

    Content is stored EXACTLY as submitted — never re-indented, re-wrapped or
    stripped — so the hash identifies what the model actually returned. Whether
    that content is syntactically valid, or truncated, is deliberately NOT this
    phase's question.

    Provenance is carried in full: ``subtask_id`` is the plan-mapped subtask
    that owns the package (the identity the root plan can check), while
    ``producer_subtask_id``/``recursion_depth`` record which nested delegation
    actually emitted it.
    """

    artifact_id: str
    path: str
    subtask_id: str
    package_id: str
    content: str = ""
    kind: ArtifactKind = ArtifactKind.OTHER
    operation: ArtifactOperation = ArtifactOperation.CREATE
    language: str = ""
    media_type: str = ""
    complete: bool = True
    attempt: int = 0
    summary: str = ""
    producer_subtask_id: str = ""
    recursion_depth: int = 0
    content_hash: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    # Phase 4D: the measured structural verdict on this exact body. Attached to
    # accepted candidates as evidence; the worker's ``complete`` flag above is
    # provider-controlled data and never overrides it.
    integrity: Optional[ArtifactIntegrityResult] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        policy = DEFAULT_RESULT_POLICY
        artifact_id = _clip_line(self.artifact_id, 300)
        if not artifact_id:
            raise ArtifactResultError("candidate artifact id must not be empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        try:
            object.__setattr__(self, "path", canonical_artifact_path(self.path))
        except ArtifactContractError as exc:
            raise ArtifactResultError(str(exc)) from exc
        subtask_id = _clip_line(self.subtask_id, 300)
        if not subtask_id:
            raise ArtifactResultError("candidate requires a producing subtask id")
        object.__setattr__(self, "subtask_id", subtask_id)
        package_id = _clip_line(self.package_id, 300)
        if not package_id:
            raise ArtifactResultError("candidate requires a producing package id")
        object.__setattr__(self, "package_id", package_id)

        if not isinstance(self.content, str):
            raise ArtifactResultError("candidate content must be a string")
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, ArtifactKind, ArtifactKind.OTHER)
        )
        object.__setattr__(
            self, "operation",
            _coerce_enum(self.operation, ArtifactOperation, ArtifactOperation.CREATE),
        )
        object.__setattr__(
            self, "language", _clip_line(self.language, policy.max_language_chars)
        )
        object.__setattr__(
            self, "media_type", _clip_line(self.media_type, policy.max_media_type_chars)
        )
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(
            self, "summary", _clip_line(self.summary, policy.max_summary_chars)
        )
        object.__setattr__(
            self, "producer_subtask_id",
            _clip_line(self.producer_subtask_id, 300) or subtask_id,
        )
        try:
            object.__setattr__(self, "attempt", max(0, int(self.attempt)))
        except (TypeError, ValueError):
            raise ArtifactResultError("candidate attempt must be an integer")
        try:
            object.__setattr__(self, "recursion_depth", max(0, int(self.recursion_depth)))
        except (TypeError, ValueError):
            raise ArtifactResultError("candidate recursion depth must be an integer")
        object.__setattr__(self, "metadata", _str_pairs(self.metadata, policy))
        if self.integrity is not None and not isinstance(
            self.integrity, ArtifactIntegrityResult
        ):
            raise ArtifactResultError(
                "candidate integrity must be an ArtifactIntegrityResult"
            )
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

        supplied_hash = _clip_line(self.content_hash, 100)
        computed = content_hash(self.content)
        if supplied_hash and supplied_hash != computed:
            raise ArtifactResultError(
                f"candidate {self.artifact_id!r} carries a content hash that does "
                "not match its content"
            )
        object.__setattr__(self, "content_hash", computed)

    # ------------------------------- queries --------------------------- #
    @property
    def provenance(self) -> tuple[str, str, str, int, int]:
        """Deterministic provenance key: package, subtask, producer, attempt, depth."""
        return (
            self.package_id,
            self.subtask_id,
            self.producer_subtask_id,
            self.attempt,
            self.recursion_depth,
        )

    def describe_provenance(self) -> str:
        text = f"package {self.package_id} / subtask {self.subtask_id}"
        if self.producer_subtask_id != self.subtask_id:
            text += f" (produced by {self.producer_subtask_id})"
        if self.attempt:
            text += f", attempt {self.attempt}"
        if self.recursion_depth:
            text += f", depth {self.recursion_depth}"
        return text

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "subtask_id": self.subtask_id,
            "package_id": self.package_id,
            "content": self.content,
            "kind": self.kind.value,
            "operation": self.operation.value,
            "language": self.language,
            "media_type": self.media_type,
            "complete": self.complete,
            "attempt": self.attempt,
            "summary": self.summary,
            "producer_subtask_id": self.producer_subtask_id,
            "recursion_depth": self.recursion_depth,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
            "integrity": None if self.integrity is None else self.integrity.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactCandidate":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("candidate must be a JSON object")
        content = data.get("content", "")
        integrity = data.get("integrity")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            path=data.get("path"),
            subtask_id=str(data.get("subtask_id") or ""),
            package_id=str(data.get("package_id") or ""),
            content="" if content is None else content,
            kind=_coerce_enum(data.get("kind"), ArtifactKind, ArtifactKind.OTHER),
            operation=_coerce_enum(
                data.get("operation"), ArtifactOperation, ArtifactOperation.CREATE
            ),
            language=data.get("language") or "",
            media_type=data.get("media_type") or "",
            complete=bool(data.get("complete", True)),
            attempt=data.get("attempt") or 0,
            summary=data.get("summary") or "",
            producer_subtask_id=data.get("producer_subtask_id") or "",
            recursion_depth=data.get("recursion_depth") or 0,
            content_hash=data.get("content_hash") or "",
            metadata=data.get("metadata"),
            integrity=(
                ArtifactIntegrityResult.from_dict(integrity)
                if isinstance(integrity, Mapping) else None
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CandidateRejection:
    """Why one submitted entry was not accepted. Never silently dropped."""

    code: CollectionErrorCode
    message: str
    artifact_id: str = ""
    path: str = ""
    subtask_id: str = ""
    index: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code",
            _coerce_enum(self.code, CollectionErrorCode,
                         CollectionErrorCode.MALFORMED_ENTRY),
        )
        object.__setattr__(self, "message", _clip_line(self.message, 300))
        object.__setattr__(self, "artifact_id", _clip_line(self.artifact_id, 300))
        object.__setattr__(self, "path", _clip_line(self.path, 300))
        object.__setattr__(self, "subtask_id", _clip_line(self.subtask_id, 300))
        try:
            object.__setattr__(self, "index", int(self.index))
        except (TypeError, ValueError):
            object.__setattr__(self, "index", -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "subtask_id": self.subtask_id,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateRejection":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("rejection must be a JSON object")
        return cls(
            code=_coerce_enum(
                data.get("code"), CollectionErrorCode,
                CollectionErrorCode.MALFORMED_ENTRY,
            ),
            message=data.get("message") or "",
            artifact_id=data.get("artifact_id") or "",
            path=data.get("path") or "",
            subtask_id=data.get("subtask_id") or "",
            index=data.get("index", -1),
        )


# --------------------------------------------------------------------------- #
# Part D — one subtask's collection result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubtaskCollectionResult:
    """What ONE subtask actually delivered, against what its scope owed.

    ``satisfied`` answers only the ARTIFACT question. It is deliberately
    independent of ``SubTaskResult.status``: a worker can return a genuine,
    verifier-approved answer (semantic success) while failing to deliver an
    owned file (artifact failure), and the two must stay distinguishable.

    ``engaged`` records whether the worker used the typed produced-artifact
    envelope at all. A legacy (pre-4C) envelope leaves it false, which keeps
    unscoped and legacy behavior byte-identical.
    """

    subtask_id: str
    package_ids: tuple[str, ...] = ()
    accepted: tuple[ArtifactCandidate, ...] = ()
    rejected: tuple[CandidateRejection, ...] = ()
    missing_required_artifact_ids: tuple[str, ...] = ()
    missing_optional_artifact_ids: tuple[str, ...] = ()
    unexpected_artifact_ids: tuple[str, ...] = ()
    duplicate_artifact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    engaged: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        subtask_id = _clip_line(self.subtask_id, 300)
        if not subtask_id:
            raise ArtifactResultError("collection result requires a subtask id")
        object.__setattr__(self, "subtask_id", subtask_id)
        object.__setattr__(
            self, "accepted",
            _tuple_of(self.accepted, ArtifactCandidate, "accepted candidates"),
        )
        object.__setattr__(
            self, "rejected",
            _tuple_of(self.rejected, CandidateRejection, "rejected candidates"),
        )
        for name in (
            "package_ids",
            "missing_required_artifact_ids",
            "missing_optional_artifact_ids",
            "unexpected_artifact_ids",
            "duplicate_artifact_ids",
            "warnings",
        ):
            value = getattr(self, name)
            items = () if value is None else value
            if isinstance(items, str) or not isinstance(items, (list, tuple)):
                raise ArtifactResultError(f"{name} must be a JSON array")
            object.__setattr__(self, name, tuple(str(item) for item in items))
        object.__setattr__(self, "engaged", bool(self.engaged))
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

    # ------------------------------- queries --------------------------- #
    @property
    def satisfied(self) -> bool:
        """Did this subtask deliver every artifact its scope obliged it to?"""
        typed_contract_present = (
            self.engaged or not self.missing_optional_artifact_ids
        )
        return (
            typed_contract_present
            and not self.missing_required_artifact_ids
            and not self.rejected
        )

    @property
    def integrity_rejected_artifact_ids(self) -> tuple[str, ...]:
        """Artifacts this subtask DID return but that were not structurally whole.

        Distinct from ``missing_required_artifact_ids``: something came back, it
        just could not be trusted as a complete file (Phase 4D).
        """
        return tuple(sorted({
            rejection.artifact_id
            for rejection in self.rejected
            if rejection.artifact_id and rejection.code in _INTEGRITY_REJECTIONS
        }))

    def accepted_by_artifact(self) -> dict[str, tuple[ArtifactCandidate, ...]]:
        grouped: dict[str, list[ArtifactCandidate]] = {}
        for candidate in self.accepted:
            grouped.setdefault(candidate.artifact_id, []).append(candidate)
        return {key: tuple(value) for key, value in grouped.items()}

    def problem_summary(self) -> str:
        """One deterministic clause naming why the obligations were not met."""
        if self.satisfied:
            return ""
        parts: list[str] = []
        if self.missing_required_artifact_ids:
            parts.append(
                "missing " + ", ".join(self.missing_required_artifact_ids[:4])
            )
        broken = self.integrity_rejected_artifact_ids
        if broken:
            parts.append("structurally incomplete " + ", ".join(broken[:4]))
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected submission(s)")
        return "; ".join(parts) or "artifact obligations unmet"

    def feedback(self, policy: ResultPolicy = DEFAULT_RESULT_POLICY) -> str:
        """Bounded corrective feedback for the EXISTING worker repair loop.

        Empty when the subtask's artifact obligations were met, so a satisfied
        worker is never retried for artifact reasons.
        """
        if self.satisfied:
            return ""
        items: list[str] = []
        # An artifact rejected for integrity is ALSO missing, but the rejection
        # message says precisely what was wrong with it — naming it twice only
        # spends the worker's bounded context.
        broken = set(self.integrity_rejected_artifact_ids)
        absent = [
            aid for aid in self.missing_required_artifact_ids if aid not in broken
        ]
        if absent:
            missing = ", ".join(absent[:6])
            items.append(
                "you did not return these owned artifacts, which your assigned "
                f"package must produce in full: {missing}"
            )
        for rejection in self.rejected[: policy.max_feedback_items]:
            items.append(rejection.message)
        if not items:
            items.append("your produced artifacts did not satisfy the assigned scope")
        text = (
            "Your reply did not satisfy the assigned artifact package: "
            + "; ".join(items[: policy.max_feedback_items])
            + ". Return the strict JSON envelope with an \"artifacts\" array "
            "containing one entry per owned artifact (artifact_id, path, "
            "complete content)."
        )
        if len(text) > policy.max_feedback_chars:
            text = text[: policy.max_feedback_chars - 1].rstrip() + "…"
        return text

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "package_ids": list(self.package_ids),
            "accepted": [c.to_dict() for c in self.accepted],
            "rejected": [r.to_dict() for r in self.rejected],
            "missing_required_artifact_ids": list(self.missing_required_artifact_ids),
            "missing_optional_artifact_ids": list(self.missing_optional_artifact_ids),
            "unexpected_artifact_ids": list(self.unexpected_artifact_ids),
            "duplicate_artifact_ids": list(self.duplicate_artifact_ids),
            "integrity_rejected_artifact_ids": list(
                self.integrity_rejected_artifact_ids
            ),
            "warnings": list(self.warnings),
            "engaged": self.engaged,
            "satisfied": self.satisfied,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubtaskCollectionResult":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("collection result must be a JSON object")
        return cls(
            subtask_id=str(data.get("subtask_id") or ""),
            package_ids=tuple(data.get("package_ids") or ()),
            accepted=tuple(
                ArtifactCandidate.from_dict(item)
                for item in (data.get("accepted") or ())
            ),
            rejected=tuple(
                CandidateRejection.from_dict(item)
                for item in (data.get("rejected") or ())
            ),
            missing_required_artifact_ids=tuple(
                data.get("missing_required_artifact_ids") or ()
            ),
            missing_optional_artifact_ids=tuple(
                data.get("missing_optional_artifact_ids") or ()
            ),
            unexpected_artifact_ids=tuple(data.get("unexpected_artifact_ids") or ()),
            duplicate_artifact_ids=tuple(data.get("duplicate_artifact_ids") or ()),
            warnings=tuple(data.get("warnings") or ()),
            engaged=bool(data.get("engaged", False)),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class RecursiveArtifactOutput:
    """A recursive child's answer PLUS the typed candidates it produced.

    The executor's ``recurse_fn`` may return this instead of plain text. A
    legacy callback returning a string keeps working unchanged.
    """

    text: str
    collection: Optional[SubtaskCollectionResult] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ArtifactResultError("recursive output text must be a string")
        if self.collection is not None and not isinstance(
            self.collection, SubtaskCollectionResult
        ):
            raise ArtifactResultError(
                "recursive output collection must be a SubtaskCollectionResult"
            )


# --------------------------------------------------------------------------- #
# Parts F/G — duplicate, conflict and assembly diagnostics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DuplicateDiagnostic:
    """Two or more candidates for one artifact with IDENTICAL content.

    Not silently discarded: ownership should be unique, so even a byte-identical
    resubmission is evidence that something produced work it did not own.
    """

    artifact_id: str
    path: str
    content_hash: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _clip_line(self.artifact_id, 300))
        object.__setattr__(self, "path", _clip_line(self.path, 300))
        object.__setattr__(self, "content_hash", _clip_line(self.content_hash, 100))
        object.__setattr__(
            self, "provenance",
            tuple(_clip_line(item, 300) for item in (self.provenance or ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "content_hash": self.content_hash,
            "provenance": list(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DuplicateDiagnostic":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("duplicate diagnostic must be a JSON object")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            path=str(data.get("path") or ""),
            content_hash=str(data.get("content_hash") or ""),
            provenance=tuple(data.get("provenance") or ()),
        )


@dataclass(frozen=True)
class ConflictEntry:
    """One side of a conflict. Complete provenance; content is NOT discarded."""

    artifact_id: str
    path: str
    content_hash: str
    package_id: str
    subtask_id: str
    producer_subtask_id: str = ""
    attempt: int = 0
    recursion_depth: int = 0
    operation: ArtifactOperation = ArtifactOperation.CREATE

    def __post_init__(self) -> None:
        for name in ("artifact_id", "path", "package_id", "subtask_id",
                     "producer_subtask_id"):
            object.__setattr__(self, name, _clip_line(getattr(self, name), 300))
        object.__setattr__(self, "content_hash", _clip_line(self.content_hash, 100))
        object.__setattr__(
            self, "operation",
            _coerce_enum(self.operation, ArtifactOperation, ArtifactOperation.CREATE),
        )
        for name in ("attempt", "recursion_depth"):
            try:
                object.__setattr__(self, name, max(0, int(getattr(self, name))))
            except (TypeError, ValueError):
                object.__setattr__(self, name, 0)

    @classmethod
    def of(cls, candidate: ArtifactCandidate) -> "ConflictEntry":
        return cls(
            artifact_id=candidate.artifact_id,
            path=candidate.path,
            content_hash=candidate.content_hash,
            package_id=candidate.package_id,
            subtask_id=candidate.subtask_id,
            producer_subtask_id=candidate.producer_subtask_id,
            attempt=candidate.attempt,
            recursion_depth=candidate.recursion_depth,
            operation=candidate.operation,
        )

    def sort_key(self) -> tuple:
        return (
            self.artifact_id, self.package_id, self.subtask_id,
            self.producer_subtask_id, self.attempt, self.recursion_depth,
            self.content_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "content_hash": self.content_hash,
            "package_id": self.package_id,
            "subtask_id": self.subtask_id,
            "producer_subtask_id": self.producer_subtask_id,
            "attempt": self.attempt,
            "recursion_depth": self.recursion_depth,
            "operation": self.operation.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConflictEntry":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("conflict entry must be a JSON object")
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            path=str(data.get("path") or ""),
            content_hash=str(data.get("content_hash") or ""),
            package_id=str(data.get("package_id") or ""),
            subtask_id=str(data.get("subtask_id") or ""),
            producer_subtask_id=str(data.get("producer_subtask_id") or ""),
            attempt=data.get("attempt") or 0,
            recursion_depth=data.get("recursion_depth") or 0,
            operation=_coerce_enum(
                data.get("operation"), ArtifactOperation, ArtifactOperation.CREATE
            ),
        )


@dataclass(frozen=True)
class ConflictDiagnostic:
    """Competing candidates for one artifact (or one path). NEVER auto-resolved.

    No merge, no latest-wins, no model judgment: the conflict is recorded with
    every hash and every provenance, and the artifact is excluded from the
    successfully assembled set.
    """

    kind: ConflictKind
    artifact_id: str
    path: str
    entries: tuple[ConflictEntry, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, ConflictKind, ConflictKind.CONTENT)
        )
        object.__setattr__(self, "artifact_id", _clip_line(self.artifact_id, 300))
        object.__setattr__(self, "path", _clip_line(self.path, 300))
        object.__setattr__(
            self, "entries", _tuple_of(self.entries, ConflictEntry, "conflict entries")
        )
        object.__setattr__(self, "message", _clip_line(self.message, 300))

    @property
    def fatal(self) -> bool:
        return self.kind in _FATAL_CONFLICTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "entries": [entry.to_dict() for entry in self.entries],
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConflictDiagnostic":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("conflict diagnostic must be a JSON object")
        return cls(
            kind=_coerce_enum(data.get("kind"), ConflictKind, ConflictKind.CONTENT),
            artifact_id=str(data.get("artifact_id") or ""),
            path=str(data.get("path") or ""),
            entries=tuple(
                ConflictEntry.from_dict(item) for item in (data.get("entries") or ())
            ),
            message=data.get("message") or "",
        )


@dataclass(frozen=True)
class AssembledDeliverable:
    """The deterministic in-memory combination of every collected artifact.

    In-memory ONLY: this phase never writes a file, never zips anything and
    never touches a workspace.
    """

    manifest_id: str
    status: AssemblyStatus = AssemblyStatus.FAILED
    artifacts: tuple[ArtifactCandidate, ...] = ()
    expected_required_artifact_ids: tuple[str, ...] = ()
    expected_optional_artifact_ids: tuple[str, ...] = ()
    missing_required_artifact_ids: tuple[str, ...] = ()
    missing_optional_artifact_ids: tuple[str, ...] = ()
    duplicates: tuple[DuplicateDiagnostic, ...] = ()
    conflicts: tuple[ConflictDiagnostic, ...] = ()
    rejected: tuple[CandidateRejection, ...] = ()
    package_status: tuple[tuple[str, str], ...] = ()
    validation_result_artifact_ids: tuple[str, ...] = ()
    # Phase 4D: artifacts that WERE returned but were not structurally whole.
    # A subset of the missing ids, kept separate so "never produced" and
    # "produced but truncated" never collapse into one indistinguishable state.
    integrity_rejected_artifact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        manifest_id = _clip_line(self.manifest_id, 300)
        if not manifest_id:
            raise ArtifactResultError("assembled deliverable requires a manifest id")
        object.__setattr__(self, "manifest_id", manifest_id)
        object.__setattr__(
            self, "status",
            _coerce_enum(self.status, AssemblyStatus, AssemblyStatus.FAILED),
        )
        object.__setattr__(
            self, "artifacts",
            _tuple_of(self.artifacts, ArtifactCandidate, "assembled artifacts"),
        )
        object.__setattr__(
            self, "duplicates",
            _tuple_of(self.duplicates, DuplicateDiagnostic, "duplicates"),
        )
        object.__setattr__(
            self, "conflicts",
            _tuple_of(self.conflicts, ConflictDiagnostic, "conflicts"),
        )
        object.__setattr__(
            self, "rejected",
            _tuple_of(self.rejected, CandidateRejection, "rejected candidates"),
        )
        for name in (
            "expected_required_artifact_ids",
            "expected_optional_artifact_ids",
            "missing_required_artifact_ids",
            "missing_optional_artifact_ids",
            "validation_result_artifact_ids",
            "integrity_rejected_artifact_ids",
            "warnings",
        ):
            value = getattr(self, name)
            items = () if value is None else value
            if isinstance(items, str) or not isinstance(items, (list, tuple)):
                raise ArtifactResultError(f"{name} must be a JSON array")
            object.__setattr__(self, name, tuple(str(item) for item in items))
        pairs = () if self.package_status is None else self.package_status
        if not isinstance(pairs, (list, tuple)):
            raise ArtifactResultError("package_status must be a JSON array of pairs")
        normalized: list[tuple[str, str]] = []
        for item in pairs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ArtifactResultError("malformed package_status entry")
            normalized.append((str(item[0]), str(item[1])))
        object.__setattr__(self, "package_status", tuple(normalized))
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

    # ------------------------------- queries --------------------------- #
    @property
    def complete(self) -> bool:
        return self.status is AssemblyStatus.COMPLETE

    def by_artifact(self) -> dict[str, ArtifactCandidate]:
        return {candidate.artifact_id: candidate for candidate in self.artifacts}

    def provenance_index(self) -> tuple[tuple[str, str], ...]:
        """Deterministic artifact_id -> provenance description index."""
        return tuple(
            (candidate.artifact_id, candidate.describe_provenance())
            for candidate in self.artifacts
        )

    def summary(self) -> dict[str, Any]:
        """Compact, serializable assembly trace for logs and the final result."""
        return {
            "manifest_id": self.manifest_id,
            "status": self.status.value,
            "expected_required": len(self.expected_required_artifact_ids),
            "expected_optional": len(self.expected_optional_artifact_ids),
            "assembled": len(self.artifacts),
            "missing_required": len(self.missing_required_artifact_ids),
            "missing_optional": len(self.missing_optional_artifact_ids),
            "conflicts": len(self.conflicts),
            "duplicates": len(self.duplicates),
            "rejected": len(self.rejected),
            "validation_results": len(self.validation_result_artifact_ids),
            "integrity_rejected": len(self.integrity_rejected_artifact_ids),
        }

    def problem_summary(self) -> str:
        """One deterministic sentence naming why the assembly is not COMPLETE."""
        if self.status is AssemblyStatus.COMPLETE:
            return ""
        parts: list[str] = []
        broken = set(self.integrity_rejected_artifact_ids)
        absent = [
            aid for aid in self.missing_required_artifact_ids if aid not in broken
        ]
        if absent:
            shown = ", ".join(absent[:6])
            more = len(absent) - 6
            if more > 0:
                shown += f" (+{more} more)"
            parts.append(f"missing required artifacts: {shown}")
        if self.integrity_rejected_artifact_ids:
            shown = ", ".join(self.integrity_rejected_artifact_ids[:6])
            more = len(self.integrity_rejected_artifact_ids) - 6
            if more > 0:
                shown += f" (+{more} more)"
            parts.append(f"structurally incomplete artifacts: {shown}")
        if self.conflicts:
            shown = ", ".join(
                sorted({conflict.artifact_id or conflict.path
                        for conflict in self.conflicts})[:6]
            )
            parts.append(f"conflicting artifacts: {shown}")
        if self.duplicates:
            shown = ", ".join(
                duplicate.artifact_id for duplicate in self.duplicates[:6]
            )
            parts.append(f"duplicate artifact submissions: {shown}")
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected artifact submission(s)")
        if not parts:
            parts.append("no usable artifact set could be assembled")
        return "; ".join(parts)

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "status": self.status.value,
            "artifacts": [c.to_dict() for c in self.artifacts],
            "expected_required_artifact_ids": list(self.expected_required_artifact_ids),
            "expected_optional_artifact_ids": list(self.expected_optional_artifact_ids),
            "missing_required_artifact_ids": list(self.missing_required_artifact_ids),
            "missing_optional_artifact_ids": list(self.missing_optional_artifact_ids),
            "duplicates": [d.to_dict() for d in self.duplicates],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "rejected": [r.to_dict() for r in self.rejected],
            "package_status": [list(pair) for pair in self.package_status],
            "validation_result_artifact_ids": list(self.validation_result_artifact_ids),
            "integrity_rejected_artifact_ids": list(
                self.integrity_rejected_artifact_ids
            ),
            "warnings": list(self.warnings),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssembledDeliverable":
        if not isinstance(data, Mapping):
            raise ArtifactResultError("assembled deliverable must be a JSON object")
        return cls(
            manifest_id=str(data.get("manifest_id") or ""),
            status=_coerce_enum(
                data.get("status"), AssemblyStatus, AssemblyStatus.FAILED
            ),
            artifacts=tuple(
                ArtifactCandidate.from_dict(item)
                for item in (data.get("artifacts") or ())
            ),
            expected_required_artifact_ids=tuple(
                data.get("expected_required_artifact_ids") or ()
            ),
            expected_optional_artifact_ids=tuple(
                data.get("expected_optional_artifact_ids") or ()
            ),
            missing_required_artifact_ids=tuple(
                data.get("missing_required_artifact_ids") or ()
            ),
            missing_optional_artifact_ids=tuple(
                data.get("missing_optional_artifact_ids") or ()
            ),
            duplicates=tuple(
                DuplicateDiagnostic.from_dict(item)
                for item in (data.get("duplicates") or ())
            ),
            conflicts=tuple(
                ConflictDiagnostic.from_dict(item)
                for item in (data.get("conflicts") or ())
            ),
            rejected=tuple(
                CandidateRejection.from_dict(item)
                for item in (data.get("rejected") or ())
            ),
            package_status=tuple(
                (pair[0], pair[1]) for pair in (data.get("package_status") or ())
            ),
            validation_result_artifact_ids=tuple(
                data.get("validation_result_artifact_ids") or ()
            ),
            integrity_rejected_artifact_ids=tuple(
                data.get("integrity_rejected_artifact_ids") or ()
            ),
            warnings=tuple(data.get("warnings") or ()),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Part C — parsing and normalization (ONE authoritative parser)
# --------------------------------------------------------------------------- #
def _scope_specs(scope: ArtifactExecutionScope) -> dict[str, ArtifactSpec]:
    return {spec.id: spec for spec in scope.owned_specs}


def _scope_owner(scope: ArtifactExecutionScope, artifact_id: str) -> str:
    for owned_id, package_id in scope.artifact_owners:
        if owned_id == artifact_id:
            return package_id
    return scope.package_ids[0] if scope.package_ids else ""


def scope_requires_artifacts(scope: Optional[ArtifactExecutionScope]) -> bool:
    """Does this scope legitimately owe produced artifacts?

    A scope that owns nothing (a pure validation-free coordination scope) may
    return an empty artifact list; every other scope must produce its outputs.
    """
    return bool(scope is not None and scope.output_artifact_ids)


def _reject(
    code: CollectionErrorCode,
    message: str,
    *,
    subtask_id: str,
    artifact_id: str = "",
    path: str = "",
    index: int = -1,
) -> CandidateRejection:
    return CandidateRejection(
        code=code, message=message, artifact_id=artifact_id, path=path,
        subtask_id=subtask_id, index=index,
    )


def _entry_artifacts(payload: Any) -> tuple[Optional[Sequence[Any]], bool]:
    """Extract the typed ``artifacts`` array from a worker envelope.

    Returns ``(entries, engaged)``. ``engaged`` is true only when the worker
    used the Phase 4C list form; a legacy mapping/string envelope (or none at
    all) leaves it false so pre-4C behavior is preserved exactly.
    """
    if not isinstance(payload, Mapping):
        return None, False
    raw = payload.get("artifacts")
    if isinstance(raw, (list, tuple)):
        return list(raw), True
    return None, False


def parse_artifact_candidates(
    payload: Any,
    scope: ArtifactExecutionScope,
    *,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    attempt: int = 0,
    producer_subtask_id: str = "",
    recursion_depth: int = 0,
) -> tuple[tuple[ArtifactCandidate, ...], tuple[CandidateRejection, ...], bool]:
    """Normalize one worker envelope into typed candidates for ONE scope.

    Deterministic: identical input yields identical candidates, hashes and
    rejections. Content is preserved EXACTLY. Path and manifest rules are not
    re-implemented here — they are reused from Phase 4A via the scope's specs.
    """
    entries, engaged = _entry_artifacts(payload)
    if not engaged:
        return (), (), False

    subtask_id = scope.subtask_id
    specs = _scope_specs(scope)
    known_ids = set(scope.known_artifact_ids) or set(specs)
    external_ids = set(scope.external_artifact_ids)
    summary = ""
    if isinstance(payload, Mapping):
        summary = _clip_line(payload.get("summary"), policy.max_summary_chars)

    candidates: list[ArtifactCandidate] = []
    rejections: list[CandidateRejection] = []
    seen: set[str] = set()
    aggregate = 0

    for index, entry in enumerate(entries or ()):
        if index >= policy.max_candidates_per_response:
            rejections.append(_reject(
                CollectionErrorCode.TOO_MANY_CANDIDATES,
                f"the reply submitted more than {policy.max_candidates_per_response} "
                "artifacts; split the work across the planned packages",
                subtask_id=subtask_id, index=index,
            ))
            break
        if not isinstance(entry, Mapping):
            rejections.append(_reject(
                CollectionErrorCode.MALFORMED_ENTRY,
                f"artifact entry {index} is not a JSON object",
                subtask_id=subtask_id, index=index,
            ))
            continue

        artifact_id = _clip_line(entry.get("artifact_id") or entry.get("id"), 300)
        raw_path = entry.get("path")
        path = "" if raw_path is None else str(raw_path)
        if not artifact_id:
            rejections.append(_reject(
                CollectionErrorCode.MISSING_ARTIFACT_ID,
                f"artifact entry {index} has no 'artifact_id'; ids from your "
                "assigned package are authoritative",
                subtask_id=subtask_id, path=_clip_line(path, 300), index=index,
            ))
            continue
        spec = specs.get(artifact_id)
        if spec is None:
            if artifact_id in external_ids:
                rejections.append(_reject(
                    CollectionErrorCode.EXTERNAL_ARTIFACT,
                    f"artifact {artifact_id!r} is external: it is supplied from "
                    "outside the plan and must not be produced",
                    subtask_id=subtask_id, artifact_id=artifact_id, index=index,
                ))
            elif artifact_id in known_ids:
                rejections.append(_reject(
                    CollectionErrorCode.OUT_OF_SCOPE,
                    f"artifact {artifact_id!r} is owned by another package; your "
                    "subtask must return only its own assigned artifacts",
                    subtask_id=subtask_id, artifact_id=artifact_id, index=index,
                ))
            else:
                rejections.append(_reject(
                    CollectionErrorCode.UNKNOWN_ARTIFACT_ID,
                    f"artifact id {artifact_id!r} is not in the artifact manifest",
                    subtask_id=subtask_id, artifact_id=artifact_id, index=index,
                ))
            continue

        if not path.strip():
            rejections.append(_reject(
                CollectionErrorCode.MISSING_PATH,
                f"artifact {artifact_id!r} was submitted without a path; it must "
                f"be {spec.path!r}",
                subtask_id=subtask_id, artifact_id=artifact_id, index=index,
            ))
            continue
        try:
            canonical = canonical_artifact_path(path)
        except ArtifactContractError as exc:
            rejections.append(_reject(
                CollectionErrorCode.INVALID_PATH,
                f"artifact {artifact_id!r} has an invalid path: {exc}",
                subtask_id=subtask_id, artifact_id=artifact_id,
                path=_clip_line(path, 300), index=index,
            ))
            continue
        if canonical != spec.path:
            rejections.append(_reject(
                CollectionErrorCode.PATH_MISMATCH,
                f"artifact {artifact_id!r} was returned at {canonical!r} but the "
                f"manifest declares {spec.path!r}",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue

        if artifact_id in seen:
            rejections.append(_reject(
                CollectionErrorCode.DUPLICATE_IN_RESPONSE,
                f"artifact {artifact_id!r} was submitted more than once in one "
                "reply; each owned artifact must appear exactly once",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue

        raw_content = entry.get("content")
        if raw_content is None:
            content = ""
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            rejections.append(_reject(
                CollectionErrorCode.INVALID_CONTENT_TYPE,
                f"artifact {artifact_id!r} has non-text content; return the "
                "complete file contents as a JSON string",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue

        raw_operation = entry.get("operation")
        if raw_operation is not None:
            try:
                submitted_operation = ArtifactOperation(
                    str(raw_operation).strip().lower()
                )
            except ValueError:
                rejections.append(_reject(
                    CollectionErrorCode.INVALID_OPERATION,
                    f"artifact {artifact_id!r} declares unknown operation "
                    f"{str(raw_operation)!r}",
                    subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                    index=index,
                ))
                continue
            if submitted_operation is not spec.operation:
                rejections.append(_reject(
                    CollectionErrorCode.INVALID_OPERATION,
                    f"artifact {artifact_id!r} declares operation "
                    f"{submitted_operation.value!r} but the manifest declares "
                    f"{spec.operation.value!r}; operations are manifest-controlled",
                    subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                    index=index,
                ))
                continue
        operation = spec.operation

        content_allowed = (
            spec.kind is not ArtifactKind.DIRECTORY
            and operation is not ArtifactOperation.DELETE
        )
        if not content_allowed and content.strip():
            code = (
                CollectionErrorCode.DIRECTORY_CONTENT
                if spec.kind is ArtifactKind.DIRECTORY
                else CollectionErrorCode.DELETE_CONTENT
            )
            reason = (
                "a directory artifact owns no file content"
                if spec.kind is ArtifactKind.DIRECTORY
                else "a delete operation carries no content"
            )
            rejections.append(_reject(
                code, f"artifact {artifact_id!r} carries content but {reason}",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue
        if content_allowed and not content.strip():
            rejections.append(_reject(
                CollectionErrorCode.MISSING_CONTENT,
                f"artifact {artifact_id!r} was returned empty; the complete "
                "content is required",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue
        if len(content) > policy.max_content_chars:
            rejections.append(_reject(
                CollectionErrorCode.CONTENT_TOO_LARGE,
                f"artifact {artifact_id!r} is {len(content)} characters; the "
                f"per-artifact limit is {policy.max_content_chars}",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue
        if aggregate + len(content) > policy.max_subtask_content_chars:
            rejections.append(_reject(
                CollectionErrorCode.AGGREGATE_TOO_LARGE,
                f"artifact {artifact_id!r} pushes this subtask past the "
                f"{policy.max_subtask_content_chars}-character collection limit",
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue

        # --- Phase 4D: structural integrity --------------------------- #
        # The LAST gate before acceptance, and the one the worker cannot talk
        # its way past: the body is measured, not asked. A candidate that fails
        # here never reaches the accepted set, never participates in a conflict,
        # and never gets rendered as a finished file — but its exact content is
        # still preserved verbatim for the diagnostics below.
        # Manifest metadata is trusted. A worker may supply a hint only when the
        # manifest is silent; it may never relabel a declared artifact to bypass
        # its structural inspector.
        language = _clip_line(
            spec.language or entry.get("language"), policy.max_language_chars
        )
        media_type = _clip_line(
            spec.media_type or entry.get("media_type"), policy.max_media_type_chars
        )
        declared_complete = bool(entry.get("complete", True))
        integrity: Optional[ArtifactIntegrityResult] = None
        if content_allowed:
            integrity = evaluate_artifact_integrity(
                content,
                artifact_id=artifact_id,
                path=canonical,
                kind=spec.kind,
                operation=operation,
                language=language,
                media_type=media_type,
                policy=policy.integrity,
            )
            if integrity.blocking(policy.integrity):
                code = (
                    CollectionErrorCode.STRUCTURALLY_INVALID
                    if integrity.status is IntegrityStatus.INVALID
                    else CollectionErrorCode.TRUNCATION_SUSPECTED
                )
                rejections.append(_reject(
                    code,
                    integrity_feedback(integrity, policy=policy.integrity),
                    subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                    index=index,
                ))
                continue
            if not declared_complete:
                # An explicit incomplete signal is honored even when the body
                # happens to hold together: the worker itself says there is more.
                rejections.append(_reject(
                    CollectionErrorCode.INCOMPLETE_DECLARATION,
                    f"artifact {artifact_id!r} at {canonical} was returned with "
                    "\"complete\": false; return the complete file contents",
                    subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                    index=index,
                ))
                continue

        try:
            candidate = ArtifactCandidate(
                artifact_id=artifact_id,
                path=canonical,
                subtask_id=subtask_id,
                package_id=_scope_owner(scope, artifact_id),
                content=content,
                kind=spec.kind,
                operation=operation,
                language=language,
                media_type=media_type,
                complete=declared_complete,
                attempt=attempt,
                summary=_clip_line(entry.get("summary"), policy.max_summary_chars)
                or summary,
                producer_subtask_id=producer_subtask_id or subtask_id,
                recursion_depth=recursion_depth,
                metadata=entry.get("metadata"),
                integrity=integrity,
            )
        except ArtifactResultError as exc:
            rejections.append(_reject(
                CollectionErrorCode.MALFORMED_ENTRY, str(exc),
                subtask_id=subtask_id, artifact_id=artifact_id, path=canonical,
                index=index,
            ))
            continue

        seen.add(artifact_id)
        aggregate += len(content)
        candidates.append(candidate)

    return tuple(candidates), tuple(rejections[: policy.max_rejections]), True


def collect_subtask_artifacts(
    payload: Any,
    scope: ArtifactExecutionScope,
    *,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    attempt: int = 0,
    producer_subtask_id: str = "",
    recursion_depth: int = 0,
    work_plan: Optional[ArtifactWorkPlan] = None,
) -> SubtaskCollectionResult:
    """Collect ONE subtask's produced artifacts against its assigned scope.

    ``work_plan`` is accepted (and cross-checked) at the root, where it exists;
    a recursive child validates against the scope it inherited, which carries
    the same Phase 4A specs. Never raises for model misbehavior: malformed or
    unauthorized submissions become typed rejections.
    """
    warnings: list[str] = []
    if work_plan is not None and work_plan.manifest.id != scope.manifest_id:
        warnings.append(
            "the assigned scope belongs to a different manifest than the work plan"
        )

    candidates, rejections, engaged = parse_artifact_candidates(
        payload, scope, policy=policy, attempt=attempt,
        producer_subtask_id=producer_subtask_id, recursion_depth=recursion_depth,
    )

    owed_required = tuple(
        spec.id for spec in scope.owned_specs
        if spec.required and spec.id in scope.output_artifact_ids
    )
    owed_optional = tuple(
        spec.id for spec in scope.owned_specs
        if not spec.required and spec.id in scope.output_artifact_ids
    )
    produced = {candidate.artifact_id for candidate in candidates}

    if not engaged:
        if scope_requires_artifacts(scope):
            warnings.append(
                "the worker did not use the typed artifact envelope, so no "
                "produced artifacts could be collected"
            )
            return SubtaskCollectionResult(
                subtask_id=scope.subtask_id,
                package_ids=scope.package_ids,
                missing_required_artifact_ids=owed_required,
                missing_optional_artifact_ids=owed_optional,
                warnings=tuple(warnings[: policy.max_warnings]),
                engaged=False,
            )
        return SubtaskCollectionResult(
            subtask_id=scope.subtask_id,
            package_ids=scope.package_ids,
            warnings=tuple(warnings[: policy.max_warnings]),
            engaged=False,
        )

    missing_required = tuple(aid for aid in owed_required if aid not in produced)
    missing_optional = tuple(aid for aid in owed_optional if aid not in produced)
    unexpected = tuple(sorted({
        rejection.artifact_id for rejection in rejections
        if rejection.artifact_id and rejection.code in (
            CollectionErrorCode.OUT_OF_SCOPE,
            CollectionErrorCode.UNKNOWN_ARTIFACT_ID,
            CollectionErrorCode.EXTERNAL_ARTIFACT,
        )
    }))
    duplicates = tuple(sorted({
        rejection.artifact_id for rejection in rejections
        if rejection.code is CollectionErrorCode.DUPLICATE_IN_RESPONSE
        and rejection.artifact_id
    }))
    accepted_unowed = tuple(
        candidate.artifact_id for candidate in candidates
        if candidate.artifact_id not in scope.output_artifact_ids
    )
    if accepted_unowed:
        warnings.append(
            "accepted artifacts outside the declared outputs of this scope: "
            + ", ".join(sorted(set(accepted_unowed))[:6])
        )

    return SubtaskCollectionResult(
        subtask_id=scope.subtask_id,
        package_ids=scope.package_ids,
        accepted=candidates,
        rejected=rejections,
        missing_required_artifact_ids=missing_required,
        missing_optional_artifact_ids=missing_optional,
        unexpected_artifact_ids=unexpected,
        duplicate_artifact_ids=duplicates,
        warnings=tuple(warnings[: policy.max_warnings]),
        engaged=True,
    )


def merge_subtask_collections(
    collections: Iterable[SubtaskCollectionResult],
    scope: ArtifactExecutionScope,
    *,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
) -> SubtaskCollectionResult:
    """Combine a recursive branch's collections into ONE result for the parent.

    Competing candidates from different branches are ALL retained (never
    de-duplicated away) so root conflict detection can see them; the duplicate
    ids are recorded here too.
    """
    ordered = [c for c in collections if c is not None]
    accepted: list[ArtifactCandidate] = []
    rejected: list[CandidateRejection] = []
    warnings: list[str] = []
    engaged = False
    seen: dict[str, ArtifactCandidate] = {}
    duplicates: set[str] = set()

    for collection in ordered:
        engaged = engaged or collection.engaged
        rejected.extend(collection.rejected)
        warnings.extend(collection.warnings)
        for candidate in collection.accepted:
            previous = seen.get(candidate.artifact_id)
            if previous is not None:
                duplicates.add(candidate.artifact_id)
                warnings.append(
                    f"artifact {candidate.artifact_id!r} was produced by more than "
                    "one delegation inside this recursive package"
                )
            else:
                seen[candidate.artifact_id] = candidate
            accepted.append(candidate)

    produced = set(seen)
    owed_required = tuple(
        spec.id for spec in scope.owned_specs
        if spec.required and spec.id in scope.output_artifact_ids
    )
    owed_optional = tuple(
        spec.id for spec in scope.owned_specs
        if not spec.required and spec.id in scope.output_artifact_ids
    )
    unexpected = tuple(sorted({
        aid
        for collection in ordered
        for aid in collection.unexpected_artifact_ids
    }))
    duplicates.update(
        aid for collection in ordered for aid in collection.duplicate_artifact_ids
    )

    return SubtaskCollectionResult(
        subtask_id=scope.subtask_id,
        package_ids=scope.package_ids,
        accepted=tuple(accepted),
        rejected=tuple(rejected[: policy.max_rejections]),
        missing_required_artifact_ids=tuple(
            aid for aid in owed_required if aid not in produced
        ),
        missing_optional_artifact_ids=tuple(
            aid for aid in owed_optional if aid not in produced
        ),
        unexpected_artifact_ids=unexpected,
        duplicate_artifact_ids=tuple(sorted(duplicates)),
        warnings=tuple(warnings[: policy.max_warnings]),
        engaged=engaged,
    )


# --------------------------------------------------------------------------- #
# Parts E/F/G — root collection, conflict policy, assembly
# --------------------------------------------------------------------------- #
def assemble_deliverable(
    work_plan: ArtifactWorkPlan,
    collections: Iterable[SubtaskCollectionResult],
    *,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
) -> AssembledDeliverable:
    """Deterministically combine every subtask's collection into ONE deliverable.

    Traversal is in manifest order, so the assembled artifacts, diagnostics and
    summary are byte-stable for identical inputs. Conflicts are recorded, never
    resolved: no merge, no latest-wins, no model judgment, no file writes.
    """
    manifest = work_plan.manifest
    specs = manifest.by_id()
    ordered_collections = [c for c in collections if c is not None]
    manifest_order = {spec.id: index for index, spec in enumerate(manifest.artifacts)}
    package_order = {
        package.id: index for index, package in enumerate(work_plan.packages)
    }
    subtask_order: dict[str, int] = {}
    for package in work_plan.packages:
        subtask_id = work_plan.subtask_for(package.id)
        if subtask_id is not None and subtask_id not in subtask_order:
            subtask_order[subtask_id] = len(subtask_order)

    def rejection_key(rejection: CandidateRejection) -> tuple:
        return (
            subtask_order.get(rejection.subtask_id, len(subtask_order)),
            rejection.subtask_id,
            rejection.index,
            rejection.code.value,
            rejection.artifact_id,
            rejection.path,
            rejection.message,
        )

    def candidate_key(candidate: ArtifactCandidate) -> tuple:
        return (
            manifest_order.get(candidate.artifact_id, len(manifest_order)),
            package_order.get(candidate.package_id, len(package_order)),
            subtask_order.get(candidate.subtask_id, len(subtask_order)),
            candidate.provenance,
            candidate.path,
            candidate.operation.value,
            candidate.kind.value,
            candidate.content_hash,
            candidate.language,
            candidate.media_type,
            candidate.complete,
            candidate.summary,
            candidate.metadata,
        )

    warnings: list[str] = sorted({
        warning
        for collection in ordered_collections
        for warning in collection.warnings
    })
    rejected: list[CandidateRejection] = sorted(
        (
            rejection
            for collection in ordered_collections
            for rejection in collection.rejected
        ),
        key=rejection_key,
    )
    submitted: dict[str, list[ArtifactCandidate]] = {}
    total_content = 0
    total_candidates = 0

    overflowed = False
    candidates = sorted(
        (
            candidate
            for collection in ordered_collections
            for candidate in collection.accepted
        ),
        key=candidate_key,
    )
    for candidate in candidates:
        if overflowed:
            continue
        if candidate.artifact_id not in specs:
            rejected.append(_reject(
                CollectionErrorCode.UNKNOWN_ARTIFACT_ID,
                f"artifact id {candidate.artifact_id!r} is not in the artifact manifest",
                subtask_id=candidate.subtask_id,
                artifact_id=candidate.artifact_id,
                path=candidate.path,
            ))
            continue
        total_candidates += 1
        if total_candidates > policy.max_total_candidates:
            warnings.append(
                "the run submitted more artifact candidates than the "
                f"{policy.max_total_candidates} allowed; the excess was refused"
            )
            overflowed = True
            continue
        if total_content + len(candidate.content) > policy.max_total_content_chars:
            warnings.append(
                "collected artifact content exceeded the aggregate limit of "
                f"{policy.max_total_content_chars} characters"
            )
            overflowed = True
            continue
        total_content += len(candidate.content)
        submitted.setdefault(candidate.artifact_id, []).append(candidate)

    conflicts: list[ConflictDiagnostic] = []
    duplicates: list[DuplicateDiagnostic] = []
    assembled: list[ArtifactCandidate] = []
    conflicted_ids: set[str] = set()

    for spec in manifest.artifacts:  # manifest order == deterministic order
        entries = submitted.get(spec.id, [])
        if not entries:
            continue
        declared_owner = work_plan.owner_of(spec.id)
        declared_subtask = (
            work_plan.subtask_for(declared_owner) if declared_owner else None
        )
        offenders = [
            c for c in entries
            if c.package_id != declared_owner or c.subtask_id != declared_subtask
        ]
        if offenders:
            conflicted_ids.add(spec.id)
            conflicts.append(ConflictDiagnostic(
                kind=ConflictKind.OWNERSHIP,
                artifact_id=spec.id,
                path=spec.path,
                entries=tuple(
                    ConflictEntry.of(c)
                    for c in sorted(entries, key=lambda c: ConflictEntry.of(c).sort_key())
                ),
                message=(
                    f"artifact {spec.id!r} is owned by package "
                    f"{declared_owner or 'none'!r} / subtask "
                    f"{declared_subtask or 'none'!r} but was submitted by "
                    + ", ".join(sorted({
                        f"{c.package_id}/{c.subtask_id}" for c in offenders
                    }))
                ),
            ))
            continue

        hashes = {c.content_hash for c in entries}
        operations = {c.operation for c in entries}
        if len(operations) > 1:
            conflicted_ids.add(spec.id)
            conflicts.append(ConflictDiagnostic(
                kind=ConflictKind.OPERATION,
                artifact_id=spec.id,
                path=spec.path,
                entries=tuple(
                    ConflictEntry.of(c)
                    for c in sorted(entries, key=lambda c: ConflictEntry.of(c).sort_key())
                ),
                message=(
                    f"artifact {spec.id!r} has incompatible operations: "
                    + ", ".join(sorted(op.value for op in operations))
                ),
            ))
            continue
        if len(hashes) > 1:
            conflicted_ids.add(spec.id)
            conflicts.append(ConflictDiagnostic(
                kind=ConflictKind.CONTENT,
                artifact_id=spec.id,
                path=spec.path,
                entries=tuple(
                    ConflictEntry.of(c)
                    for c in sorted(entries, key=lambda c: ConflictEntry.of(c).sort_key())
                ),
                message=(
                    f"artifact {spec.id!r} has {len(hashes)} competing contents "
                    "and was not assembled"
                ),
            ))
            continue

        if len(entries) > 1:
            duplicates.append(DuplicateDiagnostic(
                artifact_id=spec.id,
                path=spec.path,
                content_hash=entries[0].content_hash,
                provenance=tuple(
                    candidate.describe_provenance()
                    for candidate in sorted(
                        entries, key=lambda c: ConflictEntry.of(c).sort_key()
                    )
                ),
            ))
            warnings.append(
                f"artifact {spec.id!r} was submitted {len(entries)} times with "
                "identical content; ownership should be unique"
            )
        assembled.append(entries[0])

    # Path collisions cannot occur in a valid Phase 4A manifest, but malformed
    # data reaching this layer must still fail safely rather than overwrite.
    by_path: dict[str, ArtifactCandidate] = {}
    for candidate in assembled:
        clash = by_path.get(candidate.path)
        if clash is not None and clash.artifact_id != candidate.artifact_id:
            conflicted_ids.add(candidate.artifact_id)
            conflicted_ids.add(clash.artifact_id)
            conflicts.append(ConflictDiagnostic(
                kind=ConflictKind.PATH_COLLISION,
                artifact_id=candidate.artifact_id,
                path=candidate.path,
                entries=(ConflictEntry.of(clash), ConflictEntry.of(candidate)),
                message=(
                    f"artifacts {clash.artifact_id!r} and {candidate.artifact_id!r} "
                    f"both target {candidate.path!r}"
                ),
            ))
        else:
            by_path[candidate.path] = candidate
    if conflicted_ids:
        assembled = [c for c in assembled if c.artifact_id not in conflicted_ids]

    assembled_ids = {candidate.artifact_id for candidate in assembled}
    expected_required = tuple(
        spec.id for spec in manifest.artifacts if spec.required and not spec.external
    )
    expected_optional = tuple(
        spec.id for spec in manifest.artifacts
        if not spec.required and not spec.external
    )
    missing_required = tuple(
        aid for aid in expected_required if aid not in assembled_ids
    )
    missing_optional = tuple(
        aid for aid in expected_optional if aid not in assembled_ids
    )
    validation_results = tuple(
        spec.id for spec in manifest.artifacts
        if spec.kind in _RESULT_KINDS and spec.id in assembled_ids
    )
    # Phase 4D: an artifact that came back truncated is NOT the same failure as
    # an artifact nobody produced. Reported in manifest order, and only when the
    # artifact did not also arrive intact from a legitimate producer.
    integrity_rejected_ids = {
        rejection.artifact_id
        for rejection in rejected
        if rejection.artifact_id and rejection.code in _INTEGRITY_REJECTIONS
    }
    integrity_rejected = tuple(
        spec.id for spec in manifest.artifacts
        if spec.id in integrity_rejected_ids and spec.id not in assembled_ids
    )

    package_status: list[tuple[str, str]] = []
    for package in work_plan.packages:
        owned = tuple(package.owns)
        if not owned:
            package_status.append((package.id, "not_applicable"))
            continue
        if any(aid in conflicted_ids for aid in owned):
            package_status.append((package.id, "conflicted"))
            continue
        outstanding = [
            aid for aid in owned
            if specs[aid].required and aid not in assembled_ids
        ]
        package_status.append(
            (package.id, "incomplete" if outstanding else "complete")
        )

    rejected.sort(key=rejection_key)
    fatal_conflict = any(conflict.fatal for conflict in conflicts)
    fatal_rejection = any(
        rejection.code in _FATAL_REJECTIONS for rejection in rejected
    )
    unengaged_obligation = any(
        not collection.engaged
        and bool(
            collection.missing_required_artifact_ids
            or collection.missing_optional_artifact_ids
        )
        for collection in ordered_collections
    )
    duplicate_rejection = any(
        rejection.code is CollectionErrorCode.DUPLICATE_IN_RESPONSE
        for rejection in rejected
    )
    if fatal_conflict or fatal_rejection:
        status = AssemblyStatus.FAILED
    elif not assembled and expected_required:
        status = AssemblyStatus.FAILED
    elif conflicts or duplicates or duplicate_rejection:
        status = AssemblyStatus.CONFLICTED
    elif missing_required or rejected or unengaged_obligation:
        status = AssemblyStatus.PARTIAL
    else:
        status = AssemblyStatus.COMPLETE

    return AssembledDeliverable(
        manifest_id=manifest.id,
        status=status,
        artifacts=tuple(assembled),
        expected_required_artifact_ids=expected_required,
        expected_optional_artifact_ids=expected_optional,
        missing_required_artifact_ids=missing_required,
        missing_optional_artifact_ids=missing_optional,
        duplicates=tuple(duplicates[: policy.max_duplicates]),
        conflicts=tuple(conflicts[: policy.max_conflicts]),
        rejected=tuple(rejected[: policy.max_rejections]),
        package_status=tuple(package_status),
        validation_result_artifact_ids=validation_results,
        integrity_rejected_artifact_ids=integrity_rejected,
        warnings=tuple(warnings[: policy.max_warnings]),
    )


# --------------------------------------------------------------------------- #
# Part J — deterministic presentation for synthesis
# --------------------------------------------------------------------------- #
_STATUS_HEADLINE = {
    AssemblyStatus.COMPLETE: (
        "All required artifacts were produced and assembled."
    ),
    AssemblyStatus.PARTIAL: (
        "⚠️ INCOMPLETE DELIVERABLE — some required artifacts were never produced "
        "or were not structurally complete."
    ),
    AssemblyStatus.CONFLICTED: (
        "⚠️ CONFLICTED DELIVERABLE — duplicate or competing artifact "
        "submissions remain unresolved."
    ),
    AssemblyStatus.FAILED: (
        "❌ ASSEMBLY FAILED — no usable artifact set could be assembled."
    ),
}


def _fence(language: str, content: str) -> str:
    """Fence content without ever rewriting it (widen the fence if needed)."""
    longest = 0
    for run in re.findall(r"`+", content):
        longest = max(longest, len(run))
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{content}\n{fence}"


def render_assembled_deliverable(assembly: AssembledDeliverable) -> str:
    """Deterministic, lossless rendering of an assembled deliverable.

    Complete artifacts keep their path AND their exact content. Conflicted and
    missing artifacts are named, never silently presented as delivered. No LLM
    is called and nothing is written to disk.
    """
    lines: list[str] = [_STATUS_HEADLINE[assembly.status], ""]

    broken = {
        rejection.artifact_id: rejection
        for rejection in assembly.rejected
        if rejection.code in _INTEGRITY_REJECTIONS and rejection.artifact_id
    }
    if assembly.missing_required_artifact_ids:
        lines.append("**Missing required artifacts**")
        for artifact_id in assembly.missing_required_artifact_ids:
            note = broken.get(artifact_id)
            lines.append(
                f"- {artifact_id}"
                + (" (returned but structurally incomplete)" if note else "")
            )
        lines.append("")
    # Phase 4D: a truncated body is NEVER rendered as a delivered file. It is
    # named, its defect is stated, and it is not repaired here.
    if assembly.integrity_rejected_artifact_ids:
        lines.append("**Structurally incomplete artifacts (rejected, not repaired)**")
        for artifact_id in assembly.integrity_rejected_artifact_ids:
            rejection = broken.get(artifact_id)
            path = rejection.path if rejection else ""
            message = rejection.message if rejection else "structurally incomplete"
            lines.append(f"- {path or artifact_id}: {message}")
        lines.append("")
    if assembly.missing_optional_artifact_ids:
        lines.append("**Missing optional artifacts**")
        for artifact_id in assembly.missing_optional_artifact_ids:
            lines.append(f"- {artifact_id}")
        lines.append("")
    if assembly.conflicts:
        lines.append("**Conflicting artifacts (not assembled, not merged)**")
        for conflict in assembly.conflicts:
            hashes = ", ".join(entry.content_hash[:19] for entry in conflict.entries)
            lines.append(
                f"- {conflict.path or conflict.artifact_id} "
                f"[{conflict.kind.value}]: {conflict.message} (hashes: {hashes})"
            )
        lines.append("")
    if assembly.duplicates:
        lines.append("**Duplicate submissions (identical content)**")
        for duplicate in assembly.duplicates:
            lines.append(
                f"- {duplicate.path or duplicate.artifact_id}: "
                + "; ".join(duplicate.provenance)
            )
        lines.append("")
    if assembly.rejected:
        lines.append("**Rejected artifact submissions**")
        for rejection in assembly.rejected:
            lines.append(f"- [{rejection.code.value}] {rejection.message}")
        lines.append("")

    if assembly.artifacts:
        lines.append("## Assembled artifacts")
        lines.append("")
        for candidate in assembly.artifacts:
            lines.append(f"### {candidate.path}")
            if candidate.kind in _RESULT_KINDS:
                lines.append("")
                lines.append(candidate.content)
                lines.append("")
                continue
            if not candidate.content:
                lines.append("")
                lines.append("_(no content: directory or delete operation)_")
                lines.append("")
                continue
            lines.append("")
            lines.append(_fence(candidate.language, candidate.content))
            lines.append("")

    return "\n".join(lines).strip()
