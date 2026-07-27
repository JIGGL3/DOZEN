"""Artifact manifest and work-package contracts (production-hardening Phase 4A).

WHY THIS EXISTS
---------------
Phase 3 taught DOZEN what KIND of outcome a request owes the user
(``DeliverableContract``). Nothing yet represents WHICH concrete deliverables
constitute that outcome. Asked to "build me a React dashboard", the pipeline
still has no typed statement of "these files, grouped into these bounded work
packages, validated like this" — so decomposition of large deliverables cannot
be checked, assembled, or repaired in later phases.

This module is that missing vocabulary, and ONLY the vocabulary:

    DeliverableContract  → says an implementation is required          (Phase 3)
    ArtifactManifest     → says which concrete deliverables constitute it
    ArtifactWorkPlan     → says which bounded work packages produce and
                           validate them

Everything here is pure, deterministic, immutable and serializable. Nothing
here touches the filesystem, executes a command, calls a provider, or contains
file contents — the manifest describes EXPECTED deliverables, never produced
bytes. Population, assembly, truncation detection and repair are later phases.

This module deliberately imports nothing from the rest of ``dozen`` so any
other module (models, planner, prompts) can depend on it without cycles.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class ArtifactContractError(ValueError):
    """A structural violation of the artifact contracts.

    Deterministic and message-bearing: the planner surfaces the message as
    corrective re-plan feedback, so it must state the specific problem.
    """


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class ArtifactKind(str, Enum):
    SOURCE_FILE = "source_file"      # source code forming part of the deliverable
    TEST_FILE = "test_file"          # automated test source file
    CONFIG_FILE = "config_file"      # project or tool configuration
    DOCUMENT = "document"            # a requested documentation artifact (product output)
    DATA_FILE = "data_file"          # structured or unstructured data artifact
    DIRECTORY = "directory"          # logical directory; owns no file content
    PATCH = "patch"                  # a patch/diff deliverable
    COMMAND_RESULT = "command_result"  # evidence from a command invocation
    BUILD_RESULT = "build_result"    # evidence a build/compilation completed
    TEST_RESULT = "test_result"      # evidence tests completed
    OTHER = "other"                  # conservative extension fallback


class ArtifactOperation(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    INSPECT = "inspect"
    GENERATE = "generate"


class ValidationKind(str, Enum):
    SYNTAX = "syntax"
    TYPE_CHECK = "type_check"
    BUILD = "build"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    LINT = "lint"
    SCHEMA = "schema"
    EXISTENCE = "existence"
    NON_EMPTY = "non_empty"
    CUSTOM = "custom"


class WorkPackageKind(str, Enum):
    IMPLEMENTATION = "implementation"
    MODIFICATION = "modification"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    REVIEW = "review"
    COORDINATION = "coordination"


# Unknown enum strings deserialize to these conservative fallbacks instead of
# failing, so a newer serializer never breaks an older reader.
_KIND_FALLBACK = ArtifactKind.OTHER
_OPERATION_FALLBACK = ArtifactOperation.CREATE
_VALIDATION_FALLBACK = ValidationKind.CUSTOM
_PACKAGE_FALLBACK = WorkPackageKind.COORDINATION

SCHEMA_VERSION = 1

_RESULT_ARTIFACT_KINDS = frozenset({
    ArtifactKind.COMMAND_RESULT,
    ArtifactKind.BUILD_RESULT,
    ArtifactKind.TEST_RESULT,
})
_VALIDATION_RESULT_KIND = {
    ValidationKind.SYNTAX: ArtifactKind.COMMAND_RESULT,
    ValidationKind.TYPE_CHECK: ArtifactKind.COMMAND_RESULT,
    ValidationKind.BUILD: ArtifactKind.BUILD_RESULT,
    ValidationKind.UNIT_TESTS: ArtifactKind.TEST_RESULT,
    ValidationKind.INTEGRATION_TESTS: ArtifactKind.TEST_RESULT,
    ValidationKind.LINT: ArtifactKind.COMMAND_RESULT,
    ValidationKind.SCHEMA: ArtifactKind.COMMAND_RESULT,
    ValidationKind.EXISTENCE: ArtifactKind.COMMAND_RESULT,
    ValidationKind.NON_EMPTY: ArtifactKind.COMMAND_RESULT,
    ValidationKind.CUSTOM: ArtifactKind.COMMAND_RESULT,
}

# ---- field bounds (contract fields are declarations, not content) ---------- #
_MAX_PATH_CHARS = 240
_MAX_TEXT_CHARS = 300
_MAX_CRITERION_CHARS = 200
_MAX_CRITERIA = 12
_MAX_METADATA_ITEMS = 16
_MAX_METADATA_KEY_CHARS = 60
_MAX_METADATA_VALUE_CHARS = 200
_MAX_CAPABILITY_HINTS = 12
_MAX_CAPABILITY_HINT_CHARS = 60
_MAX_ARTIFACT_REFERENCES = 80
_MAX_PACKAGE_REFERENCES = 24
_MAX_VALIDATION_REFERENCES = 40

# ---- prompt-rendering bounds ------------------------------------------------ #
# The in-memory model always retains EVERY entry; only rendering is clipped.
_BRIEF_MAX_ARTIFACTS = 16
_BRIEF_MAX_PACKAGES = 10
_BRIEF_MAX_VALIDATIONS = 6
_BRIEF_MAX_IDS_PER_LINE = 6
_BRIEF_LINE_CHARS = 160


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _clip(value: Any, limit: int) -> str:
    """Collapse whitespace (kills newline injection) and clip one field."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return default


def _str_tuple(value: Any, *, clip: int = 0) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raise ArtifactContractError("collection fields must not be JSON objects")
    if isinstance(value, str):
        items: list[Any] = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        text = _clip(item, clip) if clip else str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _criteria_tuple(value: Any) -> tuple[str, ...]:
    return _str_tuple(value, clip=_MAX_CRITERION_CHARS)[:_MAX_CRITERIA]


def _metadata_tuple(value: Any) -> tuple[tuple[str, str], ...]:
    """Canonicalize metadata into a bounded, sorted, immutable pair tuple."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        try:
            items = [(k, v) for k, v in value]
        except (TypeError, ValueError):
            raise ArtifactContractError(
                "metadata must be a mapping or an iterable of key/value pairs"
            )
    pairs: dict[str, str] = {}
    for key, val in items:
        k = _clip(key, _MAX_METADATA_KEY_CHARS)
        if k:
            v = _clip(val, _MAX_METADATA_VALUE_CHARS)
            if k in pairs and pairs[k] != v:
                raise ArtifactContractError(
                    f"metadata keys collide after canonicalization: {k!r}"
                )
            pairs[k] = v
    return tuple(sorted(pairs.items()))[:_MAX_METADATA_ITEMS]


def _json_array(
    value: Any,
    field: str,
    *,
    max_items: int = 0,
) -> tuple[Any, ...]:
    """Require a JSON-array-shaped nested field and return an immutable copy."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ArtifactContractError(f"{field} must be a JSON array")
    if max_items and len(value) > max_items:
        raise ArtifactContractError(
            f"{field} has {len(value)} entries; the limit is {max_items}"
        )
    return tuple(value)


def _coerce_enum(value: Any, enum_cls: type, fallback: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return fallback
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return fallback


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug[:60] or fallback


def _assert_acyclic(edges: Mapping[str, tuple[str, ...]], what: str) -> None:
    """Kahn's algorithm; raises with a deterministic message on any cycle."""
    deps = {node: {d for d in edges[node] if d in edges} for node in edges}
    indegree = {node: len(node_deps) for node, node_deps in deps.items()}
    dependents: dict[str, list[str]] = {node: [] for node in edges}
    for node, node_deps in deps.items():
        for dep in node_deps:
            dependents[dep].append(node)
    queue = sorted(node for node, deg in indegree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for other in sorted(dependents[node]):
            indegree[other] -= 1
            if indegree[other] == 0:
                queue.append(other)
    if visited != len(edges):
        cyclic = sorted(node for node, deg in indegree.items() if deg > 0)
        raise ArtifactContractError(
            f"{what} dependencies contain a cycle involving: {', '.join(cyclic)}"
        )


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_WINDOWS_INVALID_PATH_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def canonical_artifact_path(path: Any) -> str:
    """Canonicalize a logical artifact path. Pure; never touches the filesystem.

    Rules: separators normalize to ``/``; absolute paths, drive letters, ``~``,
    parent traversal (``..``) and empty/control-character paths are rejected,
    so no canonical path can name anything outside the (future) project root.
    Case is preserved as supplied — case-conflict detection is the manifest's
    job, so conflicting spellings are rejected there, never silently merged.

    Logical non-path artifacts (e.g. a ``build-result``) pass through as a
    single-segment name under the same safety rules.
    """
    text = str(path if path is not None else "")
    if not text.strip():
        raise ArtifactContractError("artifact path/name must not be empty")
    if len(text) > _MAX_PATH_CHARS:
        raise ArtifactContractError(
            f"artifact path exceeds {_MAX_PATH_CHARS} characters: {text[:50]!r}…"
        )
    if any(unicodedata.category(ch) == "Cc" for ch in text):
        raise ArtifactContractError(
            f"artifact path contains control characters: {text[:50]!r}"
        )
    if text != text.strip():
        raise ArtifactContractError(
            f"artifact path must not have leading or trailing whitespace: {text!r}"
        )
    normalized = text.replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.startswith("~")
        or _WINDOWS_DRIVE_RE.match(normalized)
    ):
        raise ArtifactContractError(f"absolute paths are not allowed: {text!r}")
    segments: list[str] = []
    for segment in normalized.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            raise ArtifactContractError(
                f"parent traversal is not allowed in artifact paths: {text!r}"
            )
        if not segment.strip():
            raise ArtifactContractError(f"blank path segment in: {text!r}")
        if segment != segment.strip():
            raise ArtifactContractError(
                f"path segments must not have leading or trailing whitespace: {text!r}"
            )
        if segment.endswith("."):
            raise ArtifactContractError(
                f"path segments must not end with a dot: {text!r}"
            )
        invalid = sorted(set(segment) & _WINDOWS_INVALID_PATH_CHARS)
        if invalid:
            raise ArtifactContractError(
                f"artifact path contains invalid filename characters: {text!r}"
            )
        base_name = segment.split(".", 1)[0].casefold()
        if base_name in _WINDOWS_RESERVED_NAMES:
            raise ArtifactContractError(
                f"artifact path uses a reserved Windows device name: {text!r}"
            )
        segments.append(segment)
    if not segments:
        raise ArtifactContractError(f"path has no usable segments: {text!r}")
    return "/".join(segments)


# --------------------------------------------------------------------------- #
# Artifact specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactSpec:
    """One expected deliverable. Describes the artifact; never contains it."""

    id: str
    path: str
    kind: ArtifactKind = ArtifactKind.OTHER
    required: bool = True
    operation: ArtifactOperation = ArtifactOperation.CREATE
    description: str = ""
    # Owning work-package id, when assigned (consistency enforced by the plan).
    package_id: str = ""
    depends_on: tuple[str, ...] = ()
    # Language / media type are metadata, deliberately NOT artifact kinds.
    language: str = ""
    media_type: str = ""
    # True = supplied from outside the plan; no work package must produce it.
    external: bool = False
    completion_criteria: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ident = _clip(self.id, _MAX_TEXT_CHARS)
        if not ident:
            raise ArtifactContractError("artifact id must not be empty")
        object.__setattr__(self, "id", ident)
        object.__setattr__(self, "path", canonical_artifact_path(self.path))
        object.__setattr__(self, "kind", _coerce_enum(self.kind, ArtifactKind, _KIND_FALLBACK))
        object.__setattr__(
            self, "operation",
            _coerce_enum(self.operation, ArtifactOperation, _OPERATION_FALLBACK),
        )
        object.__setattr__(self, "required", _bool(self.required, True))
        object.__setattr__(self, "external", _bool(self.external, False))
        object.__setattr__(self, "description", _clip(self.description, _MAX_TEXT_CHARS))
        object.__setattr__(self, "package_id", _clip(self.package_id, _MAX_TEXT_CHARS))
        object.__setattr__(self, "language", _clip(self.language, 40))
        object.__setattr__(self, "media_type", _clip(self.media_type, 80))
        object.__setattr__(
            self, "depends_on", _str_tuple(self.depends_on, clip=_MAX_TEXT_CHARS)
        )
        if self.id in self.depends_on:
            raise ArtifactContractError(f"artifact {self.id!r} depends on itself")
        object.__setattr__(
            self, "completion_criteria", _criteria_tuple(self.completion_criteria)
        )
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "kind": self.kind.value,
            "required": self.required,
            "operation": self.operation.value,
            "description": self.description,
            "package_id": self.package_id,
            "depends_on": list(self.depends_on),
            "language": self.language,
            "media_type": self.media_type,
            "external": self.external,
            "completion_criteria": list(self.completion_criteria),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactSpec":
        if not isinstance(data, Mapping):
            raise ArtifactContractError("artifact spec must be a JSON object")
        return cls(
            id=str(data.get("id") or ""),
            path=data.get("path"),
            kind=_coerce_enum(data.get("kind"), ArtifactKind, _KIND_FALLBACK),
            required=_bool(data.get("required"), True),
            operation=_coerce_enum(
                data.get("operation"), ArtifactOperation, _OPERATION_FALLBACK
            ),
            description=data.get("description") or "",
            package_id=data.get("package_id") or "",
            depends_on=_json_array(
                data.get("depends_on"), "artifact depends_on"
            ),
            language=data.get("language") or "",
            media_type=data.get("media_type") or "",
            external=_bool(data.get("external"), False),
            completion_criteria=_json_array(
                data.get("completion_criteria"),
                "artifact completion_criteria",
            ),
            metadata=data.get("metadata"),
        )


# --------------------------------------------------------------------------- #
# Validation requirement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactValidation:
    """A declared validation expectation. Declarative only — NEVER executed here.

    ``command`` is a description of what a later phase may run, not a function.
    Empty ``target_artifact_ids`` + empty ``target_package_id`` targets the
    manifest as a whole.
    """

    id: str
    kind: ValidationKind = ValidationKind.CUSTOM
    required: bool = True
    target_artifact_ids: tuple[str, ...] = ()
    target_package_id: str = ""
    command: str = ""
    success_criterion: str = ""
    # Optional id of the manifest artifact this validation's evidence becomes
    # (e.g. a TEST_RESULT artifact).
    produces_artifact_id: str = ""

    def __post_init__(self) -> None:
        ident = _clip(self.id, _MAX_TEXT_CHARS)
        if not ident:
            raise ArtifactContractError("validation id must not be empty")
        object.__setattr__(self, "id", ident)
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, ValidationKind, _VALIDATION_FALLBACK)
        )
        object.__setattr__(self, "required", _bool(self.required, True))
        object.__setattr__(
            self, "target_artifact_ids",
            _str_tuple(self.target_artifact_ids, clip=_MAX_TEXT_CHARS),
        )
        object.__setattr__(
            self, "target_package_id", _clip(self.target_package_id, _MAX_TEXT_CHARS)
        )
        object.__setattr__(self, "command", _clip(self.command, _MAX_TEXT_CHARS))
        object.__setattr__(
            self, "success_criterion", _clip(self.success_criterion, _MAX_TEXT_CHARS)
        )
        object.__setattr__(
            self, "produces_artifact_id",
            _clip(self.produces_artifact_id, _MAX_TEXT_CHARS),
        )

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "required": self.required,
            "target_artifact_ids": list(self.target_artifact_ids),
            "target_package_id": self.target_package_id,
            "command": self.command,
            "success_criterion": self.success_criterion,
            "produces_artifact_id": self.produces_artifact_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactValidation":
        if not isinstance(data, Mapping):
            raise ArtifactContractError("validation must be a JSON object")
        return cls(
            id=str(data.get("id") or ""),
            kind=_coerce_enum(data.get("kind"), ValidationKind, _VALIDATION_FALLBACK),
            required=_bool(data.get("required"), True),
            target_artifact_ids=_json_array(
                data.get("target_artifact_ids"),
                "validation target_artifact_ids",
            ),
            target_package_id=data.get("target_package_id") or "",
            command=data.get("command") or "",
            success_criterion=data.get("success_criterion") or "",
            produces_artifact_id=data.get("produces_artifact_id") or "",
        )


# --------------------------------------------------------------------------- #
# Artifact manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactManifest:
    """The complete EXPECTED deliverable: what must exist, never the bytes."""

    id: str
    title: str
    artifacts: tuple[ArtifactSpec, ...]
    description: str = ""
    validations: tuple[ArtifactValidation, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    # Optional reference back to the request / parent manifest that spawned it.
    source_request: str = ""
    # A declared label only; the model never resolves it on a real filesystem.
    project_root: str = ""
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ident = _clip(self.id, _MAX_TEXT_CHARS)
        if not ident:
            raise ArtifactContractError("manifest id must not be empty")
        object.__setattr__(self, "id", ident)
        object.__setattr__(self, "title", _clip(self.title, _MAX_TEXT_CHARS))
        object.__setattr__(self, "description", _clip(self.description, _MAX_TEXT_CHARS))
        object.__setattr__(self, "source_request", _clip(self.source_request, _MAX_TEXT_CHARS))
        object.__setattr__(self, "project_root", _clip(self.project_root, 120))
        object.__setattr__(
            self, "completion_criteria", _criteria_tuple(self.completion_criteria)
        )
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

        raw_artifacts = () if self.artifacts is None else self.artifacts
        if not isinstance(raw_artifacts, (list, tuple)):
            raise ArtifactContractError("manifest artifacts must be a JSON array")
        artifacts = tuple(raw_artifacts)
        for spec in artifacts:
            if not isinstance(spec, ArtifactSpec):
                raise ArtifactContractError(
                    "manifest artifacts must be ArtifactSpec instances"
                )
        object.__setattr__(self, "artifacts", artifacts)
        raw_validations = () if self.validations is None else self.validations
        if not isinstance(raw_validations, (list, tuple)):
            raise ArtifactContractError("manifest validations must be a JSON array")
        validations = tuple(raw_validations)
        for validation in validations:
            if not isinstance(validation, ArtifactValidation):
                raise ArtifactContractError(
                    "manifest validations must be ArtifactValidation instances"
                )
        object.__setattr__(self, "validations", validations)

        # ---- invariants ------------------------------------------------ #
        ids: set[str] = set()
        for spec in artifacts:
            if spec.id in ids:
                raise ArtifactContractError(f"duplicate artifact id: {spec.id!r}")
            ids.add(spec.id)

        # Paths must be unique after canonical normalization; spellings that
        # differ only by case are an explicit conflict, never a silent merge.
        by_folded: dict[str, str] = {}
        for spec in artifacts:
            folded = spec.path.casefold()
            other = by_folded.get(folded)
            if other is not None:
                if other == spec.path:
                    raise ArtifactContractError(
                        f"duplicate artifact path: {spec.path!r}"
                    )
                raise ArtifactContractError(
                    f"case-conflicting artifact paths: {other!r} vs {spec.path!r}"
                )
            by_folded[folded] = spec.path

        for spec in artifacts:
            for dep in spec.depends_on:
                if dep not in ids:
                    raise ArtifactContractError(
                        f"artifact {spec.id!r} depends on unknown artifact {dep!r}"
                    )
        _assert_acyclic({s.id: s.depends_on for s in artifacts}, "artifact")

        validation_ids: set[str] = set()
        produced_results: dict[str, str] = {}
        artifacts_by_id = {spec.id: spec for spec in artifacts}
        for validation in validations:
            if validation.id in validation_ids:
                raise ArtifactContractError(
                    f"duplicate validation id: {validation.id!r}"
                )
            validation_ids.add(validation.id)
            for target in validation.target_artifact_ids:
                if target not in ids:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} targets unknown artifact "
                        f"{target!r}"
                    )
            if validation.produces_artifact_id and (
                validation.produces_artifact_id not in ids
            ):
                raise ArtifactContractError(
                    f"validation {validation.id!r} produces unknown artifact "
                    f"{validation.produces_artifact_id!r}"
                )
            if validation.produces_artifact_id:
                result_id = validation.produces_artifact_id
                result = artifacts_by_id[result_id]
                if result_id in validation.target_artifact_ids:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} cannot target the same "
                        f"result artifact {result_id!r} that it produces"
                    )
                if result.kind not in _RESULT_ARTIFACT_KINDS:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} produces artifact "
                        f"{result_id!r}, which is not a validation-result kind"
                    )
                expected_kind = _VALIDATION_RESULT_KIND[validation.kind]
                if result.kind is not expected_kind:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} of kind "
                        f"{validation.kind.value!r} must produce "
                        f"{expected_kind.value!r}, not {result.kind.value!r}"
                    )
                previous = produced_results.get(result_id)
                if previous is not None:
                    raise ArtifactContractError(
                        f"result artifact {result_id!r} is produced by more than "
                        f"one validation: {previous!r} and {validation.id!r}"
                    )
                if result.external:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} cannot produce external "
                        f"result artifact {result_id!r}"
                    )
                if result.required and not validation.required:
                    raise ArtifactContractError(
                        f"required result artifact {result_id!r} is produced by "
                        f"optional validation {validation.id!r}"
                    )
                produced_results[result_id] = validation.id

        missing_result_producers = sorted(
            spec.id
            for spec in artifacts
            if spec.kind in _RESULT_ARTIFACT_KINDS
            and spec.required
            and not spec.external
            and spec.id not in produced_results
        )
        if missing_result_producers:
            raise ArtifactContractError(
                "required validation-result artifacts have no producing "
                f"validation: {', '.join(missing_result_producers)}"
            )

    # ------------------------------ queries ---------------------------- #
    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(spec.id for spec in self.artifacts)

    def by_id(self) -> dict[str, ArtifactSpec]:
        return {spec.id: spec for spec in self.artifacts}

    def required_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(spec for spec in self.artifacts if spec.required)

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "artifacts": [spec.to_dict() for spec in self.artifacts],
            "validations": [v.to_dict() for v in self.validations],
            "completion_criteria": list(self.completion_criteria),
            "schema_version": self.schema_version,
            "source_request": self.source_request,
            "project_root": self.project_root,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
        if not isinstance(data, Mapping):
            raise ArtifactContractError("manifest must be a JSON object")
        raw_artifacts = _json_array(data.get("artifacts"), "manifest artifacts")
        raw_validations = _json_array(
            data.get("validations"), "manifest validations"
        )
        return cls(
            id=str(data.get("id") or ""),
            title=data.get("title") or "",
            description=data.get("description") or "",
            artifacts=tuple(ArtifactSpec.from_dict(item) for item in raw_artifacts),
            validations=tuple(
                ArtifactValidation.from_dict(item) for item in raw_validations
            ),
            completion_criteria=_json_array(
                data.get("completion_criteria"),
                "manifest completion_criteria",
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            source_request=data.get("source_request") or "",
            project_root=data.get("project_root") or "",
            metadata=data.get("metadata"),
        )

    # ------------------------------- prompts --------------------------- #
    def to_brief(self, max_artifacts: int = _BRIEF_MAX_ARTIFACTS) -> str:
        """Bounded, newline-sanitized block for prompts. Never file contents."""
        lines = [f"ARTIFACT MANIFEST: {_clip(self.title or self.id, _BRIEF_LINE_CHARS)}"]
        if self.description:
            lines.append(f"- purpose: {_clip(self.description, _BRIEF_LINE_CHARS)}")
        shown = self.artifacts[: max(0, max_artifacts)]
        for spec in shown:
            status = "required" if spec.required else "optional"
            lines.append(
                f"- {status} {spec.kind.value}: {_clip(spec.path, 120)} "
                f"[{spec.operation.value}]"
            )
        hidden = len(self.artifacts) - len(shown)
        if hidden > 0:
            lines.append(f"- … (+{hidden} more artifacts retained in the manifest)")
        for validation in self.validations[:_BRIEF_MAX_VALIDATIONS]:
            need = "required" if validation.required else "optional"
            detail = validation.success_criterion or validation.command
            suffix = f" — {_clip(detail, 100)}" if detail else ""
            lines.append(f"- validation ({need}): {validation.kind.value}{suffix}")
        hidden_validations = len(self.validations) - _BRIEF_MAX_VALIDATIONS
        if hidden_validations > 0:
            lines.append(f"- … (+{hidden_validations} more validations retained)")
        if self.completion_criteria:
            joined = "; ".join(
                _clip(c, 100) for c in self.completion_criteria[:4]
            )
            if len(self.completion_criteria) > 4:
                joined += f"; … (+{len(self.completion_criteria) - 4} more)"
            lines.append(f"- done when: {joined}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Work package
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkPackage:
    """A bounded unit of planned artifact work. Planning only — no scheduling."""

    id: str
    title: str
    objective: str = ""
    kind: WorkPackageKind = WorkPackageKind.IMPLEMENTATION
    # Artifact ids this package produces/owns (at most one owner per artifact).
    owns: tuple[str, ...] = ()
    # Package ids that must complete before this one.
    depends_on: tuple[str, ...] = ()
    # Artifact ids consumed as inputs (owned elsewhere, or external).
    input_artifact_ids: tuple[str, ...] = ()
    # Artifact ids this package is expected to emit; must be a subset of owns.
    output_artifact_ids: tuple[str, ...] = ()
    # Manifest validation ids this package is responsible for satisfying.
    validation_ids: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    # Bounded free label (e.g. "small", "≤4 files"): a boundedness signal only.
    estimated_size: str = ""
    # Optional relevant request mode (e.g. "implement"), never a provider name.
    contract_intent: str = ""
    # Optional capability hints matching SubTask.required_capabilities style.
    capability_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ident = _clip(self.id, _MAX_TEXT_CHARS)
        if not ident:
            raise ArtifactContractError("work-package id must not be empty")
        object.__setattr__(self, "id", ident)
        object.__setattr__(self, "title", _clip(self.title, _MAX_TEXT_CHARS))
        object.__setattr__(self, "objective", _clip(self.objective, _MAX_TEXT_CHARS))
        object.__setattr__(
            self, "kind", _coerce_enum(self.kind, WorkPackageKind, _PACKAGE_FALLBACK)
        )
        for name in (
            "owns",
            "depends_on",
            "input_artifact_ids",
            "output_artifact_ids",
            "validation_ids",
        ):
            object.__setattr__(
                self, name,
                _str_tuple(getattr(self, name), clip=_MAX_TEXT_CHARS),
            )
        object.__setattr__(
            self,
            "capability_hints",
            _str_tuple(
                self.capability_hints, clip=_MAX_CAPABILITY_HINT_CHARS
            )[:_MAX_CAPABILITY_HINTS],
        )
        object.__setattr__(
            self, "completion_criteria", _criteria_tuple(self.completion_criteria)
        )
        object.__setattr__(self, "estimated_size", _clip(self.estimated_size, 60))
        object.__setattr__(self, "contract_intent", _clip(self.contract_intent, 40))

        if self.id in self.depends_on:
            raise ArtifactContractError(f"package {self.id!r} depends on itself")
        seen: set[str] = set()
        for artifact_id in self.owns:
            if artifact_id in seen:
                raise ArtifactContractError(
                    f"package {self.id!r} lists artifact {artifact_id!r} twice"
                )
            seen.add(artifact_id)
        for output in self.output_artifact_ids:
            if output not in self.owns:
                raise ArtifactContractError(
                    f"package {self.id!r} declares output {output!r} it does not own"
                )

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "objective": self.objective,
            "kind": self.kind.value,
            "owns": list(self.owns),
            "depends_on": list(self.depends_on),
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "validation_ids": list(self.validation_ids),
            "completion_criteria": list(self.completion_criteria),
            "estimated_size": self.estimated_size,
            "contract_intent": self.contract_intent,
            "capability_hints": list(self.capability_hints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkPackage":
        if not isinstance(data, Mapping):
            raise ArtifactContractError("work package must be a JSON object")
        return cls(
            id=str(data.get("id") or ""),
            title=data.get("title") or "",
            objective=data.get("objective") or "",
            kind=(
                WorkPackageKind.IMPLEMENTATION
                if data.get("kind") is None
                else _coerce_enum(
                    data.get("kind"), WorkPackageKind, _PACKAGE_FALLBACK
                )
            ),
            owns=_json_array(data.get("owns"), "work package owns"),
            depends_on=_json_array(
                data.get("depends_on"), "work package depends_on"
            ),
            input_artifact_ids=_json_array(
                data.get("input_artifact_ids"),
                "work package input_artifact_ids",
            ),
            output_artifact_ids=_json_array(
                data.get("output_artifact_ids"),
                "work package output_artifact_ids",
            ),
            validation_ids=_json_array(
                data.get("validation_ids"), "work package validation_ids"
            ),
            completion_criteria=_json_array(
                data.get("completion_criteria"),
                "work package completion_criteria",
            ),
            estimated_size=data.get("estimated_size") or "",
            contract_intent=data.get("contract_intent") or "",
            capability_hints=_json_array(
                data.get("capability_hints"),
                "work package capability_hints",
            ),
        )

    # ------------------------------- prompts --------------------------- #
    def to_brief(self) -> str:
        """Bounded, newline-sanitized block for prompts."""
        lines = [f"WORK PACKAGE: {self.id}", f"- kind: {self.kind.value}"]
        if self.title and self.title != self.id:
            lines.insert(1, f"- title: {_clip(self.title, _BRIEF_LINE_CHARS)}")
        if self.objective:
            lines.append(f"- objective: {_clip(self.objective, _BRIEF_LINE_CHARS)}")
        if self.owns:
            lines.append(f"- owns: {_ids_line(self.owns)}")
        if self.depends_on:
            lines.append(f"- depends on: {_ids_line(self.depends_on)}")
        if self.input_artifact_ids:
            lines.append(f"- inputs: {_ids_line(self.input_artifact_ids)}")
        if self.completion_criteria:
            lines.append(f"- completion: {_clip('; '.join(self.completion_criteria[:3]), _BRIEF_LINE_CHARS)}")
        return "\n".join(lines)


def _ids_line(ids: tuple[str, ...]) -> str:
    shown = [_clip(item, 60) for item in ids[:_BRIEF_MAX_IDS_PER_LINE]]
    line = ", ".join(shown)
    if len(ids) > _BRIEF_MAX_IDS_PER_LINE:
        line += f" (+{len(ids) - _BRIEF_MAX_IDS_PER_LINE} more)"
    return line


# --------------------------------------------------------------------------- #
# Artifact-oriented work plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactWorkPlan:
    """One manifest plus the bounded work packages that produce/validate it.

    This is a typed CONTRACT future phases consume — not a second execution
    engine. It attaches to the existing ``Plan`` as an optional field; plans
    without one remain exactly as valid as before Phase 4A.
    """

    id: str
    manifest: ArtifactManifest
    packages: tuple[WorkPackage, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    # Optional mapping of package id -> existing SubTask id (one pair each).
    subtask_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ident = _clip(self.id, _MAX_TEXT_CHARS)
        if not ident:
            raise ArtifactContractError("artifact plan id must not be empty")
        object.__setattr__(self, "id", ident)
        if not isinstance(self.manifest, ArtifactManifest):
            raise ArtifactContractError("artifact plan requires an ArtifactManifest")
        raw_packages = () if self.packages is None else self.packages
        if not isinstance(raw_packages, (list, tuple)):
            raise ArtifactContractError("artifact plan packages must be a JSON array")
        packages = tuple(raw_packages)
        for package in packages:
            if not isinstance(package, WorkPackage):
                raise ArtifactContractError(
                    "artifact plan packages must be WorkPackage instances"
                )
        object.__setattr__(self, "packages", packages)
        object.__setattr__(
            self, "completion_criteria", _criteria_tuple(self.completion_criteria)
        )
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        raw_pairs = () if self.subtask_map is None else self.subtask_map
        if not isinstance(raw_pairs, (list, tuple)):
            raise ArtifactContractError("subtask_map must be a JSON array of pairs")
        normalized_pairs: list[tuple[str, str]] = []
        for item in raw_pairs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ArtifactContractError("malformed subtask_map entry")
            package_id, subtask_id = item
            if not isinstance(package_id, str) or not isinstance(subtask_id, str):
                raise ArtifactContractError(
                    "subtask_map package and subtask ids must be strings"
                )
            normalized_pairs.append((
                _clip(package_id, _MAX_TEXT_CHARS),
                _clip(subtask_id, _MAX_TEXT_CHARS),
            ))
        pairs = tuple(normalized_pairs)
        object.__setattr__(self, "subtask_map", pairs)

        # ---- invariants ------------------------------------------------ #
        package_ids: set[str] = set()
        for package in packages:
            if package.id in package_ids:
                raise ArtifactContractError(f"duplicate package id: {package.id!r}")
            package_ids.add(package.id)

        artifact_ids = set(self.manifest.artifact_ids())
        artifacts_by_id = self.manifest.by_id()
        packages_by_id = {package.id: package for package in packages}
        owner: dict[str, str] = {}
        for package in packages:
            for dep in package.depends_on:
                if dep not in package_ids:
                    raise ArtifactContractError(
                        f"package {package.id!r} depends on unknown package {dep!r}"
                    )
            for artifact_id in package.owns:
                if artifact_id not in artifact_ids:
                    raise ArtifactContractError(
                        f"package {package.id!r} claims artifact {artifact_id!r} "
                        "which is not in the manifest"
                    )
                if artifacts_by_id[artifact_id].external:
                    raise ArtifactContractError(
                        f"package {package.id!r} cannot own external artifact "
                        f"{artifact_id!r}; external artifacts are supplied "
                        "inputs, not package-produced outputs"
                    )
                if artifact_id in owner:
                    raise ArtifactContractError(
                        f"artifact {artifact_id!r} is owned by more than one "
                        f"package: {owner[artifact_id]!r} and {package.id!r}"
                    )
                owner[artifact_id] = package.id
            for artifact_id in package.input_artifact_ids:
                if artifact_id not in artifact_ids:
                    raise ArtifactContractError(
                        f"package {package.id!r} consumes unknown artifact "
                        f"{artifact_id!r}"
                    )
        for package in packages:
            for artifact_id in package.input_artifact_ids:
                spec = artifacts_by_id[artifact_id]
                if not spec.external and artifact_id not in owner:
                    raise ArtifactContractError(
                        f"package {package.id!r} consumes non-external artifact "
                        f"{artifact_id!r} with no owning work package"
                    )
            for owned_id in package.owns:
                for dependency_id in artifacts_by_id[owned_id].depends_on:
                    dependency = artifacts_by_id[dependency_id]
                    if not dependency.external and dependency_id not in owner:
                        raise ArtifactContractError(
                            f"package {package.id!r} owns artifact {owned_id!r}, "
                            f"which depends on non-external artifact "
                            f"{dependency_id!r} with no owning work package"
                        )
        _assert_acyclic({p.id: p.depends_on for p in packages}, "package")

        # A package that consumes another package's artifact must declare that
        # owner upstream (directly or transitively). The same rule links the
        # artifact dependency DAG to the package dependency DAG.
        upstream: dict[str, set[str]] = {}
        for package in packages:
            reachable: set[str] = set()
            pending = list(package.depends_on)
            while pending:
                dependency = pending.pop()
                if dependency in reachable:
                    continue
                reachable.add(dependency)
                pending.extend(packages_by_id[dependency].depends_on)
            upstream[package.id] = reachable
        for package in packages:
            required_upstream: set[str] = set()
            for artifact_id in package.input_artifact_ids:
                artifact_owner = owner.get(artifact_id)
                if artifact_owner and artifact_owner != package.id:
                    required_upstream.add(artifact_owner)
            for artifact_id in package.owns:
                for dependency_id in artifacts_by_id[artifact_id].depends_on:
                    dependency_owner = owner.get(dependency_id)
                    if dependency_owner and dependency_owner != package.id:
                        required_upstream.add(dependency_owner)
            missing = sorted(required_upstream - upstream[package.id])
            if missing:
                raise ArtifactContractError(
                    f"package {package.id!r} consumes artifacts owned by "
                    "packages that are not declared dependencies: "
                    f"{', '.join(missing)}"
                )

        # A spec's declared owner (when assigned) must agree with the packages.
        for spec in self.manifest.artifacts:
            if spec.package_id and owner.get(spec.id) != spec.package_id:
                raise ArtifactContractError(
                    f"artifact {spec.id!r} declares owner {spec.package_id!r}, "
                    f"but package ownership says {owner.get(spec.id)!r}"
                )

        # Every required, non-external artifact needs exactly one producer.
        orphans = sorted(
            spec.id
            for spec in self.manifest.artifacts
            if spec.required and not spec.external and spec.id not in owner
        )
        if orphans:
            raise ArtifactContractError(
                "required artifacts have no owning work package (mark them "
                f"external if supplied from outside the plan): {', '.join(orphans)}"
            )

        produced_result_ids = {
            validation.produces_artifact_id
            for validation in self.manifest.validations
            if validation.produces_artifact_id
        }
        owned_unproduced_results = sorted(
            spec.id
            for spec in self.manifest.artifacts
            if spec.kind in _RESULT_ARTIFACT_KINDS
            and not spec.external
            and spec.id in owner
            and spec.id not in produced_result_ids
        )
        if owned_unproduced_results:
            raise ArtifactContractError(
                "owned validation-result artifacts have no producing "
                f"validation: {', '.join(owned_unproduced_results)}"
            )

        known_validation_ids = {v.id for v in self.manifest.validations}
        for package in packages:
            for validation_id in package.validation_ids:
                if validation_id not in known_validation_ids:
                    raise ArtifactContractError(
                        f"package {package.id!r} references unknown validation "
                        f"{validation_id!r}"
                    )
        for validation in self.manifest.validations:
            if validation.target_package_id and (
                validation.target_package_id not in package_ids
            ):
                raise ArtifactContractError(
                    f"validation {validation.id!r} targets unknown package "
                    f"{validation.target_package_id!r}"
                )
            responsible = sorted(
                package.id
                for package in packages
                if validation.id in package.validation_ids
            )
            if validation.produces_artifact_id:
                result_id = validation.produces_artifact_id
                result_owner = owner.get(result_id, "")
                if not result_owner:
                    raise ArtifactContractError(
                        f"validation {validation.id!r} produces result artifact "
                        f"{result_id!r} with no owning work package"
                    )
                if responsible != [result_owner]:
                    rendered = ", ".join(responsible) or "none"
                    raise ArtifactContractError(
                        f"validation {validation.id!r} produces result artifact "
                        f"{result_id!r}; exactly its owner {result_owner!r} must "
                        f"claim that validation (claimed by: {rendered})"
                    )
            # A package cannot validate another package's output before that
            # producer is upstream. Validation packages may inspect artifacts
            # they do not own, but their dependency graph must declare it.
            validation_upstream: set[str] = set()
            if validation.target_package_id:
                validation_upstream.add(validation.target_package_id)
            for artifact_id in validation.target_artifact_ids:
                artifact_owner = owner.get(artifact_id)
                if artifact_owner:
                    validation_upstream.add(artifact_owner)
            for package_id in responsible:
                needed = validation_upstream - {package_id}
                missing = sorted(needed - upstream[package_id])
                if missing:
                    raise ArtifactContractError(
                        f"package {package_id!r} validates outputs of packages "
                        "that are not declared dependencies: "
                        f"{', '.join(missing)}"
                    )

        mapped: set[str] = set()
        for package_id, subtask_id in pairs:
            if package_id not in package_ids:
                raise ArtifactContractError(
                    f"subtask map references unknown package {package_id!r}"
                )
            if not subtask_id:
                raise ArtifactContractError(
                    f"subtask map entry for package {package_id!r} has no subtask id"
                )
            if package_id in mapped:
                raise ArtifactContractError(
                    f"package {package_id!r} is mapped to more than one subtask"
                )
            mapped.add(package_id)

    # ------------------------------ queries ---------------------------- #
    def owner_of(self, artifact_id: str) -> str:
        """Deterministic owning package id for an artifact ('' when unowned)."""
        for package in self.packages:
            if artifact_id in package.owns:
                return package.id
        return ""

    def subtask_for(self, package_id: str) -> str:
        for pid, subtask_id in self.subtask_map:
            if pid == package_id:
                return subtask_id
        return ""

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "manifest": self.manifest.to_dict(),
            "packages": [p.to_dict() for p in self.packages],
            "completion_criteria": list(self.completion_criteria),
            "schema_version": self.schema_version,
            "subtask_map": [list(pair) for pair in self.subtask_map],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactWorkPlan":
        if not isinstance(data, Mapping):
            raise ArtifactContractError("artifact plan must be a JSON object")
        manifest_data = data.get("manifest")
        if not isinstance(manifest_data, Mapping):
            raise ArtifactContractError("artifact plan is missing its manifest")
        raw_map = _json_array(data.get("subtask_map"), "subtask_map")
        pairs = []
        for item in raw_map:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ArtifactContractError("malformed subtask_map entry")
            package_id, subtask_id = item
            pairs.append((package_id, subtask_id))
        raw_packages = _json_array(data.get("packages"), "artifact plan packages")
        return cls(
            id=str(data.get("id") or ""),
            manifest=ArtifactManifest.from_dict(manifest_data),
            packages=tuple(
                WorkPackage.from_dict(item) for item in raw_packages
            ),
            completion_criteria=_json_array(
                data.get("completion_criteria"),
                "artifact plan completion_criteria",
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            subtask_map=tuple(pairs),
        )

    # ------------------------------- prompts --------------------------- #
    def to_brief(self, max_packages: int = _BRIEF_MAX_PACKAGES) -> str:
        """Bounded manifest + package summary for prompts. Deterministic."""
        lines = [self.manifest.to_brief(), "WORK PACKAGES:"]
        shown = self.packages[: max(0, max_packages)]
        for package in shown:
            parts = [f"- {package.id} [{package.kind.value}]"]
            if package.owns:
                parts.append(f"owns: {_ids_line(package.owns)}")
            if package.depends_on:
                parts.append(f"needs: {_ids_line(package.depends_on)}")
            lines.append(_clip(" — ".join(parts), _BRIEF_LINE_CHARS * 2))
        hidden = len(self.packages) - len(shown)
        if hidden > 0:
            lines.append(f"- … (+{hidden} more packages retained in the plan)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Planner-output parsing
# --------------------------------------------------------------------------- #
# Bounds on what a single planner reply may declare. These protect prompt and
# parser budgets; the in-memory models themselves are not the limiting factor.
_PLANNER_MAX_ARTIFACTS = _MAX_ARTIFACT_REFERENCES
_PLANNER_MAX_PACKAGES = _MAX_PACKAGE_REFERENCES
_PLANNER_MAX_VALIDATIONS = _MAX_VALIDATION_REFERENCES

# The planner must NEVER put produced file contents in the contract. Any of
# these keys carrying a non-empty value is rejected outright (one corrective
# re-plan teaches the planner; a second offense fails the plan).
_CONTENT_KEY_TOKENS = frozenset({
    "content", "contents", "filecontent", "filecontents", "body",
    "source", "sourcecode", "code", "data", "text",
})


def _first_present(data: Mapping[str, Any], *keys: str) -> Any:
    """Return the first explicitly present alias, preserving falsey values."""
    for key in keys:
        if key in data:
            return data[key]
    return None


def _find_content_field(value: Any) -> str:
    """Find an obvious content-carrying key anywhere in planner JSON."""
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            for key, child in current.items():
                token = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if token in _CONTENT_KEY_TOKENS:
                    if isinstance(child, str):
                        nonempty = bool(child.strip())
                    elif isinstance(child, (Mapping, list, tuple)):
                        nonempty = bool(child)
                    else:
                        nonempty = child not in (None, False, 0)
                    if nonempty:
                        return _clip(key, 60)
                if isinstance(child, (Mapping, list, tuple)):
                    pending.append(child)
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            pending.extend(current)
    return ""


def parse_planner_artifact_plan(
    raw: Any,
    subtask_id_map: Optional[Mapping[str, str]] = None,
) -> ArtifactWorkPlan:
    """Parse the planner's optional ``artifact_plan`` block deterministically.

    ``subtask_id_map`` maps planner-visible subtask ids (``"s1"``) to internal
    ``SubTask`` ids; package ``subtask_id`` references are remapped through it
    and unknown references fail. Raises :class:`ArtifactContractError` with a
    specific, feedback-ready message on any structural problem.
    """
    if not isinstance(raw, Mapping):
        raise ArtifactContractError("artifact_plan must be a JSON object")

    content_field = _find_content_field(raw)
    if content_field:
        raise ArtifactContractError(
            "artifact_plan must not carry file contents "
            f"(found a non-empty {content_field!r} field); declare paths and "
            "short descriptions only"
        )

    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)) or not raw_artifacts:
        raise ArtifactContractError(
            "artifact_plan must declare a non-empty 'artifacts' list"
        )
    if len(raw_artifacts) > _PLANNER_MAX_ARTIFACTS:
        raise ArtifactContractError(
            f"artifact_plan declares {len(raw_artifacts)} artifacts; the limit "
            f"is {_PLANNER_MAX_ARTIFACTS}"
        )
    raw_packages = raw.get("packages")
    if raw_packages is None:
        raw_packages = []
    if not isinstance(raw_packages, (list, tuple)):
        raise ArtifactContractError("artifact_plan 'packages' must be a list")
    if len(raw_packages) > _PLANNER_MAX_PACKAGES:
        raise ArtifactContractError(
            f"artifact_plan declares {len(raw_packages)} packages; the limit "
            f"is {_PLANNER_MAX_PACKAGES}"
        )
    raw_validations = raw.get("validations")
    if raw_validations is None:
        raw_validations = []
    if not isinstance(raw_validations, (list, tuple)):
        raise ArtifactContractError("artifact_plan 'validations' must be a list")
    if len(raw_validations) > _PLANNER_MAX_VALIDATIONS:
        raise ArtifactContractError(
            f"artifact_plan declares {len(raw_validations)} validations; the "
            f"limit is {_PLANNER_MAX_VALIDATIONS}"
        )

    # Ownership is read from the packages' 'owns' lists; a spec's own owner
    # field is then assigned from it so both views agree by construction.
    claims: dict[str, str] = {}
    for entry in raw_packages:
        if not isinstance(entry, Mapping):
            raise ArtifactContractError("each package must be a JSON object")
        package_id = str(entry.get("id") or "").strip()
        owns = _json_array(
            entry.get("owns"),
            "package owns",
            max_items=_MAX_ARTIFACT_REFERENCES,
        )
        for artifact_id in _str_tuple(owns):
            claims.setdefault(artifact_id, package_id)

    specs: list[ArtifactSpec] = []
    for entry in raw_artifacts:
        if not isinstance(entry, Mapping):
            raise ArtifactContractError("each artifact must be a JSON object")
        path_value = entry.get("path") or entry.get("name")
        artifact_id = str(entry.get("id") or "").strip()
        if not artifact_id:
            artifact_id = canonical_artifact_path(path_value)
        specs.append(
            ArtifactSpec(
                id=artifact_id,
                path=path_value,
                kind=_coerce_enum(entry.get("kind"), ArtifactKind, _KIND_FALLBACK),
                required=_bool(entry.get("required"), True),
                operation=_coerce_enum(
                    entry.get("operation"), ArtifactOperation, _OPERATION_FALLBACK
                ),
                description=entry.get("description") or "",
                package_id=claims.get(artifact_id, ""),
                depends_on=_json_array(
                    entry.get("depends_on"),
                    "artifact depends_on",
                    max_items=_MAX_ARTIFACT_REFERENCES,
                ),
                language=entry.get("language") or "",
                media_type=entry.get("media_type") or "",
                external=_bool(entry.get("external"), False),
                completion_criteria=_json_array(
                    entry.get("completion_criteria"),
                    "artifact completion_criteria",
                    max_items=_MAX_CRITERIA,
                ),
            )
        )

    validations: list[ArtifactValidation] = []
    for index, entry in enumerate(raw_validations, 1):
        if not isinstance(entry, Mapping):
            raise ArtifactContractError("each validation must be a JSON object")
        validations.append(
            ArtifactValidation(
                id=str(entry.get("id") or "").strip() or f"v{index}",
                kind=_coerce_enum(
                    entry.get("kind"), ValidationKind, _VALIDATION_FALLBACK
                ),
                required=_bool(entry.get("required"), True),
                target_artifact_ids=_json_array(
                    _first_present(entry, "targets", "target_artifact_ids"),
                    "validation targets",
                    max_items=_MAX_ARTIFACT_REFERENCES,
                ),
                target_package_id=entry.get("package")
                or entry.get("target_package_id")
                or "",
                command=entry.get("command") or "",
                success_criterion=entry.get("criterion")
                or entry.get("success_criterion")
                or "",
                produces_artifact_id=entry.get("produces")
                or entry.get("produces_artifact_id")
                or "",
            )
        )

    packages: list[WorkPackage] = []
    subtask_pairs: list[tuple[str, str]] = []
    for entry in raw_packages:
        package_id = str(entry.get("id") or "").strip()
        if not package_id:
            raise ArtifactContractError("every package needs a non-empty 'id'")
        packages.append(
            WorkPackage(
                id=package_id,
                title=entry.get("title") or package_id,
                objective=entry.get("objective") or "",
                kind=(
                    WorkPackageKind.IMPLEMENTATION
                    if entry.get("kind") is None
                    else _coerce_enum(
                        entry.get("kind"), WorkPackageKind, _PACKAGE_FALLBACK
                    )
                ),
                owns=_json_array(
                    entry.get("owns"),
                    "package owns",
                    max_items=_MAX_ARTIFACT_REFERENCES,
                ),
                depends_on=_json_array(
                    entry.get("depends_on"),
                    "package depends_on",
                    max_items=_MAX_PACKAGE_REFERENCES,
                ),
                input_artifact_ids=_json_array(
                    _first_present(entry, "inputs", "input_artifact_ids"),
                    "package inputs",
                    max_items=_MAX_ARTIFACT_REFERENCES,
                ),
                output_artifact_ids=_json_array(
                    _first_present(entry, "outputs", "output_artifact_ids"),
                    "package outputs",
                    max_items=_MAX_ARTIFACT_REFERENCES,
                ),
                validation_ids=_json_array(
                    _first_present(entry, "validations", "validation_ids"),
                    "package validations",
                    max_items=_MAX_VALIDATION_REFERENCES,
                ),
                completion_criteria=_json_array(
                    _first_present(
                        entry, "completion_criteria", "completion"
                    ),
                    "package completion_criteria",
                    max_items=_MAX_CRITERIA,
                ),
                estimated_size=entry.get("estimated_size") or "",
                capability_hints=_json_array(
                    _first_present(
                        entry, "required_capabilities", "capability_hints"
                    ),
                    "package required_capabilities",
                    max_items=_MAX_CAPABILITY_HINTS,
                ),
            )
        )
        raw_subtask_ref = entry.get("subtask_id")
        if raw_subtask_ref is not None and not isinstance(raw_subtask_ref, str):
            raise ArtifactContractError("package subtask_id must be a string")
        subtask_ref = (raw_subtask_ref or "").strip()
        if subtask_ref:
            if subtask_id_map is None:
                subtask_pairs.append((package_id, subtask_ref))
            else:
                internal = subtask_id_map.get(subtask_ref)
                if internal is None:
                    raise ArtifactContractError(
                        f"package {package_id!r} maps to unknown subtask "
                        f"{subtask_ref!r}"
                    )
                subtask_pairs.append((package_id, internal))

    completion_criteria = _json_array(
        raw.get("completion_criteria"),
        "artifact_plan completion_criteria",
        max_items=_MAX_CRITERIA,
    )
    title = str(raw.get("title") or "artifact deliverable").strip()
    manifest_id = _slug(title, "artifact-manifest")
    manifest = ArtifactManifest(
        id=manifest_id,
        title=title,
        description=raw.get("description") or "",
        artifacts=tuple(specs),
        validations=tuple(validations),
        completion_criteria=completion_criteria,
    )
    return ArtifactWorkPlan(
        id=f"{manifest_id}-plan",
        manifest=manifest,
        packages=tuple(packages),
        completion_criteria=completion_criteria,
        subtask_map=tuple(subtask_pairs),
    )
