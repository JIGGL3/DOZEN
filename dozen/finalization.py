"""Deterministic ordered finalization (production-hardening Phase 4F).

WHY THIS EXISTS
---------------
Live testing showed the exact failure this repository was built to prevent: a
"give code only" request whose final answer contained raw JSON envelopes,
worker refusals, duplicated headings and no runnable project — because model
synthesis was still a REQUIRED delivery stage on some execution paths. This
module makes the DEFAULT finalization deterministic:

    validated worker results
      -> trusted plan/manifest order
      -> deterministic rendering

An LLM synthesizer survives only as an OPTIONAL presentation enhancement for
suitable prose workflows; it is never required to deliver the user's result.

Ownership boundaries (unchanged): intent owns the deliverable type, the
manifest owns expected files, assembly owns project combination, presentation
remains the final user-visible rendering boundary. This module owns HOW the
validated results are selected and stitched — pure, deterministic, no I/O and
no provider call anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

from .artifact_results import _RESULT_KINDS, _fence, AssembledDeliverable
from .presentation import sanitize_diagnostic
from .synthesis_guard import MODEL_BASED_SYNTHESIS_ENABLED
from .validation import (
    decode_nested_envelope,
    is_canonical_protocol_data,
    is_malformed_protocol_envelope,
    looks_like_orchestration_json,
    looks_like_protocol_envelope,
)

FINALIZATION_SCHEMA_VERSION = 1
PROSE_ENVELOPE_SCHEMA_VERSION = 1
ORDERED_SECTION_SCHEMA_VERSION = 1

# The synthesizer's legacy all-failed sentinel. Reproduced here byte-identically
# so the deterministic stitcher keeps the exact historical semantics every
# presentation-layer consumer already handles.
NO_USABLE_RESULT_SENTINEL = "[No subtasks produced a usable result]"

# Part F: the clean deterministic non-delivery result for a failed code-only
# project. Deliberately free of worker text, refusals and internal JSON.
CODE_ONLY_FAILURE_HEADLINE = "The requested code project could not be completed."
CODE_ONLY_FAILURE_FOOTER = "No incomplete or unverified project was delivered."


class FinalizationError(ValueError):
    """A structural violation of the finalization contracts."""


# --------------------------------------------------------------------------- #
# Part A — typed finalization modes
# --------------------------------------------------------------------------- #
class FinalizationMode(str, Enum):
    """How validated results become the user-visible answer."""

    ARTIFACT_ASSEMBLY = "artifact_assembly"
    ORDERED_SECTIONS = "ordered_sections"
    OPTIONAL_MODEL_SYNTHESIS = "optional_model_synthesis"


# Part H — typed non-delivery classification for scoped artifact workers.
class WorkerDeliveryCode(str, Enum):
    """What a scoped artifact worker actually delivered against its scope."""

    DELIVERED = "delivered"
    MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"
    ARTIFACT_NON_DELIVERY = "artifact_non_delivery"
    RESPONSE_SIZE_REFUSAL = "response_size_refusal"
    EXPLANATION_ONLY_RESPONSE = "explanation_only_response"


# Codes that prove the worker delivered NOTHING it owed; such an attempt can
# never count as a completed subtask (a refusal is not a prose result).
FATAL_DELIVERY_CODES = frozenset({
    WorkerDeliveryCode.ARTIFACT_NON_DELIVERY,
    WorkerDeliveryCode.RESPONSE_SIZE_REFUSAL,
    WorkerDeliveryCode.EXPLANATION_ONLY_RESPONSE,
})


# --------------------------------------------------------------------------- #
# Part B — the ONE authoritative finalization policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FinalizationPolicy:
    """Immutable, authoritative finalization defaults and bounds.

    These decisions live HERE, not scattered across orchestrator, synthesizer,
    server and frontend.
    """

    # Default modes per request family.
    artifact_default_mode: FinalizationMode = FinalizationMode.ARTIFACT_ASSEMBLY
    prose_default_mode: FinalizationMode = FinalizationMode.ORDERED_SECTIONS
    # Optional model polish is DISABLED unless the user explicitly requests a
    # polished unified narrative or trusted configuration enables it.
    allow_model_polish: bool = False
    # Ordered-section bounds.
    max_sections: int = 48
    max_section_title_chars: int = 120
    max_section_diagnostic_chars: int = 300
    max_section_warnings: int = 8
    max_section_evidence_refs: int = 8
    # Failed subtasks are rendered as bounded readable notes, never dropped
    # silently and never as raw payloads.
    render_failed_sections: bool = True
    # A required code-only project is never delivered partially.
    allow_partial_artifact_delivery: bool = False
    # Code-only requests never permit explanatory prose in a successful answer.
    code_only_allows_prose: bool = False
    # A model-synthesis failure always falls back to the deterministic result.
    synthesis_failure_falls_back: bool = True
    # Bounded embedded-envelope scans per section (Part K).
    max_embedded_envelopes_per_section: int = 6
    schema_version: int = FINALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_default_mode", "prose_default_mode"):
            value = getattr(self, name)
            if not isinstance(value, FinalizationMode):
                object.__setattr__(self, name, FinalizationMode(str(value)))
        for name in (
            "allow_model_polish", "render_failed_sections",
            "allow_partial_artifact_delivery", "code_only_allows_prose",
            "synthesis_failure_falls_back",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in (
            "max_sections", "max_section_title_chars",
            "max_section_diagnostic_chars", "max_section_warnings",
            "max_section_evidence_refs",
            "max_embedded_envelopes_per_section", "schema_version",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise FinalizationError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_default_mode": self.artifact_default_mode.value,
            "prose_default_mode": self.prose_default_mode.value,
            "allow_model_polish": self.allow_model_polish,
            "max_sections": self.max_sections,
            "max_section_title_chars": self.max_section_title_chars,
            "max_section_diagnostic_chars": self.max_section_diagnostic_chars,
            "max_section_warnings": self.max_section_warnings,
            "max_section_evidence_refs": self.max_section_evidence_refs,
            "render_failed_sections": self.render_failed_sections,
            "allow_partial_artifact_delivery": self.allow_partial_artifact_delivery,
            "code_only_allows_prose": self.code_only_allows_prose,
            "synthesis_failure_falls_back": self.synthesis_failure_falls_back,
            "max_embedded_envelopes_per_section":
                self.max_embedded_envelopes_per_section,
            "schema_version": self.schema_version,
        }


DEFAULT_FINALIZATION_POLICY = FinalizationPolicy()


def _bounded_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Part A — the immutable finalization decision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FinalizationDecision:
    """Which mode was selected, from trusted runtime information only."""

    mode: FinalizationMode
    reason: str = ""
    artifacts_required: bool = False
    complete_assembly_required: bool = False
    prose_allowed: bool = True
    model_synthesis_allowed: bool = False
    deterministic_fallback_required: bool = True
    code_only: bool = False
    schema_version: int = FINALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FinalizationMode):
            object.__setattr__(self, "mode", FinalizationMode(str(self.mode)))
        object.__setattr__(self, "reason", _bounded_line(self.reason, 300))
        for name in (
            "artifacts_required", "complete_assembly_required", "prose_allowed",
            "model_synthesis_allowed", "deterministic_fallback_required",
            "code_only",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "artifacts_required": self.artifacts_required,
            "complete_assembly_required": self.complete_assembly_required,
            "prose_allowed": self.prose_allowed,
            "model_synthesis_allowed": self.model_synthesis_allowed,
            "deterministic_fallback_required":
                self.deterministic_fallback_required,
            "code_only": self.code_only,
            "schema_version": self.schema_version,
        }


# Explicit request for a polished unified narrative. Deliberately narrow: a
# request that merely says "report" or "summary" stays deterministic.
_POLISH_REQUEST_RE = re.compile(
    r"(?:\bpolished\s+(?:unified\s+)?(?:narrative|report|answer|summary|prose|"
    r"write-?up|essay|document)\b"
    r"|\bunified\s+(?:polished\s+)?narrative\b"
    r"|\bpolish\s+(?:the\s+)?(?:final\s+)?(?:answer|report|narrative|prose|"
    r"summary)\b"
    r"|\bsingle\s+polished\b"
    r"|\bas\s+one\s+polished\b)",
    re.IGNORECASE,
)


def user_requested_polish(
    prompt: str = "",
    desired_output: str = "",
    constraints: Sequence[str] = (),
) -> bool:
    """Whether the USER explicitly asked for a model-polished unified narrative."""
    surfaces = [str(prompt or ""), str(desired_output or "")]
    surfaces.extend(str(item) for item in (constraints or ()))
    return any(_POLISH_REQUEST_RE.search(surface) for surface in surfaces if surface)


_PROTOCOL_CONTENT_REQUEST_RE = re.compile(
    r"(?:\b(?:document|describe|explain|show|include|provide|write|generate|"
    r"return|create)\b[^.?!\n]{0,120}\b(?:internal\s+)?(?:result|worker|"
    r"protocol|orchestration)\s+envelope\b"
    r"|\b(?:result|worker|protocol|orchestration)\s+envelope\b"
    r"[^.?!\n]{0,120}\b(?:example|fixture|documentation|schema|format)\b)",
    re.IGNORECASE,
)
_PROTOCOL_SCHEMA_CONTEXT_RE = re.compile(
    r"\b(?:source|code|json|schema|fixture|example|documentation)\b",
    re.IGNORECASE,
)
_PROTOCOL_SCHEMA_FIELD_RE = re.compile(
    r"\b(?:summary|key_decisions|artifacts|confidence)\b",
    re.IGNORECASE,
)


def user_requested_protocol_content(
    prompt: str = "",
    desired_output: str = "",
    constraints: Sequence[str] = (),
) -> bool:
    """Narrow trusted exception for requests ABOUT the internal protocol.

    This is derived only from user-owned request surfaces. Worker output cannot
    activate it, so ordinary runs retain fail-closed envelope neutralization.
    """
    surfaces = [str(prompt or ""), str(desired_output or "")]
    surfaces.extend(str(item) for item in (constraints or ()))
    for surface in surfaces:
        if not surface:
            continue
        if _PROTOCOL_CONTENT_REQUEST_RE.search(surface):
            return True
        if (
            _PROTOCOL_SCHEMA_CONTEXT_RE.search(surface)
            and len(set(_PROTOCOL_SCHEMA_FIELD_RE.findall(surface.lower()))) >= 2
        ):
            return True
    return False


def decide_finalization(
    *,
    contract: Optional[object] = None,
    artifact_plan: Optional[object] = None,
    assembly: Optional[object] = None,
    execution_scope: Optional[object] = None,
    scope_output_required: bool = True,
    polish_requested: bool = False,
    policy: FinalizationPolicy = DEFAULT_FINALIZATION_POLICY,
) -> FinalizationDecision:
    """Select the finalization mode from TRUSTED runtime information only.

    Inputs are the resolved contract, the validated plan's artifact manifest
    presence, the run's assembly, the inherited execution scope, and the
    explicit presentation preference. Worker and provider output can never
    reach this function, so no response can select its own finalization mode.
    """
    code_only = bool(getattr(contract, "code_only", False))
    owes_artifacts = artifact_plan is not None or assembly is not None

    if owes_artifacts or code_only:
        return FinalizationDecision(
            mode=policy.artifact_default_mode,
            reason=(
                "code-only contract requires complete artifact delivery"
                if code_only else "the validated plan declares an artifact manifest"
            ),
            artifacts_required=True,
            complete_assembly_required=True,
            prose_allowed=not code_only or policy.code_only_allows_prose,
            model_synthesis_allowed=False,
            deterministic_fallback_required=True,
            code_only=code_only,
        )

    if execution_scope is not None:
        # A scoped recursive child stays fully deterministic: its typed
        # artifacts travel through the collection channel and its text is an
        # intermediate feeding the parent, never a polish candidate.
        return FinalizationDecision(
            mode=FinalizationMode.ORDERED_SECTIONS,
            reason=(
                "scoped recursive producer output"
                if scope_output_required else "scoped recursive support output"
            ),
            artifacts_required=bool(
                scope_output_required
                and getattr(execution_scope, "output_artifact_ids", ())
            ),
            complete_assembly_required=False,
            prose_allowed=True,
            model_synthesis_allowed=False,
            deterministic_fallback_required=True,
        )

    # Phase 4G: model-based synthesis is eliminated from every production path.
    # The optional-polish mode survives structurally (backward-compatible enum,
    # backward-importable synthesizer) but is unreachable while the authoritative
    # invariant is False — a polish request or the trusted config flag no longer
    # selects it. This is enforcement AT the finalization boundary, not a flag an
    # old path can ignore: the deterministic ordered result is always returned.
    if MODEL_BASED_SYNTHESIS_ENABLED and (polish_requested or policy.allow_model_polish):
        return FinalizationDecision(
            mode=FinalizationMode.OPTIONAL_MODEL_SYNTHESIS,
            reason=(
                "the user explicitly requested a polished unified narrative"
                if polish_requested
                else "trusted configuration enables optional model polish"
            ),
            artifacts_required=False,
            complete_assembly_required=False,
            prose_allowed=True,
            model_synthesis_allowed=True,
            deterministic_fallback_required=True,
        )

    return FinalizationDecision(
        mode=policy.prose_default_mode,
        reason="non-artifact request; deterministic ordered sections by default",
        artifacts_required=False,
        complete_assembly_required=False,
        prose_allowed=True,
        model_synthesis_allowed=False,
        deterministic_fallback_required=True,
    )


# --------------------------------------------------------------------------- #
# Part E — the typed prose worker envelope
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProseEnvelope:
    """One non-artifact worker's typed reply. ``content`` is the ONLY visible
    body; every other field is metadata and never rendered raw."""

    task_id: str = ""
    status: str = ""
    content: str = ""
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schema_version: int = PROSE_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _bounded_line(self.task_id, 300))
        object.__setattr__(self, "status", _bounded_line(self.status, 60))
        object.__setattr__(
            self, "content", "" if self.content is None else str(self.content)
        )
        for name in ("warnings", "evidence_refs"):
            items = getattr(self, name) or ()
            object.__setattr__(
                self,
                name,
                tuple(_bounded_line(item, 300) for item in items if str(item).strip()),
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))


_PROSE_REQUIRED_TOKEN = '"task_id"'
_PROSE_SHAPE_TOKENS = ('"status"', '"content"', '"warnings"', '"evidence_refs"')


def _fence_stripped(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        from .llm_client import _strip_code_fences  # local import avoids a cycle
        stripped = _strip_code_fences(stripped).strip()
    return stripped


def _decoded_prose_candidate(text: str) -> str:
    """Fence-strip and unwrap at most two JSON-string encoding layers."""
    candidate = _fence_stripped(text)
    for _ in range(2):
        if not candidate.startswith('"'):
            break
        try:
            decoded = json.loads(candidate)
        except (TypeError, ValueError):
            break
        if not isinstance(decoded, str):
            break
        candidate = _fence_stripped(decoded)
    return candidate


def looks_like_prose_envelope(text: str) -> bool:
    """Whether the WHOLE reply is shaped as the typed prose envelope.

    Strict about shape: the reply must BE a JSON object carrying both the
    ``task_id`` and ``content`` key tokens, and must not be a worker artifact
    envelope (those carry ``artifacts``/``summary`` and take precedence). Prose
    or code that merely mentions the keys is never flagged.
    """
    stripped = _decoded_prose_candidate(text)
    if not stripped.startswith("{"):
        return False
    head = stripped[:2000]
    if '"artifacts"' in head or '"key_decisions"' in head:
        return False
    shape_count = sum(token in head for token in _PROSE_SHAPE_TOKENS)
    return (
        (_PROSE_REQUIRED_TOKEN in head and shape_count >= 1)
        or shape_count >= 2
    )


def parse_prose_envelope(text: str) -> Optional[ProseEnvelope]:
    """Parse a typed prose envelope, or return ``None`` when it is not one.

    Unknown fields are ignored by explicit policy; ``content`` must be a
    string. The worker-declared ``task_id`` is carried for CHECKING only —
    the runtime scope remains the authoritative subtask identity.
    """
    if not looks_like_prose_envelope(text):
        return None
    stripped = _decoded_prose_candidate(text)
    try:
        data = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("content"), str):
        return None
    warnings = data.get("warnings")
    evidence = data.get("evidence_refs")
    return ProseEnvelope(
        task_id=str(data.get("task_id") or ""),
        status=str(data.get("status") or ""),
        content=data["content"],
        warnings=tuple(
            str(item) for item in (warnings if isinstance(warnings, list) else ())
        ),
        evidence_refs=tuple(
            str(item) for item in (evidence if isinstance(evidence, list) else ())
        ),
    )


def is_malformed_prose_envelope(text: str) -> bool:
    """Prose-envelope-shaped text that cannot be decoded. Fails closed: callers
    reject it into the existing bounded retry instead of displaying it."""
    if not looks_like_prose_envelope(text):
        return False
    return parse_prose_envelope(text) is None


# --------------------------------------------------------------------------- #
# Part H — refusal / non-delivery classification for scoped workers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkerDeliveryClassification:
    code: WorkerDeliveryCode
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.code, WorkerDeliveryCode):
            object.__setattr__(self, "code", WorkerDeliveryCode(str(self.code)))
        object.__setattr__(self, "reason", _bounded_line(self.reason, 300))


# Secondary evidence only: scope obligations are the primary signal. These
# refine WHY a zero-delivery response failed; they never fire when the worker
# actually delivered owned artifacts, so source code that merely contains a
# refusal-like phrase is never classified as a refusal.
_SIZE_REFUSAL_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:is|would\s+be|will\s+be|are)\s+(?:far\s+|way\s+)?too\s+"
    r"(?:large|long|big)\b",
    r"\btoo\s+(?:large|long|big)\s+to\s+"
    r"(?:provide|include|return|fit|send|paste|complete|generate)\b",
    r"\bexceeds?\b[^.\n]{0,50}\b(?:limit|maximum|budget)\b",
    r"\b(?:character|message|response|output|token|length)\s+limit\b",
    r"\bsplit\s+(?:it|this|that|them|the\s+\w+)?\s*(?:in)?to\s+"
    r"(?:multiple|several|two|separate)\b",
    r"\b(?:in|across|over)\s+(?:multiple|several|separate)\s+"
    r"(?:messages|responses|replies|parts|chunks)\b",
    r"\bcan(?:'t|not|\s+not)\s+(?:provide|include|return|produce|fit|deliver)"
    r"\s+(?:all|every|the\s+entire|the\s+whole|the\s+complete)\b",
    r"\bcontinue\s+in\s+(?:a|the)\s+(?:next|following)\s+"
    r"(?:message|response|reply)\b",
))

_NON_DELIVERY_REFUSAL_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bsource\s+material\s+is\s+(?:missing|unavailable|not\s+available)\b",
    r"\bcan(?:'t|not|\s+not)\s+produce\s+the\s+requested\s+"
    r"(?:project|files?|artifacts?|code)\b",
    r"\brequired\s+(?:input|context|files?)\s+(?:is|are|was|were)\s+"
    r"(?:missing|not\s+provided)\b",
    r"\bi\s+can(?:'t|not)\s+(?:help|assist|comply|provide|create|generate)\b",
    r"\bi\s+must\s+(?:respectfully\s+)?decline\b",
    r"\bagainst\s+my\s+(?:guidelines|programming|policy|policies|principles)\b",
))


def classify_scoped_delivery(
    scope: object,
    collection: Optional[object],
    response_text: str,
) -> WorkerDeliveryClassification:
    """Classify what a scoped artifact worker delivered against its scope.

    Primary evidence is the artifact-scope obligation (owned/accepted
    artifacts), never phrase matching: a worker that delivered its owned
    artifacts is DELIVERED (or MISSING_REQUIRED_ARTIFACTS when a subset is
    outstanding) regardless of what its file contents say.
    """
    owed = tuple(getattr(scope, "output_artifact_ids", ()) or ())
    accepted = tuple(getattr(collection, "accepted", ()) or ())
    satisfied = bool(getattr(collection, "satisfied", False))
    if not owed or satisfied:
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.DELIVERED, "the assigned scope is satisfied"
        )
    if accepted:
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.MISSING_REQUIRED_ARTIFACTS,
            f"delivered {len(accepted)} owned artifact(s); required artifacts "
            "remain outstanding",
        )
    head = " ".join(str(response_text or "")[:2000].split())
    head = head.replace("\u2019", "'").replace("\u00e2\u20ac\u2122", "'")
    if any(p.search(head) for p in _SIZE_REFUSAL_PATTERNS):
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.RESPONSE_SIZE_REFUSAL,
            f"the worker declined to return its {len(owed)} owned artifact(s), "
            "citing response size / message splitting",
        )
    if any(p.search(head) for p in _NON_DELIVERY_REFUSAL_PATTERNS):
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.ARTIFACT_NON_DELIVERY,
            f"the worker declared it cannot produce its {len(owed)} owned "
            "artifact(s)",
        )
    if bool(getattr(collection, "engaged", False)):
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.ARTIFACT_NON_DELIVERY,
            f"the typed artifact envelope contained none of the {len(owed)} "
            "owned artifact(s)",
        )

    # Preserve the approved Phase 4C compatibility path for a legacy worker
    # that returned concrete code without the typed artifact envelope. It still
    # has an unmet (repairable) artifact obligation, but it is not an
    # explanation-only substitution. Plain prose remains fatal below.
    from .intent import has_real_code
    if has_real_code(str(response_text or "")):
        return WorkerDeliveryClassification(
            WorkerDeliveryCode.MISSING_REQUIRED_ARTIFACTS,
            f"the reply contains concrete code but none of the {len(owed)} "
            "owned artifact(s) were delivered through the typed envelope",
        )

    return WorkerDeliveryClassification(
        WorkerDeliveryCode.EXPLANATION_ONLY_RESPONSE,
        f"the reply was free text with none of the {len(owed)} owned "
        "artifact(s) in the typed envelope",
    )


# --------------------------------------------------------------------------- #
# Part K — embedded protocol-envelope protection
# --------------------------------------------------------------------------- #
_ENVELOPE_TOKEN_RE = re.compile(
    r'"(?:summary|key_decisions|artifacts|confidence)"'
)
_EMBEDDED_SCAN_LIMIT_NOTE = (
    "embedded internal result envelopes exceeded the bounded scan limit"
)


def neutralize_embedded_envelopes(
    text: str,
    *,
    max_scans: int = DEFAULT_FINALIZATION_POLICY.max_embedded_envelopes_per_section,
) -> tuple[str, tuple[str, ...]]:
    """Remove internal protocol envelopes EMBEDDED inside otherwise valid text.

    The live failure interleaved headings with raw ``{"summary": ...}`` JSON.
    Detection uses trusted envelope boundaries and typed parsing — never broad
    regular-expression deletion:

    * only a ``{`` that STARTS a line is a candidate;
    * the span must parse completely (``json.JSONDecoder.raw_decode``) into the
      canonical four-field protocol object to be replaced by its flattened
      content — any other JSON (source code literals, requested data, fixtures
      assigned inline) stays byte-identical;
    * an unparseable candidate is removed only when it is protocol-shaped AND
      runs to the end of the text (the truncated-envelope failure mode).

    Returns ``(cleaned_text, notes)``; notes are bounded diagnostics, never
    payload fragments.
    """
    from .validation import render_artifact  # local import keeps top imports lean

    decoder = json.JSONDecoder()
    notes: list[str] = []
    out = text
    scan_from = 0
    for _ in range(max(0, int(max_scans))):
        start = _find_embedded_candidate(out, scan_from)
        if start is None:
            return out, tuple(notes)
        try:
            data, end = decoder.raw_decode(out, start)
        except ValueError:
            tail = out[start:]
            if is_malformed_protocol_envelope(tail):
                out = out[:start].rstrip()
                notes.append(
                    "a truncated internal result envelope was removed from the "
                    f"section ({len(tail)} chars)"
                )
                continue
            scan_from = start + 1  # not decodable, not protocol: skip past it
            continue
        if isinstance(data, dict) and is_canonical_protocol_data(data):
            flattened = decode_nested_envelope(render_artifact(data)).strip()
            before = out[:start].rstrip("\n")
            after = out[end:].lstrip("\n")
            pieces = [piece for piece in (before, flattened, after) if piece]
            scan_from = len("\n\n".join(piece for piece in (before, flattened) if piece))
            out = "\n\n".join(pieces)
            notes.append("an embedded internal result envelope was flattened")
            continue
        # Parsed, but not protocol traffic: legitimate JSON stays byte-identical.
        scan_from = end
    # The scan bound is a security boundary, not permission for a seventh
    # envelope to pass through. Quarantine the unresolved tail; the ordered
    # section builder turns this note into an explicit failed section.
    remaining = _find_embedded_candidate(out, scan_from)
    if remaining is not None:
        out = out[:remaining].rstrip()
        notes.append(_EMBEDDED_SCAN_LIMIT_NOTE)
    return out, tuple(notes)


def _find_embedded_candidate(text: str, scan_from: int = 0) -> Optional[int]:
    """Offset of the next line-leading ``{`` whose head carries >= 2 protocol
    key tokens — the trusted boundary shape of an embedded envelope."""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("{"):
            start = offset + (len(line) - len(stripped))
            if start >= scan_from:
                head = text[start: start + 2000]
                if len(_ENVELOPE_TOKEN_RE.findall(head)) >= 2:
                    return start
        offset += len(line)
    return None


# --------------------------------------------------------------------------- #
# Parts C + D — trusted ordering and the deterministic section stitcher
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderedSection:
    """One deterministic section of the final answer. Typed and bounded;
    internal metadata (scores, decisions, envelopes) can never appear here."""

    subtask_id: str
    title: str
    status: str
    content: str = ""
    diagnostic: str = ""
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    # Runtime cleanup/identity diagnostics belong in orchestration metadata,
    # never in the user-visible warning/evidence channel.
    internal_notes: tuple[str, ...] = ()
    provenance: str = ""
    schema_version: int = ORDERED_SECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtask_id", _bounded_line(self.subtask_id, 300))
        object.__setattr__(self, "title", str(self.title or ""))
        object.__setattr__(self, "status", _bounded_line(self.status, 40))
        object.__setattr__(
            self, "content", "" if self.content is None else str(self.content)
        )
        object.__setattr__(self, "diagnostic", str(self.diagnostic or ""))
        for name in ("warnings", "evidence_refs", "internal_notes"):
            items = getattr(self, name) or ()
            object.__setattr__(
                self, name, tuple(str(item) for item in items if str(item).strip())
            )
        object.__setattr__(self, "provenance", _bounded_line(self.provenance, 120))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def usable(self) -> bool:
        return self.status == "completed" and bool(self.content.strip())


def build_ordered_sections(
    plan: object,
    results: Sequence[object],
    *,
    policy: FinalizationPolicy = DEFAULT_FINALIZATION_POLICY,
    json_requested: bool = False,
    protocol_content_requested: bool = False,
) -> tuple[OrderedSection, ...]:
    """Build typed sections in TRUSTED plan order.

    The order comes exclusively from the validated plan's subtask allocation —
    never worker completion time, provider response order, worker-supplied
    order fields, confidence values or agent names. Results whose subtask id is
    not in the plan are dropped (a worker cannot inject a section), and only
    the FIRST result per subtask id is used (a duplicate cannot displace it).
    """
    subtasks = list(getattr(plan, "subtasks", None) or [])
    plan_ids: list[str] = []
    seen_plan_ids: set[str] = set()
    duplicate_plan_ids: set[str] = set()
    for subtask in subtasks:
        sid = str(getattr(subtask, "id", "") or "")
        if not sid:
            raise FinalizationError("trusted plan contains an empty subtask id")
        if sid in seen_plan_ids:
            duplicate_plan_ids.add(sid)
        seen_plan_ids.add(sid)
        plan_ids.append(sid)
    if duplicate_plan_ids:
        raise FinalizationError(
            "trusted plan contains duplicate subtask id(s): "
            + ", ".join(sorted(duplicate_plan_ids)[:8])
        )

    plan_id_set = set(plan_ids)
    by_id: dict[str, object] = {}
    for result in results or ():
        sid = str(getattr(result, "subtask_id", "") or "")
        if not sid:
            raise FinalizationError("a subtask result has an empty runtime identity")
        if sid not in plan_id_set:
            raise FinalizationError(
                f"subtask result identity {sid!r} does not exist in the trusted plan"
            )
        if sid in by_id:
            raise FinalizationError(
                f"duplicate subtask result identity {sid!r} cannot be finalized"
            )
        by_id[sid] = result

    sections: list[OrderedSection] = []
    for subtask in subtasks[: policy.max_sections]:
        sid = str(getattr(subtask, "id", "") or "")
        title = _bounded_line(
            getattr(subtask, "title", "") or sid, policy.max_section_title_chars
        )
        result = by_id.get(sid)
        warnings: list[str] = []
        evidence_refs: list[str] = []
        internal_notes: list[str] = []
        if result is None:
            continue  # never scheduled/returned: nothing to render for it

        status_value = getattr(getattr(result, "status", None), "value", None)
        status = str(status_value or getattr(result, "status", "") or "")
        provenance = _bounded_line(getattr(result, "agent_name", "") or "", 120)
        for note in tuple(getattr(result, "worker_warnings", ()) or ())[
            : policy.max_section_warnings
        ]:
            warnings.append(_bounded_line(note, policy.max_section_diagnostic_chars))
        for ref in tuple(getattr(result, "worker_evidence_refs", ()) or ())[
            : policy.max_section_evidence_refs
        ]:
            evidence_refs.append(_bounded_line(ref, policy.max_section_diagnostic_chars))
        for note in tuple(getattr(result, "worker_internal_notes", ()) or ())[
            : policy.max_section_warnings
        ]:
            internal_notes.append(
                _bounded_line(note, policy.max_section_diagnostic_chars)
            )

        if status != "completed" or not str(getattr(result, "output", "") or "").strip():
            diagnostic = sanitize_diagnostic(
                getattr(result, "error", "") or status or "not completed",
                policy.max_section_diagnostic_chars,
            )
            sections.append(OrderedSection(
                subtask_id=sid, title=title, status=status or "failed",
                diagnostic=diagnostic, warnings=tuple(warnings),
                evidence_refs=tuple(evidence_refs),
                internal_notes=tuple(internal_notes),
                provenance=provenance,
            ))
            continue

        (
            content, clean_evidence, clean_warnings, clean_notes, failure,
        ) = _clean_section_content(
            str(getattr(result, "output", "")), sid, policy,
            json_requested=json_requested,
            protocol_content_requested=protocol_content_requested,
        )
        warnings.extend(clean_warnings)
        evidence_refs.extend(clean_evidence)
        internal_notes.extend(clean_notes)
        if failure:
            sections.append(OrderedSection(
                subtask_id=sid, title=title, status="failed",
                diagnostic=failure, warnings=tuple(warnings),
                evidence_refs=tuple(evidence_refs),
                internal_notes=tuple(internal_notes),
                provenance=provenance,
            ))
            continue

        sections.append(OrderedSection(
            subtask_id=sid, title=title, status="completed", content=content,
            warnings=tuple(dict.fromkeys(warnings))[: policy.max_section_warnings],
            evidence_refs=tuple(dict.fromkeys(evidence_refs))[
                : policy.max_section_evidence_refs
            ],
            internal_notes=tuple(dict.fromkeys(internal_notes))[
                : policy.max_section_warnings
            ],
            provenance=provenance,
        ))

    if len(subtasks) > policy.max_sections:
        omitted = len(subtasks) - policy.max_sections
        sections.append(OrderedSection(
            subtask_id="__overflow__",
            title="Additional sections",
            status="failed",
            diagnostic=f"{omitted} additional section(s) exceeded the "
                       f"{policy.max_sections}-section policy bound",
        ))
    return tuple(sections)


def _clean_section_content(
    output: str,
    subtask_id: str,
    policy: FinalizationPolicy,
    *,
    json_requested: bool,
    protocol_content_requested: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    """One section body: typed envelopes parsed, internal traffic fails closed.

    Returns ``(content, evidence_refs, warnings, internal_notes,
    failure_reason)`` where a
    non-empty ``failure_reason`` means the section must render as a bounded
    diagnostic, never as content.
    """
    warnings: list[str] = []
    internal_notes: list[str] = []
    evidence_refs: tuple[str, ...] = ()
    text = output

    # Exact-data and protocol-documentation requests are trusted USER intent.
    # Their JSON/code fixtures are deliverable content, not runtime traffic.
    if json_requested or protocol_content_requested:
        if not text.strip():
            return "", (), (), (), "the reply contained no usable content"
        return text, (), (), (), ""

    prose = parse_prose_envelope(text)
    if prose is not None:
        if prose.task_id and prose.task_id != subtask_id:
            internal_notes.append(
                "the worker-declared task id did not match the assigned "
                "subtask; the runtime identity was used"
            )
        warnings.extend(prose.warnings[: policy.max_section_warnings])
        evidence_refs = prose.evidence_refs
        text = prose.content
    elif is_malformed_prose_envelope(text):
        return "", (), tuple(warnings), tuple(internal_notes), (
            "the worker reply was a malformed typed result envelope "
            f"({len(text)} chars)"
        )

    # The embedded scanner runs FIRST, before any whole-text malformed check:
    # it already handles both shapes correctly — a truncated embedded envelope
    # keeps its preceding heading text (the live failure), and a whole-text
    # malformed envelope (no surrounding text) degrades to empty, caught by
    # the final "no usable content" check below. Running a whole-text malformed
    # check BEFORE this would discard legitimate surrounding text instead.
    text, notes = neutralize_embedded_envelopes(
        text, max_scans=policy.max_embedded_envelopes_per_section
    )
    internal_notes.extend(notes)
    if _EMBEDDED_SCAN_LIMIT_NOTE in notes:
        return "", (), tuple(warnings), tuple(internal_notes), (
            "the worker reply exceeded the bounded internal-envelope scan limit"
        )

    text = decode_nested_envelope(text)
    if is_malformed_protocol_envelope(text):
        return "", (), tuple(warnings), tuple(internal_notes), (
            f"the worker reply was an undecodable internal envelope "
            f"({len(text)} chars)"
        )
    if looks_like_protocol_envelope(text):
        # A valid but non-canonical envelope (e.g. ``artifacts`` without the
        # other three fields) survives the bounded nested decode; flatten it
        # here so no whole-section protocol object can render raw.
        from .validation import parse_worker_artifact, render_artifact
        envelope = parse_worker_artifact(text)
        if envelope is not None:
            flattened = decode_nested_envelope(render_artifact(envelope)).strip()
            if not flattened or flattened == text.strip():
                return "", (), tuple(warnings), tuple(internal_notes), (
                    "the reply contained only an internal envelope with no "
                    "usable content"
                )
            internal_notes.append("a whole-section internal envelope was flattened")
            text = flattened
    if looks_like_orchestration_json(text):
        return "", (), tuple(warnings), tuple(internal_notes), (
            "the captured output was internal planning data, not an answer"
        )
    if not text.strip():
        return (
            "", (), tuple(warnings), tuple(internal_notes),
            "the reply contained no usable content",
        )
    return text, evidence_refs, tuple(warnings), tuple(internal_notes), ""


def render_ordered_sections(
    sections: Sequence[OrderedSection],
    *,
    policy: FinalizationPolicy = DEFAULT_FINALIZATION_POLICY,
) -> str:
    """Deterministically stitch typed sections into the visible answer.

    Stable plan order, bounded headings and diagnostics, no model call, no
    JSON fallback, no ``str(mapping)`` of anything. Failed sections render as
    short readable notes (when the policy says so), never as raw payloads.
    """
    ordered = list(sections or ())
    usable = [section for section in ordered if section.usable]
    if not usable:
        failed = "; ".join(
            f"{section.title or section.subtask_id}: "
            f"{section.diagnostic or section.status}"
            for section in ordered
        )
        return f"{NO_USABLE_RESULT_SENTINEL} {failed}".rstrip()

    failed_sections = [
        section for section in ordered
        if not section.usable and policy.render_failed_sections
    ]
    if len(usable) == 1 and not failed_sections:
        return _render_section_body(usable[0], policy)

    blocks: list[str] = []
    for section in ordered:
        if section.usable:
            blocks.append(
                f"## {section.title}\n\n{_render_section_body(section, policy)}"
            )
        elif policy.render_failed_sections:
            note = _bounded_line(
                section.diagnostic or f"status: {section.status}",
                policy.max_section_diagnostic_chars,
            )
            blocks.append(
                f"## {section.title}\n\n_(This section was not completed: "
                f"{note})_"
            )
    return "\n\n".join(blocks)


def _render_section_body(
    section: OrderedSection,
    policy: FinalizationPolicy,
) -> str:
    """Render user-owned body plus typed, bounded warning/evidence metadata."""
    blocks = [section.content]
    metadata = _render_section_metadata(section, policy)
    if metadata:
        blocks.append(metadata)
    return "\n\n".join(blocks)


def _render_section_metadata(
    section: OrderedSection,
    policy: FinalizationPolicy,
) -> str:
    """Render only the typed user-facing metadata for one section."""
    blocks: list[str] = []
    warnings = [
        _escape_metadata(item, policy.max_section_diagnostic_chars)
        for item in section.warnings[: policy.max_section_warnings]
        if str(item).strip()
    ]
    evidence = [
        _escape_metadata(item, policy.max_section_diagnostic_chars)
        for item in section.evidence_refs[: policy.max_section_evidence_refs]
        if str(item).strip()
    ]
    if warnings:
        blocks.append("**Warnings**\n\n" + "\n".join(f"- {item}" for item in warnings))
    if evidence:
        blocks.append("**Evidence**\n\n" + "\n".join(f"- {item}" for item in evidence))
    return "\n\n".join(blocks)


def render_ordered_metadata(
    sections: Sequence[OrderedSection],
    *,
    policy: FinalizationPolicy = DEFAULT_FINALIZATION_POLICY,
) -> str:
    """Deterministic appendix used when optional polish replaces section text."""
    entries = [
        (section, _render_section_metadata(section, policy))
        for section in sections
        if section.usable
    ]
    entries = [(section, body) for section, body in entries if body]
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0][1]
    blocks = ["## Warnings and evidence"]
    for section, body in entries:
        blocks.append(f"### {section.title}\n\n{body}")
    return "\n\n".join(blocks)


def _escape_metadata(value: Any, limit: int) -> str:
    line = _bounded_line(value, limit)
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", line)


# --------------------------------------------------------------------------- #
# Part F — deterministic artifact finalization
# --------------------------------------------------------------------------- #
def finalize_artifact_delivery(
    assembly: Optional[AssembledDeliverable],
    decision: FinalizationDecision,
    *,
    policy: FinalizationPolicy = DEFAULT_FINALIZATION_POLICY,
) -> str:
    """Render an artifact-bearing run's final answer deterministically.

    Never calls a model. A complete assembly is delivered exactly; a code-only
    project that is not COMPLETE yields ONE clean failure result — no partial
    project presented as success, no prose fallback, no refusal text.
    """
    if assembly is None:
        return "\n\n".join([
            CODE_ONLY_FAILURE_HEADLINE if decision.code_only else
            "The requested deliverable could not be completed.",
            "No artifact assembly was produced for this artifact-bearing "
            "request.",
            CODE_ONLY_FAILURE_FOOTER,
        ])
    if decision.code_only:
        if assembly.complete:
            return render_code_only_deliverable(assembly)
        if not policy.allow_partial_artifact_delivery:
            return render_artifact_failure(assembly)
    from .artifact_results import render_assembled_deliverable
    return render_assembled_deliverable(assembly)


def render_code_only_deliverable(assembly: AssembledDeliverable) -> str:
    """Code-only rendering: every accepted file exactly once, in manifest
    order, with stable labels and exact content — nothing else.

    No headline, no summaries, no confidence, no key decisions, no worker
    prose. Result-kind artifacts (declared build/test evidence) render as their
    exact text under their label.
    """
    blocks: list[str] = []
    for candidate in assembly.artifacts:
        label = f"### {candidate.path}"
        if candidate.kind in _RESULT_KINDS:
            body = candidate.content
        elif not candidate.content:
            body = "_(no content: directory or delete operation)_"
        else:
            body = _fence(candidate.language, candidate.content)
        blocks.append(f"{label}\n\n{body}")
    if not blocks:
        return render_artifact_failure(assembly)
    return "\n\n".join(blocks)


def render_artifact_failure(assembly: AssembledDeliverable) -> str:
    """ONE clean deterministic failure result for an incomplete project.

    Names what is missing/broken/conflicted by artifact id or path only —
    never rejection payloads, worker refusals, or partial file bodies.
    """
    lines: list[str] = [CODE_ONLY_FAILURE_HEADLINE, ""]
    broken = set(assembly.integrity_rejected_artifact_ids)
    absent = [
        artifact_id
        for artifact_id in assembly.missing_required_artifact_ids
        if artifact_id not in broken
    ]
    if absent:
        lines.append("Missing required files:")
        lines.extend(f"- {artifact_id}" for artifact_id in absent)
        lines.append("")
    if assembly.integrity_rejected_artifact_ids:
        lines.append("Files returned but structurally incomplete:")
        lines.extend(
            f"- {artifact_id}"
            for artifact_id in assembly.integrity_rejected_artifact_ids
        )
        lines.append("")
    if assembly.conflicts:
        conflicted = sorted({
            conflict.path or conflict.artifact_id
            for conflict in assembly.conflicts
        })
        lines.append("Files with unresolved conflicting submissions:")
        lines.extend(f"- {name}" for name in conflicted)
        lines.append("")
    lines.append(CODE_ONLY_FAILURE_FOOTER)
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Part L — unbypassable code-only root guards
#
# These make the code-only contract enforceable at the authoritative runtime
# boundaries, not merely in prompt instructions. They are pure, deterministic,
# duck-typed (no import from models.py) and see only TRUSTED runtime state —
# never worker or provider text — so no response can talk its way past them.
# --------------------------------------------------------------------------- #

# The clean, deterministic result for a run whose code-only obligation was lost
# before finalization (a dropped/legacy contract, or a stale workspace). It
# carries no worker prose, no refusal and no partial source.
CODE_ONLY_CONTRACT_LOST_DIAGNOSTIC = (
    "the code-only delivery contract was not preserved for this request"
)
CODE_ONLY_NO_ARTIFACTS_DIAGNOSTIC = (
    "the code-only request produced no artifact project to deliver"
)


def code_only_contract_intact(contract: Optional[object]) -> bool:
    """Whether a contract still carries the FULL code-only obligation.

    A genuine code-only contract must resolve to a code-producing intent with
    ``code_required`` and ``code_only`` both true. IMPLEMENT, MODIFY and DEBUG
    are all authoritative code contracts; anything less means the obligation
    was dropped somewhere upstream and the root guard fails closed.
    """
    if contract is None:
        return False
    if not bool(getattr(contract, "code_only", False)):
        return False
    if not bool(getattr(contract, "code_required", False)):
        return False
    intent_value = getattr(getattr(contract, "intent", None), "value", "")
    return intent_value in {"implement", "modify", "debug"}


def assert_code_only_decision(decision: FinalizationDecision) -> None:
    """Fail closed unless a code-only finalization decision is pure assembly.

    Code-only work may finalize ONLY through ARTIFACT_ASSEMBLY: no ordered
    sections, no optional model synthesis, no prose. A decision that violates
    this is a structural bug, not a recoverable state, so it raises.
    """
    if not getattr(decision, "code_only", False):
        return
    if decision.mode is not FinalizationMode.ARTIFACT_ASSEMBLY:
        raise FinalizationError(
            "code-only finalization must use ARTIFACT_ASSEMBLY, not "
            f"{decision.mode.value}"
        )
    if decision.model_synthesis_allowed:
        raise FinalizationError(
            "code-only finalization must not permit model synthesis"
        )
    if decision.prose_allowed:
        raise FinalizationError(
            "code-only finalization must not permit explanatory prose"
        )


def assert_no_model_synthesis(decision: FinalizationDecision) -> None:
    """Fail closed unless a finalization decision honors the no-synthesis invariant.

    Phase 4G: while ``MODEL_BASED_SYNTHESIS_ENABLED`` is False, NO finalization
    decision may permit model synthesis or select the optional-synthesis mode.
    A decision that does is a structural bug (some path fabricated a synthesis
    decision behind ``decide_finalization``'s back), not a recoverable state, so
    it raises. Enforced at the root finalization boundary alongside
    :func:`assert_code_only_decision`.
    """
    if MODEL_BASED_SYNTHESIS_ENABLED:
        return
    if decision.model_synthesis_allowed:
        raise FinalizationError(
            "model-based synthesis is disabled but the finalization decision "
            "permits it"
        )
    if decision.mode is FinalizationMode.OPTIONAL_MODEL_SYNTHESIS:
        raise FinalizationError(
            "model-based synthesis is disabled but the finalization mode is "
            "optional_model_synthesis"
        )


def code_only_delivery_ok(
    assembly: Optional[AssembledDeliverable],
) -> tuple[bool, str]:
    """Whether a code-only assembly is a COMPLETE, deliverable project.

    Successful delivery requires a complete assembly with no unresolved
    integrity rejection, no required missing artifact and no unresolved
    conflict. Returns ``(ok, diagnostic)``; the diagnostic is one clean
    reason naming ids/paths only — never worker text.
    """
    if assembly is None:
        return False, CODE_ONLY_NO_ARTIFACTS_DIAGNOSTIC
    reasons: list[str] = []
    broken = set(getattr(assembly, "integrity_rejected_artifact_ids", ()) or ())
    absent = [
        artifact_id
        for artifact_id in getattr(assembly, "missing_required_artifact_ids", ()) or ()
        if artifact_id not in broken
    ]
    if absent:
        reasons.append("missing required file(s): " + ", ".join(absent[:6]))
    if broken:
        reasons.append(
            "structurally incomplete file(s): "
            + ", ".join(sorted(broken)[:6])
        )
    if getattr(assembly, "conflicts", ()):
        reasons.append("unresolved conflicting submission(s)")
    if not getattr(assembly, "complete", False) and not reasons:
        reasons.append(
            assembly.problem_summary() or "the assembly is not complete"
        )
    if reasons:
        return False, "; ".join(reasons)
    return True, ""
