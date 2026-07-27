"""Final user-facing presentation — the ONE authoritative rendering boundary.

Every orchestration outcome (success, partial artifact deliverable, diagnostic)
passes through :func:`render_final_presentation` before it reaches terminal
SSE, conversation persistence, or any frontend. The boundary guarantees:

* prose renders as readable prose;
* artifact deliverables keep the deterministic Phase 4 rendering;
* raw JSON reaches the user ONLY when the request explicitly asked for JSON;
* internal protocol envelopes (worker/synthesizer JSON, planner echoes) are
  decoded or replaced with a clean diagnostic — never displayed raw;
* parser failures surface as bounded, sanitized diagnostics, never payloads.

The module is deliberately small and provider-neutral: it owns presentation
DECISIONS, while the deterministic artifact rendering itself stays in
``dozen.artifact_results`` and protocol parsing stays in ``dozen.validation``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

from .validation import (
    decode_nested_envelope,
    is_canonical_protocol_data,
    is_malformed_protocol_envelope,
    looks_like_protocol_envelope,
    looks_like_orchestration_json,
    parse_worker_artifact,
    render_artifact,
)

PRESENTATION_SCHEMA_VERSION = 1

# Bound for any diagnostic string that may reach a user, an SSE event or a log.
MAX_DIAGNOSTIC_CHARS = 500


class PresentationKind(str, Enum):
    """What the final answer IS — the smallest useful presentation taxonomy."""

    PROSE = "prose"                      # readable answer / explanation
    ARTIFACT = "artifact"                # complete artifact/project deliverable
    PARTIAL_ARTIFACT = "partial_artifact"  # incomplete/conflicted deliverable
    JSON_DATA = "json_data"              # the user explicitly requested JSON
    LITERAL = "literal"                  # explicitly requested protocol fixture/docs
    DIAGNOSTIC = "diagnostic"            # clean failure/error presentation


@dataclass(frozen=True)
class FinalPresentation:
    """The typed, immutable user-facing result representation.

    ``text`` is the ONLY user-visible surface. ``diagnostics`` carries bounded,
    sanitized internal notes for logs and tests — it is metadata and must never
    be rendered into chat as raw JSON.
    """

    kind: PresentationKind
    text: str
    diagnostics: tuple[str, ...] = ()
    schema_version: int = PRESENTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PresentationKind):
            object.__setattr__(
                self, "kind", PresentationKind(str(self.kind))
            )
        # ``None`` is absence, never user-facing prose.  Preserve other scalar
        # values losslessly while preventing the literal string "None" from
        # leaking through serialization or terminal transport.
        object.__setattr__(self, "text", "" if self.text is None else str(self.text))
        items = self.diagnostics or ()
        object.__setattr__(
            self,
            "diagnostics",
            tuple(sanitize_diagnostic(item) for item in items),
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "diagnostics": list(self.diagnostics),
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #
def sanitize_diagnostic(message: object, max_chars: int = MAX_DIAGNOSTIC_CHARS) -> str:
    """Bound a diagnostic so no raw payload can ride inside it.

    Collapses whitespace, elides any long braced span (an embedded protocol
    object), and clips to ``max_chars``. Deterministic and idempotent.
    """
    text = str(message)
    # Stack traces are implementation detail and commonly contain local paths,
    # request bodies and credentials. Keep only a preceding human reason.
    marker = "Traceback (most recent call last):"
    if marker in text:
        text = text.split(marker, 1)[0].strip() or "Internal operation failed."
    # Redact common credential forms before collapsing newlines; header values
    # must not consume the following line during matching.
    text = re.sub(
        r"(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*[:=][^\r\n]*",
        "credential=[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|"
        r"secret|session(?:id)?)\s*[:=]\s*[^\s,;]+",
        lambda m: f"{m.group(1)}=[redacted]",
        text,
    )
    # Remove non-whitespace control bytes, then normalize all whitespace.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    text = " ".join(text.split())
    # A long braced span inside a diagnostic is an embedded payload, not
    # information the user needs. Elide it, keeping a short identifying head.
    text = re.sub(r"\{.{120,}", lambda m: m.group(0)[:60] + " …payload elided…", text)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


# Explicit request for a JSON deliverable. Deliberately anchored on output
# phrasing ("as JSON", "output raw JSON", "JSON format", ".json file") so a
# request that merely MENTIONS JSON ("parse this JSON") does not flip the
# whole answer into pass-through mode.
_JSON_REQUEST_RE = re.compile(
    r"(?:\b(?:as|in|into|only|just|raw|strict|valid|pure)\s+(?:a\s+|the\s+)?json\b"
    r"|\bjson\s+(?:format|output|object|array|response|file|document|payload|"
    r"structure|schema|deliverable)\b"
    r"|\b(?:output|return|respond|reply|give|produce|emit|format|deliver)\b"
    r"[^.?!\n]{0,60}\bjson\b"
    r"|\.json\b)",
    re.IGNORECASE,
)

_NEGATED_JSON_RE = re.compile(
    r"\b(?:do\s+not|don['â€™]?t|never|without)\b[^.?!\n]{0,80}\bjson\b",
    re.IGNORECASE,
)


def user_requested_json(
    prompt: str = "",
    desired_output: str = "",
    constraints: Sequence[str] = (),
    contract: Optional[object] = None,
) -> bool:
    """Whether the REQUEST explicitly names JSON as the deliverable."""
    surfaces = [str(prompt or ""), str(desired_output or "")]
    surfaces.extend(str(item) for item in (constraints or ()))
    if contract is not None:
        surfaces.append(str(getattr(contract, "deliverable", "") or ""))
        surfaces.extend(
            str(item) for item in (getattr(contract, "user_constraints", ()) or ())
        )
    surfaces = [_NEGATED_JSON_RE.sub("", surface) for surface in surfaces]
    # ``desired_output`` is an authoritative output slot: a bare "JSON" there
    # is already an explicit format request.
    clean_desired = _NEGATED_JSON_RE.sub("", str(desired_output or ""))
    if re.search(r"\bjson\b", clean_desired, re.IGNORECASE):
        return True
    matches = any(_JSON_REQUEST_RE.search(surface) for surface in surfaces if surface)
    if not matches:
        return False
    # A typed code/change contract prevents incidental wording such as "build
    # an API that returns JSON" from changing the WHOLE answer into raw-JSON
    # mode. An explicit JSON deliverable in the contract remains authoritative.
    if contract is not None and (
        bool(getattr(contract, "code_required", False))
        or bool(getattr(contract, "repo_changes_required", False))
    ):
        deliverable = _NEGATED_JSON_RE.sub(
            "", str(getattr(contract, "deliverable", "") or "")
        )
        explicit_format = re.search(
            r"\b(?:as|in|only|just|raw|strict|valid|pure)\s+(?:a\s+|the\s+)?json\b",
            _NEGATED_JSON_RE.sub("", str(prompt or "")),
            re.IGNORECASE,
        )
        if not re.search(r"\bjson\b", deliverable, re.IGNORECASE) and not explicit_format:
            return False
    return True


def _looks_like_bare_json(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _requested_json_data(text: str) -> bool:
    """Valid requested JSON that is not the complete internal envelope schema."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        # Validate the fenced payload, but return/classify the original text so
        # the user's requested formatting stays byte-identical.
        from .llm_client import _strip_code_fences
        stripped = _strip_code_fences(stripped).strip()
    try:
        if _looks_like_bare_json(stripped):
            data = json.loads(stripped)
        else:
            # A provider may add a short or long prose lead-in despite an
            # explicit JSON request. Validate the embedded object while
            # preserving the original response bytes for presentation.
            from .llm_client import _extract_json
            data = _extract_json(stripped)
    except (TypeError, ValueError):
        return False
    except Exception:
        return False
    return not is_canonical_protocol_data(data)


def render_results_fallback(results: Sequence[object]) -> str:
    """Deterministic, envelope-free stitching of completed subtask outputs.

    Used when synthesis is unavailable or its answer had to be discarded.
    Stable order (caller passes plan order), subtask headings when there is
    more than one section, and any output that is itself protocol JSON is
    excluded rather than displayed.
    """
    sections: list[tuple[str, str]] = []
    for result in results or ():
        status = getattr(result, "status", None)
        output = str(getattr(result, "output", "") or "")
        if getattr(status, "value", status) != "completed" or not output.strip():
            continue
        cleaned = decode_nested_envelope(output)
        if looks_like_orchestration_json(cleaned) or is_malformed_protocol_envelope(cleaned):
            continue
        sections.append((str(getattr(result, "title", "") or ""), cleaned))
    if not sections:
        return "[No usable subtask outputs produced.]"
    if len(sections) == 1:
        return sections[0][1]
    return "\n\n".join(
        f"## {title}\n\n{body}" if title else body for title, body in sections
    )


# --------------------------------------------------------------------------- #
# THE rendering boundary
# --------------------------------------------------------------------------- #
_INTERNAL_DATA_MESSAGE = (
    "The run finished, but every captured output was internal "
    "planning data instead of an answer. Please run the task "
    "again — the model pool likely returned a stale page."
)

_MALFORMED_FINAL_MESSAGE = (
    "The run finished, but the final response was an internal result envelope "
    "that could not be decoded. Please run the task again."
)


def render_final_presentation(
    final_text: str,
    results: Sequence[object] = (),
    *,
    assembly: Optional[object] = None,
    json_requested: bool = False,
    protocol_content_requested: bool = False,
) -> FinalPresentation:
    """Render one orchestration outcome into the approved user-facing form.

    Returns a :class:`FinalPresentation` whose ``text`` is safe for terminal
    SSE, persistence and the frontend. A non-empty diagnostic in
    ``diagnostics`` signals that internal data had to be replaced (callers
    surface it as the run error, preserving pre-Phase-6 semantics).
    """
    text = str(final_text or "")
    diagnostics: list[str] = []

    # Artifact-bearing runs were rendered deterministically by the Phase 4
    # bypass; classify them without re-touching the content.
    if assembly is not None:
        complete = bool(getattr(assembly, "complete", False))
        kind = (
            PresentationKind.ARTIFACT if complete
            else PresentationKind.PARTIAL_ARTIFACT
        )
        return FinalPresentation(kind=kind, text=text)

    # A narrow request explicitly ABOUT the protocol makes envelope examples
    # user-owned content. The caller derives this flag only from trusted request
    # surfaces; worker output cannot grant itself this exception.
    if protocol_content_requested:
        return FinalPresentation(kind=PresentationKind.LITERAL, text=text)

    # The synthesizer's all-failed sentinel is a diagnostic, not successful
    # prose. Bound the aggregate list of worker reasons at the one diagnostic
    # boundary so a large plan cannot create an unbounded terminal payload.
    if text.startswith("[No subtasks produced a usable result]"):
        return FinalPresentation(
            # Preserve the historical error/status semantics for legacy tasks:
            # this sentinel was readable fallback prose, not a run error.
            kind=PresentationKind.PROSE,
            text=sanitize_diagnostic(text),
        )

    # A valid explicitly requested JSON document is user data, even when it
    # happens to use one or two protocol-looking field names. The complete
    # canonical four-field envelope remains internal and is flattened below.
    if json_requested and _requested_json_data(text):
        return FinalPresentation(kind=PresentationKind.JSON_DATA, text=text)

    # 1) A well-formed envelope (possibly double-encoded) is flattened to its
    #    actual content through bounded explicit rules.
    envelope = (
        parse_worker_artifact(text) if looks_like_protocol_envelope(text) else None
    )
    if envelope is not None:
        flattened = decode_nested_envelope(render_artifact(envelope))
        if flattened.strip():
            text = flattened
            diagnostics.append("final answer was an artifact envelope; flattened")

    # An explicitly requested JSON document nested inside the worker envelope
    # is still user data; preserve its exact bytes after unwrapping.
    if json_requested and _requested_json_data(text):
        return FinalPresentation(
            kind=PresentationKind.JSON_DATA,
            text=text,
            diagnostics=tuple(diagnostics),
        )

    # Decoding is intentionally depth-bounded. If canonical protocol traffic
    # remains after that bound, fail closed rather than displaying the residual
    # envelope or recursing without limit.
    if looks_like_protocol_envelope(text) and parse_worker_artifact(text) is not None:
        stitched = render_results_fallback(results)
        diagnostics.append("nested protocol-envelope depth limit reached")
        if stitched != "[No usable subtask outputs produced.]":
            return FinalPresentation(
                kind=PresentationKind.PROSE,
                text=stitched,
                diagnostics=tuple(diagnostics),
            )
        return FinalPresentation(
            kind=PresentationKind.DIAGNOSTIC,
            text=_MALFORMED_FINAL_MESSAGE,
            diagnostics=tuple(diagnostics),
        )

    # 2) A malformed envelope must fail closed: replace it with the stitched
    #    worker outputs, or a clean diagnostic when none exist.
    if is_malformed_protocol_envelope(text):
        stitched = render_results_fallback(results)
        # Length only — a diagnostic can propagate into parent errors and SSE,
        # so it must never carry even a fragment of the payload.
        note = f"malformed protocol envelope ({len(text)} chars)"
        if stitched != "[No usable subtask outputs produced.]":
            diagnostics.append(
                f"final answer was a {note}; replaced with stitched worker outputs"
            )
            text = stitched
        else:
            diagnostics.append(
                f"final answer validation failed: {note} and no usable worker "
                "outputs"
            )
            return FinalPresentation(
                kind=PresentationKind.DIAGNOSTIC,
                text=_MALFORMED_FINAL_MESSAGE,
                diagnostics=tuple(diagnostics),
            )

    # 3) The system's own control JSON (plan/verdict echo) is never an answer.
    if looks_like_orchestration_json(text):
        stitched = render_results_fallback(results) if results else ""
        if (
            stitched.strip()
            and stitched != "[No usable subtask outputs produced.]"
            and not looks_like_orchestration_json(stitched)
        ):
            diagnostics.append(
                "final answer was orchestration JSON; replaced with stitched "
                "worker outputs"
            )
            text = stitched
        else:
            diagnostics.append(
                "Final answer validation failed: the run produced internal "
                "orchestration JSON instead of a user answer."
            )
            return FinalPresentation(
                kind=PresentationKind.DIAGNOSTIC,
                text=_INTERNAL_DATA_MESSAGE,
                diagnostics=tuple(diagnostics),
            )

    # 4) Classify what remains. JSON text is a legitimate deliverable only
    #    when the request explicitly asked for JSON.
    if json_requested and _looks_like_bare_json(text):
        return FinalPresentation(
            kind=PresentationKind.JSON_DATA, text=text,
            diagnostics=tuple(diagnostics),
        )
    return FinalPresentation(
        kind=PresentationKind.PROSE, text=text, diagnostics=tuple(diagnostics)
    )
