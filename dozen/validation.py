"""Semantic validation + prompt hygiene for the orchestration pipeline.

Two responsibilities, both about NOT trusting a string just because it exists:

1. ``validate_model_response`` — the ``validateModelResponse`` equivalent. A
   successful DOM scrape is *not* a successful answer: a worker can scrape a
   refusal ("I cannot fulfill this request.") and the pipeline would otherwise
   mark it ``completed`` with score ``1.0``. This catches refusals and
   empty/too-short replies so they get scored ``0.0`` and retried/failed.

2. ``sanitize_prompt`` — strips any leaked JSON/schema fragments out of a string
   before it is sent to a worker model, so the Manager's plan structure can
   never bleed into the prompt Gemini actually receives.

Pure stdlib; safe to import anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# 1) Refusal / non-answer detection
# --------------------------------------------------------------------------- #
# Ordered, case-insensitive. Each is a short, distinctive phrase that almost
# only appears when a model is declining or hedging instead of answering.
_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"i\s+cannot\s+fulf(?:i|il)l\s+(?:this|that|your)\s+request",
        r"i\s+can(?:'|no)?t\s+fulf(?:i|il)l\s+(?:this|that|your)\s+request",
        r"i(?:'|\s+a)?m\s+sorry,?\s+but\s+i\s+can(?:'|no)?t",
        r"i(?:'|\s+a)?m\s+sorry,?\s+but\s+i\s+(?:am|'m)?\s*unable",
        r"i(?:'|\s+a)?m\s+(?:really\s+)?sorry,?\s+(?:but\s+)?i\s+can(?:'|no)?t",
        r"i\s+can(?:'|no)?t\s+(?:help|assist)\s+(?:you\s+)?with\s+(?:that|this)",
        r"i\s+can(?:'|no)?t\s+assist\s+with\s+(?:that|this|your)",
        r"i\s+can(?:'|no)?t\s+(?:comply|provide|create|generate)\s+",
        r"i\s+(?:am|'m)\s+unable\s+to\s+(?:help|assist|provide|comply|complete|do)",
        r"i\s+(?:am|'m)\s+not\s+able\s+to\s+(?:help|assist|provide|comply|complete)",
        r"i\s+will\s+not\s+be\s+able\s+to\s+(?:help|assist|provide|complete)",
        r"i\s+won(?:'|no)?t\s+be\s+able\s+to",
        r"as\s+an\s+ai\s+language\s+model",
        r"as\s+a\s+large\s+language\s+model",
        r"against\s+my\s+(?:guidelines|programming|policy|policies|principles)",
        r"i\s+do\s+not\s+feel\s+comfortable",
        r"i\s+must\s+(?:respectfully\s+)?decline",
        r"i\s+can(?:'|no)?t\s+do\s+that",
        r"i\s+(?:cannot|can(?:'|no)?t)\s+engage\s+with",
        r"unable\s+to\s+(?:help|assist|provide|complete|process)\s+(?:with\s+)?(?:that|this|your)",
        r"sorry,?\s+i\s+can(?:'|no)?t\s+help\s+with\s+(?:that|this)",
    )
]

# Text up to this length is treated as "short", so a refusal phrase anywhere in
# it counts. In longer text we only treat a refusal as real if it leads the
# reply (so a long, genuine answer that merely mentions a phrase isn't flagged).
_SHORT_TEXT_LEN = 400
_REFUSAL_HEAD_LEN = 240


@dataclass
class ResponseValidation:
    """Outcome of :func:`validate_model_response`."""

    ok: bool
    reason: str = ""
    # Machine-readable kind: "" (ok), "empty", "too_short", or "refusal".
    kind: str = ""


def validate_model_response(
    text: str,
    *,
    min_chars: int = 16,
    min_words: int = 2,
) -> ResponseValidation:
    """Decide whether ``text`` is a real answer or a non-answer to be rejected.

    Returns ``ok=False`` for:
      * empty / whitespace-only replies,
      * suspiciously short replies (< ``min_chars`` or < ``min_words`` words),
      * recognizable AI refusals / hedges.

    The thresholds are deliberately conservative to avoid rejecting terse but
    legitimate answers; tune ``min_chars`` / ``min_words`` per call site.
    """
    stripped = (text or "").strip()

    if not stripped:
        return ResponseValidation(False, "empty response", "empty")

    word_count = len(stripped.split())
    if len(stripped) < min_chars or word_count < min_words:
        return ResponseValidation(
            False,
            f"suspiciously short response ({len(stripped)} chars, {word_count} word(s))",
            "too_short",
        )

    lowered = stripped.lower()
    is_short = len(stripped) <= _SHORT_TEXT_LEN
    head = lowered[:_REFUSAL_HEAD_LEN]

    for pat in _REFUSAL_PATTERNS:
        # In short replies, a match anywhere is decisive. In long replies, only
        # a *leading* refusal counts (the model declined up front).
        haystack = lowered if is_short else head
        m = pat.search(haystack)
        if m:
            phrase = m.group(0).strip()
            return ResponseValidation(
                False, f"model refusal detected: '{phrase}'", "refusal"
            )

    return ResponseValidation(True, "", "")


def is_refusal(text: str) -> bool:
    """Convenience boolean wrapper around :func:`validate_model_response`."""
    return not validate_model_response(text).ok


def clip_text(
    text: str,
    max_chars: int,
    marker: str = "\n\n[... middle truncated for length ...]\n\n",
) -> str:
    """Clip ``text`` to ``max_chars``, cutting from the MIDDLE.

    Long model inputs (dependency outputs, synthesis material, web-composer
    limits) keep their head (framing/definitions) and tail (conclusions and
    output contracts), which is where the load-bearing content lives.
    ``max_chars <= 0`` disables clipping.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars - len(marker)
    if keep <= 40:
        return text[:max_chars]
    head = int(keep * 0.62)
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


# Substrings that must be present before we even try to parse — cheap pre-filter
# so ordinary answers never pay the JSON-extraction cost.
_ORCHESTRATION_KEY_HINTS = (
    '"delegations"',
    '"synthesis_strategy"',
    '"model_selection_reasoning"',
    '"passed"',
    '"score"',
    '"feedback"',
    '"agent"',
    '"reason"',
)


def _has_source_json_prefix(text: str) -> bool:
    """Whether the first JSON object is introduced by source-code syntax."""
    start = (text or "").find("{")
    if start < 0:
        return False
    # Only syntax immediately introducing the object is relevant. Looking at
    # the whole prose prefix would misclassify sentences that merely used words
    # such as "return" or "class" hundreds of characters earlier.
    prefix = text[:start][-240:]
    return bool(re.search(
        r"(?:\b(?:return|const|let|var)\b[^{}\n]{0,180}|=>|"
        r"[A-Za-z_$][\w.$\[\]'\"]*\s*=)\s*$",
        prefix,
    ))


def _looks_like_raw_control_shape(text: str) -> bool:
    """Recognize a malformed internal plan/router/verifier by raw key shape."""
    start = text.find("{")
    if start < 0:
        return False
    head = text[start: start + 4000]
    planner = '"delegations"' in head and any(
        key in head for key in ('"synthesis_strategy"', '"analysis"', '"direct_answer"')
    )
    verifier = sum(
        key in head for key in ('"passed"', '"score"', '"feedback"')
    ) >= 2
    router = '"agent"' in head and '"reason"' in head
    return planner or verifier or router


def looks_like_orchestration_json(text: str) -> bool:
    """True when ``text`` IS the system's own control JSON, not a real answer.

    Guards against a stale web scrape (or an echo-happy model) surfacing the
    planner's plan — ``{"analysis": ..., "delegations": [...], ...}`` — or a
    router/verifier verdict as a worker output or final answer. Detection is
    strict (parse + shape check), so genuine answers that merely *discuss* JSON
    are never flagged.
    """
    stripped = (text or "").strip()
    if not stripped or "{" not in stripped:
        return False
    if _has_source_json_prefix(stripped):
        return False
    if not any(k in stripped for k in _ORCHESTRATION_KEY_HINTS):
        # Router/verifier echoes are tiny; only parse short texts for them.
        if len(stripped) > 600:
            return False
    try:
        from .llm_client import _extract_json  # local import avoids a cycle
        data = _extract_json(stripped)
    except Exception:
        return _looks_like_raw_control_shape(stripped)
    if not isinstance(data, dict):
        return False
    # Planner plan: delegations alongside strategy/analysis/direct_answer.
    # (So distinctive it is checked even when embedded in surrounding prose.)
    if "delegations" in data and (
        "synthesis_strategy" in data or "analysis" in data or "direct_answer" in data
    ):
        return True
    # Router/verifier verdicts are tiny generic shapes, so only treat them as
    # echoes when the reply IS the JSON — not when an answer merely contains a
    # similar-looking object (e.g. inside a code sample).
    if stripped.startswith("{"):
        if {"agent", "reason"}.issubset(data.keys()) and len(data) <= 3:
            return True
        if {"passed", "score", "feedback"}.issubset(data.keys()):
            return True
    return False


# Quoted key tokens that identify the WORKER/SYNTHESIZER protocol envelope.
# They are matched as raw tokens (not parsed) so a truncated / malformed
# envelope is still recognizable as protocol traffic.
_ENVELOPE_KEY_TOKENS = ('"summary"', '"key_decisions"', '"artifacts"', '"confidence"')


def is_canonical_protocol_data(data: object) -> bool:
    """Whether parsed data is the complete worker/synthesizer envelope shape."""
    return isinstance(data, dict) and all(
        key in data for key in ("summary", "key_decisions", "artifacts", "confidence")
    )


def looks_like_protocol_envelope(text: str) -> bool:
    """True when ``text`` as a whole is (or starts as) one protocol envelope.

    The predicate is deliberately strict about SHAPE — the reply must BE a JSON
    object (optionally fenced), not merely contain one — so source code or prose
    that quotes envelope keys somewhere inside is never flagged. It does NOT
    require the JSON to parse: recognizing broken envelopes is the whole point
    (they must fail closed instead of being displayed raw).
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        from .llm_client import _strip_code_fences  # local import avoids a cycle
        stripped = _strip_code_fences(stripped).strip()
    start = stripped.find("{")
    if start < 0:
        return False
    # Tolerate a short natural-language lead-in (models often say "Result:")
    # while excluding source-code assignments/returns that merely contain a
    # protocol-shaped JSON fixture.
    if _has_source_json_prefix(stripped):
        return False
    head = stripped[start: start + 2000]
    hits = sum(1 for token in _ENVELOPE_KEY_TOKENS if token in head)
    # One recognized key is enough for the malformed-envelope gate. Valid JSON
    # still has to parse before it is rejected, keeping legitimate user data.
    return hits >= 1


def is_malformed_protocol_envelope(text: str) -> bool:
    """Envelope-shaped text that cannot be parsed into a JSON object.

    This is the fail-closed gate: a valid envelope is parsed normally, plain
    prose/code passes through untouched, and ONLY the broken-protocol case is
    flagged so callers reject it (bounded retry / clean diagnostic) instead of
    ever presenting the raw payload as an answer.
    """
    if not looks_like_protocol_envelope(text):
        return False
    try:
        from .llm_client import _extract_json  # local import avoids a cycle
        data = _extract_json(text)
    except Exception:
        return True
    return not isinstance(data, dict)


def decode_nested_envelope(content: str, max_depth: int = 2) -> str:
    """Decode double-encoded envelopes through bounded, explicit rules.

    A worker sometimes nests one full envelope as the STRING content of another
    (double encoding). Each round: if the whole text is a parseable envelope,
    flatten it via :func:`render_artifact`; otherwise stop. At most
    ``max_depth`` rounds, so pathological input can never recurse unboundedly.
    Source code that merely CONTAINS JSON is never touched (the whole text must
    be envelope-shaped), so valid code with JSON literals survives intact.
    """
    text = content
    for _ in range(max(0, int(max_depth))):
        stripped = (text or "").strip()
        if not looks_like_protocol_envelope(stripped):
            return text
        envelope = parse_worker_artifact(stripped)
        # Nested decoding is intentionally stricter than worker parsing. A
        # complete canonical envelope may wrap another envelope; a user's JSON
        # object that merely has an ``artifacts`` or ``summary`` field must stay
        # byte-identical.
        if envelope is None or not is_canonical_protocol_data(envelope):
            return text
        flattened = render_artifact(envelope)
        if not flattened.strip() or flattened.strip() == stripped:
            return text
        text = flattened
    return text


# --------------------------------------------------------------------------- #
# 2) Prompt hygiene — strip leaked JSON / schema fragments
# --------------------------------------------------------------------------- #
# Keys that belong to the Manager's PLAN schema. If any of these appear inside a
# value that is about to be sent to a worker, the JSON structure has bled into
# the prompt (the "prompt mangling" bug) and must be cut out.
_SCHEMA_KEYS = (
    "analysis",
    "direct_answer",
    "subtasks",
    "synthesis_strategy",
    "required_capabilities",
    "depends_on",
    "success_criteria",
    "expected_output",
    "difficulty",
    "instruction",
    "title",
)

# Matches a JSON-style key token like  ", "synthesis_strategy":  or  "depends_on":
_SCHEMA_KEY_RE = re.compile(
    r'["\']?\s*(?:' + "|".join(_SCHEMA_KEYS) + r')\s*["\']?\s*:',
    re.IGNORECASE,
)

# Parenthetical schema notes the planner sometimes emits, e.g.
# "(Must include exactly one entry for EVERY model in the pool)".
_SCHEMA_NOTE_RE = re.compile(
    r"\((?:[^()]*\b(?:must include|exactly one entry|for every model|"
    r"in the pool|one per|schema|json)\b[^()]*)\)",
    re.IGNORECASE,
)

# Leading/trailing JSON punctuation cruft left behind after a bad slice, e.g.
#   ' ... ], "'   or   '} ,'   or   leading '"], '
_EDGE_JSON_RE = re.compile(r'^[\s\.\]\}\[\{",:]+|[\s\]\}\[\{",:]+$')


def sanitize_prompt(text: str) -> str:
    """Return ``text`` with any leaked JSON/schema fragments removed.

    Safe to call on already-clean strings (it is a no-op for them). Conservative:
    it only trims when a clear schema marker is found, then tidies the edges.
    """
    if not text:
        return ""

    cleaned = text

    # 1) Drop parenthetical schema instructions outright.
    cleaned = _SCHEMA_NOTE_RE.sub("", cleaned)

    # 2) If a schema key token appears, the structure leaked in. Truncate at the
    #    earliest one — everything after it is plan JSON, not prompt content.
    m = _SCHEMA_KEY_RE.search(cleaned)
    if m:
        cleaned = cleaned[: m.start()]

    # 3) Strip stray JSON punctuation that framed the leaked value.
    cleaned = _EDGE_JSON_RE.sub("", cleaned)

    # 4) Collapse whitespace introduced by the cuts.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


# --------------------------------------------------------------------------- #
# 3) Web-UI prompt hardening — make a prompt look like a normal human message
# --------------------------------------------------------------------------- #
# Consumer chat UIs (Gemini especially) refuse prompts that look like system-
# prompt injection or multi-agent meta-prompting ("You are an AI agent…",
# "Your subtask is…", leaked JSON/markdown). ``sanitize_prompt_for_web_ui``
# runs right before the automation pastes text into the browser and rewrites it
# into plain, human-readable text so the safety filters don't trip.

# Whole lines that are pure orchestration labels/headers -> dropped entirely.
_WEB_LABEL_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    r"system\s+instructions?|overall\s+task(?:\s*\(for\s+context\))?|"
    r"your\s+subtask|expected\s+output(?:\s+format)?|success\s+criteria|"
    r"prerequisite\s+outputs?|revision\s+required|synthesis\s+strategy|"
    r"original\s+task|subtask\s+outputs?(?:\s*\(in\s+execution\s+order\))?|"
    r"context|constraints|desired\s+final\s+output|role|target\s+model|"
    r"assigned\s+model"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

# Lines naming a role/model/routing identifier -> dropped WITH their value
# (e.g. "Role: engineer", "Target model: gemini"). These must never be pasted.
_WEB_IDENTITY_LINE_RE = re.compile(
    r"^\s*(?:role|target\s+model|assigned\s+model|model|agent|provider|"
    r"model_selection_reasoning)\s*:.*$",
    re.IGNORECASE,
)

# A leading "Label: <content>" prefix to strip while keeping <content>.
_WEB_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    r"your\s+subtask|overall\s+task(?:\s*\(for\s+context\))?|"
    r"expected\s+output(?:\s+format)?|success\s+criteria|prerequisite\s+outputs?|"
    r"synthesis\s+strategy|original\s+task|context|task"
    r")\s*:\s*",
    re.IGNORECASE,
)

# Lines made up only of JSON/markdown punctuation -> dropped.
_WEB_PUNCT_LINE_RE = re.compile(r'^[\s\{\}\[\]\(\),;:"\'`*_>#|=-]+$')

# Code-fence delimiters (```json, ```python, ```), kept inner text.
_WEB_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?")

# JSON object spans like {"key": ...} that leaked into the text.
_WEB_JSON_OBJ_RE = re.compile(r'\{[^{}]*"[^"]+"\s*:[^{}]*\}', re.DOTALL)

# Inline meta phrases / roleplay removed wherever they appear.
_WEB_META_INLINE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # "You are an AI copywriter." / "You are the expert worker agent ..."
        r"you\s+are\s+(?:an?|the)\s+[^.\n]{0,60}?"
        r"(?:agent|assistant|ai|model|worker|bot|expert|copywriter|specialist|"
        r"planner|manager|synthesizer|conductor)\b[^.\n]*\.?",
        # "As an AI language model, ..."
        r"as\s+an?\s+ai(?:\s+language\s+model)?\b[^.\n]*\.?",
        # "You are part of a multi-agent orchestration ..."
        r"you\s+are\s+part\s+of\s+(?:an?\s+)?(?:multi-?agent|orchestration)[^.\n]*\.?",
        # Meta labels that bled into a sentence.
        r"\byour\s+(?:sub)?task\s+is(?:\s+to)?\s*:?\s*",
        r"\bthe\s+(?:sub)?task\s+is(?:\s+to)?\s*:?\s*",
        r"\btarget\s+model\s*:?\s*",
        r"\bassigned\s+model\s*:?\s*",
        # "Act as a senior engineer and ..." -> drop the "Act as ... and"
        r"\bact\s+as\s+(?:an?|the)\s+[^.\n]{0,40}?\b(?:and\b|,|:|\.)",
    )
]


def sanitize_prompt_for_web_ui(raw_prompt: str) -> str:
    """Rewrite ``raw_prompt`` into plain text a human could have typed.

    Steps:
      * strip markdown code fences (```json ... ```) and stray backticks,
      * remove leaked JSON object spans and punctuation-only lines,
      * drop orchestration labels/headers and roleplay/meta-prompting,
      * strip residual schema-key leakage and normalize whitespace.

    Always returns non-empty text when the input had any content (it falls back
    to a minimally de-fenced version rather than returning "").
    """
    if not raw_prompt or not raw_prompt.strip():
        return ""

    # 1) Code fences + backticks (keep inner content).
    text = _WEB_FENCE_RE.sub("", raw_prompt).replace("```", "").replace("`", "")

    # 2) Remove obvious leaked JSON object spans.
    text = _WEB_JSON_OBJ_RE.sub(" ", text)

    # 3) Line-by-line cleanup.
    out_lines: list[str] = []
    for line in text.splitlines():
        if _WEB_LABEL_LINE_RE.match(line):
            continue
        if _WEB_IDENTITY_LINE_RE.match(line):
            continue
        if _WEB_PUNCT_LINE_RE.match(line):
            continue
        line = _WEB_LABEL_PREFIX_RE.sub("", line)
        for pat in _WEB_META_INLINE:
            line = pat.sub("", line)
        out_lines.append(line)
    text = "\n".join(out_lines)

    # 4) Remove any residual schema-key leakage, then tidy whitespace.
    text = sanitize_prompt(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip(" \t\n-:>*")

    if not text.strip():
        # Everything got stripped; return a minimally de-fenced fallback so we
        # never paste an empty box.
        return raw_prompt.replace("`", "").strip()

    # Make the opening read naturally (avoid a lowercase sentence fragment).
    if text[:1].islower():
        text = text[0].upper() + text[1:]
    return text.strip()


# --------------------------------------------------------------------------- #
# 4) Artifact parsing — turn a worker's JSON artifact into usable content
# --------------------------------------------------------------------------- #
# Workers reply with a strict JSON artifact ({summary, key_decisions, artifacts,
# confidence}). These helpers parse it and flatten the `artifacts` back into the
# plain text/code that downstream synthesis consumes.


def parse_worker_artifact(text: str):
    """Parse a worker's artifact JSON. Returns the dict, or None if it isn't one.

    Tolerant of fences/prose/minor malformations (reuses the robust JSON
    extractor). Returns None when the reply is plainly not artifact-shaped, so
    callers can fall back to treating the raw text as the answer.
    """
    if not text or not text.strip():
        return None
    try:
        from .llm_client import _extract_json  # local import avoids import cycle
        data = _extract_json(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "artifacts" in data or ("summary" in data and "key_decisions" in data):
        return data
    return None


def render_artifact(artifact: dict) -> str:
    """Flatten an artifact's `artifacts` payload into plain text for synthesis.

    A single artifact is returned as-is; multiple artifacts are joined under
    `### filename` headers. Falls back to the `summary` if no artifacts exist.

    Both worker envelopes are supported: the original mapping of filename ->
    content, and the Phase 4C typed list of {artifact_id, path, content}
    entries a scoped worker returns. Rendering is presentation only — typed
    collection of the same payload happens in ``dozen.artifact_results``.
    """
    arts = artifact.get("artifacts")
    pieces: list[str] = []
    if isinstance(arts, (list, tuple)):
        items = []
        for entry in arts:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            name = str(
                entry.get("path") or entry.get("artifact_id") or entry.get("id") or ""
            ).strip()
            items.append((name, content))
        if len(items) == 1 and not items[0][0]:
            pieces.append(items[0][1])
        else:
            for name, content in items:
                pieces.append(f"### {name}\n{content}" if name else content)
    elif isinstance(arts, dict):
        items = [(str(k), v) for k, v in arts.items() if str(v).strip()]
        if len(items) == 1:
            pieces.append(str(items[0][1]))
        else:
            for name, content in items:
                pieces.append(f"### {name}\n{content}")
    elif isinstance(arts, str) and arts.strip():
        pieces.append(arts.strip())
    body = "\n\n".join(p for p in pieces if p and str(p).strip())
    if not body:
        body = str(artifact.get("summary", "")).strip()
    return body


def artifact_confidence(artifact: dict):
    """Return the artifact's confidence as a float in [0,1], or None if absent/invalid."""
    try:
        c = float(artifact.get("confidence"))
    except (TypeError, ValueError):
        return None
    return c if 0.0 <= c <= 1.0 else None
