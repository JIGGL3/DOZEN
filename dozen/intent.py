"""Request intent and the deliverable contract (production-hardening Phase 3).

WHY THIS EXISTS
---------------
``Task`` carried only free text (prompt/context/constraints/desired_output), so
nothing in the pipeline knew what KIND of deliverable a request demanded. Asked
to "build me a React dashboard", the planner was free to decompose into
"describe the architecture" / "recommend libraries", workers produced prose, and
every existing guard (orchestration-JSON, artifact-envelope, refusal) passed it
— because those guards check FORMAT, never MODE. An implementation request could
silently become an architecture essay ending in "No application code is
included."

This module is the ONE authoritative place that answers: what is being asked
for, and what would count as an invalid substitution. It is pure, deterministic,
does no I/O and makes no provider call. Every other component (planner, worker,
verifier, synthesizer, final guard) CONSUMES the resolved contract instead of
re-interpreting the user's wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class RequestIntent(str, Enum):
    IMPLEMENT = "implement"          # new working implementation
    MODIFY = "modify"                # change an existing codebase
    DEBUG = "debug"                  # diagnose + fix a defect
    REVIEW = "review"                # analyze without changing
    ARCHITECTURE = "architecture"    # design/plan/spec, explicitly not code
    EXPLAIN = "explain"              # teaching / conceptual guidance
    RESEARCH = "research"            # investigate / compare / gather evidence
    CONTENT = "content"              # non-code writing artifact
    UNKNOWN = "unknown"              # unclassifiable — safe, permissive defaults


# --------------------------------------------------------------------------- #
# Signal vocabulary
#
# Deliberately verb-anchored: nouns alone ("code", "architecture") are terrible
# classifiers — "review this code" is not an implementation request and "review
# the architecture" is not an architecture request. Every pattern below is
# matched with word boundaries against the lowercased request text.
# --------------------------------------------------------------------------- #

# Objects that make a generic verb ("write", "make") a CODE request.
_CODE_OBJECT = (
    r"(?:code|script|program|app|application|website|web ?app|dashboard|api|"
    r"endpoint|service|server|backend|frontend|ui|cli|tool|library|package|"
    r"module|component|function|class|method|test|tests|suite|feature|page|"
    r"form|game|bot|scraper|parser|pipeline|migration|schema|query|algorithm|"
    r"implementation|prototype|mvp|repo|repository|project|integration|"
    r"authentication|authorization|generator|configuration|config|terraform|"
    r"regular expression|regex|source files?)"
)

# Objects that make a generic verb ("write", "draft") a CONTENT request.
_PROSE_OBJECT = (
    r"(?:blog ?post|article|essay|copy|email|newsletter|readme|documentation|"
    r"docs|report|summary|story|script for|press release|proposal|memo|"
    r"user guide|tutorial|explanation|examples?|announcement|caption|tagline|"
    r"slogan)"
)

# Greenfield construction verbs.
#
# STRONG verbs stand alone — but only as VERBS: the negative lookbehinds stop
# "the build pipeline" / "a design" (noun phrases) from reading as commands.
_IMPLEMENT_STRONG = (
    r"(?<!the )(?<!a )(?<!an )(?<!our )(?<!my )(?<!this )(?<!that )"
    r"\b(?:build|create|implement|develop|scaffold|bootstrap|code up)\b",
    r"\bset up\b",
    r"\b(?:spin|stand) up\b",
)
# GENERIC verbs only count when a code object is nearby — so "write about
# databases" is not an implementation request (a real regression this guards).
_IMPLEMENT_GENERIC = (
    rf"\b(?:write|make|generate|produce|return|deliver|give me|need|want)\b"
    rf"[^.?!]{{0,40}}\b{_CODE_OBJECT}\b",
)
_IMPLEMENT_PATTERNS = _IMPLEMENT_STRONG + _IMPLEMENT_GENERIC

# Change-an-existing-thing verbs.
_MODIFY_PATTERNS = (
    r"\b(?:modify|change|update|refactor|extend|upgrade|migrate|rename|"
    r"replace|remove|delete|port|convert|rewrite|improve|optimi[sz]e)\b",
    r"\badd\b",
    r"\bintegrate\b",
)

# Evidence that a codebase already exists (turns "add X" into MODIFY rather than
# part of a greenfield build).
_EXISTING_CODE_PATTERNS = (
    r"\b(?:existing|current|already) (?:code|file|function|endpoint|module|"
    r"class|repo|repository|project|component|script|app|application|"
    r"implementation|method|service|codebase|code ?base)\b",
    r"\bthis (?:code|file|function|endpoint|module|class|repo|repository|"
    r"project|component|script|app|implementation|method|service)\b",
    r"\b(?:the|our|my) (?:repo|repository|codebase|code ?base|project|"
    r"existing code)\b",
    r"\blegacy\b",
)

_DEBUG_ACTION_PATTERNS = (
    r"\b(?:fix|debug|repair|troubleshoot|diagnose)\b",
    r"\bfind (?:out )?(?:why|the (?:cause|root cause))\b",
)

_DEBUG_PATTERNS = _DEBUG_ACTION_PATTERNS + (
    r"\b(?:bug|crash|crashes|crashing|defect|regression|race condition|"
    r"memory leak|deadlock|traceback|stack ?trace)\b",
    r"\b(?:not working|doesn'?t work|does not work|isn'?t working|is broken|"
    r"are broken|fails|failing|throws|erroring|errors out)\b",
    r"\bwhy (?:does|is|are|do|did|isn'?t|doesn'?t).{0,40}\b(?:fail|crash|break|"
    r"error|wrong|not work)",
)

_REVIEW_PATTERNS = (
    r"\b(?:review|audit|critique|assess|appraise|inspect)\b",
    r"\bcode review\b",
    r"\blook over\b",
    r"\bwhat'?s wrong with\b",
)

_ARCHITECTURE_PATTERNS = (
    r"\b(?:architect|architecture|architectural)\b",
    r"\bdesign\b",
    r"\b(?:sadd|adr|rfc|tech(?:nical)? spec|specification|system design|"
    r"high[- ]level design|implementation plan|design doc(?:ument)?|blueprint)\b",
    r"\bplan (?:for|out|the)\b",
)

_EXPLAIN_PATTERNS = (
    r"\b(?:explain|teach|clarify|elaborate|discuss|outline)\b",
    r"\b(?:show|tell) me how\b",
    r"\bwalk me through\b",
    r"\bhow (?:does|do|did|is|are|can|should|would)\b",
    r"\bwhat (?:is|are|does|do)\b",
    r"\bwhy (?:is|are|does|do)\b",
    r"\bhelp me understand\b",
    r"\bdifference between\b",
)

_RESEARCH_PATTERNS = (
    r"\b(?:research|investigate|compare|benchmark|survey|evaluate)\b",
    r"\bpros and cons\b",
    r"\b(?:which|what) (?:is|are) (?:better|best|the best)\b",
    r"\brecommend\b",
    r"\bfind (?:out|information|sources|evidence)\b",
)

_CONTENT_PATTERNS = (
    rf"\b(?:write|draft|compose|create|make|produce|generate|add|edit|revise|improve|"
    rf"update|change|rewrite|fix|document)\b[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
    r"\bdocument\b[^.?!]{0,40}\b(?:api|endpoint|function|class|module|"
    r"service|component)\b",
    r"\bwrite about\b",
    r"\bmake\b[^.?!]{0,30}\bexplanation\b[^.?!]{0,20}\bclearer\b",
)

_CONTENT_ACTION_WORD = (
    r"(?:write|draft|compose|create|make|produce|generate|add|edit|revise|"
    r"improve|update|change|rewrite|fix|document)"
)
_CONTENT_NON_ACTION_PATTERNS = (
    rf"\b(?:explain|teach|discuss|outline|review|show(?: me)?|tell(?: me)?|"
    rf"walk me through)\b[^.?!]{{0,60}}\bhow"
    rf"(?:\s+to|\s+(?:one|someone|we|you)\s+"
    rf"(?:might|could|would|should))\b[^.?!]{{0,30}}\b"
    rf"{_CONTENT_ACTION_WORD}\b[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
    rf"\bhow\s+(?:do|can|should|would)\s+(?:i|we|you)\s+"
    rf"{_CONTENT_ACTION_WORD}\b[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
    rf"\b(?:should\s+(?:we|i|you)|do\s+(?:i|we|you)\s+need\s+to)\s+"
    rf"{_CONTENT_ACTION_WORD}\b[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
    rf"\breview\b[^.?!]{{0,50}}\bwhether\s+(?:we|i|you)\s+should\s+"
    rf"{_CONTENT_ACTION_WORD}\b[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
)

_GENERATOR_REQUEST = (
    rf"\b(?:build|create|implement|develop|write|make|generate|produce)\b"
    rf"[^.?!]{{0,45}}\b{_PROSE_OBJECT}\b[^.?!]{{0,20}}\b(?:generator|tool|"
    r"script|application|app|service)\b",
    r"\b(?:build|create|implement|develop|write|make|generate|produce)\b"
    r"[^.?!]{0,45}\b(?:sadd|adr|rfc|blueprint|architecture(?: diagram)?|"
    r"design doc(?:ument)?|implementation plan|deployment plan|specification|"
    r"spec|plan)\b[^.?!]{0,20}\b(?:generator|tool|script|application|app|"
    r"service)\b",
)

_CONTENT_EDIT_REQUEST = (
    rf"\b(?:add|edit|revise|improve|update|change|rewrite|make|fix)\b"
    rf"[^.?!]{{0,40}}\b{_PROSE_OBJECT}\b",
)

# A construction verb whose OBJECT is a design document, not software.
_DESIGN_ARTIFACT_REQUEST = (
    r"\b(?:build|create|implement|develop|write|make|give me|need|want|produce|"
    r"draft|prepare)\b[^.?!]{0,30}\b(?:sadd|adr|rfc|blueprint|architecture|"
    r"design doc(?:ument)?|implementation plan|plan|spec|specification)\b",
)

# --------------------------------------------------------------------------- #
# Explicit output constraints — these are USER INSTRUCTIONS and outrank every
# inferred default (Part B: explicit-instruction precedence).
# --------------------------------------------------------------------------- #
_NO_CODE_PATTERNS = (
    r"\b(?:don'?t|do not|no need to|without) (?:write|writing|produce|"
    r"include|including|generate) (?:any )?(?:source )?code\b"
    r"(?!\s+comments?\b)",
    r"\bno (?:code|implementation|source code)\b(?!\s+comments?\b)",
    r"\b(?:architecture|design|plan|spec(?:ification)?) (?:only|alone)\b",
    r"\bonly (?:the |a |an )?(?:architecture|design|plan|spec(?:ification)?)\b",
    r"\bjust (?:the |a |an )?(?:architecture|design|plan|spec(?:ification)?)\b",
    r"\bwithout (?:any )?(?:code|implementation)\b(?!\s+comments?\b)",
    r"\bwithout (?:actually )?(?:implementing|applying) (?:it|the fix|"
    r"the change|the changes)\b",
    r"\bdon'?t implement\b",
    r"\bdo not implement\b",
)

_NO_CHANGE_PATTERNS = (
    r"\b(?:don'?t|do not) (?:change|modify|edit|touch|alter|update) "
    r"(?:anything|it|this|the (?:code|file|module|repo|repository|project|"
    r"implementation)|existing code)\b",
    r"\bwithout (?:changing|modifying|editing|altering) "
    r"(?:anything|it|the (?:code|file|module|repo|repository|project|"
    r"implementation)|existing code)\b",
    r"\b(?:no|zero) (?:changes|modifications|edits)\b(?!\s+to\b)",
    r"\bread[- ]only\b",
    r"\bleave (?:it|everything|the code) (?:as is|alone|unchanged)\b",
)

# Negated actions must not become positive intent signals. These are deliberately
# separate from output constraints: "do not build X; explain Y" negates an
# action, but does not necessarily establish an architecture-only deliverable.
_NEGATED_ACTION_PATTERNS = (
    r"\bdo not\s*,\s*under any circumstances\s*,?\s*"
    r"(?=[^,;.?!]*\b(?:build|create|implement|develop|scaffold|bootstrap|"
    r"write|draft|compose|make|generate|produce|modify|change|update|edit|"
    r"revise|improve|rewrite|document|refactor|add|integrate|fix|debug|"
    r"repair|design)\b)[^,;.?!]*",
    r"\b(?:there is )?no need to\b"
    r"(?=[^,;.?!]*\b(?:build|create|implement|develop|scaffold|bootstrap|"
    r"write|draft|compose|make|generate|produce|modify|change|update|edit|"
    r"revise|improve|rewrite|document|refactor|add|integrate|fix|debug|"
    r"repair|design)\b)[^,;.?!]*",
    r"\b(?:don'?t|do not|never|under no circumstances)\b"
    r"(?=[^,;.?!]*\b(?:build|create|implement|develop|scaffold|bootstrap|"
    r"write|draft|compose|make|generate|produce|modify|change|update|edit|"
    r"revise|improve|rewrite|document|refactor|add|integrate|fix|debug|"
    r"repair|design)\b)[^,;.?!]*",
    r"\b(?:don'?t|do not) want (?:(?:me|us|you|the system)\s+)?to "
    r"(?:build|create|implement|develop|write|make|generate|produce|modify|"
    r"change|fix|design)\b",
)

_NON_ACTION_WORD = (
    r"(?:build|create|implement|develop|scaffold|bootstrap|write|draft|"
    r"compose|make|generate|produce|modify|change|update|edit|revise|improve|"
    r"rewrite|document|refactor|add|integrate|fix|debug|repair|design)"
)
_NON_ACTION_PATTERNS = (
    # Questions/hypotheticals about the speaker or a third party are not
    # requests to execute.  Deliberately exclude "you": "Could you create X?"
    # is a polite imperative and must remain actionable.
    rf"\b(?:(?:can|could|may|might|should|would)\s+"
    rf"(?:i|we|they|he|she|it)|(?:i|we|they|he|she|it)\s+"
    rf"(?:can|could|may|might|would))\s+{_NON_ACTION_WORD}\b[^,;.?!]*",
    rf"\bdoes\s+(?:this|that|the)\s+(?:tool|system|app|application|service|"
    rf"script|program|code|feature)\s+{_NON_ACTION_WORD}\b[^,;.?!]*",
    rf"\b(?:tell me|explain|describe|discuss)\b[^.?!]{{0,24}}\bwhether\b"
    rf"[^.?!]{{0,36}}\b(?:can|could|may|might|should|would)\s+"
    rf"{_NON_ACTION_WORD}\b[^,;.?!]*",
    rf"\b(?:(?:you|we|they|i)\s+must\s+not|you\s+are\s+not\s+to|"
    rf"(?:(?:my|our|the)\s+)?goal\s+is\s+not\s+to|do\s+anything\s+but)\s+"
    rf"{_NON_ACTION_WORD}\b[^,;.?!]*",
)

_TESTS_REQUIRED_PATTERNS = (
    r"\b(?:with|including|include|add|plus|and) (?:unit |integration |"
    r"regression )?tests?\b",
    r"\btest(?:ed|s)? (?:coverage|suite)\b",
    r"\bwrite tests?\b",
)

_NO_TESTS_PATTERNS = (
    r"\b(?:no|without|skip|don'?t (?:write|add)|do not (?:write|add)) "
    r"(?:unit |integration )?tests?\b",
)

# Phase 4F (Part G): an explicit code-only OUTPUT constraint. The exact live
# failure "(give code only)" must bind the trusted contract: the deliverable is
# the source code itself, explanatory prose is not an acceptable component of a
# successful answer, and the artifact pipeline becomes mandatory downstream.
_CODE_ONLY_PATTERNS = (
    r"\bcode\s+only\b",
    r"\bcode\s+onli\b",
    r"(?:^|[(:;.!?]\s)only\s+(?:the\s+)?(?:source\s+)?code\b",
    r"\b(?:complete\s+)?implementation\s+only\b",
    # "only the code" counts when it is what the response should carry — not
    # noun phrases such as "the only code that changed".
    r"\b(?:give|output|return|provide|send|show|write|produce|deliver|want|"
    r"need|respond\s+with|reply\s+with)\b[^.?!]{0,40}\bonly\s+(?:the\s+)?"
    r"(?:source\s+)?code\b",
    r"\bjust\s+(?:the\s+)?(?:source\s+)?code\b(?!\s+review)",
    r"\bnothing\s+but\s+(?:the\s+)?(?:source\s+)?code\b",
    r"\bcode\s+and\s+nothing\s+else\b",
    r"\bno\s+(?:prose|explanations?|commentary)\b[^.?!]{0,20}\b(?:just|only)\s+"
    r"(?:the\s+)?code\b",
    r"\b(?:no|without)\s+(?:explanations?|commentary|prose)\b[^.?!]{0,40}"
    r"\b(?:just|only)\s+(?:the\s+)?source\s+files?\b",
)

# These phrases use "only" to scope a review/explanation, not to demand that
# source code be the response format. Blank them before applying the explicit
# output detector so the root integrity guard cannot promote a valid prose task.
_NON_DELIVERY_CODE_ONLY_PATTERNS = (
    r"\b(?:review|audit|inspect|analy[sz]e|explain|describe|discuss|"
    r"summari[sz]e)\s+(?:the\s+)?code\s+only\b",
)

# Quoted examples are data, not instructions. These match paired spans only,
# so apostrophes in contractions do not hide real commands.
_QUOTED_INSTRUCTION_PATTERNS = (
    r'"[^"\n]*"',
    r"(?<!\w)'[^'\n]*'(?!\w)",
    r"`[^`\n]*`",
    "\u201c[^\u201d\n]*\u201d",
    "\u2018[^\u2019\n]*\u2019",
)

_NO_COMMENTARY_PATTERNS = (
    r"\b(?:no|without)\s+(?:explanations?|commentary|prose)\b",
    r"\b(?:do not|don'?t)\s+(?:include|add|provide|write)\s+"
    r"(?:any\s+)?(?:explanations?|commentary|prose)\b",
)

# Intent strength order. An action-oriented requirement is NEVER discarded in
# favor of a weaker descriptive one (Part B): a request that both explains and
# fixes is a DEBUG request; one that both designs and implements is IMPLEMENT.
_STRENGTH: tuple[RequestIntent, ...] = (
    RequestIntent.DEBUG,
    RequestIntent.MODIFY,
    RequestIntent.IMPLEMENT,
    RequestIntent.REVIEW,
    RequestIntent.ARCHITECTURE,
    RequestIntent.CONTENT,
    RequestIntent.RESEARCH,
    RequestIntent.EXPLAIN,
)

_CODE_INTENTS = (RequestIntent.IMPLEMENT, RequestIntent.MODIFY, RequestIntent.DEBUG)

_MAX_BRIEF_ITEMS = 4          # keeps the prompt block bounded (Part G)
_MAX_ITEM_CHARS = 120
_MAX_DELIVERABLE_CHARS = 240
_MAX_SHORT_LINE_CHARS = 360


def _bounded_text(value: Any, limit: int) -> str:
    """Collapse message-structure characters and clip one rendered field."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _string_tuple(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _bool_value(value: Any, default: bool) -> bool:
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


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    """Every pattern that fires, returned as evidence."""
    hits: list[str] = []
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            hits.append(found.group(0).strip())
    return hits


def _mask_quoted_instruction_text(text: str) -> str:
    """Remove quoted/example spans from the explicit-instruction surface."""
    masked = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    for pattern in _QUOTED_INSTRUCTION_PATTERNS:
        masked = re.sub(pattern, " ", masked)
    return masked


def _explicit_code_only_hits(text: str) -> list[str]:
    """One shared strong code-only signal for resolution and root guarding."""
    signal_text = _mask_quoted_instruction_text(text)
    for pattern in _NON_DELIVERY_CODE_ONLY_PATTERNS:
        signal_text = re.sub(pattern, " ", signal_text)
    return _matches(_CODE_ONLY_PATTERNS, signal_text)


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeliverableContract:
    """What DOZEN owes the user for this request. Immutable and serializable."""

    intent: RequestIntent = RequestIntent.UNKNOWN
    # ``None`` is an internal constructor sentinel. ``__post_init__`` replaces
    # it with the selected intent's safe default, while an explicit False/empty
    # value remains explicit and is never overwritten.
    deliverable: str = None  # type: ignore[assignment]
    code_required: bool = None  # type: ignore[assignment]
    repo_changes_required: bool = None  # type: ignore[assignment]
    tests_required: bool = None  # type: ignore[assignment]
    architecture_only_allowed: bool = None  # type: ignore[assignment]
    explanation_only_allowed: bool = None  # type: ignore[assignment]
    completion_criteria: tuple[str, ...] = None  # type: ignore[assignment]
    user_constraints: tuple[str, ...] = None  # type: ignore[assignment]
    rationale: str = None  # type: ignore[assignment]
    # Phase 4F (Part G): the user explicitly demanded code as the ONLY visible
    # deliverable ("give code only"). Plain default (never a sentinel) so every
    # existing construction site keeps its exact prior behavior.
    code_only: bool = False

    def __post_init__(self) -> None:
        """Canonicalize public inputs so frozen instances are deeply immutable."""
        if not isinstance(self.intent, RequestIntent):
            try:
                object.__setattr__(self, "intent", RequestIntent(str(self.intent)))
            except ValueError:
                object.__setattr__(self, "intent", RequestIntent.UNKNOWN)
        mode = _defaults(self.intent)
        generic = {
            "deliverable": "a direct, complete answer to the request",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (),
            "rationale": (
                "no intent signals detected; permissive defaults applied"
                if self.intent is RequestIntent.UNKNOWN
                else f"explicit {self.intent.value} contract; safe intent defaults applied"
            ),
        }
        for name, fallback in generic.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, mode.get(name, fallback))
        object.__setattr__(self, "deliverable", str(self.deliverable))
        object.__setattr__(self, "rationale", str(self.rationale))
        for name in (
            "code_required",
            "repo_changes_required",
            "tests_required",
            "architecture_only_allowed",
            "explanation_only_allowed",
        ):
            object.__setattr__(
                self,
                name,
                _bool_value(getattr(self, name), bool(mode.get(name, generic[name]))),
            )
        object.__setattr__(
            self,
            "completion_criteria",
            _string_tuple(
                self.completion_criteria, mode.get("completion_criteria", ())
            ),
        )
        object.__setattr__(
            self, "user_constraints", _string_tuple(self.user_constraints)
        )
        object.__setattr__(self, "code_only", _bool_value(self.code_only, False))

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "deliverable": self.deliverable,
            "code_required": self.code_required,
            "repo_changes_required": self.repo_changes_required,
            "tests_required": self.tests_required,
            "architecture_only_allowed": self.architecture_only_allowed,
            "explanation_only_allowed": self.explanation_only_allowed,
            "completion_criteria": list(self.completion_criteria),
            "user_constraints": list(self.user_constraints),
            "rationale": self.rationale,
            "code_only": self.code_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeliverableContract":
        intent_value = data.get("intent", RequestIntent.UNKNOWN.value)
        if isinstance(intent_value, RequestIntent):
            intent = intent_value
        else:
            try:
                intent = RequestIntent(str(intent_value))
            except ValueError:
                intent = RequestIntent.UNKNOWN
        default = cls()
        mode = _defaults(intent)
        return cls(
            intent=intent,
            # Preserve ``None`` as the constructor sentinel so JSON ``null``
            # receives the intent-safe default instead of becoming the literal
            # string "None" in prompts and summaries.
            deliverable=data.get(
                "deliverable", mode.get("deliverable", default.deliverable)
            ),
            code_required=_bool_value(
                data.get("code_required"), mode.get("code_required", False)
            ),
            repo_changes_required=_bool_value(
                data.get("repo_changes_required"),
                mode.get("repo_changes_required", False),
            ),
            tests_required=_bool_value(
                data.get("tests_required"), mode.get("tests_required", False)
            ),
            architecture_only_allowed=_bool_value(
                data.get("architecture_only_allowed"),
                mode.get("architecture_only_allowed", True),
            ),
            explanation_only_allowed=_bool_value(
                data.get("explanation_only_allowed"),
                mode.get("explanation_only_allowed", True),
            ),
            completion_criteria=_string_tuple(
                data.get("completion_criteria"),
                mode.get("completion_criteria", ()),
            ),
            user_constraints=_string_tuple(data.get("user_constraints")),
            rationale=data.get(
                "rationale",
                (
                    default.rationale
                    if intent is RequestIntent.UNKNOWN
                    else f"explicit {intent.value} contract; safe intent defaults applied"
                ),
            ),
            code_only=_bool_value(data.get("code_only"), False),
        )

    # ------------------------------- prompts --------------------------- #
    def to_brief(self) -> str:
        """The concise, bounded block every role prompt receives."""
        lines = [
            "DELIVERABLE CONTRACT (authoritative — it defines what counts as done):",
            f"- Request intent: {self.intent.value.upper()}",
            f"- Expected deliverable: "
            f"{_bounded_text(self.deliverable, _MAX_DELIVERABLE_CHARS)}",
            f"- Working code required: {'YES' if self.code_required else 'no'}",
            f"- Changes to existing code required: "
            f"{'YES' if self.repo_changes_required else 'no'}",
            f"- Tests/validation required: {'YES' if self.tests_required else 'no'}",
        ]
        if not self.architecture_only_allowed:
            lines.append(
                "- INVALID SUBSTITUTION: architecture/design prose, a plan, a "
                "library survey, or a description of what the code would look "
                "like does NOT satisfy this request. The actual artifact must "
                "be produced."
            )
        if not self.explanation_only_allowed:
            lines.append(
                "- INVALID SUBSTITUTION: explanation alone does not satisfy this "
                "request; the corrective change itself must be produced."
            )
        if self.code_only:
            lines.append(
                "- CODE ONLY: the user explicitly asked for code only. The "
                "deliverable is the source files themselves — no introductory "
                "prose, no summaries, no concluding commentary."
            )
        if self.completion_criteria:
            criteria = "; ".join(
                _bounded_text(c, _MAX_ITEM_CHARS)
                for c in self.completion_criteria[:_MAX_BRIEF_ITEMS]
            )
            if len(self.completion_criteria) > _MAX_BRIEF_ITEMS:
                criteria += (
                    f"; … (+{len(self.completion_criteria) - _MAX_BRIEF_ITEMS} "
                    "more retained in the contract)"
                )
            lines.append(f"- Done when: {criteria}")
        if self.user_constraints:
            constraints = "; ".join(
                _bounded_text(c, _MAX_ITEM_CHARS)
                for c in self.user_constraints[:_MAX_BRIEF_ITEMS]
            )
            if len(self.user_constraints) > _MAX_BRIEF_ITEMS:
                constraints += (
                    f"; … (+{len(self.user_constraints) - _MAX_BRIEF_ITEMS} "
                    "more retained in the contract)"
                )
            lines.append(f"- Explicit user constraints (these win): {constraints}")
        return "\n".join(lines)

    def short_line(self) -> str:
        """One-line form for worker prompts, where space is tightest."""
        needs = []
        if self.code_required:
            needs.append("working code")
        if self.repo_changes_required:
            needs.append("actual changes")
        if self.tests_required:
            needs.append("tests")
        requirement = (
            ", ".join(needs)
            if needs
            else _bounded_text(self.deliverable, 150)
        )
        line = f"{self.intent.value.upper()} request — must deliver {requirement}."
        if self.user_constraints:
            constraints = "; ".join(
                _bounded_text(item, 90) for item in self.user_constraints[:2]
            )
            if len(self.user_constraints) > 2:
                constraints += f"; … (+{len(self.user_constraints) - 2} more)"
            line += f" Explicit constraints: {constraints}."
        return _bounded_text(line, _MAX_SHORT_LINE_CHARS)


# --------------------------------------------------------------------------- #
# Per-intent defaults
# --------------------------------------------------------------------------- #
def _defaults(intent: RequestIntent) -> dict[str, Any]:
    if intent is RequestIntent.IMPLEMENT:
        return {
            "deliverable": "a working implementation (complete, runnable source code)",
            "code_required": True,
            "repo_changes_required": False,
            "tests_required": True,
            "architecture_only_allowed": False,
            "explanation_only_allowed": False,
            "completion_criteria": (
                "every file needed to run the deliverable is provided in full",
                "no placeholders, stubs, or 'code would go here' descriptions",
                "the implementation is validated (tests, or a runnable entry point)",
            ),
        }
    if intent is RequestIntent.MODIFY:
        return {
            "deliverable": "the concrete change applied to the existing code",
            "code_required": True,
            "repo_changes_required": True,
            "tests_required": True,
            "architecture_only_allowed": False,
            "explanation_only_allowed": False,
            "completion_criteria": (
                "the actual modified/added code is produced, not just described",
                "existing behavior is preserved unless the user asked to change it",
                "tests cover the change",
            ),
        }
    if intent is RequestIntent.DEBUG:
        return {
            "deliverable": "a root-cause diagnosis AND the corrective fix",
            "code_required": True,
            "repo_changes_required": True,
            "tests_required": True,
            "architecture_only_allowed": False,
            "explanation_only_allowed": False,
            "completion_criteria": (
                "the root cause is identified specifically, not guessed at",
                "the corrective code change is provided",
                "a regression test or validation proves the fix",
            ),
        }
    if intent is RequestIntent.REVIEW:
        return {
            "deliverable": "an analysis with specific, actionable findings",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "findings are specific and reference concrete evidence",
                "no code changes are made unless the user asked for them",
            ),
        }
    if intent is RequestIntent.ARCHITECTURE:
        return {
            "deliverable": "a design/architecture document (code not required)",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "components, responsibilities and interactions are specified",
                "key decisions and trade-offs are justified",
            ),
        }
    if intent is RequestIntent.RESEARCH:
        return {
            "deliverable": "an evidence-backed investigation or comparison",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "options are compared on stated criteria",
                "a clear conclusion or recommendation is given",
            ),
        }
    if intent is RequestIntent.CONTENT:
        return {
            "deliverable": "the finished written piece itself",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "the complete piece is delivered, not an outline of it",
            ),
        }
    if intent is RequestIntent.EXPLAIN:
        return {
            "deliverable": "a clear explanation (code only as supporting illustration)",
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "the concept is explained accurately and understandably",
            ),
        }
    return {}  # UNKNOWN keeps the permissive dataclass defaults


# --------------------------------------------------------------------------- #
# THE RESOLVER — one authoritative entry point
# --------------------------------------------------------------------------- #
def resolve_contract(
    prompt: str,
    context: str = "",
    desired_output: str = "",
    constraints: Optional[list[str]] = None,
    repo_context: bool = False,
) -> DeliverableContract:
    """Resolve the deliverable contract for a request. Pure and deterministic.

    ``context`` is background (e.g. injected conversation history) and is NOT
    scanned for intent verbs — only the user's own instruction surface
    (prompt + desired_output + constraints) decides what is being asked for.
    ``repo_context=True`` tells the resolver an existing codebase is in play.
    """
    raw_constraints = [str(item) for item in (constraints or []) if str(item).strip()]
    instruction = " ".join(
        part for part in [prompt or "", desired_output or "", *raw_constraints]
        if part
    )
    # Normalize typographic apostrophes before matching contractions. This is
    # punctuation normalization, not fuzzy classification, so determinism holds.
    text = instruction.lower().replace("’", "'").replace("‘", "'").strip()
    if not text:
        return DeliverableContract()  # empty request: permissive defaults

    evidence: list[str] = []

    # ---- 1. Explicit output constraints (highest precedence) ------------- #
    signal_text = _mask_quoted_instruction_text(text)
    no_code = _matches(_NO_CODE_PATTERNS, signal_text)
    no_change = _matches(_NO_CHANGE_PATTERNS, signal_text)
    tests_wanted = _matches(_TESTS_REQUIRED_PATTERNS, text)
    tests_declined = _matches(_NO_TESTS_PATTERNS, text)
    # Phase 4F: "code only" is a positive output demand, not a prohibition. An
    # explicit no-code / no-change instruction outranks it (a prohibition beats
    # a format preference when the user contradicts themselves).
    code_only_hits = _explicit_code_only_hits(text)
    no_commentary_hits = _matches(_NO_COMMENTARY_PATTERNS, signal_text)

    # ---- 2. Action signals ---------------------------------------------- #
    # Constraint phrases are blanked out FIRST, because a negation carries the
    # very verb it forbids: "do not modify anything" contains "modify", and "do
    # not write code" contains "write ... code". Matching actions against the
    # raw text would read those prohibitions as requ sts.
    action_text = text
    for pattern in (
        _NO_CODE_PATTERNS
        + _NO_CHANGE_PATTERNS
        + _NO_TESTS_PATTERNS
        + _NEGATED_ACTION_PATTERNS
        + _NON_ACTION_PATTERNS
    ):
        action_text = re.sub(pattern, " ", action_text)

    hits: dict[RequestIntent, list[str]] = {}
    for intent, patterns in (
        (RequestIntent.DEBUG, _DEBUG_PATTERNS),
        (RequestIntent.MODIFY, _MODIFY_PATTERNS),
        (RequestIntent.IMPLEMENT, _IMPLEMENT_PATTERNS),
        (RequestIntent.REVIEW, _REVIEW_PATTERNS),
        (RequestIntent.ARCHITECTURE, _ARCHITECTURE_PATTERNS),
        (RequestIntent.CONTENT, _CONTENT_PATTERNS),
        (RequestIntent.RESEARCH, _RESEARCH_PATTERNS),
        (RequestIntent.EXPLAIN, _EXPLAIN_PATTERNS),
    ):
        candidate_text = action_text
        if intent is RequestIntent.CONTENT:
            for pattern in _CONTENT_NON_ACTION_PATTERNS:
                candidate_text = re.sub(pattern, " ", candidate_text)
        found = _matches(patterns, candidate_text)
        if found:
            hits[intent] = found

    generator_request = bool(_matches(_GENERATOR_REQUEST, action_text))

    # The requested OBJECT outranks a generic construction verb. Creating a
    # README/document/tutorial is CONTENT; creating a README generator is code.
    if generator_request:
        hits[RequestIntent.IMPLEMENT] = ["software generator request"]
        hits.pop(RequestIntent.CONTENT, None)
    elif RequestIntent.IMPLEMENT in hits and RequestIntent.CONTENT in hits:
        hits.pop(RequestIntent.IMPLEMENT)
    if (
        RequestIntent.MODIFY in hits
        and RequestIntent.CONTENT in hits
        and _matches(_CONTENT_EDIT_REQUEST, action_text)
    ):
        hits.pop(RequestIntent.MODIFY)
    if (
        RequestIntent.DEBUG in hits
        and RequestIntent.CONTENT in hits
        and _matches(_CONTENT_EDIT_REQUEST, action_text)
        and not re.search(
            r"\b(?:bug|crash|defect|regression|race condition|memory leak|"
            r"deadlock|traceback|stack ?trace|fails?|failing|error|broken|"
            r"not work(?:ing)?)\b",
            action_text,
        )
    ):
        hits.pop(RequestIntent.DEBUG)

    # Interrogative how-to wording requests guidance, not execution.
    if (
        RequestIntent.IMPLEMENT in hits
        and (
            re.search(
                r"\bhow (?:do|can|should|would) i\b[^.?!]{0,40}"
                r"\b(?:build|create|implement|develop|write|make|set up)\b",
                action_text,
            )
            or re.search(
                r"\b(?:explain|show me|teach me|tell me|walk me through)\b"
                r"[^.?!]{0,45}\bhow to\b[^.?!]{0,20}"
                r"\b(?:build|create|implement|develop|write|make|set up)\b",
                action_text,
            )
            or re.search(
                r"\b(?:outline|describe|explain|discuss|teach)\b"
                r"[^.?!]{0,35}\bhow "
                r"(?:one|someone|we|you) (?:might|could|would|can|should)\b"
                r"[^.?!]{0,25}\b(?:build|create|implement|develop|write|make|"
                r"set up)\b",
                action_text,
            )
            or re.search(
                r"\b(?:should\s+(?:we|i|you)|"
                r"do\s+(?:i|we|you)\s+need\s+to)\s+"
                r"(?:build|create|implement|develop|write|make|set up)\b",
                action_text,
            )
        )
    ):
        hits.pop(RequestIntent.IMPLEMENT)
        hits.setdefault(RequestIntent.EXPLAIN, ["interrogative how-to request"])

    # "build" is a noun in build pipelines/files. Do not let it override an
    # explicit explain/review verb.
    if (
        RequestIntent.IMPLEMENT in hits
        and any(i in hits for i in (RequestIntent.EXPLAIN, RequestIntent.REVIEW))
        and re.search(
            r"(?:\bbuild\.py\b|\bbuild\s+(?:pipeline|process|system|step|script|"
            r"configuration|config)\b)",
            action_text,
        )
    ):
        hits.pop(RequestIntent.IMPLEMENT)

    # Symptom/design nouns remain useful when no stronger communicative verb is
    # present ("the button crashes"), but "explain a crash report" and
    # "explain architecture" are explanations, not mandatory code work.
    if (
        RequestIntent.DEBUG in hits
        and any(i in hits for i in (RequestIntent.EXPLAIN, RequestIntent.REVIEW))
        and not _matches(_DEBUG_ACTION_PATTERNS, action_text)
    ):
        hits.pop(RequestIntent.DEBUG)
    if (
        RequestIntent.ARCHITECTURE in hits
        and re.search(
            r"\b(?:explain|teach|clarify|walk me through|summari[sz]e|describe|"
            r"discuss|outline)\b",
            action_text,
        )
    ):
        hits.pop(RequestIntent.ARCHITECTURE)
        hits.setdefault(RequestIntent.EXPLAIN, ["communicative architecture request"])

    # "Create a SADD" / "Give me an implementation plan": the construction verb
    # is building a DESIGN ARTIFACT, not software. The artifact being asked for
    # decides the mode. ("Design and implement the dashboard" is unaffected —
    # there the verb's object is the dashboard, not a document.)
    direct_software = bool(re.search(
        r"\b(?:build|implement|develop|scaffold|bootstrap|code up)\b"
        r"[^.?!;]{0,45}\b(?:app|application|dashboard|api|endpoint|service|"
        r"server|backend|frontend|ui|cli|tool|library|package|module|component|"
        r"function|class|method|feature|page|game|bot|scraper|parser|migration|"
        r"schema|query|algorithm|prototype|mvp|repo|repository|project|"
        r"integration|authentication|authorization|generator)\b",
        action_text,
    ))
    if (
        RequestIntent.IMPLEMENT in hits
        and _matches(_DESIGN_ARTIFACT_REQUEST, action_text)
        and not generator_request
        and not direct_software
    ):
        hits.pop(RequestIntent.IMPLEMENT)
        hits.setdefault(RequestIntent.ARCHITECTURE, ["design-artifact request"])

    existing_code = repo_context or bool(_matches(_EXISTING_CODE_PATTERNS, action_text))

    # A change-verb only means MODIFY when there is something to change: an
    # existing codebase. Otherwise "build an app and add auth" is one greenfield
    # implementation, not a modification.
    if RequestIntent.MODIFY in hits and not existing_code and RequestIntent.IMPLEMENT in hits:
        hits.pop(RequestIntent.MODIFY)
    elif (
        existing_code
        and RequestIntent.IMPLEMENT in hits
        and RequestIntent.CONTENT not in hits
    ):
        hits.pop(RequestIntent.IMPLEMENT)
        hits.setdefault(RequestIntent.MODIFY, ["implementation in existing codebase"])

    # ---- 3. Strongest action wins (never discard the stronger one) -------- #
    intent = RequestIntent.UNKNOWN
    for candidate in _STRENGTH:
        if candidate in hits:
            intent = candidate
            evidence.append(f"{candidate.value} signals: {', '.join(hits[candidate][:3])}")
            break
    if intent is RequestIntent.UNKNOWN:
        evidence.append("no action signals matched")

    # ---- 4. Explicit no-code override ------------------------------------ #
    # "Build me a dashboard, but only the architecture — do not write code."
    # The user's explicit instruction beats the inferred default: the request
    # becomes the DESIGN of the thing they asked for.
    overridden = False
    diagnostic_only = False
    if no_code and intent in _CODE_INTENTS:
        evidence.append(f"explicit no-code constraint: {no_code[0]!r} — overrides "
                        f"inferred {intent.value}")
        overridden = True
        if intent is RequestIntent.DEBUG:
            diagnostic_only = True
        elif intent is RequestIntent.MODIFY:
            intent = RequestIntent.REVIEW
        else:
            intent = RequestIntent.ARCHITECTURE

    if no_change and intent in _CODE_INTENTS:
        if intent is RequestIntent.DEBUG:
            diagnostic_only = True
        elif intent is RequestIntent.MODIFY:
            intent = RequestIntent.REVIEW
        else:
            intent = RequestIntent.ARCHITECTURE

    fields = dict(_defaults(intent))
    if diagnostic_only:
        fields.update({
            "deliverable": (
                "a root-cause diagnosis and proposed corrective approach, "
                "without implementing or applying changes"
            ),
            "code_required": False,
            "repo_changes_required": False,
            "tests_required": False,
            "architecture_only_allowed": True,
            "explanation_only_allowed": True,
            "completion_criteria": (
                "the root cause is identified specifically, not guessed at",
                "a concrete corrective approach is proposed",
                "no code or repository changes are applied",
            ),
        })
    constraints_out: list[str] = list(raw_constraints)

    # ---- 4b. Explicit code-only demand (Phase 4F, Part G) ---------------- #
    # "give code only" makes code the ONLY acceptable deliverable surface. It
    # never overrides an explicit prohibition (no-code / no-change wins), but it
    # DOES outrank a weak or missing action classification: the live failure
    # resolved to UNKNOWN and silently lost the whole artifact pipeline.
    code_only = bool(
        code_only_hits
        or (no_commentary_hits and intent in _CODE_INTENTS)
    ) and not no_code and not no_change
    if code_only:
        code_only_evidence = (
            code_only_hits[0] if code_only_hits else no_commentary_hits[0]
        )
        if intent not in _CODE_INTENTS:
            previous = intent
            intent = RequestIntent.IMPLEMENT
            fields = dict(_defaults(intent))
            evidence.append(
                f"explicit code-only constraint: {code_only_evidence!r} — "
                + ("no action signal matched; code is the demanded deliverable"
                   if previous is RequestIntent.UNKNOWN
                   else f"outranks inferred {previous.value}; code is the "
                        "demanded deliverable")
            )
        else:
            evidence.append(f"explicit code-only constraint: {code_only_evidence!r}")
        fields["architecture_only_allowed"] = False
        fields["explanation_only_allowed"] = False
        constraints_out.append(
            "the user explicitly asked for code only — no explanatory prose "
            "in the deliverable"
        )

    if overridden:
        constraints_out.append("the user explicitly asked for NO code in this response")
    if no_change:
        # Applies to any intent: an explicit "don't change anything" forbids
        # repository modification (Review-without-modification, Part B).
        fields["repo_changes_required"] = False
        constraints_out.append("the user explicitly forbade changing existing code")
        evidence.append(f"explicit no-change constraint: {no_change[0]!r}")
    if tests_wanted and not tests_declined:
        fields["tests_required"] = True
        constraints_out.append("the user explicitly asked for tests")
    if tests_declined:
        fields["tests_required"] = False
        constraints_out.append("the user explicitly declined tests")

    contract = DeliverableContract(
        intent=intent,
        **{k: v for k, v in fields.items() if k != "user_constraints"},
        user_constraints=tuple(constraints_out),
        rationale="; ".join(evidence),
        code_only=code_only,
    )
    return contract


def demands_code_only(
    prompt: str,
    desired_output: str = "",
    constraints: Optional[list[str]] = None,
) -> bool:
    """Whether the user's OWN instruction surface explicitly demands code-only.

    Mirrors :func:`resolve_contract`'s explicit code-only signal (the strong
    ``code only`` / ``give code only`` / ``just the code`` family, defeated by
    an explicit no-code or no-change prohibition), but WITHOUT resolving a full
    contract. The root integrity guard uses it to detect a dropped/lost
    code-only obligation while leaving a legitimately supplied contract
    untouched — so a supplied contract is still never re-resolved.
    """
    raw_constraints = [str(item) for item in (constraints or []) if str(item).strip()]
    instruction = " ".join(
        part for part in [prompt or "", desired_output or "", *raw_constraints] if part
    )
    text = instruction.lower().replace("’", "'").replace("‘", "'").strip()
    if not text:
        return False
    signal_text = _mask_quoted_instruction_text(text)
    if (_matches(_NO_CODE_PATTERNS, signal_text)
            or _matches(_NO_CHANGE_PATTERNS, signal_text)):
        return False
    return bool(_explicit_code_only_hits(text))


# --------------------------------------------------------------------------- #
# Contract guards
#
# Both are deterministic, pure, and duck-typed (no imports from models.py, so
# Task can own a contract without a circular import).
# --------------------------------------------------------------------------- #

# Concrete production work in a plan.
_DELIVERY_PATTERNS = (
    r"\b(?:write|create|implement|build|add|apply|modify|update|fix|correct|"
    r"patch|generate|produce|develop|refactor|scaffold)\b[^.\n]{0,60}"
    r"\b(?:code|file|files|script|"
    r"component|components|module|function|class|endpoint|api|app|test|tests|"
    r"suite|schema|config|migration|page|hook|service|route|routes|css|"
    r"stylesheet|implementation|source|correction|fix|patch|bug|defect)\b",
    r"\b(?:source code|working code|full code|complete code|runnable|"
    r"code file|implementation of)\b",
    r"\.(?:py|js|jsx|ts|tsx|java|go|rs|rb|css|html|sql|json|yaml|yml|sh)\b",
)

_TERSE_DELIVERY_PATTERNS = (
    rf"\b{_CODE_OBJECT}\b",
    r"\b(?:auth|authentication|authorization|application shell)\b",
)

_NON_DELIVERY_PATTERNS = (
    r"\b(?:describe|explain|outline|recommend|discuss|survey|propose|advise)\b"
    r"[^.\n]{0,45}\b(?:how to|ways? to|options?|structure|approach|would)\b",
    r"\bwithout (?:actually )?(?:implementing|writing|creating|changing|"
    r"producing|applying)\b",
)

_DEBUG_PLAN_DIAGNOSIS_PATTERNS = (
    r"\b(?:reproduce|diagnose|investigate|trace|isolate|identify|analy[sz]e)\b"
    r"[^.\n]{0,45}\b(?:failure|crash|bug|defect|cause|root cause|error)\b",
    r"\b(?:root cause|reproduce (?:the )?failure)\b",
    r"\b(?:diagnose|investigate|troubleshoot)\b",
)

_DEBUG_PLAN_FIX_PATTERNS = (
    r"\b(?:apply|implement|write|make|produce)\b[^.\n]{0,45}"
    r"\b(?:fix|correction|patch|change)\b",
    r"\b(?:fix|repair|correct|patch)\b[^.\n]{0,45}"
    r"\b(?:bug|defect|failure|crash|code|source|file|implementation)\b",
    r"\b(?:fix|repair|correct|patch|apply (?:the )?correction)\b",
)

# Work that only talks about the thing.
_PROSE_ONLY_PATTERNS = (
    r"\b(?:describe|explain|outline|recommend|document|summari[sz]e|discuss|"
    r"compare|research|survey|propose|advise|analy[sz]e)\b",
    r"\b(?:architecture|design|specification|spec|overview|high[- ]level|"
    r"best practices|considerations|approach|strategy|roadmap|plan)\b",
)

_VALIDATION_PATTERNS = (
    r"\b(?:test|tests|testing|unit test|integration test|validate|validation|"
    r"verify|verification|run|check|lint|typecheck|coverage)\b",
)

# Answers that openly admit no code was produced. These are the exact failure
# mode this phase exists to stop ("No application code is included.").
_NO_CODE_DISCLAIMERS = (
    r"\bno (?:application |source |actual |real )?code (?:is |was |has been )?"
    r"(?:included|provided|written|generated|produced|attached)\b",
    r"\b(?:code|implementation) (?:is |was |has been )?(?:not|omitted)"
    r"(?: included| provided| written| shown)?\b",
    r"\bthis (?:document|response|answer|deliverable) (?:does not|doesn'?t) "
    r"(?:include|contain|provide) (?:any )?(?:code|implementation)\b",
    r"\bwithout (?:writing|providing|including) (?:any )?code\b",
    r"\b(?:code|implementation) (?:is )?(?:beyond|outside) the scope\b",
    r"\bwhat the code would look like\b",
    r"\bcode would (?:go|be placed|look)\b",
)

_FENCE_RE = re.compile(
    r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL
)
_NON_CODE_FENCE_LANGS = {
    "text", "txt", "plaintext", "markdown", "md", "pseudo", "pseudocode",
}
_KNOWN_CODE_FENCE_LANGS = {
    "python", "py", "javascript", "js", "jsx", "typescript", "ts", "tsx",
    "java", "go", "rust", "rs", "ruby", "rb", "c", "cpp", "c++", "csharp",
    "cs", "kotlin", "swift", "php", "sql", "sh", "bash", "shell",
    "powershell", "ps1", "html", "css", "scss", "json", "yaml", "yml",
    "toml", "xml",
}
_PLACEHOLDER_LINE_RE = re.compile(
    r"^\s*(?:(?://|#|/\*+|\*|<!--)\s*)?"
    r"(?:todo|fixme|tbd|pass\b|\.\.\.|not implemented|code (?:goes|would go) here)",
    re.IGNORECASE,
)
_FILE_REFERENCE_RE = re.compile(
    r"\b[\w./\\-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|rb|css|html|sql|json|"
    r"yaml|yml|sh|ps1|toml|xml)\b",
    re.IGNORECASE,
)
_CHANGE_ASSERTION_RE = re.compile(
    r"\b(?:changed|modified|updated|added|removed|fixed|patched|implemented|"
    r"refactored|corrected)\b",
    re.IGNORECASE,
)
_VALIDATION_EVIDENCE_RE = re.compile(
    r"\b(?:(?:tests?|suite|checks?|validation)\b[^.\n]{0,45}"
    r"\b(?:pass(?:ed|ing)?|green|succeed(?:ed|ing)?|ok)\b|"
    r"(?:pass(?:ed|ing)?|green)\b[^.\n]{0,30}\b(?:tests?|suite|checks?)\b)",
    re.IGNORECASE,
)
_DEBUG_DIAGNOSIS_RE = re.compile(
    r"\b(?:root cause|caused by|because|failure occurs|crash occurs|the bug was|"
    r"the defect was)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContractCheck:
    """Result of a contract guard. ``fatal`` blocks; ``advisory`` only warns."""

    fatal: tuple[str, ...] = ()
    advisory: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.fatal

    def feedback(self) -> str:
        return "; ".join(self.fatal + self.advisory)


def _has_unified_diff(text: str) -> bool:
    lines = text.splitlines()
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_hunk = any(re.match(r"^@@ -\d", line) for line in lines)
    has_change = any(
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
        for line in lines
    )
    return has_old and has_new and has_hunk and has_change


def _python_has_substance(text: str) -> bool:
    """Reject declaration/TODO skeletons while accepting concise real Python."""
    useful: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or _PLACEHOLDER_LINE_RE.match(stripped)
            or re.match(r"^(?:async\s+def|def|class)\b", stripped)
            or stripped.startswith("@")
        ):
            continue
        useful.append(stripped)
    return any(
        re.match(
            r"^(?:return\s+\S|raise\s+\S|yield\b|"
            r"[A-Za-z_]\w*(?::\s*[^=]+)?\s*(?:=|\+=|-=|\*=|/=)\s*\S|"
            r"[A-Za-z_][\w.]*\s*\()",
            line,
        )
        for line in useful
    )


def _looks_like_raw_code(text: str) -> bool:
    """Recognize source flattened from a compliant worker artifact."""
    candidate = text.strip()
    if candidate[:1] in {"{", "["}:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                return True
    if _python_has_substance(text):
        return True
    patterns = (
        r"(?m)^\s*(?:export\s+default\s+)?(?:async\s+)?function\s+\w+\s*\(",
        r"(?m)^\s*(?:const|let|var)\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|function)",
        r"(?im)^\s*(?:select\b.+|insert\s+into\b|update\s+\w+\s+set\b|"
        r"create\s+(?:table|index)\b)",
        r"(?m)^\s*#!\s*/(?:usr/)?bin/(?:env\s+)?(?:ba|z|k)?sh\b",
        r"(?is)<(?:!doctype\s+html|html\b|style\b|script\b|div\b)[^>]*>",
        r"(?m)^\s*[.#]?[A-Za-z][\w .#,:>+~-]*\s*\{[^}]+\}",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _fence_has_code(language: str, body: str) -> bool:
    language = language.lower().strip()
    stripped = body.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines or language in _NON_CODE_FENCE_LANGS:
        return False
    substantive = [line for line in lines if not _PLACEHOLDER_LINE_RE.match(line)]
    if not substantive:
        return False
    if language == "json":
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return False
        return isinstance(parsed, (dict, list))
    if language in {"python", "py"}:
        return _python_has_substance(stripped)
    if language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        return bool(re.search(
            r"(?m)^\s*(?:export\b|import\b|function\b|class\b|interface\b|"
            r"(?:const|let|var)\s+\w+\s*=|return\b)|=>|[{};]",
            stripped,
        ))
    if language == "sql":
        return bool(re.search(
            r"(?i)\b(?:select|insert\s+into|update\s+\w+\s+set|delete\s+from|"
            r"create\s+(?:table|index|view)|alter\s+table|with\s+\w+\s+as)\b",
            stripped,
        ))
    if language in {"html", "xml"}:
        return bool(re.search(r"<[/!?]?[A-Za-z][^>]*>", stripped))
    if language in {"css", "scss"}:
        return bool(re.search(r"[^{}]+\{[^}]+\}", stripped, re.DOTALL))
    if language in {"yaml", "yml", "toml"}:
        return bool(re.search(r"(?m)^\s*[\w.-]+\s*[:=]\s*\S", stripped))
    if language in {"sh", "bash", "shell"}:
        return bool(re.search(
            r"(?m)^\s*(?:#!|set\s+-|(?:if|for|while|case)\b|"
            r"[\w./-]+\s+(?:--?[\w-]+|[\w./'\"]+))",
            stripped,
        ))
    if language in {"powershell", "ps1"}:
        return bool(re.search(
            r"(?mi)^\s*(?:\$\w+\s*=|(?:Get|Set|New|Remove|Invoke|Write)-\w+|"
            r"if\s*\(|foreach\s*\()",
            stripped,
        ))
    if language in _KNOWN_CODE_FENCE_LANGS:
        return bool(re.search(
            r"(?m)^\s*(?:package|import|class|func|fn|struct|enum|interface|"
            r"public|private|protected|static|return)\b|[{};]",
            stripped,
        ))
    return _looks_like_raw_code(stripped)


def has_real_code(text: str) -> bool:
    """Detect a substantive source/config artifact or internally valid diff.

    This remains a mode check, not a compiler. It rejects placeholder/comment
    fences and pseudocode, accepts recognized language artifacts (including
    concise SQL/config), and recognizes raw source flattened from worker JSON.
    """
    if not text:
        return False
    if _has_unified_diff(text):
        return True
    if text.count("```") % 2:
        # An unmatched fence is truncation evidence, not a complete artifact.
        return False
    for language, body in _FENCE_RE.findall(text):
        if _fence_has_code(language, body):
            return True
    return _looks_like_raw_code(text)


def _has_repo_change_summary(text: str, *, debug: bool) -> bool:
    """Interim evidence for an already-applied repository change.

    This is deliberately an assertion check, not proof. Trusted mutation
    attestation belongs to the future artifact system; until then a concrete
    file/change/test summary is indeterminate rather than an automatic failure.
    """
    if not (
        _FILE_REFERENCE_RE.search(text)
        and _CHANGE_ASSERTION_RE.search(text)
        and _VALIDATION_EVIDENCE_RE.search(text)
    ):
        return False
    return not debug or bool(_DEBUG_DIAGNOSIS_RE.search(text))


def validate_plan_against_contract(contract: DeliverableContract, plan: Any) -> ContractCheck:
    """Reject plans that cannot possibly satisfy a code-bearing contract.

    Only code-oriented contracts (IMPLEMENT/MODIFY/DEBUG) are constrained; an
    architecture or explanation request may legitimately plan pure prose work.
    """
    if not (contract.code_required or contract.repo_changes_required):
        return ContractCheck()
    subtasks = list(getattr(plan, "subtasks", None) or [])
    if not subtasks:
        # A direct answer bypasses decomposition; the final guard judges it.
        return ContractCheck()

    fatal: list[str] = []
    advisory: list[str] = []
    delivery_titles: list[str] = []
    validation_present = False
    debug_diagnosis_present = False
    debug_fix_present = False

    for st in subtasks:
        title = str(getattr(st, "title", "") or "").lower()
        blob = " ".join(
            str(getattr(st, attr, "") or "")
            for attr in ("title", "instruction", "expected_output", "success_criteria")
        ).lower()
        prose_substitution = bool(_matches(_NON_DELIVERY_PATTERNS, blob))
        concrete = bool(_matches(_DELIVERY_PATTERNS, blob)) and not prose_substitution
        terse = (
            len(title.split()) <= 5
            and bool(_matches(_TERSE_DELIVERY_PATTERNS, title))
            and not _matches(_PROSE_ONLY_PATTERNS, blob)
            and not prose_substitution
        )
        if concrete or terse:
            delivery_titles.append(str(getattr(st, "title", "")))
        if _matches(_VALIDATION_PATTERNS, blob):
            validation_present = True
        if _matches(_DEBUG_PLAN_DIAGNOSIS_PATTERNS, blob):
            debug_diagnosis_present = True
        if _matches(_DEBUG_PLAN_FIX_PATTERNS, blob):
            debug_fix_present = True

    if not delivery_titles:
        prose_only = any(
            _matches(_PROSE_ONLY_PATTERNS,
                     " ".join(str(getattr(st, a, "") or "")
                              for a in ("title", "instruction")).lower())
            for st in subtasks
        )
        detail = (" every subtask only describes, explains or designs the work"
                  if prose_only else " no subtask produces the artifact")
        fatal.append(
            f"the plan contains no concrete implementation work —{detail}. "
            f"This is a {contract.intent.value.upper()} request: at least one "
            "subtask must actually produce the required code/changes"
        )

    if contract.intent is RequestIntent.DEBUG:
        if not debug_diagnosis_present:
            fatal.append(
                "the DEBUG plan never reproduces or identifies the root cause"
            )
        if not debug_fix_present:
            fatal.append(
                "the DEBUG plan never applies a corrective fix"
            )

    if contract.tests_required and not validation_present:
        # Advisory, not fatal: a plan that builds the thing but forgets to name
        # a test step is incomplete, not a MODE violation. The planner gets one
        # corrective retry; if it still omits tests the run proceeds (the
        # executor's verify/repair loop remains the validation backstop).
        advisory.append(
            "no subtask covers tests or validation, which this contract requires"
        )

    return ContractCheck(fatal=tuple(fatal), advisory=tuple(advisory))


def validate_final_answer_against_contract(
    contract: DeliverableContract, answer: str
) -> ContractCheck:
    """The minimal final-answer mode guard.

    Catches obvious substitutions only. It does NOT judge code correctness,
    completeness across files, or truncation — those are later phases.
    """
    text = (answer or "").strip()
    if not text:
        return ContractCheck(fatal=("the run produced no final answer",))
    lowered = text.lower()
    fatal: list[str] = []
    advisory: list[str] = []
    code_present = has_real_code(text)

    constraint_text = " ".join(contract.user_constraints).lower()
    code_forbidden = (
        "explicitly asked for no code" in constraint_text
        or bool(_matches(_NO_CODE_PATTERNS, constraint_text))
    )
    if code_forbidden and code_present:
        fatal.append(
            "the user explicitly forbade code in this response, but a substantive "
            "code artifact was produced"
        )

    if not (contract.code_required or contract.repo_changes_required):
        return ContractCheck(fatal=tuple(fatal))

    disclaimers = _matches(_NO_CODE_DISCLAIMERS, lowered)
    if not code_present and disclaimers:
        fatal.append(
            f"the answer explicitly declines to deliver code ({disclaimers[0]!r}), "
            f"but this is a {contract.intent.value.upper()} request where working "
            "code is the deliverable"
        )
    elif not code_present:
        repo_summary = (
            contract.repo_changes_required
            and _has_repo_change_summary(
                text, debug=contract.intent is RequestIntent.DEBUG
            )
        )
        if repo_summary:
            advisory.append(
                "the answer gives concrete file/change/test evidence, but the "
                "current architecture cannot independently attest repository mutations"
            )
        else:
            fatal.append(
                f"this is a {contract.intent.value.upper()} request requiring working "
                "code or concrete applied-change evidence, but the answer contains "
                "no code or such evidence — only prose"
            )

    if (
        contract.intent is RequestIntent.DEBUG
        and code_present
        and not _DEBUG_DIAGNOSIS_RE.search(text)
    ):
        fatal.append(
            "this is a DEBUG request: corrective code is present, but the required "
            "root-cause diagnosis is missing"
        )

    if (
        contract.tests_required
        and code_present
        and not _matches(_VALIDATION_PATTERNS, lowered)
    ):
        advisory.append(
            "the contract requires tests or validation, but the final answer does "
            "not state any validation evidence"
        )

    return ContractCheck(fatal=tuple(fatal), advisory=tuple(advisory))
