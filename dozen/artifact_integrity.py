"""Artifact truncation and structural-integrity detection (Phase 4D).

WHY THIS EXISTS
---------------
Phase 4C collects typed produced artifacts, checks ownership and provenance,
and assembles them deterministically — but it still TRUSTS the worker's own
``"complete": true``. A provider that stops mid-token, a model that ends a file
with ``// ... rest of the component`` or a reply that gets cut off inside a JSX
attribute all sail straight into the accepted set and get presented as a
finished file.

This module answers exactly ONE question about ONE submitted body:

    Does this content LOOK structurally complete, deterministically?

It never repairs, never regenerates, never rewrites, never calls a provider,
never touches the filesystem and never executes submitted code. It reports.
Everything here is pure, deterministic, immutable, serializable and linear in
content length.

Conservatism is the design rule: an artifact is only INVALID when a
deterministic structural violation is proven (a JSON parse failure, an
unterminated string, an unclosed fence, an unfinished diff hunk). Content the
detector cannot reason about is NOT_APPLICABLE, never invalid — an unsupported
language must not be punished for the detector's ignorance.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from enum import Enum
from posixpath import splitext
from typing import Any, Mapping, Optional, Sequence

from .artifacts import ArtifactKind, ArtifactOperation
from .validation import validate_model_response

try:  # Python 3.11+; absent on 3.10, where TOML is simply not inspected.
    import tomllib as _tomllib
except ImportError:  # pragma: no cover - depends on interpreter version
    _tomllib = None  # type: ignore[assignment]

SCHEMA_VERSION = 1


class ArtifactIntegrityError(ValueError):
    """A structural violation of the integrity models themselves."""


# --------------------------------------------------------------------------- #
# Part A — vocabulary
# --------------------------------------------------------------------------- #
class IntegrityStatus(str, Enum):
    """The verdict on ONE submitted artifact body.

    VALID          no supported structural problem was detected. This does NOT
                   prove semantic, type or compiler correctness.
    INVALID        a deterministic structural violation was detected.
    SUSPICIOUS     strong evidence of truncation exists, but the lightweight
                   detector cannot prove invalidity safely.
    NOT_APPLICABLE the artifact kind or format does not require (or does not
                   support) structural inspection. Never a punishment.
    """

    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    NOT_APPLICABLE = "not_applicable"


class IntegritySeverity(str, Enum):
    FATAL = "fatal"          # proves INVALID
    SUSPECT = "suspect"      # proves SUSPICIOUS
    ADVISORY = "advisory"    # informational only; never blocks on its own


class InspectionMode(str, Enum):
    """How one body was inspected. Recorded so a verdict is explainable."""

    NONE = "none"                  # no content expected (directory / delete)
    TEXT = "text"                  # generic: universal indicators only
    PROSE = "prose"                # documents: fences + universal indicators
    RESULT = "result"              # build/test/command evidence
    JSON = "json"
    JSONC = "jsonc"
    JSONL = "jsonl"
    TOML = "toml"
    XML = "xml"
    YAML = "yaml"                  # declared but not structurally supported
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    SQL = "sql"
    DIFF = "diff"


class IntegrityIssueCode(str, Enum):
    EMPTY_CONTENT = "empty_content"
    TRUNCATION_MARKER = "truncation_marker"
    PLACEHOLDER_ENDING = "placeholder_ending"
    REFUSAL_CONTENT = "refusal_content"
    CONTINUATION_ENDING = "continuation_ending"
    PARTIAL_ESCAPE = "partial_escape"
    REPLACEMENT_CHARACTER = "replacement_character"
    UNCLOSED_FENCE = "unclosed_fence"
    EMPTY_FENCE = "empty_fence"
    MULTIPLE_FENCES = "multiple_fences"
    TRAILING_PROSE = "trailing_prose"
    FENCED_CONTENT = "fenced_content"
    UNTERMINATED_STRING = "unterminated_string"
    UNTERMINATED_TEMPLATE = "unterminated_template"
    UNTERMINATED_COMMENT = "unterminated_comment"
    UNBALANCED_DELIMITER = "unbalanced_delimiter"
    UNCLOSED_TAG = "unclosed_tag"
    INCOMPLETE_TAG = "incomplete_tag"
    UNTERMINATED_ATTRIBUTE = "unterminated_attribute"
    UNFINISHED_STATEMENT = "unfinished_statement"
    MALFORMED_JSON = "malformed_json"
    MALFORMED_JSONL = "malformed_jsonl"
    MALFORMED_TOML = "malformed_toml"
    MALFORMED_XML = "malformed_xml"
    PYTHON_INCOMPLETE = "python_incomplete"
    PYTHON_SYNTAX_ERROR = "python_syntax_error"
    MALFORMED_DIFF = "malformed_diff"
    INCOMPLETE_HUNK = "incomplete_hunk"
    NESTING_LIMIT = "nesting_limit"
    JSONC_COMMENTS = "jsonc_comments"
    UNSUPPORTED_FORMAT = "unsupported_format"
    NOT_INSPECTED = "not_inspected"


# Modes whose bodies are code: a trailing ellipsis comment or a dangling
# backslash there is a truncation, not a prose flourish.
_CODE_MODES = frozenset({
    InspectionMode.JSON, InspectionMode.JSONC, InspectionMode.JSONL,
    InspectionMode.TOML,
    InspectionMode.XML, InspectionMode.YAML, InspectionMode.PYTHON,
    InspectionMode.JAVASCRIPT, InspectionMode.TYPESCRIPT, InspectionMode.HTML,
    InspectionMode.CSS, InspectionMode.SHELL, InspectionMode.SQL,
    InspectionMode.DIFF,
})


# --------------------------------------------------------------------------- #
# Part B — the ONE authoritative integrity policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegrityPolicy:
    """Every bound and switch the detector obeys. Nothing is hard-coded elsewhere.

    Defaults are conservative: they favour accepting an odd-but-plausible file
    over rejecting a good one, because a false INVALID burns a real repair
    attempt and can fail an otherwise complete deliverable.
    """

    # Bounded scanning. Content beyond these bounds is not inspected, and the
    # result says so (``conclusive`` becomes False) instead of guessing.
    max_inspect_chars: int = 262_144
    max_inspect_lines: int = 20_000
    max_nesting_depth: int = 256
    # Bounded diagnostics.
    max_issues: int = 8
    max_evidence_chars: int = 120
    max_message_chars: int = 240
    # Heuristics.
    # Below this length, the SUSPICIOUS ending heuristics (dangling token, unfinished
    # declaration) are not applied: a tiny body carries too little signal to accuse,
    # and a false rejection costs a real repair attempt.
    min_meaningful_content_chars: int = 8
    min_result_content_chars: int = 2       # build/test evidence must say something
    # Acceptance switches (Part I).
    block_suspicious: bool = True           # SUSPICIOUS candidates are not accepted
    advisories_in_feedback: bool = False    # advisories never enter repair feedback
    inspect_documents: bool = True          # documents get fence + marker checks
    require_result_content: bool = True


DEFAULT_INTEGRITY_POLICY = IntegrityPolicy()


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _clip(value: Any, limit: int) -> str:
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


def _line_of(text: str, offset: int) -> int:
    """1-based line number of ``offset``. Linear, bounded by the caller."""
    if offset < 0:
        return -1
    return text.count("\n", 0, min(offset, len(text))) + 1


# --------------------------------------------------------------------------- #
# Part A — the integrity models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegrityIssue:
    """ONE detected problem. Bounded evidence; never a slice of the whole file."""

    code: IntegrityIssueCode
    severity: IntegritySeverity
    message: str
    evidence: str = ""
    offset: int = -1
    line: int = -1
    truncation: bool = False

    def __post_init__(self) -> None:
        policy = DEFAULT_INTEGRITY_POLICY
        object.__setattr__(
            self, "code",
            _coerce_enum(self.code, IntegrityIssueCode,
                         IntegrityIssueCode.NOT_INSPECTED),
        )
        object.__setattr__(
            self, "severity",
            _coerce_enum(self.severity, IntegritySeverity, IntegritySeverity.ADVISORY),
        )
        object.__setattr__(self, "message", _clip(self.message, policy.max_message_chars))
        object.__setattr__(
            self, "evidence", _clip(self.evidence, policy.max_evidence_chars)
        )
        for name in ("offset", "line"):
            try:
                object.__setattr__(self, name, int(getattr(self, name)))
            except (TypeError, ValueError):
                object.__setattr__(self, name, -1)
        object.__setattr__(self, "truncation", bool(self.truncation))

    @property
    def fatal(self) -> bool:
        return self.severity is IntegritySeverity.FATAL

    def sort_key(self) -> tuple:
        rank = {
            IntegritySeverity.FATAL: 0,
            IntegritySeverity.SUSPECT: 1,
            IntegritySeverity.ADVISORY: 2,
        }[self.severity]
        return (rank, self.offset if self.offset >= 0 else 1 << 40, self.code.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
            "offset": self.offset,
            "line": self.line,
            "truncation": self.truncation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrityIssue":
        if not isinstance(data, Mapping):
            raise ArtifactIntegrityError("integrity issue must be a JSON object")
        return cls(
            code=_coerce_enum(data.get("code"), IntegrityIssueCode,
                              IntegrityIssueCode.NOT_INSPECTED),
            severity=_coerce_enum(data.get("severity"), IntegritySeverity,
                                  IntegritySeverity.ADVISORY),
            message=data.get("message") or "",
            evidence=data.get("evidence") or "",
            offset=data.get("offset", -1),
            line=data.get("line", -1),
            truncation=bool(data.get("truncation", False)),
        )


@dataclass(frozen=True)
class ArtifactIntegrityResult:
    """The verdict on ONE submitted body. Immutable, serializable, evidence-bearing.

    ``structurally_complete`` is the question Phase 4C could not answer, and is
    deliberately independent of the worker's ``complete`` flag: the flag is
    provider-controlled data, this is measured evidence.
    """

    status: IntegrityStatus
    artifact_id: str = ""
    path: str = ""
    mode: InspectionMode = InspectionMode.TEXT
    issues: tuple[IntegrityIssue, ...] = ()
    conclusive: bool = True
    content_chars: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        raw_status = self.status
        status_known = isinstance(raw_status, IntegrityStatus)
        if not status_known:
            try:
                IntegrityStatus(str(raw_status).strip().lower())
                status_known = True
            except (TypeError, ValueError):
                status_known = False
        object.__setattr__(
            self, "status",
            _coerce_enum(self.status, IntegrityStatus, IntegrityStatus.NOT_APPLICABLE),
        )
        object.__setattr__(self, "artifact_id", _clip(self.artifact_id, 300))
        object.__setattr__(self, "path", _clip(self.path, 300))
        object.__setattr__(
            self, "mode", _coerce_enum(self.mode, InspectionMode, InspectionMode.TEXT)
        )
        issues = () if self.issues is None else self.issues
        if isinstance(issues, str) or not isinstance(issues, (list, tuple)):
            raise ArtifactIntegrityError("integrity issues must be a JSON array")
        for issue in issues:
            if not isinstance(issue, IntegrityIssue):
                raise ArtifactIntegrityError(
                    "integrity issues must contain IntegrityIssue items"
                )
        object.__setattr__(
            self, "issues",
            tuple(sorted(issues, key=lambda issue: issue.sort_key()))[
                : DEFAULT_INTEGRITY_POLICY.max_issues
            ],
        )
        canonical_issues = self.issues
        has_fatal = any(
            issue.severity is IntegritySeverity.FATAL for issue in canonical_issues
        )
        has_suspect = any(
            issue.severity is IntegritySeverity.SUSPECT for issue in canonical_issues
        )
        status = self.status
        if has_fatal:
            status = IntegrityStatus.INVALID
        elif has_suspect:
            status = IntegrityStatus.SUSPICIOUS
        elif status is IntegrityStatus.VALID and any(
            issue.code in (
                IntegrityIssueCode.NOT_INSPECTED,
                IntegrityIssueCode.NESTING_LIMIT,
                IntegrityIssueCode.UNSUPPORTED_FORMAT,
            )
            for issue in canonical_issues
        ):
            status = IntegrityStatus.NOT_APPLICABLE
        elif status is IntegrityStatus.VALID and not bool(self.conclusive):
            status = IntegrityStatus.NOT_APPLICABLE
            canonical_issues = canonical_issues + (IntegrityIssue(
                code=IntegrityIssueCode.NOT_INSPECTED,
                severity=IntegritySeverity.ADVISORY,
                message="the serialized result is non-conclusive and cannot be valid",
            ),)
        elif not status_known:
            status = IntegrityStatus.SUSPICIOUS
            canonical_issues = canonical_issues + (IntegrityIssue(
                code=IntegrityIssueCode.NOT_INSPECTED,
                severity=IntegritySeverity.SUSPECT,
                message="the serialized integrity status is unknown and was not trusted",
            ),)
        elif status in (IntegrityStatus.INVALID, IntegrityStatus.SUSPICIOUS):
            severity = (
                IntegritySeverity.FATAL
                if status is IntegrityStatus.INVALID
                else IntegritySeverity.SUSPECT
            )
            canonical_issues = canonical_issues + (IntegrityIssue(
                code=IntegrityIssueCode.NOT_INSPECTED,
                severity=severity,
                message="the serialized integrity verdict did not include supporting evidence",
            ),)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "issues",
            tuple(sorted(canonical_issues, key=lambda issue: issue.sort_key()))[
                : DEFAULT_INTEGRITY_POLICY.max_issues
            ],
        )
        not_fully_inspected = any(
            issue.code in (
                IntegrityIssueCode.NOT_INSPECTED,
                IntegrityIssueCode.NESTING_LIMIT,
                IntegrityIssueCode.UNSUPPORTED_FORMAT,
            )
            for issue in self.issues
        )
        object.__setattr__(
            self, "conclusive", bool(self.conclusive) and not not_fully_inspected
        )
        try:
            object.__setattr__(self, "content_chars", max(0, int(self.content_chars)))
        except (TypeError, ValueError):
            object.__setattr__(self, "content_chars", 0)
        try:
            object.__setattr__(self, "schema_version", int(self.schema_version))
        except (TypeError, ValueError):
            object.__setattr__(self, "schema_version", SCHEMA_VERSION)

    # ------------------------------- queries --------------------------- #
    @property
    def fatal_issues(self) -> tuple[IntegrityIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IntegritySeverity.FATAL)

    @property
    def suspect_issues(self) -> tuple[IntegrityIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IntegritySeverity.SUSPECT)

    @property
    def advisories(self) -> tuple[IntegrityIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IntegritySeverity.ADVISORY)

    @property
    def structurally_complete(self) -> bool:
        return self.status in (IntegrityStatus.VALID, IntegrityStatus.NOT_APPLICABLE)

    @property
    def truncation_suspected(self) -> bool:
        return any(issue.truncation for issue in self.issues)

    def blocking(self, policy: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY) -> bool:
        """Must this body be kept out of the accepted artifact set?"""
        if self.status is IntegrityStatus.INVALID:
            return True
        return (
            self.status is IntegrityStatus.SUSPICIOUS and policy.block_suspicious
        )

    def problem_summary(self) -> str:
        """One deterministic clause naming the strongest detected problem."""
        for issue in self.issues:
            if issue.severity in (IntegritySeverity.FATAL, IntegritySeverity.SUSPECT):
                return issue.message
        return ""

    # ---------------------------- serialization ----------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "mode": self.mode.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "structurally_complete": self.structurally_complete,
            "truncation_suspected": self.truncation_suspected,
            "conclusive": self.conclusive,
            "content_chars": self.content_chars,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactIntegrityResult":
        if not isinstance(data, Mapping):
            raise ArtifactIntegrityError("integrity result must be a JSON object")
        return cls(
            status=data.get("status"),
            artifact_id=data.get("artifact_id") or "",
            path=data.get("path") or "",
            mode=_coerce_enum(data.get("mode"), InspectionMode, InspectionMode.TEXT),
            issues=tuple(
                IntegrityIssue.from_dict(item) for item in (data.get("issues") or ())
            ),
            conclusive=bool(data.get("conclusive", True)),
            content_chars=data.get("content_chars") or 0,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Part G — the bounded generic lexical scanner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LexDialect:
    """What ONE family of languages considers a string, comment or delimiter."""

    name: str
    line_comments: tuple[str, ...] = ()
    block_comments: tuple[tuple[str, str], ...] = ()
    quotes: tuple[str, ...] = ()
    template_quotes: tuple[str, ...] = ()
    escapes: bool = True
    doubled_quote_escape: bool = False     # SQL: '' inside '...' is one quote
    strings_span_lines: bool = False       # JS/CSS strings may not; SQL/shell may
    regex_literals: bool = False           # JS/TS: /.../ is not a division
    # Quotes that may not directly follow an identifier character. In JS/TS no
    # string literal can ever begin right after a word character, while JSX TEXT
    # is full of apostrophes ("It's fine"). Without this rule an ordinary
    # apostrophe in rendered text would open a phantom string.
    quote_needs_boundary: bool = False
    brackets: tuple[tuple[str, str], ...] = (("(", ")"), ("[", "]"), ("{", "}"))


@dataclass(frozen=True)
class LexicalScan:
    """The immutable outcome of one bounded lexical pass.

    ``masked`` is the body with every string, template and comment body replaced
    by spaces (newlines preserved, so offsets and line numbers still line up).
    It is used for delimiter, tag and trailing-token analysis — and it is NEVER
    substituted for the submitted content anywhere.
    """

    masked: str
    unterminated_string: int = -1
    unterminated_template: int = -1
    unterminated_comment: int = -1
    unbalanced_open: tuple[tuple[str, int], ...] = ()
    unbalanced_close: tuple[tuple[str, int], ...] = ()
    depth_exceeded: bool = False

    @property
    def balanced(self) -> bool:
        return not self.unbalanced_open and not self.unbalanced_close

    @property
    def clean(self) -> bool:
        return (
            self.balanced
            and self.unterminated_string < 0
            and self.unterminated_template < 0
            and self.unterminated_comment < 0
        )


# Characters after which a '/' begins a regex literal rather than a division.
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%~^<>\n\t ")
_REGEX_KEYWORDS = (
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "case", "do", "else", "yield", "await", "throw",
)


def scan_lexical(
    content: str,
    dialect: LexDialect,
    *,
    policy: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY,
) -> LexicalScan:
    """One linear, bounded, non-executing pass over ``content``.

    Delimiters inside strings and comments are ignored — that is the whole point
    of masking, and it is why ``const cls = "p-4 { }"`` does not read as an
    unbalanced brace. The scanner claims NO language validity: it reports only
    what it can prove about quotes, comments and delimiter balance.
    """
    length = len(content)
    out: list[str] = []
    stack: list[tuple[str, int]] = []
    unbalanced_close: list[tuple[str, int]] = []
    openers = {pair[0]: pair[1] for pair in dialect.brackets}
    closers = {pair[1]: pair[0] for pair in dialect.brackets}
    unterminated_string = -1
    unterminated_template = -1
    unterminated_comment = -1
    depth_exceeded = False

    i = 0
    prev_significant = ""
    prev_word = ""
    while i < length:
        ch = content[i]

        # --- block comments ------------------------------------------- #
        matched_block = False
        for opener, closer in dialect.block_comments:
            if content.startswith(opener, i):
                end = content.find(closer, i + len(opener))
                if end < 0:
                    unterminated_comment = i
                    out.append(_blank(content[i:]))
                    i = length
                else:
                    out.append(_blank(content[i:end + len(closer)]))
                    i = end + len(closer)
                matched_block = True
                break
        if matched_block:
            continue

        # --- line comments -------------------------------------------- #
        matched_line = False
        for opener in dialect.line_comments:
            if content.startswith(opener, i):
                end = content.find("\n", i)
                end = length if end < 0 else end
                out.append(_blank(content[i:end]))
                i = end
                matched_line = True
                break
        if matched_line:
            continue

        # --- template strings (backtick, with ${...} interpolation) ---- #
        if ch in dialect.template_quotes:
            end = _skip_template(content, i, dialect)
            if end < 0:
                unterminated_template = i
                out.append(_blank(content[i:]))
                i = length
            else:
                out.append(_blank(content[i:end]))
                i = end
            continue

        # --- simple quoted strings ------------------------------------ #
        if ch in dialect.quotes and not (
            dialect.quote_needs_boundary
            and i > 0
            and (content[i - 1].isalnum() or content[i - 1] in "_$")
        ):
            end, terminated = _skip_string(content, i, ch, dialect)
            if terminated:
                out.append(_blank(content[i:end]))
                i = end
                prev_significant = "'"
                prev_word = ""
                continue
            if end >= length:
                # Ran off the end of the content: this is the truncation signal.
                unterminated_string = i
                out.append(_blank(content[i:]))
                i = length
                break
            # Broken at a newline. In a real language this is a syntax error, but
            # it is also what an apostrophe in JSX/HTML text looks like, so the
            # span is left UNMASKED and unreported rather than risking a false
            # rejection cascading through the rest of the file.
            out.append(content[i:end])
            i = end
            continue

        # --- regex literals (JS/TS only) ------------------------------- #
        if dialect.regex_literals and ch == "/":
            # ``</Tag>`` and ``</>`` are JSX closes, never regex literals.
            closes_jsx = i > 0 and content[i - 1] == "<"
            if not closes_jsx and (
                prev_significant in _REGEX_PRECEDERS or prev_word in _REGEX_KEYWORDS
            ):
                end = _skip_regex(content, i)
                if end > 0:
                    out.append(_blank(content[i:end]))
                    i = end
                    prev_significant = "/"
                    prev_word = ""
                    continue

        # --- delimiters ------------------------------------------------ #
        if ch in openers:
            if len(stack) < policy.max_nesting_depth:
                stack.append((ch, i))
            else:
                depth_exceeded = True
        elif ch in closers:
            if stack and stack[-1][0] == closers[ch]:
                stack.pop()
            elif not depth_exceeded:
                unbalanced_close.append((ch, i))

        out.append(ch)
        if not ch.isspace():
            prev_significant = ch
        if ch.isalnum() or ch == "_":
            prev_word += ch
        else:
            prev_word = ""
        i += 1

    return LexicalScan(
        masked="".join(out),
        unterminated_string=unterminated_string,
        unterminated_template=unterminated_template,
        unterminated_comment=unterminated_comment,
        unbalanced_open=tuple(stack),
        unbalanced_close=tuple(unbalanced_close),
        depth_exceeded=depth_exceeded,
    )


def _blank(text: str) -> str:
    """Replace a span with spaces, preserving length AND newlines (offsets hold)."""
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _skip_string(
    content: str, start: int, quote: str, dialect: LexDialect
) -> tuple[int, bool]:
    """Return (end_offset_after_string, terminated)."""
    i = start + 1
    length = len(content)
    while i < length:
        ch = content[i]
        if dialect.escapes and ch == "\\":
            i += 2
            continue
        if ch == quote:
            if dialect.doubled_quote_escape and content.startswith(quote * 2, i):
                i += 2
                continue
            return i + 1, True
        if ch == "\n" and not dialect.strings_span_lines:
            # An unescaped newline closes nothing: in JS/TS/CSS this string is
            # broken. Report it and resume at the newline so one bad line cannot
            # cascade into a whole-file misparse.
            return i, False
        i += 1
    return length, False


def _skip_template(content: str, start: int, dialect: LexDialect) -> int:
    """Return the offset AFTER a closing backtick, or -1 if unterminated."""
    i = start + 1
    length = len(content)
    while i < length:
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "$" and content.startswith("${", i):
            depth = 1
            i += 2
            while i < length and depth:
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                i += 1
            if depth:
                return -1
            continue
        if ch == "`":
            return i + 1
        i += 1
    return -1


def _skip_regex(content: str, start: int) -> int:
    """Return the offset after a /regex/flags literal, or -1 if it is not one."""
    i = start + 1
    length = len(content)
    in_class = False
    while i < length:
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "\n":
            return -1  # a regex literal may not span lines: it was a division
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            i += 1
            while i < length and content[i].isalpha():
                i += 1
            return i
        i += 1
    return -1


_JS_DIALECT = LexDialect(
    name="javascript",
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    quotes=("'", '"'),
    template_quotes=("`",),
    regex_literals=True,
    quote_needs_boundary=True,
)
_CSS_DIALECT = LexDialect(
    name="css",
    block_comments=(("/*", "*/"),),
    quotes=("'", '"'),
    brackets=(("(", ")"), ("[", "]"), ("{", "}")),
)
_SQL_DIALECT = LexDialect(
    name="sql",
    line_comments=("--",),
    block_comments=(("/*", "*/"),),
    quotes=("'", '"'),
    escapes=False,
    doubled_quote_escape=True,
    strings_span_lines=True,
    brackets=(("(", ")"),),
)
_SHELL_DIALECT = LexDialect(
    name="shell",
    line_comments=("#",),
    quotes=("'", '"', "`"),
    strings_span_lines=True,
    brackets=(("(", ")"), ("{", "}")),
)


# --------------------------------------------------------------------------- #
# Part C — universal truncation indicators
# --------------------------------------------------------------------------- #
# Only END-OF-CONTENT evidence is fatal. A model that writes "truncated" in the
# middle of a paragraph is discussing truncation; a file that ENDS with it was.
_END_TRUNCATION_PHRASES = (
    "response truncated", "output truncated", "content truncated",
    "message truncated", "response was cut off", "output cut off",
    "truncated for brevity", "truncated for length", "[truncated]", "<truncated>",
    "rest of file omitted", "rest of the file omitted", "rest of file unchanged",
    "rest of code omitted", "rest of the code omitted", "remaining code here",
    "remaining code omitted", "remaining lines omitted", "implementation omitted",
    "omitted for brevity", "todo: continue", "to be continued",
    "continued in next", "continued in the next", "continued...", "continued…",
    "(continued)", "rest of the implementation", "rest of the component",
    "rest of the file here", "same as before", "unchanged from above",
)
_ELLIPSIS_ONLY = re.compile(r"^(?:\.{2,}|…)$")
_COMMENT_PREFIX = re.compile(r"^\s*(?://+|#+|/\*+|\*+|<!--|--|;+)\s*")
_COMMENT_SUFFIX = re.compile(r"\s*(?:\*/|-->)\s*$")
# A dangling escape at end-of-content: "\", "\u12", "\x", "\U0001F6".
_PARTIAL_ESCAPE = re.compile(r"(?<!\\)\\(?:u[0-9A-Fa-f]{0,3}|U[0-9A-Fa-f]{0,7}|x[0-9A-Fa-f]?)$")

_REFUSAL_LEADS = (
    "i cannot ", "i can't ", "i cant ", "i am sorry", "i'm sorry",
    "im sorry", "sorry,", "as an ai ", "as a large language model",
    "i am unable ", "i'm unable ", "unable to ",
)


def _inside_quoted_span(line: str, offset: int) -> bool:
    """Return whether an offset is inside a simple quote on this line."""
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if index >= offset:
            return bool(quote)
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"', "`"):
            quote = char
    return bool(quote)


def _last_nonempty_line(text: str) -> tuple[str, int]:
    """Return (line, offset_of_line_start). ('' , -1) when there is none."""
    end = len(text)
    while end > 0:
        start = text.rfind("\n", 0, end)
        line = text[start + 1:end]
        if line.strip():
            return line, start + 1
        if start < 0:
            return "", -1
        end = start
    return "", -1


def _universal_issues(
    content: str, mode: InspectionMode, policy: IntegrityPolicy
) -> list[IntegrityIssue]:
    """Provider-independent truncation evidence. Conservative by construction."""
    issues: list[IntegrityIssue] = []
    code_like = mode in _CODE_MODES

    # 1) A model refusal returned INSTEAD of file content.
    refusal_sample = content if len(content) <= 400 else content[:240]
    verdict = validate_model_response(refusal_sample)
    refusal_leads = refusal_sample.lstrip().lower().startswith(_REFUSAL_LEADS)
    if verdict.kind == "refusal" and (not code_like or refusal_leads):
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.REFUSAL_CONTENT,
            severity=IntegritySeverity.FATAL,
            message=(
                "the submitted content is a model refusal, not artifact content "
                f"({verdict.reason})"
            ),
            evidence=content[:policy.max_evidence_chars],
            offset=0, line=1,
        ))

    line, offset = _last_nonempty_line(content)
    if not line:
        return issues
    lowered = line.lower()

    # 2) An explicit end-of-content truncation marker.
    for phrase in _END_TRUNCATION_PHRASES:
        phrase_offset = lowered.find(phrase)
        if phrase_offset >= 0 and not _inside_quoted_span(line, phrase_offset):
            issues.append(IntegrityIssue(
                code=IntegrityIssueCode.TRUNCATION_MARKER,
                severity=IntegritySeverity.FATAL,
                message=(
                    "the content ends with an explicit truncation marker "
                    f"({phrase!r}); the complete artifact was never returned"
                ),
                evidence=line, offset=offset, line=_line_of(content, offset),
                truncation=True,
            ))
            break

    # 3) A placeholder ending: a bare ellipsis, or an ellipsis in the final
    #    comment line. Ordinary ellipses INSIDE code or prose are untouched.
    stripped = line.strip()
    bare = _COMMENT_SUFFIX.sub("", _COMMENT_PREFIX.sub("", stripped)).strip()
    placeholder = _ELLIPSIS_ONLY.match(bare)
    if placeholder:
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.PLACEHOLDER_ENDING,
            severity=(
                IntegritySeverity.FATAL if code_like else IntegritySeverity.SUSPECT
            ),
            message=(
                "the content ends with a placeholder rather than real content: "
                f"{stripped!r}"
            ),
            evidence=stripped, offset=offset, line=_line_of(content, offset),
            truncation=True,
        ))

    # 4) A dangling continuation / partial escape at the very end.
    tail = content.rstrip("\r\n")
    if tail.endswith("\\"):
        match = _PARTIAL_ESCAPE.search(tail[-12:])
        partial = bool(match and len(match.group(0)) > 1)
        issues.append(IntegrityIssue(
            code=(
                IntegrityIssueCode.PARTIAL_ESCAPE if partial
                else IntegrityIssueCode.CONTINUATION_ENDING
            ),
            severity=(
                IntegritySeverity.FATAL if code_like else IntegritySeverity.SUSPECT
            ),
            message=(
                "the content ends inside an unfinished escape sequence" if partial
                else "the content ends with a dangling line-continuation character"
            ),
            evidence=tail[-policy.max_evidence_chars:],
            offset=len(tail) - 1, line=_line_of(content, len(tail) - 1),
            truncation=True,
        ))
    elif _PARTIAL_ESCAPE.search(tail[-12:]):
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.PARTIAL_ESCAPE,
            severity=(
                IntegritySeverity.FATAL if code_like else IntegritySeverity.SUSPECT
            ),
            message="the content ends inside an unfinished escape sequence",
            evidence=tail[-policy.max_evidence_chars:],
            offset=max(0, len(tail) - 1), line=_line_of(content, max(0, len(tail) - 1)),
            truncation=True,
        ))

    # 5) A trailing Unicode replacement character: a byte-level cut mid-codepoint.
    if tail.endswith("�"):
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.REPLACEMENT_CHARACTER,
            severity=IntegritySeverity.SUSPECT,
            message=(
                "the content ends with a Unicode replacement character, which "
                "usually means the reply was cut mid-character"
            ),
            evidence=tail[-24:], offset=len(tail) - 1,
            line=_line_of(content, len(tail) - 1), truncation=True,
        ))

    return issues


# --------------------------------------------------------------------------- #
# Part D — fenced-content integrity
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(\S*)[^\n]*$")
# How far into a non-prose body a wrapping fence may open before we stop reading
# backticks as Markdown (a provider preamble is short; a template literal is not).
_FENCE_LOOKAHEAD_LINES = 3


@dataclass(frozen=True)
class FenceAnalysis:
    """What the fences say about a body. The submitted content is NEVER rewritten.

    ``body`` is a LOGICAL view used only for further inspection: when a provider
    wraps a whole file in one fence, the file itself is what deserves the
    structural checks. Hashing, storage, diagnostics and assembly all continue
    to use the exact submitted content.
    """

    body: str
    fenced: bool = False
    info: str = ""
    issues: tuple[IntegrityIssue, ...] = ()


def analyze_fences(
    content: str, mode: InspectionMode, policy: IntegrityPolicy
) -> FenceAnalysis:
    """Detect fence problems and, when present, expose the logical fenced body.

    Outside PROSE, fence analysis only engages when a fence opens in the first
    few lines — the shape of a provider wrapping a whole file (optionally after
    "Here is the file:"). A stray triple backtick deep inside a template literal
    must never be mistaken for an unclosed Markdown fence.
    """
    lines = content.splitlines()
    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    if not nonblank:
        return FenceAnalysis(body=content)
    opens_early = any(
        _FENCE_RE.match(lines[index]) for index in nonblank[:_FENCE_LOOKAHEAD_LINES]
    )
    if mode is not InspectionMode.PROSE and not opens_early:
        return FenceAnalysis(body=content)

    issues: list[IntegrityIssue] = []
    blocks: list[tuple[int, int, str]] = []  # (open_line, close_line, info)
    open_line = -1
    open_marker = ""
    info = ""
    for index, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        marker, tag = match.group(1), match.group(2)
        if open_line < 0:
            open_line, open_marker, info = index, marker, tag
            continue
        # A closing fence must use the same character and be at least as long,
        # and it carries no info string. Anything else does not close.
        if marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not tag:
            blocks.append((open_line, index, info))
            open_line, open_marker, info = -1, "", ""

    if open_line >= 0:
        offset = sum(len(line) + 1 for line in lines[:open_line])
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNCLOSED_FENCE,
            severity=IntegritySeverity.FATAL,
            message=(
                "the content opens a code fence that is never closed; the reply "
                "was cut off before the fenced block ended"
            ),
            evidence=lines[open_line], offset=offset, line=open_line + 1,
            truncation=True,
        ))
        return FenceAnalysis(
            body=content, fenced=True, info=info, issues=tuple(issues)
        )

    if not blocks:
        return FenceAnalysis(body=content)

    if len(blocks) > 1:
        if mode is not InspectionMode.PROSE:
            issues.append(IntegrityIssue(
                code=IntegrityIssueCode.MULTIPLE_FENCES,
                severity=IntegritySeverity.FATAL,
                message=(
                    f"the content contains {len(blocks)} fenced blocks but one "
                    "complete file was expected"
                ),
                evidence=lines[blocks[0][0]], offset=0, line=blocks[0][0] + 1,
            ))
        return FenceAnalysis(body=content, fenced=True, issues=tuple(issues))

    start, end, tag = blocks[0]
    body = "\n".join(lines[start + 1:end])
    if body.strip() == "":
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.EMPTY_FENCE,
            severity=IntegritySeverity.FATAL,
            message="the content is an empty fenced block: no file was returned",
            evidence=lines[start], offset=0, line=start + 1,
        ))
        return FenceAnalysis(body=body, fenced=True, info=tag, issues=tuple(issues))

    outside = [
        line for index, line in enumerate(lines)
        if (index < start or index > end) and line.strip()
    ]
    if outside and mode is not InspectionMode.PROSE:
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.TRAILING_PROSE,
            severity=IntegritySeverity.FATAL,
            message=(
                "the file content is wrapped in a fenced block with surrounding "
                "prose; return raw file contents"
            ),
            evidence=outside[-1], offset=-1, line=-1,
        ))
    elif mode is not InspectionMode.PROSE:
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.FENCED_CONTENT,
            severity=IntegritySeverity.FATAL,
            message=(
                "the artifact is a raw file but its content is wrapped in a "
                "Markdown code fence; return raw file contents"
            ),
            evidence=lines[start], offset=0, line=start + 1,
        ))
    if mode is InspectionMode.PROSE:
        return FenceAnalysis(body=content, fenced=True, info=tag, issues=tuple(issues))
    return FenceAnalysis(body=body, fenced=True, info=tag, issues=tuple(issues))


# --------------------------------------------------------------------------- #
# Part E — structured-data integrity
# --------------------------------------------------------------------------- #
_JSON_TRUNCATION_HINTS = (
    "unterminated string",
    "expecting ',' delimiter",
    "expecting ':' delimiter",
    "expecting value",
    "expecting property name",
    "expecting object",
    "unexpected end",
)


def _json_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    try:
        json.loads(body)
        return []
    except ValueError as exc:
        strict_error = exc

    message = str(strict_error)
    lowered = message.lower()
    offset = getattr(strict_error, "pos", -1)
    truncated = (
        any(hint in lowered for hint in _JSON_TRUNCATION_HINTS)
        and offset >= len(body.rstrip())
    ) or "unterminated string" in lowered
    return [IntegrityIssue(
        code=IntegrityIssueCode.MALFORMED_JSON,
        severity=IntegritySeverity.FATAL,
        message=(
            "the JSON body is structurally invalid and was not accepted: "
            f"{message}"
        ),
        evidence=body[max(0, offset - 40):offset + 40] if offset >= 0 else "",
        offset=offset,
        line=getattr(strict_error, "lineno", -1),
        truncation=bool(truncated),
    )]


def _jsonc_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    """Parse explicitly declared JSONC; plain JSON never reaches this path."""
    try:
        json.loads(body)
        return []
    except ValueError:
        pass

    stripped = _strip_json_comments(body)
    if stripped != body:
        try:
            json.loads(stripped)
        except ValueError:
            pass
        else:
            return [IntegrityIssue(
                code=IntegrityIssueCode.JSONC_COMMENTS,
                severity=IntegritySeverity.ADVISORY,
                message="the JSON body parses only once comments are removed (JSONC)",
            )]

    return _json_issues(body, policy)


def _strip_json_comments(body: str) -> str:
    """Blank `//` and `/* */` comments that are OUTSIDE JSON strings.

    Length and newlines are preserved so reported offsets stay meaningful, and a
    "//" inside a string value survives untouched.
    """
    out: list[str] = []
    i = 0
    length = len(body)
    while i < length:
        ch = body[i]
        if ch == '"':
            end, _terminated = _skip_string(
                body, i, '"', LexDialect(name="json", quotes=('"',))
            )
            out.append(body[i:end])
            i = end
            continue
        if body.startswith("//", i):
            end = body.find("\n", i)
            end = length if end < 0 else end
            out.append(_blank(body[i:end]))
            i = end
            continue
        if body.startswith("/*", i):
            end = body.find("*/", i + 2)
            if end < 0:
                out.append(_blank(body[i:]))
                i = length
                continue
            out.append(_blank(body[i:end + 2]))
            i = end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _jsonl_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    lines = body.splitlines()
    last_index = max(
        (i for i, line in enumerate(lines) if line.strip()), default=-1
    )
    if last_index < 0:
        return []
    offset = 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                json.loads(line)
            except ValueError as exc:
                final = index == last_index
                return [IntegrityIssue(
                    code=IntegrityIssueCode.MALFORMED_JSONL,
                    severity=IntegritySeverity.FATAL,
                    message=(
                        ("the final JSON Lines record is incomplete: " if final
                         else f"JSON Lines record {index + 1} is malformed: ")
                        + str(exc)
                    ),
                    evidence=line, offset=offset, line=index + 1,
                    truncation=final,
                )]
        offset += len(line) + 1
    return []


def _toml_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    if _tomllib is None:  # pragma: no cover - only on Python < 3.11
        return [IntegrityIssue(
            code=IntegrityIssueCode.UNSUPPORTED_FORMAT,
            severity=IntegritySeverity.ADVISORY,
            message="TOML structural inspection is unavailable on this interpreter",
        )]
    try:
        _tomllib.loads(body)
        return []
    except Exception as exc:  # noqa: BLE001 - tomllib raises TOMLDecodeError
        message = str(exc)
        lowered = message.lower()
        return [IntegrityIssue(
            code=IntegrityIssueCode.MALFORMED_TOML,
            severity=IntegritySeverity.FATAL,
            message=f"the TOML body is structurally invalid: {message}",
            evidence=_last_nonempty_line(body)[0],
            truncation=(
                "unterminated" in lowered
                or "expected" in lowered
                or "invalid" in lowered and "end" in lowered
            ),
        )]


_XML_ENTITY_DECL = re.compile(r"<!ENTITY\b", re.IGNORECASE)


def _xml_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    # Entity declarations are the one XML feature with an expansion hazard. We
    # never expand them: content that declares entities is simply not parsed.
    if _XML_ENTITY_DECL.search(body):
        return [IntegrityIssue(
            code=IntegrityIssueCode.UNSUPPORTED_FORMAT,
            severity=IntegritySeverity.ADVISORY,
            message=(
                "the XML body declares entities, which are never expanded here; "
                "structural inspection was skipped"
            ),
        )]
    from xml.etree import ElementTree  # stdlib, imported locally to keep imports lean

    try:
        ElementTree.fromstring(body)
        return []
    except ElementTree.ParseError as exc:
        message = str(exc)
        lowered = message.lower()
        line, column = getattr(exc, "position", (-1, -1))
        return [IntegrityIssue(
            code=IntegrityIssueCode.MALFORMED_XML,
            severity=IntegritySeverity.FATAL,
            message=f"the XML body is structurally invalid: {message}",
            evidence=_last_nonempty_line(body)[0],
            line=line,
            truncation=(
                "no element found" in lowered
                or "unclosed token" in lowered
                or "mismatched tag" in lowered
            ),
        )]


def _yaml_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    # No YAML dependency is declared by this project, and adding one for a
    # detector is not this phase's business. YAML therefore gets the universal
    # indicators and an honest "not inspected" — never a false invalidity.
    return [IntegrityIssue(
        code=IntegrityIssueCode.UNSUPPORTED_FORMAT,
        severity=IntegritySeverity.ADVISORY,
        message="YAML structural inspection is not supported; only truncation "
                "markers were checked",
    )]


# --------------------------------------------------------------------------- #
# Part F — source-code structural checks
# --------------------------------------------------------------------------- #
_PY_INCOMPLETE_HINTS = (
    "unexpected eof",
    "was never closed",
    "unterminated string literal",
    "unterminated triple-quoted string literal",
    "expected an indented block",
    "invalid syntax at end",
    "incomplete input",
)


def _python_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    """Parse only. ``ast.parse`` compiles nothing and executes nothing."""
    try:
        ast.parse(body)
        return []
    except SyntaxError as exc:
        message = (exc.msg or "syntax error").strip()
        lowered = message.lower()
        line = exc.lineno or -1
        tail = _last_nonempty_line(body)[0].strip()
        incomplete = any(hint in lowered for hint in _PY_INCOMPLETE_HINTS) or bool(
            re.match(r"^@[A-Za-z_][\w.]*\s*(?:\([^)]*\))?$", tail)
            or re.search(r"(?:\bimport|[=+\-*/%&|^,:.]|->)\s*$", tail)
        )
        return [IntegrityIssue(
            code=(
                IntegrityIssueCode.PYTHON_INCOMPLETE if incomplete
                else IntegrityIssueCode.PYTHON_SYNTAX_ERROR
            ),
            severity=IntegritySeverity.FATAL,
            message=(
                (f"the Python body ends in an incomplete construct: {message}"
                 if incomplete
                 else f"the Python body has a syntax error: {message}")
                + f" (line {line})"
            ),
            evidence=(exc.text or "").strip(),
            line=line,
            truncation=incomplete,
        )]
    except ValueError as exc:  # e.g. source containing NUL bytes
        return [IntegrityIssue(
            code=IntegrityIssueCode.PYTHON_SYNTAX_ERROR,
            severity=IntegritySeverity.FATAL,
            message=f"the Python body could not be parsed: {exc}",
        )]


# `=>`, `>=` and `<=` are masked before JSX tag scanning: an arrow function in a
# JSX attribute (`onClick={() => x}`) would otherwise close the opening tag early
# and read as an unclosed element. Masking keeps offsets and never touches the
# submitted content.
_ARROW_MASK = re.compile(r"=>|>=|<=")
_JSX_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][\w.$-]*)?([^<>]*?)(/?)>")
_JSX_CUT_TAG = re.compile(r"<\s*/?\s*[A-Za-z][^<>]*$")
# HTML/JSX elements that legitimately never need a closing tag.
_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr", "!doctype", "!--",
})
# Tags whose closing tag is optional in HTML: never report these as unclosed.
_OPTIONAL_CLOSE_TAGS = frozenset({
    "li", "p", "td", "tr", "th", "option", "thead", "tbody", "tfoot", "dt", "dd",
    "colgroup", "rt", "rp", "optgroup", "caption",
})
# A balanced, comment-free file whose last CODE line ends on one of these is
# almost certainly cut off mid-statement. '<' and '>' are deliberately excluded:
# a trailing JSX close (`</Router>`) is ordinary.
_UNFINISHED_TAIL = re.compile(
    r"(?:[=+\-*/%&|^!?:,.]|\b(?:from|import|export|return|const|let|var|new|"
    r"typeof|await|async|function|class|extends|case|default|else|do|in|of)\s*)$"
)


def _js_issues(
    body: str, mode: InspectionMode, policy: IntegrityPolicy
) -> list[IntegrityIssue]:
    """Conservative lexical structure for JS/TS/JSX/TSX. NOT a compiler.

    This proves nothing about types, imports or semantics — only about quotes,
    template literals, comments, delimiter balance and JSX tag balance, each of
    which is a deterministic, language-independent structural fact.
    """
    scan = scan_lexical(body, _JS_DIALECT, policy=policy)
    issues = _scan_issues(scan, body, policy)
    if issues:
        return issues

    masked = _ARROW_MASK.sub("==", scan.masked)
    stack: list[tuple[str, int]] = []
    for match in _JSX_TAG.finditer(masked):
        closing, name, attrs, self_closing = match.groups()
        name = (name or "").lower()
        if name in _VOID_TAGS or name.startswith("!") or self_closing:
            continue
        if closing:
            if not stack:
                continue  # a stray close proves nothing; stay conservative
            if stack[-1][0] == name:
                stack.pop()
                continue
            if any(entry[0] == name for entry in stack):
                # An ancestor is closing while an inner element never did. In JSX
                # every element must close, so this is a real structural break.
                inner, offset = stack[-1]
                issues.append(IntegrityIssue(
                    code=IntegrityIssueCode.UNCLOSED_TAG,
                    severity=IntegritySeverity.FATAL,
                    message=f"the JSX element <{inner}> is never closed",
                    evidence=body[offset:offset + 60], offset=offset,
                    line=_line_of(body, offset), truncation=True,
                ))
                return issues
            continue
        if re.match(r"\s*extends\b", attrs or ""):
            # A TSX generic arrow such as ``<T extends {}>(x: T) => x`` is not
            # an opening JSX element. Ambiguity must fail toward acceptance.
            continue
        if _is_comparison(masked, match.start()):
            continue  # `a < b`, `useState<Record<string, number>>` — not a tag
        if len(stack) >= policy.max_nesting_depth:
            continue
        stack.append((name, match.start()))
    if stack:
        name, offset = stack[-1]
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNCLOSED_TAG,
            severity=IntegritySeverity.FATAL,
            message=(
                f"the file ends with an unclosed JSX element <{name or 'fragment'}>"
            ),
            evidence=body[offset:offset + 60],
            offset=offset, line=_line_of(body, offset),
            truncation=True,
        ))
        return issues

    if _JSX_CUT_TAG.search(masked.rstrip()):
        offset = masked.rstrip().rfind("<")
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.INCOMPLETE_TAG,
            severity=IntegritySeverity.FATAL,
            message="the file ends inside an unfinished JSX/HTML opening tag",
            evidence=body[offset:offset + 60],
            offset=offset, line=_line_of(body, offset),
            truncation=True,
        ))
        return issues

    tail, offset = _last_nonempty_line(masked)
    if (
        tail
        and len(body.strip()) >= policy.min_meaningful_content_chars
        and _UNFINISHED_TAIL.search(tail.rstrip())
    ):
        line_end = body.find("\n", offset)
        original = body[offset:] if line_end < 0 else body[offset:line_end]
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNFINISHED_STATEMENT,
            severity=IntegritySeverity.SUSPECT,
            message=(
                "the file's last line ends on a dangling token, which usually "
                "means the reply stopped mid-statement"
            ),
            evidence=(original or tail).strip(),
            offset=offset, line=_line_of(body, offset),
            truncation=True,
        ))
    return issues


def _is_comparison(masked: str, start: int) -> bool:
    """Is this '<' a less-than / generic argument rather than a JSX tag?

    A JSX element never opens straight after a value: `useState<Record<...>>`,
    `arr.length < max` and `Foo<Bar>` all do. An element opens after '(', '{',
    '>', ',', '=', a newline or the start of the file.
    """
    index = start - 1
    while index >= 0 and masked[index] in " \t":
        index -= 1
    if index < 0:
        return False
    previous = masked[index]
    return previous.isalnum() or previous in "_$)]"


def _scan_issues(
    scan: LexicalScan, body: str, policy: IntegrityPolicy
) -> list[IntegrityIssue]:
    """Turn one lexical scan into bounded, deterministic issues."""
    issues: list[IntegrityIssue] = []
    if scan.unterminated_comment >= 0:
        offset = scan.unterminated_comment
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNTERMINATED_COMMENT,
            severity=IntegritySeverity.FATAL,
            message="the content ends inside an unterminated block comment",
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset), truncation=True,
        ))
    if scan.unterminated_template >= 0:
        offset = scan.unterminated_template
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNTERMINATED_TEMPLATE,
            severity=IntegritySeverity.FATAL,
            message="the content ends inside an unterminated template literal",
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset), truncation=True,
        ))
    if scan.unterminated_string >= 0:
        offset = scan.unterminated_string
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNTERMINATED_STRING,
            severity=IntegritySeverity.FATAL,
            message="the content contains an unterminated quoted string",
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset), truncation=True,
        ))
    if scan.unbalanced_open:
        char, offset = scan.unbalanced_open[-1]
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNBALANCED_DELIMITER,
            severity=IntegritySeverity.FATAL,
            message=(
                f"the content ends with {len(scan.unbalanced_open)} unclosed "
                f"delimiter(s); the last unclosed {char!r} was never matched"
            ),
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset), truncation=True,
        ))
    if scan.unbalanced_close:
        char, offset = scan.unbalanced_close[0]
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNBALANCED_DELIMITER,
            severity=IntegritySeverity.FATAL,
            message=f"the content closes a delimiter {char!r} that was never opened",
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset),
        ))
    if scan.depth_exceeded:
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.NESTING_LIMIT,
            severity=IntegritySeverity.ADVISORY,
            message=(
                "delimiter nesting exceeded the inspection limit of "
                f"{policy.max_nesting_depth}; balance was not fully verified"
            ),
        ))
    return issues


_HTML_COMMENT_OPEN = "<!--"
_RAW_TEXT_TAGS = ("script", "style")
_HTML_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z!][\w.:-]*)", re.IGNORECASE)


def _html_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    """Conservative HTML structure: comments, attributes, raw text, closing tags.

    Optional-closing-tag behavior (``<li>``, ``<p>``, ``<td>`` …) is explicitly
    respected: valid HTML that never closes them must not be condemned.
    """
    issues: list[IntegrityIssue] = []
    length = len(body)
    lowered = body.lower()   # folded ONCE: raw-text lookups stay linear overall
    stack: list[tuple[str, int]] = []
    i = 0
    while i < length:
        char = body[i]
        if char != "<":
            i += 1
            continue
        if body.startswith(_HTML_COMMENT_OPEN, i):
            end = body.find("-->", i + 4)
            if end < 0:
                issues.append(IntegrityIssue(
                    code=IntegrityIssueCode.UNTERMINATED_COMMENT,
                    severity=IntegritySeverity.FATAL,
                    message="the HTML body ends inside an unterminated comment",
                    evidence=body[i:i + 60], offset=i, line=_line_of(body, i),
                    truncation=True,
                ))
                return issues
            i = end + 3
            continue

        match = _HTML_TAG.match(body, i)
        if not match:
            i += 1
            continue
        closing, name = match.group(1), match.group(2).lower()
        end, attr_offset = _skip_html_tag(body, match.end())
        if attr_offset >= 0:
            issues.append(IntegrityIssue(
                code=IntegrityIssueCode.UNTERMINATED_ATTRIBUTE,
                severity=IntegritySeverity.FATAL,
                message="the HTML body ends inside an unterminated attribute value",
                evidence=body[attr_offset:attr_offset + 60], offset=attr_offset,
                line=_line_of(body, attr_offset), truncation=True,
            ))
            return issues
        if end < 0:
            issues.append(IntegrityIssue(
                code=IntegrityIssueCode.INCOMPLETE_TAG,
                severity=IntegritySeverity.FATAL,
                message="the HTML body ends inside an unfinished opening tag",
                evidence=body[i:i + 60], offset=i, line=_line_of(body, i),
                truncation=True,
            ))
            return issues

        self_closing = body[max(i, end - 2):end].rstrip(">").endswith("/")
        if name in _VOID_TAGS or name.startswith("!") or self_closing:
            i = end
            continue
        if not closing and name in _RAW_TEXT_TAGS:
            close = lowered.find(f"</{name}", end)
            if close < 0:
                issues.append(IntegrityIssue(
                    code=IntegrityIssueCode.UNCLOSED_TAG,
                    severity=IntegritySeverity.FATAL,
                    message=f"the HTML body never closes its <{name}> element",
                    evidence=body[i:i + 60], offset=i, line=_line_of(body, i),
                    truncation=True,
                ))
                return issues
            i = close
            continue
        if closing:
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == name:
                    del stack[index:]
                    break
        elif name not in _OPTIONAL_CLOSE_TAGS:
            if len(stack) < policy.max_nesting_depth:
                stack.append((name, i))
        i = end

    if stack:
        name, offset = stack[-1]
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNCLOSED_TAG,
            severity=IntegritySeverity.FATAL,
            message=f"the HTML body ends with an unclosed <{name}> element",
            evidence=body[offset:offset + 60], offset=offset,
            line=_line_of(body, offset), truncation=True,
        ))
    return issues


def _skip_html_tag(body: str, start: int) -> tuple[int, int]:
    """Return (offset_after_tag, unterminated_attribute_offset).

    ``offset_after_tag`` is -1 when the tag never closes; the attribute offset is
    -1 unless a quoted attribute value runs off the end of the content.
    """
    i = start
    length = len(body)
    while i < length:
        char = body[i]
        if char in "\"'":
            end = body.find(char, i + 1)
            if end < 0:
                return -1, i
            i = end + 1
            continue
        if char == ">":
            return i + 1, -1
        i += 1
    return -1, -1


def _css_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    scan = scan_lexical(body, _CSS_DIALECT, policy=policy)
    issues = _scan_issues(scan, body, policy)
    if issues:
        return issues
    tail, offset = _last_nonempty_line(scan.masked)
    stripped = tail.strip()
    if (
        stripped
        and len(body.strip()) >= policy.min_meaningful_content_chars
        and not stripped.endswith(("}", "{", ";", ",", "*/"))
    ):
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.UNFINISHED_STATEMENT,
            severity=IntegritySeverity.SUSPECT,
            message=(
                "the CSS body's last line ends without a terminator, which "
                "usually means the reply stopped mid-declaration"
            ),
            evidence=stripped, offset=offset, line=_line_of(body, offset),
            truncation=True,
        ))
    return issues


_HEREDOC = re.compile(r"<<-?\s*[\"']?\w+")


def _shell_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    if _HEREDOC.search(body):
        # A heredoc body is arbitrary text and may legally contain lone quotes.
        # Quote analysis there is unreliable, so it is honestly not attempted.
        return [IntegrityIssue(
            code=IntegrityIssueCode.NOT_INSPECTED,
            severity=IntegritySeverity.ADVISORY,
            message="the shell script uses a heredoc; quote balance was not checked",
        )]
    scan = scan_lexical(body, _SHELL_DIALECT, policy=policy)
    return _scan_issues(scan, body, policy)


def _sql_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    masked, unterminated = _mask_sql_dollar_quotes(body)
    if unterminated >= 0:
        return [IntegrityIssue(
            code=IntegrityIssueCode.UNTERMINATED_STRING,
            severity=IntegritySeverity.FATAL,
            message="the SQL body ends inside an unterminated dollar-quoted string",
            evidence=body[unterminated:unterminated + 60],
            offset=unterminated,
            line=_line_of(body, unterminated),
            truncation=True,
        )]
    scan = scan_lexical(masked, _SQL_DIALECT, policy=policy)
    return _scan_issues(scan, body, policy)


_SQL_DOLLAR_OPEN = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


def _mask_sql_dollar_quotes(body: str) -> tuple[str, int]:
    """Mask PostgreSQL dollar-quoted strings without interpreting their body."""
    out: list[str] = []
    cursor = 0
    while True:
        match = _SQL_DOLLAR_OPEN.search(body, cursor)
        if match is None:
            out.append(body[cursor:])
            return "".join(out), -1
        marker = match.group(0)
        end = body.find(marker, match.end())
        if end < 0:
            out.append(body[cursor:match.start()])
            out.append(_blank(body[match.start():]))
            return "".join(out), match.start()
        end += len(marker)
        out.append(body[cursor:match.start()])
        out.append(_blank(body[match.start():end]))
        cursor = end


# --------------------------------------------------------------------------- #
# Part F — unified diffs
# --------------------------------------------------------------------------- #
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_HEADERS = (
    "diff --git", "index ", "--- ", "+++ ", "--- ", "old mode", "new mode",
    "new file mode", "deleted file mode", "similarity index", "rename from",
    "rename to", "copy from", "copy to", "Binary files", "GIT binary patch",
)


def _diff_issues(body: str, policy: IntegrityPolicy) -> list[IntegrityIssue]:
    """Structural validation of a unified diff. Nothing is ever applied."""
    lines = body.splitlines()
    has_git_header = any(line.startswith("diff --git ") for line in lines)
    if has_git_header and any(
        line.startswith(("Binary files ", "GIT binary patch")) for line in lines
    ):
        return []
    rename_only = (
        has_git_header
        and any(line.startswith("rename from ") for line in lines)
        and any(line.startswith("rename to ") for line in lines)
    )
    saw_file_header = False
    saw_hunk = False
    old_left = 0
    new_left = 0
    hunk_line = -1
    for index, line in enumerate(lines):
        header = _HUNK_HEADER.match(line)
        if header:
            if old_left > 0 or new_left > 0:
                return [_incomplete_hunk(body, hunk_line, old_left, new_left)]
            saw_hunk = True
            hunk_line = index + 1
            old_left = int(header.group(2)) if header.group(2) is not None else 1
            new_left = int(header.group(4)) if header.group(4) is not None else 1
            continue
        if old_left > 0 or new_left > 0:
            marker = line[:1]
            if marker == "\\":       # "\ No newline at end of file"
                continue
            if marker == "+":
                new_left -= 1
            elif marker == "-":
                old_left -= 1
            elif marker in (" ", ""):
                old_left -= 1
                new_left -= 1
            else:
                return [IntegrityIssue(
                    code=IntegrityIssueCode.MALFORMED_DIFF,
                    severity=IntegritySeverity.FATAL,
                    message=(
                        f"diff line {index + 1} has an invalid hunk prefix "
                        f"{marker!r}; unified-diff body lines must start with "
                        "' ', '+', '-' or '\\'"
                    ),
                    evidence=line, line=index + 1,
                )]
            if old_left < 0 or new_left < 0:
                return [IntegrityIssue(
                    code=IntegrityIssueCode.MALFORMED_DIFF,
                    severity=IntegritySeverity.FATAL,
                    message=(
                        f"the hunk starting on line {hunk_line} declares fewer "
                        "lines than it contains"
                    ),
                    evidence=line, line=index + 1,
                )]
            continue
        if line.startswith("\\"):
            continue  # "\ No newline at end of file" may trail a finished hunk
        if any(line.startswith(prefix) for prefix in _DIFF_HEADERS):
            saw_file_header = saw_file_header or line.startswith(
                ("diff --git", "--- ", "+++ ")
            )
            continue
        if line.strip():
            # Free prose between hunks is tolerated by git apply only as leading
            # commentary; treat a non-header line inside the patch as malformed
            # only when it precedes no file header at all.
            if not saw_file_header:
                continue
            return [IntegrityIssue(
                code=IntegrityIssueCode.MALFORMED_DIFF,
                severity=IntegritySeverity.FATAL,
                message=(
                    f"diff line {index + 1} is neither a header nor a hunk line"
                ),
                evidence=line, line=index + 1,
            )]

    if old_left > 0 or new_left > 0:
        return [_incomplete_hunk(body, hunk_line, old_left, new_left)]
    if not saw_file_header:
        return [IntegrityIssue(
            code=IntegrityIssueCode.MALFORMED_DIFF,
            severity=IntegritySeverity.FATAL,
            message="the patch has no unified-diff file header (--- / +++)",
            evidence=_last_nonempty_line(body)[0],
        )]
    if not saw_hunk:
        if rename_only:
            return []
        return [IntegrityIssue(
            code=IntegrityIssueCode.MALFORMED_DIFF,
            severity=IntegritySeverity.FATAL,
            message=(
                "the patch declares file headers but contains no hunk (@@ …@@); "
                "no change was actually returned"
            ),
            evidence=_last_nonempty_line(body)[0],
            truncation=True,
        )]
    return []


def _incomplete_hunk(
    body: str, hunk_line: int, old_left: int, new_left: int
) -> IntegrityIssue:
    return IntegrityIssue(
        code=IntegrityIssueCode.INCOMPLETE_HUNK,
        severity=IntegritySeverity.FATAL,
        message=(
            f"the hunk starting on line {hunk_line} is incomplete: "
            f"{max(old_left, 0)} removed-side and {max(new_left, 0)} added-side "
            "line(s) are missing"
        ),
        evidence=_last_nonempty_line(body)[0],
        line=hunk_line,
        truncation=True,
    )


# --------------------------------------------------------------------------- #
# Part H — inspection-mode detection
# --------------------------------------------------------------------------- #
_EXTENSION_MODES: dict[str, InspectionMode] = {
    ".py": InspectionMode.PYTHON, ".pyi": InspectionMode.PYTHON,
    ".js": InspectionMode.JAVASCRIPT, ".mjs": InspectionMode.JAVASCRIPT,
    ".cjs": InspectionMode.JAVASCRIPT, ".jsx": InspectionMode.JAVASCRIPT,
    ".ts": InspectionMode.TYPESCRIPT, ".tsx": InspectionMode.TYPESCRIPT,
    ".mts": InspectionMode.TYPESCRIPT, ".cts": InspectionMode.TYPESCRIPT,
    ".json": InspectionMode.JSON, ".jsonc": InspectionMode.JSONC,
    ".jsonl": InspectionMode.JSONL, ".ndjson": InspectionMode.JSONL,
    ".toml": InspectionMode.TOML,
    ".xml": InspectionMode.XML, ".svg": InspectionMode.XML,
    ".xhtml": InspectionMode.XML, ".xsd": InspectionMode.XML,
    ".html": InspectionMode.HTML, ".htm": InspectionMode.HTML,
    ".css": InspectionMode.CSS, ".scss": InspectionMode.CSS,
    ".less": InspectionMode.CSS,
    ".sh": InspectionMode.SHELL, ".bash": InspectionMode.SHELL,
    ".zsh": InspectionMode.SHELL,
    ".sql": InspectionMode.SQL,
    ".diff": InspectionMode.DIFF, ".patch": InspectionMode.DIFF,
    ".yaml": InspectionMode.YAML, ".yml": InspectionMode.YAML,
    ".md": InspectionMode.PROSE, ".markdown": InspectionMode.PROSE,
    ".rst": InspectionMode.PROSE, ".txt": InspectionMode.TEXT,
}

_LANGUAGE_MODES: dict[str, InspectionMode] = {
    "python": InspectionMode.PYTHON, "py": InspectionMode.PYTHON,
    "javascript": InspectionMode.JAVASCRIPT, "js": InspectionMode.JAVASCRIPT,
    "jsx": InspectionMode.JAVASCRIPT, "node": InspectionMode.JAVASCRIPT,
    "typescript": InspectionMode.TYPESCRIPT, "ts": InspectionMode.TYPESCRIPT,
    "tsx": InspectionMode.TYPESCRIPT,
    "json": InspectionMode.JSON, "jsonc": InspectionMode.JSONC,
    "jsonl": InspectionMode.JSONL, "ndjson": InspectionMode.JSONL,
    "toml": InspectionMode.TOML,
    "xml": InspectionMode.XML, "svg": InspectionMode.XML,
    "html": InspectionMode.HTML,
    "css": InspectionMode.CSS, "scss": InspectionMode.CSS,
    "sh": InspectionMode.SHELL, "bash": InspectionMode.SHELL,
    "shell": InspectionMode.SHELL, "zsh": InspectionMode.SHELL,
    "sql": InspectionMode.SQL,
    "diff": InspectionMode.DIFF, "patch": InspectionMode.DIFF,
    "yaml": InspectionMode.YAML, "yml": InspectionMode.YAML,
    "markdown": InspectionMode.PROSE, "md": InspectionMode.PROSE,
    "text": InspectionMode.TEXT, "plaintext": InspectionMode.TEXT,
}

_MEDIA_MODES: tuple[tuple[str, InspectionMode], ...] = (
    ("application/jsonc", InspectionMode.JSONC),
    ("application/json", InspectionMode.JSON),
    ("application/x-ndjson", InspectionMode.JSONL),
    ("application/jsonl", InspectionMode.JSONL),
    ("application/toml", InspectionMode.TOML),
    ("text/html", InspectionMode.HTML),
    ("text/css", InspectionMode.CSS),
    ("text/markdown", InspectionMode.PROSE),
    ("text/x-diff", InspectionMode.DIFF),
    ("text/x-patch", InspectionMode.DIFF),
    ("text/x-python", InspectionMode.PYTHON),
    ("text/x-sql", InspectionMode.SQL),
    ("application/x-sh", InspectionMode.SHELL),
    ("application/xml", InspectionMode.XML),
    ("text/xml", InspectionMode.XML),
    ("application/yaml", InspectionMode.YAML),
    ("text/yaml", InspectionMode.YAML),
)


def detect_inspection_mode(
    *,
    path: str = "",
    kind: ArtifactKind = ArtifactKind.OTHER,
    operation: ArtifactOperation = ArtifactOperation.CREATE,
    language: str = "",
    media_type: str = "",
) -> InspectionMode:
    """Decide HOW one artifact body should be inspected.

    The manifest-controlled path wins over the provider-supplied ``language``
    hint: a worker cannot relabel ``package.json`` as ``text`` to dodge JSON
    validation.
    """
    kind = _coerce_enum(kind, ArtifactKind, ArtifactKind.OTHER)
    operation = _coerce_enum(operation, ArtifactOperation, ArtifactOperation.CREATE)
    if kind is ArtifactKind.DIRECTORY or operation is ArtifactOperation.DELETE:
        return InspectionMode.NONE
    if kind is ArtifactKind.PATCH:
        return InspectionMode.DIFF
    if kind is ArtifactKind.DOCUMENT:
        # A trusted document declaration owns its semantics even when a strange
        # filename suffix resembles source code. Documents may contain fences.
        return InspectionMode.PROSE

    extension = splitext(str(path or "").lower())[1]
    result_kind = kind in (
        ArtifactKind.COMMAND_RESULT, ArtifactKind.BUILD_RESULT,
        ArtifactKind.TEST_RESULT,
    )
    media = str(media_type or "").strip().lower()
    hint = str(language or "").strip().lower()

    if result_kind:
        # Evidence text, unless the manifest explicitly declares it structured.
        for prefix, mode in _MEDIA_MODES:
            if media.startswith(prefix) and mode in (
                InspectionMode.JSON, InspectionMode.JSONC,
                InspectionMode.JSONL, InspectionMode.XML
            ):
                return mode
        if hint in ("json", "jsonc", "jsonl", "ndjson") or extension in (
            ".json", ".jsonc", ".jsonl"
        ):
            return _LANGUAGE_MODES.get(hint) or _EXTENSION_MODES[extension]
        return InspectionMode.RESULT

    if extension in _EXTENSION_MODES:
        return _EXTENSION_MODES[extension]
    for prefix, mode in _MEDIA_MODES:
        if media.startswith(prefix):
            return mode
    if media.endswith("+xml"):
        return InspectionMode.XML
    if hint in _LANGUAGE_MODES:
        return _LANGUAGE_MODES[hint]
    if kind is ArtifactKind.DOCUMENT:
        return InspectionMode.PROSE
    # Unknown or unsupported: generic text. Universal truncation evidence still
    # applies, but an unrecognized language is NEVER condemned for being unknown.
    return InspectionMode.TEXT


# --------------------------------------------------------------------------- #
# The one public entry point
# --------------------------------------------------------------------------- #
def evaluate_artifact_integrity(
    content: str,
    *,
    artifact_id: str = "",
    path: str = "",
    kind: ArtifactKind = ArtifactKind.OTHER,
    operation: ArtifactOperation = ArtifactOperation.CREATE,
    language: str = "",
    media_type: str = "",
    policy: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY,
) -> ArtifactIntegrityResult:
    """Evaluate ONE submitted body. Pure, deterministic, bounded, side-effect free.

    The worker's ``complete`` flag is never consulted here: this layer measures
    the content itself, so ``"complete": true`` cannot buy a truncated file its
    way into the deliverable.
    """
    if not isinstance(content, str):
        raise ArtifactIntegrityError("artifact content must be a string")
    mode = detect_inspection_mode(
        path=path, kind=kind, operation=operation,
        language=language, media_type=media_type,
    )
    if mode is InspectionMode.NONE or (
        mode is InspectionMode.PROSE and not policy.inspect_documents
    ):
        return ArtifactIntegrityResult(
            status=IntegrityStatus.NOT_APPLICABLE, artifact_id=artifact_id,
            path=path, mode=mode, content_chars=len(content),
        )

    # Bounded inspection: past the caps we do not guess, we say so.
    conclusive = True
    text = content
    if len(text) > policy.max_inspect_chars:
        text = text[: policy.max_inspect_chars]
        conclusive = False
    physical_lines = text.splitlines(keepends=True)
    if len(physical_lines) > policy.max_inspect_lines:
        text = "".join(physical_lines[: policy.max_inspect_lines])
        conclusive = False

    if not text.strip():
        return ArtifactIntegrityResult(
            status=IntegrityStatus.INVALID, artifact_id=artifact_id, path=path,
            mode=mode, content_chars=len(content), conclusive=conclusive,
            issues=(IntegrityIssue(
                code=IntegrityIssueCode.EMPTY_CONTENT,
                severity=IntegritySeverity.FATAL,
                message="no content was returned for an artifact that owes content",
            ),),
        )

    issues: list[IntegrityIssue] = []
    # Never interpret an inspection-boundary cut as the artifact's real end.
    # Strong universal indicators still inspect the actual submitted tail below.
    fences = (
        analyze_fences(text, mode, policy)
        if conclusive else FenceAnalysis(body=text)
    )
    issues.extend(fences.issues)
    body = fences.body
    issues.extend(_universal_issues(content, mode, policy))
    if fences.fenced and body is not text:
        issues.extend(_universal_issues(body, mode, policy))

    fatal_so_far = any(issue.fatal for issue in issues)
    inspect_body = not fatal_so_far and not any(
        issue.code is IntegrityIssueCode.MULTIPLE_FENCES for issue in issues
    )
    if inspect_body:
        issues.extend(_mode_issues(body, mode, policy, conclusive))
    elif not fatal_so_far:
        conclusive = False

    if (
        mode is InspectionMode.RESULT
        and policy.require_result_content
        and len(body.strip()) < policy.min_result_content_chars
    ):
        issues.append(IntegrityIssue(
            code=IntegrityIssueCode.EMPTY_CONTENT,
            severity=IntegritySeverity.FATAL,
            message="the result artifact carries no evidence of what happened",
            evidence=body.strip(),
        ))

    return _finalize(issues, artifact_id, path, mode, len(content), conclusive, policy)


def _mode_issues(
    body: str, mode: InspectionMode, policy: IntegrityPolicy, conclusive: bool
) -> list[IntegrityIssue]:
    """Dispatch the format-aware check. Unknown formats add nothing."""
    if not conclusive:
        # A clipped body will always look unterminated. Refusing to guess beats
        # rejecting a large-but-valid file on an artefact of our own clipping.
        return [IntegrityIssue(
            code=IntegrityIssueCode.NOT_INSPECTED,
            severity=IntegritySeverity.ADVISORY,
            message=(
                "the content exceeded the inspection bounds; structural checks "
                "were not applied"
            ),
        )]
    if mode is InspectionMode.JSON:
        return _json_issues(body, policy)
    if mode is InspectionMode.JSONC:
        return _jsonc_issues(body, policy)
    if mode is InspectionMode.JSONL:
        return _jsonl_issues(body, policy)
    if mode is InspectionMode.TOML:
        return _toml_issues(body, policy)
    if mode is InspectionMode.XML:
        return _xml_issues(body, policy)
    if mode is InspectionMode.YAML:
        return _yaml_issues(body, policy)
    if mode is InspectionMode.PYTHON:
        return _python_issues(body, policy)
    if mode in (InspectionMode.JAVASCRIPT, InspectionMode.TYPESCRIPT):
        return _js_issues(body, mode, policy)
    if mode is InspectionMode.HTML:
        return _html_issues(body, policy)
    if mode is InspectionMode.CSS:
        return _css_issues(body, policy)
    if mode is InspectionMode.SHELL:
        return _shell_issues(body, policy)
    if mode is InspectionMode.SQL:
        return _sql_issues(body, policy)
    if mode is InspectionMode.DIFF:
        return _diff_issues(body, policy)
    return []


def _finalize(
    issues: Sequence[IntegrityIssue],
    artifact_id: str,
    path: str,
    mode: InspectionMode,
    content_chars: int,
    conclusive: bool,
    policy: IntegrityPolicy,
) -> ArtifactIntegrityResult:
    """Bound, order and score the issues. Fatal beats suspect beats advisory."""
    seen: set[tuple] = set()
    unique: list[IntegrityIssue] = []
    bounded_issues = (
        IntegrityIssue(
            code=issue.code,
            severity=issue.severity,
            message=_clip(issue.message, policy.max_message_chars),
            evidence=_clip(issue.evidence, policy.max_evidence_chars),
            offset=issue.offset,
            line=issue.line,
            truncation=issue.truncation,
        )
        for issue in issues
    )
    for issue in sorted(bounded_issues, key=lambda i: i.sort_key()):
        key = (issue.code, issue.offset)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    bounded = tuple(unique[: policy.max_issues])

    if any(issue.severity is IntegritySeverity.FATAL for issue in bounded):
        status = IntegrityStatus.INVALID
    elif any(issue.severity is IntegritySeverity.SUSPECT for issue in bounded):
        status = IntegrityStatus.SUSPICIOUS
    elif mode in (InspectionMode.TEXT, InspectionMode.YAML) or any(
        issue.code in (
            IntegrityIssueCode.UNSUPPORTED_FORMAT,
            IntegrityIssueCode.NOT_INSPECTED,
            IntegrityIssueCode.NESTING_LIMIT,
        )
        for issue in bounded
    ):
        # Nothing was proven either way: an unsupported or uninspected format is
        # NOT_APPLICABLE, never invalid.
        status = IntegrityStatus.NOT_APPLICABLE
    else:
        status = IntegrityStatus.VALID

    return ArtifactIntegrityResult(
        status=status, artifact_id=artifact_id, path=path, mode=mode,
        issues=bounded, conclusive=conclusive, content_chars=content_chars,
    )


def integrity_feedback(
    result: ArtifactIntegrityResult,
    *,
    policy: IntegrityPolicy = DEFAULT_INTEGRITY_POLICY,
) -> str:
    """Bounded corrective text for the EXISTING worker repair loop.

    Names the artifact, states the detected structural problem, and asks for the
    complete artifact back. It never asks for unrelated files, never proposes a
    fix and never requests a fragment/continuation.
    """
    if result.structurally_complete:
        return ""
    problems = [
        issue for issue in result.issues
        if issue.severity is IntegritySeverity.FATAL
        or issue.severity is IntegritySeverity.SUSPECT
    ]
    if policy.advisories_in_feedback:
        problems.extend(result.advisories)
    if not problems:
        return ""
    detail = problems[0].message
    if result.artifact_id and result.path and result.artifact_id != result.path:
        where = f"{result.artifact_id} at {result.path}"
    else:
        where = result.path or result.artifact_id or "the artifact"
    return _clip(
        f"Artifact {where} is incomplete: {detail}. Return the complete, "
        "untruncated contents of every artifact assigned to this package.",
        policy.max_message_chars * 2,
    )
