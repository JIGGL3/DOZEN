"""Domain models of the Context Management System (pure data, no I/O).

Serialization contract shared by every model:

* ``to_dict()``   -> JSON-safe dict including ``schema_version``.
* ``from_dict()`` -> tolerant parse: unknown keys are preserved in ``extra``
  and re-emitted by ``to_dict`` (forward compatibility — data written by a
  newer schema survives a round-trip through this one); unknown enum values
  map to the enum's ``UNKNOWN`` member with the raw value kept in ``extra``.
* ``to_json()`` / ``from_json()`` -> string conveniences.
* ``validate()``  -> list of structural problems ('' == valid). Structural
  only: types, presence, ranges. Business rules live in later phases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, TypeVar

from .enums import (
    ContextPurpose,
    ConversationStatus,
    EstimateMethod,
    EventType,
    MessageRole,
    OriginKind,
    SectionOrigin,
    StageStatus,
    SummaryStatus,
    TrimAction,
)
from .types import (
    AgentId,
    ConversationId,
    EventId,
    MessageId,
    ProviderId,
    RunId,
    SectionRef,
    SummaryId,
    Timestamp,
    TokenCount,
    WorkflowId,
)

E = TypeVar("E", bound=Enum)

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def _parse_enum(enum_cls: Type[E], value: object, extra: dict[str, object], key: str) -> E:
    """Parse an enum member; unknown values become UNKNOWN with the raw kept."""
    try:
        return enum_cls(str(value))
    except ValueError:
        extra[f"{key}__raw"] = value
        return enum_cls("unknown")  # every context enum defines UNKNOWN


def _extra_of(data: dict[str, object], known: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in known and k != "schema_version"}


def _is_timestamp(value: object) -> bool:
    return isinstance(value, str) and bool(_ISO_UTC_RE.match(value))


def _validate_id(problems: list[str], name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{name} must be a non-empty string")


class _JsonMixin:
    """to_json/from_json in terms of each model's to_dict/from_dict."""

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)  # type: ignore[attr-defined]

    @classmethod
    def from_json(cls, payload: str) -> "object":
        return cls.from_dict(json.loads(payload))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Statistics & metadata
# --------------------------------------------------------------------------- #
@dataclass
class ConversationStatistics(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("message_count", "approx_tokens", "summary_count", "last_message_at")

    message_count: int = 0
    approx_tokens: int = 0
    summary_count: int = 0
    last_message_at: Optional[Timestamp] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "message_count": self.message_count,
            "approx_tokens": self.approx_tokens,
            "summary_count": self.summary_count,
            "last_message_at": self.last_message_at,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationStatistics":
        last = data.get("last_message_at")
        return cls(
            message_count=int(data.get("message_count", 0) or 0),
            approx_tokens=int(data.get("approx_tokens", 0) or 0),
            summary_count=int(data.get("summary_count", 0) or 0),
            last_message_at=Timestamp(str(last)) if isinstance(last, str) else None,
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for name in ("message_count", "approx_tokens", "summary_count"):
            if int(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        if self.last_message_at is not None and not _is_timestamp(self.last_message_at):
            problems.append("last_message_at must be ISO-8601 UTC (…Z)")
        return problems


@dataclass
class ConversationMetadata(_JsonMixin):
    """Open string-keyed map with reserved namespaces (the extension surface)."""

    SCHEMA_VERSION = 1
    RESERVED_NAMESPACES = ("workflow.", "planner.", "agent.", "user.")
    _KNOWN = ("values",)

    values: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def namespace(self, prefix: str) -> dict[str, object]:
        return {k: v for k, v in self.values.items() if k.startswith(prefix)}

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "values": dict(self.values),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationMetadata":
        values = data.get("values")
        return cls(
            values=dict(values) if isinstance(values, dict) else {},
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for k in self.values:
            if not isinstance(k, str) or not k:
                problems.append("metadata keys must be non-empty strings")
                break
        return problems


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
@dataclass
class Conversation(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "id", "title", "created_at", "updated_at", "status", "parent_id",
        "forked_at_message_id", "version", "metadata", "stats",
    )

    id: ConversationId
    title: str
    created_at: Timestamp
    updated_at: Timestamp
    status: ConversationStatus = ConversationStatus.ACTIVE
    parent_id: Optional[ConversationId] = None
    forked_at_message_id: Optional[MessageId] = None
    version: int = 1
    metadata: ConversationMetadata = field(default_factory=ConversationMetadata)
    stats: ConversationStatistics = field(default_factory=ConversationStatistics)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "forked_at_message_id": self.forked_at_message_id,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "stats": self.stats.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Conversation":
        extra = _extra_of(data, cls._KNOWN)
        meta = data.get("metadata")
        stats = data.get("stats")
        parent = data.get("parent_id")
        fork_msg = data.get("forked_at_message_id")
        return cls(
            id=ConversationId(str(data.get("id", ""))),
            title=str(data.get("title", "")),
            created_at=Timestamp(str(data.get("created_at", ""))),
            updated_at=Timestamp(str(data.get("updated_at", ""))),
            status=_parse_enum(ConversationStatus, data.get("status", "active"), extra, "status"),
            parent_id=ConversationId(str(parent)) if isinstance(parent, str) else None,
            forked_at_message_id=MessageId(str(fork_msg)) if isinstance(fork_msg, str) else None,
            version=int(data.get("version", 1) or 1),
            metadata=ConversationMetadata.from_dict(meta) if isinstance(meta, dict) else ConversationMetadata(),
            stats=ConversationStatistics.from_dict(stats) if isinstance(stats, dict) else ConversationStatistics(),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        if not isinstance(self.title, str):
            problems.append("title must be a string")
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        if not _is_timestamp(self.updated_at):
            problems.append("updated_at must be ISO-8601 UTC (…Z)")
        if self.version < 1:
            problems.append("version must be >= 1")
        if self.status is ConversationStatus.UNKNOWN:
            problems.append("status is unknown")
        if (self.parent_id is None) != (self.forked_at_message_id is None):
            problems.append("parent_id and forked_at_message_id must be set together")
        problems.extend(self.metadata.validate())
        problems.extend(self.stats.validate())
        return problems


# --------------------------------------------------------------------------- #
# Message
# --------------------------------------------------------------------------- #
@dataclass
class MessageOrigin(_JsonMixin):
    """Provenance: who produced a message, in which run/subtask."""

    SCHEMA_VERSION = 1
    _KNOWN = ("kind", "agent_name", "provider", "run_id", "subtask_id")

    kind: OriginKind = OriginKind.USER
    agent_name: Optional[str] = None
    provider: Optional[ProviderId] = None
    run_id: Optional[RunId] = None
    subtask_id: Optional[str] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "kind": self.kind.value,
            "agent_name": self.agent_name,
            "provider": self.provider,
            "run_id": self.run_id,
            "subtask_id": self.subtask_id,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MessageOrigin":
        extra = _extra_of(data, cls._KNOWN)
        prov = data.get("provider")
        run = data.get("run_id")
        return cls(
            kind=_parse_enum(OriginKind, data.get("kind", "user"), extra, "kind"),
            agent_name=str(data["agent_name"]) if isinstance(data.get("agent_name"), str) else None,
            provider=ProviderId(str(prov)) if isinstance(prov, str) else None,
            run_id=RunId(str(run)) if isinstance(run, str) else None,
            subtask_id=str(data["subtask_id"]) if isinstance(data.get("subtask_id"), str) else None,
            extra=extra,
        )

    def validate(self) -> list[str]:
        return ["origin.kind is unknown"] if self.kind is OriginKind.UNKNOWN else []


@dataclass
class Message(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "id", "conversation_id", "role", "content", "origin", "pinned",
        "superseded_by", "created_at", "token_estimate_cache", "metadata",
    )

    id: MessageId
    conversation_id: ConversationId
    role: MessageRole
    content: str
    created_at: Timestamp
    origin: MessageOrigin = field(default_factory=MessageOrigin)
    pinned: bool = False
    superseded_by: Optional[SummaryId] = None
    # provider family -> cached TokenEstimate (filled lazily by later phases)
    token_estimate_cache: dict[str, "TokenEstimate"] = field(default_factory=dict)
    metadata: ConversationMetadata = field(default_factory=ConversationMetadata)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at,
            "origin": self.origin.to_dict(),
            "pinned": self.pinned,
            "superseded_by": self.superseded_by,
            "token_estimate_cache": {k: v.to_dict() for k, v in self.token_estimate_cache.items()},
            "metadata": self.metadata.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Message":
        extra = _extra_of(data, cls._KNOWN)
        origin = data.get("origin")
        meta = data.get("metadata")
        sup = data.get("superseded_by")
        cache_raw = data.get("token_estimate_cache")
        cache: dict[str, TokenEstimate] = {}
        if isinstance(cache_raw, dict):
            for k, v in cache_raw.items():
                if isinstance(v, dict):
                    cache[str(k)] = TokenEstimate.from_dict(v)
        return cls(
            id=MessageId(str(data.get("id", ""))),
            conversation_id=ConversationId(str(data.get("conversation_id", ""))),
            role=_parse_enum(MessageRole, data.get("role", "user"), extra, "role"),
            content=str(data.get("content", "")),
            created_at=Timestamp(str(data.get("created_at", ""))),
            origin=MessageOrigin.from_dict(origin) if isinstance(origin, dict) else MessageOrigin(),
            pinned=bool(data.get("pinned", False)),
            superseded_by=SummaryId(str(sup)) if isinstance(sup, str) else None,
            token_estimate_cache=cache,
            metadata=ConversationMetadata.from_dict(meta) if isinstance(meta, dict) else ConversationMetadata(),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        _validate_id(problems, "conversation_id", self.conversation_id)
        if self.role is MessageRole.UNKNOWN:
            problems.append("role is unknown")
        if not isinstance(self.content, str):
            problems.append("content must be a string")
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        problems.extend(self.origin.validate())
        return problems


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
@dataclass
class MessageRange(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("from_message_id", "to_message_id", "count")

    from_message_id: MessageId
    to_message_id: MessageId
    count: int = 0
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "from_message_id": self.from_message_id,
            "to_message_id": self.to_message_id,
            "count": self.count,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MessageRange":
        return cls(
            from_message_id=MessageId(str(data.get("from_message_id", ""))),
            to_message_id=MessageId(str(data.get("to_message_id", ""))),
            count=int(data.get("count", 0) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "from_message_id", self.from_message_id)
        _validate_id(problems, "to_message_id", self.to_message_id)
        if self.count < 0:
            problems.append("count must be >= 0")
        return problems


@dataclass
class Summary(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "id", "conversation_id", "source_range", "level", "content",
        "status", "tokens", "created_at", "model",
    )

    id: SummaryId
    conversation_id: ConversationId
    source_range: MessageRange
    created_at: Timestamp
    level: int = 1
    content: str = ""
    status: SummaryStatus = SummaryStatus.PENDING
    tokens: TokenCount = TokenCount(0)
    model: Optional[str] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "conversation_id": self.conversation_id,
            "source_range": self.source_range.to_dict(),
            "created_at": self.created_at,
            "level": self.level,
            "content": self.content,
            "status": self.status.value,
            "tokens": int(self.tokens),
            "model": self.model,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Summary":
        extra = _extra_of(data, cls._KNOWN)
        rng = data.get("source_range")
        return cls(
            id=SummaryId(str(data.get("id", ""))),
            conversation_id=ConversationId(str(data.get("conversation_id", ""))),
            source_range=MessageRange.from_dict(rng) if isinstance(rng, dict) else MessageRange(MessageId(""), MessageId("")),
            created_at=Timestamp(str(data.get("created_at", ""))),
            level=int(data.get("level", 1) or 1),
            content=str(data.get("content", "")),
            status=_parse_enum(SummaryStatus, data.get("status", "pending"), extra, "status"),
            tokens=TokenCount(int(data.get("tokens", 0) or 0)),
            model=str(data["model"]) if isinstance(data.get("model"), str) else None,
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        _validate_id(problems, "conversation_id", self.conversation_id)
        if self.level < 1:
            problems.append("level must be >= 1")
        if int(self.tokens) < 0:
            problems.append("tokens must be >= 0")
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        if self.status is SummaryStatus.UNKNOWN:
            problems.append("status is unknown")
        problems.extend(self.source_range.validate())
        return problems


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
@dataclass
class TokenEstimate(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("tokens", "method", "family", "confidence", "per_section")

    tokens: TokenCount = TokenCount(0)
    method: EstimateMethod = EstimateMethod.HEURISTIC
    family: str = ""
    confidence: float = 0.0
    per_section: dict[str, int] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "tokens": int(self.tokens),
            "method": self.method.value,
            "family": self.family,
            "confidence": self.confidence,
            "per_section": dict(self.per_section),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TokenEstimate":
        extra = _extra_of(data, cls._KNOWN)
        per = data.get("per_section")
        return cls(
            tokens=TokenCount(int(data.get("tokens", 0) or 0)),
            method=_parse_enum(EstimateMethod, data.get("method", "heuristic"), extra, "method"),
            family=str(data.get("family", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            per_section={str(k): int(v) for k, v in per.items()} if isinstance(per, dict) else {},
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if int(self.tokens) < 0:
            problems.append("tokens must be >= 0")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be within [0, 1]")
        return problems


@dataclass
class TokenBudget(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("model_max", "transport_max", "reserved_for_response", "usable")

    model_max: TokenCount = TokenCount(0)
    transport_max: TokenCount = TokenCount(0)
    reserved_for_response: TokenCount = TokenCount(0)
    usable: TokenCount = TokenCount(0)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "model_max": int(self.model_max),
            "transport_max": int(self.transport_max),
            "reserved_for_response": int(self.reserved_for_response),
            "usable": int(self.usable),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TokenBudget":
        return cls(
            model_max=TokenCount(int(data.get("model_max", 0) or 0)),
            transport_max=TokenCount(int(data.get("transport_max", 0) or 0)),
            reserved_for_response=TokenCount(int(data.get("reserved_for_response", 0) or 0)),
            usable=TokenCount(int(data.get("usable", 0) or 0)),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for name in ("model_max", "transport_max", "reserved_for_response", "usable"):
            if int(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        return problems


# --------------------------------------------------------------------------- #
# Context window & sections
# --------------------------------------------------------------------------- #
@dataclass
class EvictionRecord(_JsonMixin):
    """The audit trail: nothing leaves context without a trace."""

    SCHEMA_VERSION = 1
    _KNOWN = ("section_ref", "action", "replacement_summary_id")

    section_ref: SectionRef
    action: TrimAction = TrimAction.DROPPED
    replacement_summary_id: Optional[SummaryId] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "section_ref": self.section_ref,
            "action": self.action.value,
            "replacement_summary_id": self.replacement_summary_id,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EvictionRecord":
        extra = _extra_of(data, cls._KNOWN)
        rep = data.get("replacement_summary_id")
        return cls(
            section_ref=SectionRef(str(data.get("section_ref", ""))),
            action=_parse_enum(TrimAction, data.get("action", "dropped"), extra, "action"),
            replacement_summary_id=SummaryId(str(rep)) if isinstance(rep, str) else None,
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "section_ref", self.section_ref)
        if self.action is TrimAction.UNKNOWN:
            problems.append("action is unknown")
        return problems


@dataclass
class ContextWindow(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("budget", "used", "kept", "evicted", "policy_name")

    budget: TokenBudget = field(default_factory=TokenBudget)
    used: TokenEstimate = field(default_factory=TokenEstimate)
    kept: list[SectionRef] = field(default_factory=list)
    evicted: list[EvictionRecord] = field(default_factory=list)
    policy_name: str = ""
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "budget": self.budget.to_dict(),
            "used": self.used.to_dict(),
            "kept": list(self.kept),
            "evicted": [e.to_dict() for e in self.evicted],
            "policy_name": self.policy_name,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextWindow":
        budget = data.get("budget")
        used = data.get("used")
        kept = data.get("kept")
        evicted = data.get("evicted")
        return cls(
            budget=TokenBudget.from_dict(budget) if isinstance(budget, dict) else TokenBudget(),
            used=TokenEstimate.from_dict(used) if isinstance(used, dict) else TokenEstimate(),
            kept=[SectionRef(str(x)) for x in kept] if isinstance(kept, list) else [],
            evicted=[EvictionRecord.from_dict(e) for e in evicted if isinstance(e, dict)]
            if isinstance(evicted, list) else [],
            policy_name=str(data.get("policy_name", "")),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems = self.budget.validate() + self.used.validate()
        for e in self.evicted:
            problems.extend(e.validate())
        return problems


@dataclass
class ContextSection(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("ref", "origin", "priority", "content", "source_ids", "pinned")

    ref: SectionRef
    origin: SectionOrigin = SectionOrigin.HISTORY
    priority: int = 0          # lower evicts first; REQUEST sections never evict
    content: str = ""
    source_ids: list[str] = field(default_factory=list)
    pinned: bool = False
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "ref": self.ref,
            "origin": self.origin.value,
            "priority": self.priority,
            "content": self.content,
            "source_ids": list(self.source_ids),
            "pinned": self.pinned,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextSection":
        extra = _extra_of(data, cls._KNOWN)
        src = data.get("source_ids")
        return cls(
            ref=SectionRef(str(data.get("ref", ""))),
            origin=_parse_enum(SectionOrigin, data.get("origin", "history"), extra, "origin"),
            priority=int(data.get("priority", 0) or 0),
            content=str(data.get("content", "")),
            source_ids=[str(x) for x in src] if isinstance(src, list) else [],
            pinned=bool(data.get("pinned", False)),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "ref", self.ref)
        if self.origin is SectionOrigin.UNKNOWN:
            problems.append("origin is unknown")
        return problems


# --------------------------------------------------------------------------- #
# Context request / result
# --------------------------------------------------------------------------- #
@dataclass
class CurrentInput(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("content", "role")

    content: str = ""
    role: MessageRole = MessageRole.USER
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "content": self.content,
            "role": self.role.value,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CurrentInput":
        extra = _extra_of(data, cls._KNOWN)
        return cls(
            content=str(data.get("content", "")),
            role=_parse_enum(MessageRole, data.get("role", "user"), extra, "role"),
            extra=extra,
        )

    def validate(self) -> list[str]:
        return ["role is unknown"] if self.role is MessageRole.UNKNOWN else []


@dataclass
class ContextRequest(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "request_id", "conversation_id", "purpose", "current_input",
        "target_provider", "target_model", "token_budget_override",
        "policy_override", "extra_sections",
    )

    conversation_id: ConversationId
    purpose: ContextPurpose = ContextPurpose.CHAT
    current_input: CurrentInput = field(default_factory=CurrentInput)
    target_provider: Optional[ProviderId] = None
    target_model: str = ""
    token_budget_override: Optional[TokenBudget] = None
    policy_override: Optional[str] = None
    extra_sections: list[ContextSection] = field(default_factory=list)
    request_id: str = ""
    # Transient (never serialized): a cooperative cancellation token supplied
    # by the caller. The pipeline checks it between stages.
    cancel_token: Optional[object] = field(default=None, compare=False, repr=False)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "purpose": self.purpose.value,
            "current_input": self.current_input.to_dict(),
            "target_provider": self.target_provider,
            "target_model": self.target_model,
            "token_budget_override": self.token_budget_override.to_dict()
            if self.token_budget_override else None,
            "policy_override": self.policy_override,
            "extra_sections": [s.to_dict() for s in self.extra_sections],
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextRequest":
        extra = _extra_of(data, cls._KNOWN)
        cur = data.get("current_input")
        budget = data.get("token_budget_override")
        prov = data.get("target_provider")
        sections = data.get("extra_sections")
        return cls(
            conversation_id=ConversationId(str(data.get("conversation_id", ""))),
            purpose=_parse_enum(ContextPurpose, data.get("purpose", "chat"), extra, "purpose"),
            current_input=CurrentInput.from_dict(cur) if isinstance(cur, dict) else CurrentInput(),
            target_provider=ProviderId(str(prov)) if isinstance(prov, str) else None,
            target_model=str(data.get("target_model", "")),
            token_budget_override=TokenBudget.from_dict(budget) if isinstance(budget, dict) else None,
            policy_override=str(data["policy_override"]) if isinstance(data.get("policy_override"), str) else None,
            extra_sections=[ContextSection.from_dict(s) for s in sections if isinstance(s, dict)]
            if isinstance(sections, list) else [],
            request_id=str(data.get("request_id", "")),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "conversation_id", self.conversation_id)
        if self.purpose is ContextPurpose.UNKNOWN:
            problems.append("purpose is unknown")
        problems.extend(self.current_input.validate())
        if self.token_budget_override is not None:
            problems.extend(self.token_budget_override.validate())
        for s in self.extra_sections:
            problems.extend(s.validate())
        return problems


@dataclass
class RenderedMessage(_JsonMixin):
    """Provider-neutral {role, content} — maps 1:1 onto the LLM seam."""

    SCHEMA_VERSION = 1
    _KNOWN = ("role", "content")

    role: MessageRole = MessageRole.USER
    content: str = ""
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "role": self.role.value,
            "content": self.content,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RenderedMessage":
        extra = _extra_of(data, cls._KNOWN)
        return cls(
            role=_parse_enum(MessageRole, data.get("role", "user"), extra, "role"),
            content=str(data.get("content", "")),
            extra=extra,
        )

    def validate(self) -> list[str]:
        return ["role is unknown"] if self.role is MessageRole.UNKNOWN else []


@dataclass
class PipelineStageResult(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = (
        "stage", "status", "duration_ms", "tokens_before", "tokens_after",
        "notes", "error",
    )

    stage: str = ""
    status: StageStatus = StageStatus.OK
    duration_ms: float = 0.0
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "stage": self.stage,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "notes": list(self.notes),
            "error": self.error,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PipelineStageResult":
        extra = _extra_of(data, cls._KNOWN)
        tb = data.get("tokens_before")
        ta = data.get("tokens_after")
        notes = data.get("notes")
        return cls(
            stage=str(data.get("stage", "")),
            status=_parse_enum(StageStatus, data.get("status", "ok"), extra, "status"),
            duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
            tokens_before=int(tb) if isinstance(tb, (int, float)) else None,
            tokens_after=int(ta) if isinstance(ta, (int, float)) else None,
            notes=[str(n) for n in notes] if isinstance(notes, list) else [],
            error=str(data["error"]) if isinstance(data.get("error"), str) else None,
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.stage:
            problems.append("stage must be non-empty")
        if self.duration_ms < 0:
            problems.append("duration_ms must be >= 0")
        return problems


@dataclass
class PipelineTrace(_JsonMixin):
    """Per-request observability record: timings, token deltas, evictions."""

    SCHEMA_VERSION = 1
    _KNOWN = ("request_id", "stages", "started_at", "finished_at")

    request_id: str = ""
    stages: list[PipelineStageResult] = field(default_factory=list)
    started_at: Optional[Timestamp] = None
    finished_at: Optional[Timestamp] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "stages": [s.to_dict() for s in self.stages],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PipelineTrace":
        stages = data.get("stages")
        started = data.get("started_at")
        finished = data.get("finished_at")
        return cls(
            request_id=str(data.get("request_id", "")),
            stages=[PipelineStageResult.from_dict(s) for s in stages if isinstance(s, dict)]
            if isinstance(stages, list) else [],
            started_at=Timestamp(str(started)) if isinstance(started, str) else None,
            finished_at=Timestamp(str(finished)) if isinstance(finished, str) else None,
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for s in self.stages:
            problems.extend(s.validate())
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None and not _is_timestamp(value):
                problems.append(f"{name} must be ISO-8601 UTC (…Z)")
        return problems


@dataclass
class ContextResult(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("sections", "rendered_messages", "window", "trace")

    sections: list[ContextSection] = field(default_factory=list)
    rendered_messages: list[RenderedMessage] = field(default_factory=list)
    window: ContextWindow = field(default_factory=ContextWindow)
    trace: PipelineTrace = field(default_factory=PipelineTrace)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "sections": [s.to_dict() for s in self.sections],
            "rendered_messages": [m.to_dict() for m in self.rendered_messages],
            "window": self.window.to_dict(),
            "trace": self.trace.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextResult":
        sections = data.get("sections")
        rendered = data.get("rendered_messages")
        window = data.get("window")
        trace = data.get("trace")
        return cls(
            sections=[ContextSection.from_dict(s) for s in sections if isinstance(s, dict)]
            if isinstance(sections, list) else [],
            rendered_messages=[RenderedMessage.from_dict(m) for m in rendered if isinstance(m, dict)]
            if isinstance(rendered, list) else [],
            window=ContextWindow.from_dict(window) if isinstance(window, dict) else ContextWindow(),
            trace=PipelineTrace.from_dict(trace) if isinstance(trace, dict) else PipelineTrace(),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for s in self.sections:
            problems.extend(s.validate())
        for m in self.rendered_messages:
            problems.extend(m.validate())
        problems.extend(self.window.validate())
        problems.extend(self.trace.validate())
        return problems


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class Event(_JsonMixin):
    SCHEMA_VERSION = 1
    _KNOWN = ("id", "type", "conversation_id", "payload", "created_at")

    id: EventId
    type: EventType
    created_at: Timestamp
    conversation_id: Optional[ConversationId] = None
    payload: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "id": self.id,
            "type": self.type.value,
            "created_at": self.created_at,
            "conversation_id": self.conversation_id,
            "payload": dict(self.payload),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Event":
        extra = _extra_of(data, cls._KNOWN)
        conv = data.get("conversation_id")
        payload = data.get("payload")
        return cls(
            id=EventId(str(data.get("id", ""))),
            type=_parse_enum(EventType, data.get("type", "unknown"), extra, "type"),
            created_at=Timestamp(str(data.get("created_at", ""))),
            conversation_id=ConversationId(str(conv)) if isinstance(conv, str) else None,
            payload=dict(payload) if isinstance(payload, dict) else {},
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        _validate_id(problems, "id", self.id)
        if self.type is EventType.UNKNOWN:
            problems.append("type is unknown")
        if not _is_timestamp(self.created_at):
            problems.append("created_at must be ISO-8601 UTC (…Z)")
        return problems


# --------------------------------------------------------------------------- #
# Storage manifest
# --------------------------------------------------------------------------- #
@dataclass
class StorageManifest(_JsonMixin):
    """What `manifest.json` holds: the conversation record + storage format tag."""

    SCHEMA_VERSION = 1
    _KNOWN = ("storage_format_version", "conversation", "updated_at")

    conversation: Conversation
    updated_at: Timestamp
    storage_format_version: int = 1
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "storage_format_version": self.storage_format_version,
            "conversation": self.conversation.to_dict(),
            "updated_at": self.updated_at,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StorageManifest":
        conv = data.get("conversation")
        return cls(
            conversation=Conversation.from_dict(conv) if isinstance(conv, dict)
            else Conversation(
                id=ConversationId(""), title="",
                created_at=Timestamp(""), updated_at=Timestamp(""),
            ),
            updated_at=Timestamp(str(data.get("updated_at", ""))),
            storage_format_version=int(data.get("storage_format_version", 1) or 1),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.storage_format_version < 1:
            problems.append("storage_format_version must be >= 1")
        if not _is_timestamp(self.updated_at):
            problems.append("updated_at must be ISO-8601 UTC (…Z)")
        problems.extend(self.conversation.validate())
        return problems


# --------------------------------------------------------------------------- #
# Execution context (the future object threaded through the orchestrator)
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionContext(_JsonMixin):
    """References (by id, never by object) to everything one execution step
    may need: conversation, workflow, run, provider, agent, task/step, budget,
    validation notes and future memory. Pure model — behavior arrives in later
    phases. Holding ids keeps the domain serializable and free of imports from
    the orchestrator."""

    SCHEMA_VERSION = 1
    _KNOWN = (
        "conversation_id", "workflow_id", "run_id", "provider_id", "agent_id",
        "current_task_id", "current_task_title", "current_step_id",
        "current_step_title", "token_budget", "metadata", "validation_notes",
        "memory_keys",
    )

    conversation_id: Optional[ConversationId] = None
    workflow_id: Optional[WorkflowId] = None
    run_id: Optional[RunId] = None
    provider_id: Optional[ProviderId] = None
    agent_id: Optional[AgentId] = None
    current_task_id: Optional[str] = None
    current_task_title: Optional[str] = None
    current_step_id: Optional[str] = None
    current_step_title: Optional[str] = None
    token_budget: Optional[TokenBudget] = None
    metadata: ConversationMetadata = field(default_factory=ConversationMetadata)
    validation_notes: list[str] = field(default_factory=list)
    # Future memory integration: opaque keys a MemoryProvider will resolve.
    memory_keys: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "conversation_id": self.conversation_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "provider_id": self.provider_id,
            "agent_id": self.agent_id,
            "current_task_id": self.current_task_id,
            "current_task_title": self.current_task_title,
            "current_step_id": self.current_step_id,
            "current_step_title": self.current_step_title,
            "token_budget": self.token_budget.to_dict() if self.token_budget else None,
            "metadata": self.metadata.to_dict(),
            "validation_notes": list(self.validation_notes),
            "memory_keys": list(self.memory_keys),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExecutionContext":
        def _opt(key: str) -> Optional[str]:
            value = data.get(key)
            return str(value) if isinstance(value, str) else None

        budget = data.get("token_budget")
        meta = data.get("metadata")
        notes = data.get("validation_notes")
        mem = data.get("memory_keys")
        conv, wf, run, prov, agent = (
            _opt("conversation_id"), _opt("workflow_id"), _opt("run_id"),
            _opt("provider_id"), _opt("agent_id"),
        )
        return cls(
            conversation_id=ConversationId(conv) if conv else None,
            workflow_id=WorkflowId(wf) if wf else None,
            run_id=RunId(run) if run else None,
            provider_id=ProviderId(prov) if prov else None,
            agent_id=AgentId(agent) if agent else None,
            current_task_id=_opt("current_task_id"),
            current_task_title=_opt("current_task_title"),
            current_step_id=_opt("current_step_id"),
            current_step_title=_opt("current_step_title"),
            token_budget=TokenBudget.from_dict(budget) if isinstance(budget, dict) else None,
            metadata=ConversationMetadata.from_dict(meta) if isinstance(meta, dict) else ConversationMetadata(),
            validation_notes=[str(n) for n in notes] if isinstance(notes, list) else [],
            memory_keys=[str(k) for k in mem] if isinstance(mem, list) else [],
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.token_budget is not None:
            problems.extend(self.token_budget.validate())
        problems.extend(self.metadata.validate())
        return problems
