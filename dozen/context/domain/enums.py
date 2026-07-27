"""All enums of the context domain.

Every enum carries an ``UNKNOWN`` member: readers built against this schema
must not crash when a *newer* writer introduces a value they don't know
(forward compatibility). Deserialization maps unrecognized values to
``UNKNOWN`` and preserves the raw string alongside the model's ``extra`` bag,
so a round-trip never loses information.
"""

from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    UNKNOWN = "unknown"


class SummaryStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    UNKNOWN = "unknown"


class OriginKind(str, Enum):
    """Who produced a message (provenance)."""

    USER = "user"
    WORKER = "worker"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"
    VERIFIER = "verifier"
    REPAIR = "repair"
    MEMORY = "memory"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class SectionOrigin(str, Enum):
    """Which layer of the structured window a context section belongs to."""

    SYSTEM = "system"
    PINNED = "pinned"
    MEMORY = "memory"
    SUMMARY = "summary"
    HISTORY = "history"
    PAYLOAD = "payload"      # role-specific extras (e.g. dependency outputs)
    REQUEST = "request"      # the current input — never evicted
    UNKNOWN = "unknown"


class TrimAction(str, Enum):
    DROPPED = "dropped"
    SUMMARIZED = "summarized"
    CLIPPED = "clipped"
    UNKNOWN = "unknown"


class ContextPurpose(str, Enum):
    CHAT = "chat"
    WORKER = "worker"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"
    VERIFIER = "verifier"
    REPAIR = "repair"
    UNKNOWN = "unknown"


class PipelineStageName(str, Enum):
    LOAD = "load"
    ASSEMBLE = "assemble"
    MEMORY = "memory"
    ESTIMATE = "estimate"
    FIT = "fit"
    RENDER = "render"
    UNKNOWN = "unknown"


class StageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class EstimateMethod(str, Enum):
    HEURISTIC = "heuristic"
    EXACT = "exact"
    UNKNOWN = "unknown"


class ProviderType(str, Enum):
    """Provider families; mirrors the keys of the existing provider registry."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COPILOT = "copilot"
    GROK = "grok"
    META = "meta"
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    POE = "poe"
    PI = "pi"
    LOCAL = "local"
    OTHER = "other"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_UPDATED = "conversation.updated"
    CONVERSATION_FORKED = "conversation.forked"
    CONVERSATION_ARCHIVED = "conversation.archived"
    CONVERSATION_DELETED = "conversation.deleted"
    # Members below were added in Phase 1.3 (additive — readers built against
    # the 1.1 schema map them to UNKNOWN and keep the raw value, by design).
    CONVERSATION_RENAMED = "conversation.renamed"
    CONVERSATION_LOADED = "conversation.loaded"
    MESSAGE_APPENDED = "message.appended"
    MESSAGE_UPDATED = "message.updated"
    CONTEXT_BUILT = "context.built"
    SUMMARY_REQUESTED = "summary.requested"
    SUMMARY_READY = "summary.ready"
    SUMMARY_FAILED = "summary.failed"
    SUMMARY_ADDED = "summary.added"
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    # Members below were added in Phase 2.1.3 (additive; observation only).
    ATTEMPT_STARTED = "attempt.started"
    ATTEMPT_FINISHED = "attempt.finished"
    ATTEMPT_FAILED = "attempt.failed"
    ATTEMPT_CANCELLED = "attempt.cancelled"
    ATTEMPT_RECORDED = "attempt.recorded"
    ATTEMPT_EVICTED = "attempt.evicted"
    # Members below were added in Phase 2.2.2 (additive; observation only).
    PROVIDER_HEALTH_CHANGED = "provider.health_changed"
    PROVIDER_RECOVERED = "provider.recovered"
    PROVIDER_DEGRADED = "provider.degraded"
    PROVIDER_QUARANTINED = "provider.quarantined"
    HEALTH_SNAPSHOT = "health.snapshot"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    CONFLICT = "conflict"                    # optimistic version mismatch
    VALIDATION_FAILED = "validation_failed"
    SERIALIZATION_FAILED = "serialization_failed"
    VERSION_UNSUPPORTED = "version_unsupported"
    IO_ERROR = "io_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
