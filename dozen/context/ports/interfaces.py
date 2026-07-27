"""Ports — the dependency-inversion line of the context package.

Interfaces only (``typing.Protocol``): concrete implementations are adapters
that arrive in later phases (filesystem repository, system clock, ULID
generator, no-op memory provider, …). The domain and future managers depend on
THESE, never on adapters. All are ``runtime_checkable`` so contract tests can
assert structural conformance of any implementation.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from ..config.models import ContextConfig
from ..domain.enums import ConversationStatus, EventType
from ..domain.models import (
    ContextRequest,
    ContextSection,
    Event,
    Message,
    StorageManifest,
    Summary,
)
from ..domain.results import OperationResult, RepositoryResult
from ..domain.types import ConversationId, MessageId, Timestamp, TokenCount

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Persistence ports
# --------------------------------------------------------------------------- #
@runtime_checkable
class ConversationRepository(Protocol):
    """Durable CRUD for conversation manifests with optimistic concurrency."""

    def create_conversation(self, manifest: StorageManifest) -> RepositoryResult[StorageManifest]: ...

    def read_manifest(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]: ...

    def update_manifest(
        self, manifest: StorageManifest, expected_version: int
    ) -> RepositoryResult[StorageManifest]: ...

    def list_conversations(
        self,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RepositoryResult[list[StorageManifest]]: ...

    def delete_conversation(self, conversation_id: ConversationId) -> OperationResult: ...


@runtime_checkable
class MessageRepository(Protocol):
    """Append-only message log per conversation."""

    def append_messages(
        self, conversation_id: ConversationId, messages: Sequence[Message]
    ) -> RepositoryResult[int]: ...

    def read_messages(
        self,
        conversation_id: ConversationId,
        limit: int = 200,
        after_id: Optional[MessageId] = None,
    ) -> RepositoryResult[list[Message]]: ...

    def count_messages(self, conversation_id: ConversationId) -> RepositoryResult[int]: ...


@runtime_checkable
class SummaryRepository(Protocol):
    """Append-only summary records per conversation."""

    def append_summary(
        self, conversation_id: ConversationId, summary: Summary
    ) -> RepositoryResult[Summary]: ...

    def read_summaries(self, conversation_id: ConversationId) -> RepositoryResult[list[Summary]]: ...


@runtime_checkable
class PersistenceProvider(Protocol):
    """Factory/lifecycle owner for one storage medium's repositories."""

    def conversations(self) -> ConversationRepository: ...

    def messages(self) -> MessageRepository: ...

    def summaries(self) -> SummaryRepository: ...

    def health_check(self) -> OperationResult: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Integration ports (future subsystems plug in here)
# --------------------------------------------------------------------------- #
@runtime_checkable
class EventBus(Protocol):
    """Lifecycle/telemetry fan-out. Returns an unsubscribe callable."""

    def publish(self, event: Event) -> None: ...

    def subscribe(
        self, handler: Callable[[Event], None], event_type: Optional[EventType] = None
    ) -> Callable[[], None]: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Future memory system: contributes sections during context assembly."""

    def inject(
        self, request: ContextRequest, history: Sequence[Message]
    ) -> list[ContextSection]: ...


@runtime_checkable
class Summarizer(Protocol):
    """LLM summarization behind a seam (domain never imports provider code)."""

    def summarize(self, content: str, target_tokens: TokenCount) -> RepositoryResult[str]: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Exact token counting for one provider family (optional adapter)."""

    @property
    def family(self) -> str: ...

    def count_tokens(self, text: str) -> TokenCount: ...


@runtime_checkable
class PipelineStage(Protocol):
    """One stage of the ContextPipeline. State type lands with the pipeline in
    Phase 1.2; the contract is fixed now so stages written later conform."""

    @property
    def name(self) -> str: ...

    def execute(self, state: object) -> object: ...


# --------------------------------------------------------------------------- #
# Environment ports (determinism seams for testing)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Clock(Protocol):
    def now(self) -> Timestamp: ...


@runtime_checkable
class IdGenerator(Protocol):
    def new_id(self) -> str: ...


@runtime_checkable
class ConfigurationProvider(Protocol):
    def context_config(self) -> ContextConfig: ...


@runtime_checkable
class Logger(Protocol):
    def debug(self, message: str, **fields: object) -> None: ...

    def info(self, message: str, **fields: object) -> None: ...

    def warning(self, message: str, **fields: object) -> None: ...

    def error(self, message: str, **fields: object) -> None: ...


@runtime_checkable
class ContextSerializer(Protocol):
    """Pluggable model <-> string codec (JSON in v1)."""

    def serialize(self, model: object) -> str: ...

    def deserialize(self, payload: str, model_type: type) -> RepositoryResult[object]: ...
