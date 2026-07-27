"""ConversationFactory — the only place that mints conversation-layer models.

Centralizing construction here guarantees two invariants everywhere:

* Every id is a monotonic ULID from one generator, so **message ordering is
  total**: append order == id sort order == read order.
* Every timestamp is the canonical ISO-8601 UTC shape from one clock.

``SystemClock`` and ``UlidIdGenerator`` are the production adapters for the
``Clock`` / ``IdGenerator`` ports from Phase 1.1; tests substitute fakes.
"""

from __future__ import annotations

from typing import Optional

from ..domain.enums import MessageRole, OriginKind
from ..domain.models import (
    Conversation,
    Message,
    MessageOrigin,
    StorageManifest,
)
from ..domain.types import (
    ConversationId,
    MessageId,
    ProviderId,
    RunId,
    Timestamp,
)
from ..ports import Clock, IdGenerator
from ..utils import generate_ulid, utc_now_iso

_MAX_TITLE_CHARS = 80


def sanitize_text(text: str) -> str:
    """Replace unpaired UTF-16 surrogates with U+FFFD.

    JSON clients can legally deliver lone surrogates (``"\\ud800"`` parses!),
    e.g. from text broken mid-emoji. Such strings cannot be UTF-8 encoded and
    would poison every write they touch; one replacement character keeps the
    message recordable and the rest of its content intact.
    """
    if all(not 0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return text
    return "".join(
        "�" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text
    )


class SystemClock:
    """Production ``Clock``: real UTC time in the canonical format."""

    def now(self) -> Timestamp:
        return Timestamp(utc_now_iso())


class UlidIdGenerator:
    """Production ``IdGenerator``: monotonic Crockford ULIDs."""

    def new_id(self) -> str:
        return generate_ulid()


def derive_title(prompt: str) -> str:
    """A human-scannable conversation title from the first prompt."""
    title = " ".join(sanitize_text(prompt or "").split())
    if not title:
        return "Untitled conversation"
    if len(title) > _MAX_TITLE_CHARS:
        title = title[: _MAX_TITLE_CHARS - 1].rstrip() + "…"
    return title


class ConversationFactory:
    def __init__(
        self,
        ids: Optional[IdGenerator] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.ids = ids or UlidIdGenerator()
        self.clock = clock or SystemClock()

    # --------------------------- conversations ------------------------- #
    def new_conversation(
        self, title: str = "", metadata: Optional[dict[str, object]] = None
    ) -> StorageManifest:
        now = self.clock.now()
        conversation = Conversation(
            id=ConversationId(self.ids.new_id()),
            title=sanitize_text(title) or "Untitled conversation",
            created_at=now,
            updated_at=now,
        )
        if metadata:
            conversation.metadata.values.update(metadata)
        return StorageManifest(conversation=conversation, updated_at=now)

    # ------------------------------ messages --------------------------- #
    def new_message(
        self,
        conversation_id: ConversationId,
        role: MessageRole,
        content: str,
        origin: Optional[MessageOrigin] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Message:
        message = Message(
            id=MessageId(self.ids.new_id()),
            conversation_id=conversation_id,
            role=role,
            content=sanitize_text(content),
            created_at=self.clock.now(),
            origin=origin or MessageOrigin(kind=OriginKind.USER),
        )
        if metadata:
            message.metadata.values.update(metadata)
        return message

    def new_user_message(
        self,
        conversation_id: ConversationId,
        content: str,
        run_id: Optional[RunId] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Message:
        return self.new_message(
            conversation_id, MessageRole.USER, content,
            origin=MessageOrigin(kind=OriginKind.USER, run_id=run_id),
            metadata=metadata,
        )

    def new_assistant_message(
        self,
        conversation_id: ConversationId,
        content: str,
        run_id: Optional[RunId] = None,
        provider: Optional[ProviderId] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Message:
        return self.new_message(
            conversation_id, MessageRole.ASSISTANT, content,
            origin=MessageOrigin(
                kind=OriginKind.SYNTHESIZER,  # final answers come from synthesis
                run_id=run_id, provider=provider, agent_name=agent_name,
            ),
            metadata=metadata,
        )
