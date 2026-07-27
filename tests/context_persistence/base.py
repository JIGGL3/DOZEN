"""Shared fixtures for the persistence-layer tests.

Every test case gets its own temporary storage root (isolated, removed on
teardown) and a fully wired FileSystemPersistenceProvider with fsync disabled
(durability syscalls add nothing to logic tests and slow CI down).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import FileSystemPersistenceProvider
from dozen.context.config.models import PersistenceConfig, StorageConfig
from dozen.context.domain.enums import MessageRole, OriginKind, SummaryStatus
from dozen.context.domain.models import (
    Conversation,
    Message,
    MessageOrigin,
    MessageRange,
    StorageManifest,
    Summary,
)
from dozen.context.domain.types import (
    ConversationId,
    MessageId,
    SummaryId,
    Timestamp,
)
from dozen.context.utils import generate_ulid, utc_now_iso


def make_conversation(title: str = "Test conversation") -> Conversation:
    now = Timestamp(utc_now_iso())
    return Conversation(
        id=ConversationId(generate_ulid()),
        title=title,
        created_at=now,
        updated_at=now,
    )


def make_manifest(conversation: Conversation | None = None) -> StorageManifest:
    conv = conversation or make_conversation()
    return StorageManifest(conversation=conv, updated_at=Timestamp(utc_now_iso()))


def make_message(
    conversation_id: ConversationId,
    content: str = "Hello, orchestrator!",
    role: MessageRole = MessageRole.USER,
    created_at: Timestamp | None = None,
) -> Message:
    return Message(
        id=MessageId(generate_ulid()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=created_at or Timestamp(utc_now_iso()),
        origin=MessageOrigin(kind=OriginKind.USER),
    )


def make_summary(conversation_id: ConversationId, content: str = "summary body") -> Summary:
    anchor = MessageId(generate_ulid())
    return Summary(
        id=SummaryId(generate_ulid()),
        conversation_id=conversation_id,
        source_range=MessageRange(anchor, anchor, count=1),
        created_at=Timestamp(utc_now_iso()),
        content=content,
        status=SummaryStatus.READY,
    )


class PersistenceTestCase(unittest.TestCase):
    """Base: isolated temp root + wired provider per test."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-ctx-store-"))
        self.config = PersistenceConfig(
            storage=StorageConfig(root_path=str(self.tmp / "conversations"), fsync_appends=False)
        )
        self.provider = FileSystemPersistenceProvider(self.config)
        self.conversations = self.provider.conversations()
        self.messages = self.provider.messages()
        self.summaries = self.provider.summaries()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def create_conversation(self, title: str = "Test conversation") -> StorageManifest:
        manifest = make_manifest(make_conversation(title))
        created = self.conversations.create_conversation(manifest)
        self.assertTrue(created.ok, getattr(created, "error", None))
        return created.unwrap()
