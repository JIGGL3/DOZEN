"""Port contract tests.

In-memory fakes implement each Protocol; ``runtime_checkable`` isinstance
checks assert structural conformance. Phase 1.2's real adapters (filesystem
repository, system clock, ULID id generator) must pass these same shape checks
— and the repository fakes double as the reference behavior (optimistic
versioning, append ordering) that adapter contract tests will reuse.
"""

from __future__ import annotations

import unittest
from typing import Callable, Optional, Sequence

from dozen.context.config.models import ContextConfig
from dozen.context.domain.enums import ConversationStatus, ErrorCode, EventType, MessageRole
from dozen.context.domain.models import (
    ContextRequest,
    ContextSection,
    Event,
    Message,
    StorageManifest,
    Summary,
)
from dozen.context.domain.results import (
    Failure,
    RepositoryResult,
    Success,
    done,
    fail,
    ok,
)
from dozen.context.domain.types import ConversationId, MessageId, Timestamp, TokenCount
from dozen.context.ports import (
    Clock,
    ConfigurationProvider,
    ConversationRepository,
    EventBus,
    IdGenerator,
    MemoryProvider,
    MessageRepository,
    PipelineStage,
    SummaryRepository,
    Summarizer,
    Tokenizer,
)
from dozen.context.utils import generate_ulid, utc_now_iso

from .test_models import make_conversation, make_message


# --------------------------------------------------------------------------- #
# In-memory fakes (reference implementations of the contracts)
# --------------------------------------------------------------------------- #
class FakeConversationRepository:
    def __init__(self) -> None:
        self._manifests: dict[str, StorageManifest] = {}

    def create_conversation(self, manifest: StorageManifest) -> RepositoryResult[StorageManifest]:
        cid = manifest.conversation.id
        if cid in self._manifests:
            return fail(ErrorCode.ALREADY_EXISTS, "conversation exists", conversation_id=cid)
        self._manifests[cid] = manifest
        return ok(manifest)

    def read_manifest(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]:
        found = self._manifests.get(conversation_id)
        if found is None:
            return fail(ErrorCode.NOT_FOUND, "no such conversation", conversation_id=conversation_id)
        return ok(found)

    def update_manifest(
        self, manifest: StorageManifest, expected_version: int
    ) -> RepositoryResult[StorageManifest]:
        cid = manifest.conversation.id
        current = self._manifests.get(cid)
        if current is None:
            return fail(ErrorCode.NOT_FOUND, "no such conversation", conversation_id=cid)
        if current.conversation.version != expected_version:
            return fail(ErrorCode.CONFLICT, "version mismatch",
                        expected=expected_version, actual=current.conversation.version)
        manifest.conversation.version = expected_version + 1
        self._manifests[cid] = manifest
        return ok(manifest)

    def list_conversations(
        self,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RepositoryResult[list[StorageManifest]]:
        items = [
            m for m in self._manifests.values()
            if status is None or m.conversation.status == status
        ]
        return ok(items[offset:offset + limit])

    def delete_conversation(self, conversation_id: ConversationId):
        self._manifests.pop(conversation_id, None)
        return done()


class FakeMessageRepository:
    def __init__(self) -> None:
        self._log: dict[str, list[Message]] = {}

    def append_messages(
        self, conversation_id: ConversationId, messages: Sequence[Message]
    ) -> RepositoryResult[int]:
        self._log.setdefault(conversation_id, []).extend(messages)
        return ok(len(messages))

    def read_messages(
        self,
        conversation_id: ConversationId,
        limit: int = 200,
        after_id: Optional[MessageId] = None,
    ) -> RepositoryResult[list[Message]]:
        msgs = self._log.get(conversation_id, [])
        if after_id is not None:
            idx = next((i for i, m in enumerate(msgs) if m.id == after_id), None)
            msgs = msgs[idx + 1:] if idx is not None else []
        return ok(msgs[:limit])

    def count_messages(self, conversation_id: ConversationId) -> RepositoryResult[int]:
        return ok(len(self._log.get(conversation_id, [])))


class FakeSummaryRepository:
    def __init__(self) -> None:
        self._summaries: dict[str, list[Summary]] = {}

    def append_summary(self, conversation_id: ConversationId, summary: Summary):
        self._summaries.setdefault(conversation_id, []).append(summary)
        return ok(summary)

    def read_summaries(self, conversation_id: ConversationId):
        return ok(list(self._summaries.get(conversation_id, [])))


class FakeEventBus:
    def __init__(self) -> None:
        self.published: list[Event] = []

    def publish(self, event: Event) -> None:
        self.published.append(event)

    def subscribe(
        self, handler: Callable[[Event], None], event_type: Optional[EventType] = None
    ) -> Callable[[], None]:
        return lambda: None


class FakeMemoryProvider:
    def inject(self, request: ContextRequest, history: Sequence[Message]) -> list[ContextSection]:
        return []


class FakeSummarizer:
    def summarize(self, content: str, target_tokens: TokenCount):
        return ok(content[: int(target_tokens)])


class FakeTokenizer:
    @property
    def family(self) -> str:
        return "fake"

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(len(text) // 4)


class FakeStage:
    @property
    def name(self) -> str:
        return "fake-stage"

    def execute(self, state: object) -> object:
        return state


class FakeClock:
    def now(self) -> Timestamp:
        return Timestamp(utc_now_iso())


class FakeIdGenerator:
    def new_id(self) -> str:
        return generate_ulid()


class FakeConfigProvider:
    def context_config(self) -> ContextConfig:
        return ContextConfig.defaults()


class TestProtocolConformance(unittest.TestCase):
    """Every fake structurally satisfies its port (adapters must too)."""

    def test_all_ports_satisfied(self) -> None:
        checks: list[tuple[object, type]] = [
            (FakeConversationRepository(), ConversationRepository),
            (FakeMessageRepository(), MessageRepository),
            (FakeSummaryRepository(), SummaryRepository),
            (FakeEventBus(), EventBus),
            (FakeMemoryProvider(), MemoryProvider),
            (FakeSummarizer(), Summarizer),
            (FakeTokenizer(), Tokenizer),
            (FakeStage(), PipelineStage),
            (FakeClock(), Clock),
            (FakeIdGenerator(), IdGenerator),
            (FakeConfigProvider(), ConfigurationProvider),
        ]
        for instance, port in checks:
            with self.subTest(port=port.__name__):
                self.assertIsInstance(instance, port)


class TestRepositoryContract(unittest.TestCase):
    """Reference behavior every real repository adapter must reproduce."""

    def setUp(self) -> None:
        self.repo = FakeConversationRepository()
        self.messages = FakeMessageRepository()
        self.conv = make_conversation()
        self.manifest = StorageManifest(conversation=self.conv, updated_at=Timestamp(utc_now_iso()))

    def test_create_then_read(self) -> None:
        self.assertTrue(self.repo.create_conversation(self.manifest).ok)
        read = self.repo.read_manifest(self.conv.id)
        self.assertTrue(read.ok)
        self.assertEqual(read.unwrap().conversation.id, self.conv.id)

    def test_duplicate_create_fails_without_exception(self) -> None:
        self.repo.create_conversation(self.manifest)
        dup = self.repo.create_conversation(self.manifest)
        self.assertIsInstance(dup, Failure)
        self.assertEqual(dup.error.code, ErrorCode.ALREADY_EXISTS)

    def test_read_missing_is_not_found(self) -> None:
        missing = self.repo.read_manifest(ConversationId("nope"))
        self.assertIsInstance(missing, Failure)
        self.assertEqual(missing.error.code, ErrorCode.NOT_FOUND)

    def test_optimistic_versioning(self) -> None:
        self.repo.create_conversation(self.manifest)
        updated = self.repo.update_manifest(self.manifest, expected_version=1)
        self.assertTrue(updated.ok)
        self.assertEqual(updated.unwrap().conversation.version, 2)
        stale = self.repo.update_manifest(self.manifest, expected_version=1)
        self.assertIsInstance(stale, Failure)
        self.assertEqual(stale.error.code, ErrorCode.CONFLICT)

    def test_message_append_preserves_order_and_pagination(self) -> None:
        msgs = [make_message(self.conv) for _ in range(5)]
        self.assertEqual(self.messages.append_messages(self.conv.id, msgs).unwrap(), 5)
        all_msgs = self.messages.read_messages(self.conv.id).unwrap()
        self.assertEqual([m.id for m in all_msgs], [m.id for m in msgs])
        after = self.messages.read_messages(self.conv.id, after_id=msgs[2].id).unwrap()
        self.assertEqual([m.id for m in after], [m.id for m in msgs[3:]])
        self.assertEqual(self.messages.count_messages(self.conv.id).unwrap(), 5)

    def test_message_roles_round_trip_through_repository(self) -> None:
        msg = make_message(self.conv)
        msg.role = MessageRole.SUMMARY
        self.messages.append_messages(self.conv.id, [msg])
        stored = self.messages.read_messages(self.conv.id).unwrap()[0]
        self.assertEqual(stored.role, MessageRole.SUMMARY)


if __name__ == "__main__":
    unittest.main()
