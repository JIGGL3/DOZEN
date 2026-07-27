"""Foundation-layer tests: model creation, serialization, validation,
schema versioning and forward compatibility."""

from __future__ import annotations

import unittest

from dozen.context.domain.enums import (
    ContextPurpose,
    ConversationStatus,
    MessageRole,
    OriginKind,
    SummaryStatus,
    TrimAction,
)
from dozen.context.domain.models import (
    ContextRequest,
    ContextResult,
    ContextSection,
    ContextWindow,
    Conversation,
    ConversationMetadata,
    ConversationStatistics,
    CurrentInput,
    Event,
    EvictionRecord,
    ExecutionContext,
    Message,
    MessageOrigin,
    MessageRange,
    PipelineStageResult,
    PipelineTrace,
    RenderedMessage,
    StorageManifest,
    Summary,
    TokenBudget,
    TokenEstimate,
)
from dozen.context.domain.types import (
    ConversationId,
    EventId,
    MessageId,
    SummaryId,
    Timestamp,
    TokenCount,
)
from dozen.context.utils import generate_ulid, utc_now_iso

NOW = Timestamp(utc_now_iso())


def make_conversation() -> Conversation:
    return Conversation(
        id=ConversationId(generate_ulid()),
        title="Test conversation",
        created_at=NOW,
        updated_at=NOW,
    )


def make_message(conv: Conversation) -> Message:
    return Message(
        id=MessageId(generate_ulid()),
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="Hello, orchestrator!",
        created_at=NOW,
        origin=MessageOrigin(kind=OriginKind.USER),
    )


class TestModelCreationAndValidation(unittest.TestCase):
    def test_valid_conversation(self) -> None:
        conv = make_conversation()
        self.assertEqual(conv.validate(), [])
        self.assertEqual(conv.status, ConversationStatus.ACTIVE)
        self.assertEqual(conv.version, 1)

    def test_invalid_conversation_reports_problems(self) -> None:
        conv = Conversation(
            id=ConversationId(""), title="x",
            created_at=Timestamp("not-a-time"), updated_at=NOW, version=0,
        )
        problems = conv.validate()
        self.assertTrue(any("id" in p for p in problems))
        self.assertTrue(any("created_at" in p for p in problems))
        self.assertTrue(any("version" in p for p in problems))

    def test_fork_fields_must_pair(self) -> None:
        conv = make_conversation()
        conv.parent_id = ConversationId(generate_ulid())  # no forked_at_message_id
        self.assertTrue(any("together" in p for p in conv.validate()))

    def test_valid_message_and_summary(self) -> None:
        conv = make_conversation()
        msg = make_message(conv)
        self.assertEqual(msg.validate(), [])
        summary = Summary(
            id=SummaryId(generate_ulid()),
            conversation_id=conv.id,
            source_range=MessageRange(msg.id, msg.id, count=1),
            created_at=NOW,
            status=SummaryStatus.PENDING,
        )
        self.assertEqual(summary.validate(), [])

    def test_execution_context_defaults_valid(self) -> None:
        ec = ExecutionContext()
        self.assertEqual(ec.validate(), [])
        ec2 = ExecutionContext(token_budget=TokenBudget(model_max=TokenCount(-1)))
        self.assertTrue(any("model_max" in p for p in ec2.validate()))

    def test_context_request_validation(self) -> None:
        req = ContextRequest(
            conversation_id=ConversationId(generate_ulid()),
            purpose=ContextPurpose.WORKER,
            current_input=CurrentInput(content="do the thing"),
        )
        self.assertEqual(req.validate(), [])


class TestSerializationRoundTrip(unittest.TestCase):
    def test_conversation_round_trip(self) -> None:
        conv = make_conversation()
        conv.metadata.values["workflow.name"] = "stress-test"
        again = Conversation.from_dict(conv.to_dict())
        self.assertEqual(again.to_dict(), conv.to_dict())
        self.assertEqual(again.metadata.values["workflow.name"], "stress-test")

    def test_message_round_trip_with_estimate_cache(self) -> None:
        conv = make_conversation()
        msg = make_message(conv)
        msg.token_estimate_cache["openai"] = TokenEstimate(
            tokens=TokenCount(42), family="openai", confidence=0.8
        )
        again = Message.from_dict(msg.to_dict())
        self.assertEqual(again.token_estimate_cache["openai"].tokens, 42)
        self.assertEqual(again.to_dict(), msg.to_dict())

    def test_full_context_result_round_trip(self) -> None:
        result = ContextResult(
            sections=[ContextSection(ref="s1", content="sys", priority=100)],
            rendered_messages=[RenderedMessage(role=MessageRole.SYSTEM, content="sys")],
            window=ContextWindow(
                budget=TokenBudget(TokenCount(8000), TokenCount(9000), TokenCount(1000), TokenCount(7000)),
                evicted=[EvictionRecord(section_ref="s9", action=TrimAction.SUMMARIZED)],
                policy_name="structured-v1",
            ),
            trace=PipelineTrace(
                request_id="r1",
                stages=[PipelineStageResult(stage="fit", duration_ms=1.5)],
                started_at=NOW,
            ),
        )
        again = ContextResult.from_json(result.to_json())
        self.assertEqual(again.to_dict(), result.to_dict())

    def test_event_and_manifest_round_trip(self) -> None:
        conv = make_conversation()
        manifest = StorageManifest(conversation=conv, updated_at=NOW)
        self.assertEqual(StorageManifest.from_dict(manifest.to_dict()).to_dict(), manifest.to_dict())
        from dozen.context.domain.enums import EventType
        ev = Event(id=EventId(generate_ulid()), type=EventType.MESSAGE_APPENDED,
                   created_at=NOW, conversation_id=conv.id, payload={"n": 1})
        self.assertEqual(Event.from_dict(ev.to_dict()).to_dict(), ev.to_dict())


class TestVersioningAndForwardCompatibility(unittest.TestCase):
    def test_schema_version_emitted(self) -> None:
        conv = make_conversation()
        self.assertEqual(conv.to_dict()["schema_version"], 1)

    def test_unknown_fields_survive_round_trip(self) -> None:
        data = make_conversation().to_dict()
        data["field_from_the_future"] = {"nested": True}
        parsed = Conversation.from_dict(data)
        self.assertEqual(parsed.to_dict()["field_from_the_future"], {"nested": True})

    def test_unknown_enum_value_maps_to_unknown_and_keeps_raw(self) -> None:
        conv = make_conversation()
        msg = make_message(conv)
        data = msg.to_dict()
        data["role"] = "critic"  # role added by a future schema
        parsed = Message.from_dict(data)
        self.assertEqual(parsed.role, MessageRole.UNKNOWN)
        self.assertEqual(parsed.extra["role__raw"], "critic")

    def test_missing_optional_fields_use_defaults(self) -> None:
        minimal = {"id": generate_ulid(), "title": "t", "created_at": NOW, "updated_at": NOW}
        parsed = Conversation.from_dict(minimal)
        self.assertEqual(parsed.status, ConversationStatus.ACTIVE)
        self.assertEqual(parsed.stats.message_count, 0)


if __name__ == "__main__":
    unittest.main()
