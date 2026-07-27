"""RepositoryFactory / FileSystemPersistenceProvider: wiring, port
conformance, backend dispatch, lifecycle, health, and the serializer codec."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import (
    RepositoryFactory,
    StorageSerializer,
    create_persistence,
)
from dozen.context.config.models import ContextConfig, PersistenceConfig, StorageConfig
from dozen.context.domain.enums import ErrorCode
from dozen.context.domain.models import Conversation, Message
from dozen.context.ports import (
    ContextSerializer,
    ConversationRepository,
    MessageRepository,
    PersistenceProvider,
    SummaryRepository,
)

from .base import PersistenceTestCase, make_conversation, make_message


def _config(root: Path) -> PersistenceConfig:
    return PersistenceConfig(storage=StorageConfig(root_path=str(root), fsync_appends=False))


class TestRepositoryFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-factory-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_filesystem_provider_satisfying_all_ports(self) -> None:
        provider = RepositoryFactory.create(_config(self.tmp / "c")).unwrap()
        self.assertIsInstance(provider, PersistenceProvider)
        self.assertIsInstance(provider.conversations(), ConversationRepository)
        self.assertIsInstance(provider.messages(), MessageRepository)
        self.assertIsInstance(provider.summaries(), SummaryRepository)

    def test_accepts_full_context_config(self) -> None:
        cfg = ContextConfig.defaults()
        cfg.persistence.storage.root_path = str(self.tmp / "via-context-config")
        provider = RepositoryFactory.create(cfg).unwrap()
        self.assertTrue(provider.health_check().ok)

    def test_defaults_when_no_config_given(self) -> None:
        self.assertTrue(RepositoryFactory.create().ok)

    def test_unknown_backend_fails_structurally(self) -> None:
        result = RepositoryFactory.create(PersistenceConfig(backend="redis"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)
        self.assertEqual(result.error.details.get("backend"), "redis")
        self.assertIn("filesystem", result.error.details.get("supported", []))

    def test_invalid_config_fails_structurally(self) -> None:
        bad = PersistenceConfig(storage=StorageConfig(root_path="   "))
        result = RepositoryFactory.create(bad)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_wrong_config_type_fails_structurally(self) -> None:
        result = RepositoryFactory.create({"backend": "filesystem"})  # type: ignore[arg-type]
        self.assertFalse(result.ok)

    def test_module_level_helper_matches_factory(self) -> None:
        provider = create_persistence(_config(self.tmp / "helper")).unwrap()
        self.assertTrue(provider.health_check().ok)

    def test_supported_backends(self) -> None:
        self.assertEqual(RepositoryFactory.supported_backends(), ["filesystem"])


class TestProviderLifecycle(PersistenceTestCase):
    def test_health_check_green_on_writable_root(self) -> None:
        self.assertTrue(self.provider.health_check().ok)

    def test_close_is_idempotent_and_fences_use(self) -> None:
        self.provider.close()
        self.provider.close()  # second close: no error
        health = self.provider.health_check()
        self.assertFalse(health.ok)
        with self.assertRaises(RuntimeError):
            self.provider.conversations()

    def test_repositories_are_shared_instances(self) -> None:
        # One lock registry / manifest manager per provider — repositories
        # must be the same wired objects on every call.
        self.assertIs(self.provider.conversations(), self.provider.conversations())
        self.assertIs(self.provider.messages(), self.provider.messages())
        self.assertIs(self.provider.summaries(), self.provider.summaries())

    def test_end_to_end_through_ports_only(self) -> None:
        """The whole storage stack driven purely through port-typed calls."""
        manifest = self.create_conversation("port-driven")
        cid = manifest.conversation.id
        self.assertEqual(self.messages.append_messages(cid, [make_message(cid)]).unwrap(), 1)
        self.assertEqual(self.messages.count_messages(cid).unwrap(), 1)
        self.assertEqual(len(self.conversations.list_conversations().unwrap()), 1)
        self.assertTrue(self.conversations.delete_conversation(cid).ok)
        self.assertEqual(self.conversations.list_conversations().unwrap(), [])


class TestStorageSerializer(unittest.TestCase):
    def setUp(self) -> None:
        self.serializer = StorageSerializer()

    def test_model_round_trip(self) -> None:
        conv = make_conversation("codec check")
        payload = self.serializer.serialize(conv)
        again = self.serializer.deserialize(payload, Conversation).unwrap()
        self.assertEqual(again.to_dict(), conv.to_dict())

    def test_conforms_to_serializer_port(self) -> None:
        self.assertIsInstance(self.serializer, ContextSerializer)

    def test_invalid_json_is_a_failure(self) -> None:
        result = self.serializer.deserialize("{ nope", Conversation)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.SERIALIZATION_FAILED)

    def test_non_object_payload_is_a_failure(self) -> None:
        result = self.serializer.deserialize("[1, 2, 3]", Conversation)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.SERIALIZATION_FAILED)

    def test_non_model_raises_programmer_error(self) -> None:
        with self.assertRaises(TypeError):
            self.serializer.serialize(object())

    def test_strict_mode_rejects_invalid_models(self) -> None:
        strict = StorageSerializer(strict=True)
        conv = make_conversation()
        data = conv.to_dict()
        data["created_at"] = "not-a-timestamp"
        result = strict.from_dict(data, Conversation)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VALIDATION_FAILED)

    def test_tolerant_mode_preserves_future_fields(self) -> None:
        conv = make_conversation()
        message = make_message(conv.id)
        data = message.to_dict()
        data["hologram"] = {"future": True}
        parsed = self.serializer.from_dict(data, Message).unwrap()
        self.assertEqual(parsed.to_dict()["hologram"], {"future": True})


if __name__ == "__main__":
    unittest.main()
