"""Shared fixture for Conversation-Management tests: an isolated temp root,
a fully wired ConversationManager, and an event recorder."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import RepositoryFactory
from dozen.context.config.models import PersistenceConfig, StorageConfig
from dozen.context.domain.enums import EventType
from dozen.context.domain.models import Event
from dozen.context.manager import ConversationManager


class RecordedEvents:
    """Subscribes to the manager's bus and keeps everything it hears."""

    def __init__(self, manager: ConversationManager) -> None:
        self.events: list[Event] = []
        self._unsubscribe = manager.events.bus.subscribe(self.events.append)

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]

    def types(self) -> list[EventType]:
        return [e.type for e in self.events]

    def stop(self) -> None:
        self._unsubscribe()


class ManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-convmgr-"))
        self.root = str(self.tmp / "conversations")
        self.manager = self.new_manager()
        self.recorded = RecordedEvents(self.manager)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def new_manager(self) -> ConversationManager:
        """A fresh manager over the SAME root (simulates another process or a
        restart — separate cache, separate registry, shared disk)."""
        provider = RepositoryFactory.create(
            PersistenceConfig(storage=StorageConfig(root_path=self.root, fsync_appends=False))
        ).unwrap()
        return ConversationManager(provider)

    def create(self, title: str = "test conversation"):
        created = self.manager.create_conversation(title=title)
        self.assertTrue(created.ok, getattr(created, "error", None))
        return created.unwrap()
