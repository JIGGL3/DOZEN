"""ConversationCache (read/write-through, invalidation, pinning, LRU, stats)
and the InProcessEventBus."""

from __future__ import annotations

import threading
import unittest

from dozen.context.domain.enums import EventType
from dozen.context.manager import ConversationCache, InProcessEventBus
from dozen.context.manager.events import ConversationEvents

from .base import ManagerTestCase


class TestReadWriteThrough(ManagerTestCase):
    def test_create_is_write_through(self) -> None:
        manifest = self.create()
        before = self.manager.cache_statistics()
        found = self.manager.get_conversation(manifest.conversation.id)
        after = self.manager.cache_statistics()
        self.assertTrue(found.ok)
        self.assertEqual(after.hits, before.hits + 1)   # served from cache
        self.assertEqual(after.misses, before.misses)

    def test_read_through_fills_cache(self) -> None:
        cid = self.create().conversation.id
        cold = self.new_manager()  # fresh cache, same disk
        first = cold.cache_statistics()
        cold.get_conversation(cid)
        after_miss = cold.cache_statistics()
        self.assertEqual(after_miss.misses, first.misses + 1)
        cold.get_conversation(cid)
        after_hit = cold.cache_statistics()
        self.assertEqual(after_hit.hits, after_miss.hits + 1)

    def test_message_append_is_write_through(self) -> None:
        cid = self.create().conversation.id
        self.manager.read_messages(cid)  # lazy-load: working set now cached
        self.manager.append_user_message(cid, "cached instantly")
        # Prove the next read is served purely from memory: blank the on-disk
        # log; a cached read must not notice the difference.
        from pathlib import Path
        log = Path(self.root) / cid / "messages.jsonl"
        log.write_bytes(b"")
        stored = self.manager.read_messages(cid).unwrap()
        self.assertEqual([m.content for m in stored], ["cached instantly"])
        # And durability really did happen BEFORE the wipe: the write-through
        # counterpart is proven by a fresh manager in test_invalidation below.

    def test_cache_hit_and_miss_events(self) -> None:
        cid = self.create().conversation.id
        cold = self.new_manager()
        heard: list[EventType] = []
        cold.events.bus.subscribe(lambda e: heard.append(e.type))
        cold.read_messages(cid)   # miss (lazy load)
        cold.read_messages(cid)   # hit
        self.assertIn(EventType.CACHE_MISS, heard)
        self.assertIn(EventType.CACHE_HIT, heard)

    def test_invalidation_forces_reload(self) -> None:
        cid = self.create().conversation.id
        self.manager.read_messages(cid)
        # Another manager (separate cache) appends — our cache is now stale.
        self.new_manager().append_user_message(cid, "written elsewhere")
        self.manager.cache.invalidate_messages(cid)
        reloaded = self.manager.read_messages(cid).unwrap()
        self.assertEqual([m.content for m in reloaded], ["written elsewhere"])


class TestCacheUnit(unittest.TestCase):
    def test_pinning_survives_lru_eviction(self) -> None:
        from dozen.context.domain.models import Conversation, StorageManifest
        from dozen.context.domain.types import ConversationId, Timestamp
        from dozen.context.utils import generate_ulid, utc_now_iso

        cache = ConversationCache(max_conversations=2)

        def manifest() -> StorageManifest:
            now = Timestamp(utc_now_iso())
            return StorageManifest(
                conversation=Conversation(
                    id=ConversationId(generate_ulid()), title="x",
                    created_at=now, updated_at=now,
                ),
                updated_at=now,
            )

        first, second, third = manifest(), manifest(), manifest()
        cache.put_manifest(first)
        cache.pin(first.conversation.id)
        cache.put_manifest(second)
        cache.put_manifest(third)  # over capacity: LRU unpinned (second) evicts
        stats = cache.statistics()
        self.assertEqual(stats.evictions, 1)
        self.assertIsNotNone(cache.get_manifest(first.conversation.id))   # pinned
        self.assertIsNone(cache.get_manifest(second.conversation.id))     # evicted
        self.assertIsNotNone(cache.get_manifest(third.conversation.id))

    def test_statistics_snapshot(self) -> None:
        cache = ConversationCache()
        from dozen.context.domain.types import ConversationId
        cache.get_manifest(ConversationId("missing"))
        stats = cache.statistics()
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.hit_rate, 0.0)

    def test_thread_safety_smoke(self) -> None:
        from dozen.context.domain.types import ConversationId
        cache = ConversationCache(max_conversations=8)
        errors: list[BaseException] = []

        def hammer(seed: int) -> None:
            try:
                for i in range(200):
                    cid = ConversationId(f"conv-{(seed + i) % 16}")
                    cache.get_manifest(cid)
                    cache.get_messages(cid)
                    cache.pin(cid) if i % 7 == 0 else cache.unpin(cid)
                    if i % 5 == 0:
                        cache.invalidate(cid)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])


class TestEventBus(unittest.TestCase):
    def test_subscribe_all_and_by_type(self) -> None:
        bus = InProcessEventBus()
        events = ConversationEvents(bus)
        everything: list[EventType] = []
        only_created: list[EventType] = []
        bus.subscribe(lambda e: everything.append(e.type))
        bus.subscribe(lambda e: only_created.append(e.type), EventType.CONVERSATION_CREATED)
        from dozen.context.domain.types import ConversationId
        cid = ConversationId("c1")
        events.conversation_created(cid, "t")
        events.conversation_deleted(cid)
        self.assertEqual(everything,
                         [EventType.CONVERSATION_CREATED, EventType.CONVERSATION_DELETED])
        self.assertEqual(only_created, [EventType.CONVERSATION_CREATED])

    def test_unsubscribe(self) -> None:
        bus = InProcessEventBus()
        events = ConversationEvents(bus)
        heard: list[EventType] = []
        unsubscribe = bus.subscribe(lambda e: heard.append(e.type))
        from dozen.context.domain.types import ConversationId
        events.conversation_created(ConversationId("c1"), "t")
        unsubscribe()
        unsubscribe()  # double-unsubscribe is harmless
        events.conversation_created(ConversationId("c2"), "t")
        self.assertEqual(len(heard), 1)

    def test_raising_handler_is_contained(self) -> None:
        bus = InProcessEventBus()
        events = ConversationEvents(bus)
        survived: list[EventType] = []

        def bomb(_event) -> None:
            raise RuntimeError("handler bug")

        bus.subscribe(bomb)
        bus.subscribe(lambda e: survived.append(e.type))
        from dozen.context.domain.types import ConversationId
        events.conversation_created(ConversationId("c1"), "t")  # must not raise
        self.assertEqual(len(survived), 1)

    def test_recent_ring(self) -> None:
        bus = InProcessEventBus()
        events = ConversationEvents(bus)
        from dozen.context.domain.types import ConversationId
        for i in range(5):
            events.conversation_created(ConversationId(f"c{i}"), "t")
        recent = bus.recent(EventType.CONVERSATION_CREATED)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1].conversation_id, "c4")


if __name__ == "__main__":
    unittest.main()
