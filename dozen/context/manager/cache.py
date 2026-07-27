"""ConversationCache — the thread-safe in-memory layer over persistence.

Three keyed stores (manifest, messages, metadata snapshots), one policy:

* **Read-through**: callers ask the cache first; on a miss the owner
  (ConversationManager / MessageStore) loads from the repository and calls
  ``put_*`` — the next read is a hit.
* **Write-through**: every successful repository write immediately updates
  the cache, so the cache can never serve data that was not persisted first.
* **Invalidation**: any operation that can change disk state outside the
  cached view (delete, optimistic-conflict retry, external corruption
  recovery) calls ``invalidate``.
* **Pinning**: pinned conversations are never evicted.
* **LRU architecture**: entries live in ``OrderedDict``s and every access
  moves them to the MRU end; ``max_conversations`` bounds the cache and
  evicts from the LRU end (skipping pins). Default is unbounded — Phase 1.4+
  can turn the knob without touching the mechanism.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from ..domain.models import Message, StorageManifest
from ..domain.types import ConversationId


@dataclass(frozen=True)
class CacheStatistics:
    """Immutable snapshot of cache behavior since start (or last reset)."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalidations: int = 0
    evictions: int = 0
    manifest_entries: int = 0
    message_entries: int = 0
    metadata_entries: int = 0
    pinned: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


@dataclass
class _Counters:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalidations: int = 0
    evictions: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ConversationCache:
    def __init__(self, max_conversations: Optional[int] = None) -> None:
        self._lock = threading.RLock()
        self.max_conversations = max_conversations
        self._manifests: "OrderedDict[str, StorageManifest]" = OrderedDict()
        self._messages: "OrderedDict[str, list[Message]]" = OrderedDict()
        self._metadata: "OrderedDict[str, dict[str, object]]" = OrderedDict()
        self._pinned: set[str] = set()
        self._counters = _Counters()

    # ------------------------------ manifests -------------------------- #
    def get_manifest(self, conversation_id: ConversationId) -> Optional[StorageManifest]:
        with self._lock:
            found = self._manifests.get(conversation_id)
            if found is not None:
                self._manifests.move_to_end(conversation_id)
        self._count(hit=found is not None)
        return found

    def put_manifest(self, manifest: StorageManifest) -> None:
        cid = manifest.conversation.id
        with self._lock:
            self._manifests[cid] = manifest
            self._manifests.move_to_end(cid)
            # Keep the metadata snapshot coherent with the manifest.
            self._metadata[cid] = dict(manifest.conversation.metadata.values)
            self._metadata.move_to_end(cid)
            self._evict_if_needed()
        with self._counters.lock:
            self._counters.writes += 1

    # ------------------------------ messages --------------------------- #
    def get_messages(self, conversation_id: ConversationId) -> Optional[list[Message]]:
        """The FULL cached message list, or None when not loaded yet.
        Returns a copy so callers can slice/filter without racing writers."""
        with self._lock:
            found = self._messages.get(conversation_id)
            if found is not None:
                self._messages.move_to_end(conversation_id)
                found = list(found)
        self._count(hit=found is not None)
        return found

    def put_messages(self, conversation_id: ConversationId, messages: list[Message]) -> None:
        with self._lock:
            self._messages[conversation_id] = list(messages)
            self._messages.move_to_end(conversation_id)
            self._evict_if_needed()
        with self._counters.lock:
            self._counters.writes += 1

    def append_messages(self, conversation_id: ConversationId, new: list[Message]) -> bool:
        """Write-through append: extend the cached list IF it is loaded.
        Returns False when the conversation isn't cached (no partial lists —
        the next read lazily loads the complete log instead)."""
        with self._lock:
            cached = self._messages.get(conversation_id)
            if cached is None:
                return False
            cached.extend(new)
            # Keep the working set in canonical id order even when threads
            # append out of mint order (timsort is ~O(n) on the sorted case).
            cached.sort(key=lambda m: m.id)
            self._messages.move_to_end(conversation_id)
        with self._counters.lock:
            self._counters.writes += 1
        return True

    # ------------------------------ metadata --------------------------- #
    def get_metadata(self, conversation_id: ConversationId) -> Optional[dict[str, object]]:
        with self._lock:
            found = self._metadata.get(conversation_id)
            if found is not None:
                self._metadata.move_to_end(conversation_id)
                found = dict(found)
        self._count(hit=found is not None)
        return found

    # ------------------------------ pinning ---------------------------- #
    def pin(self, conversation_id: ConversationId) -> None:
        with self._lock:
            self._pinned.add(conversation_id)

    def unpin(self, conversation_id: ConversationId) -> None:
        with self._lock:
            self._pinned.discard(conversation_id)

    def is_pinned(self, conversation_id: ConversationId) -> bool:
        with self._lock:
            return conversation_id in self._pinned

    # ---------------------------- invalidation ------------------------- #
    def invalidate(self, conversation_id: ConversationId) -> None:
        with self._lock:
            self._manifests.pop(conversation_id, None)
            self._messages.pop(conversation_id, None)
            self._metadata.pop(conversation_id, None)
            self._pinned.discard(conversation_id)
        with self._counters.lock:
            self._counters.invalidations += 1

    def invalidate_messages(self, conversation_id: ConversationId) -> None:
        with self._lock:
            self._messages.pop(conversation_id, None)
        with self._counters.lock:
            self._counters.invalidations += 1

    def clear(self) -> None:
        with self._lock:
            self._manifests.clear()
            self._messages.clear()
            self._metadata.clear()
            self._pinned.clear()
        with self._counters.lock:
            self._counters.invalidations += 1

    # ------------------------------- stats ----------------------------- #
    def statistics(self) -> CacheStatistics:
        with self._counters.lock:
            hits, misses = self._counters.hits, self._counters.misses
            writes = self._counters.writes
            invalidations = self._counters.invalidations
            evictions = self._counters.evictions
        with self._lock:
            return CacheStatistics(
                hits=hits, misses=misses, writes=writes,
                invalidations=invalidations, evictions=evictions,
                manifest_entries=len(self._manifests),
                message_entries=len(self._messages),
                metadata_entries=len(self._metadata),
                pinned=len(self._pinned),
            )

    # ------------------------------ internals -------------------------- #
    def _count(self, hit: bool) -> None:
        with self._counters.lock:
            if hit:
                self._counters.hits += 1
            else:
                self._counters.misses += 1

    def _evict_if_needed(self) -> None:
        """LRU eviction (caller holds self._lock). Pinned entries survive."""
        if self.max_conversations is None:
            return
        while len(self._manifests) > self.max_conversations:
            victim = next(
                (cid for cid in self._manifests if cid not in self._pinned), None
            )
            if victim is None:
                return  # everything is pinned: over-capacity but untouchable
            self._manifests.pop(victim, None)
            self._messages.pop(victim, None)
            self._metadata.pop(victim, None)
            with self._counters.lock:
                self._counters.evictions += 1
