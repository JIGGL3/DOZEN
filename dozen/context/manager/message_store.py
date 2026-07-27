"""MessageStore — the in-memory working set of conversation messages.

Sits between the ConversationManager and the MessageRepository port:

* **Lazy loading** — nothing is read from disk until the first read of a
  conversation; then the full log is loaded once and served from memory.
* **Read-through** — cache miss -> repository -> cache -> caller.
* **Write-through** — appends go to the repository FIRST; only a durable
  write updates the in-memory list, so the store never claims unwritten data.
* **Invalidation** — external mutations (delete, recovery) drop the entry.
* Emits CACHE_HIT / CACHE_MISS events so cache behavior is observable.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..domain.models import Message
from ..domain.results import RepositoryResult, ok
from ..domain.types import ConversationId, MessageId
from ..ports import MessageRepository
from .cache import CacheStatistics, ConversationCache
from .events import ConversationEvents


class MessageStore:
    def __init__(
        self,
        repository: MessageRepository,
        cache: ConversationCache,
        events: ConversationEvents,
    ) -> None:
        self._repository = repository
        self._cache = cache
        self._events = events

    # ------------------------------- reads ----------------------------- #
    def read(
        self,
        conversation_id: ConversationId,
        limit: int = 200,
        after_id: Optional[MessageId] = None,
    ) -> RepositoryResult[list[Message]]:
        loaded = self._working_set(conversation_id)
        if not loaded.ok:
            return loaded
        messages = loaded.unwrap()
        if after_id is not None:
            anchor = next((i for i, m in enumerate(messages) if m.id == after_id), None)
            messages = messages[anchor + 1:] if anchor is not None else []
        # Clamp: a negative limit must mean "nothing", not Python's
        # from-the-end slice semantics (limit=-1 would return all but one).
        return ok(messages[:max(0, limit)])

    def count(self, conversation_id: ConversationId) -> RepositoryResult[int]:
        cached = self._cache.get_messages(conversation_id)
        if cached is not None:
            self._events.cache_hit(conversation_id, kind="messages")
            return ok(len(cached))
        # Not loaded: a repo count is cheaper than materializing the log.
        self._events.cache_miss(conversation_id, kind="messages")
        return self._repository.count_messages(conversation_id)

    # ------------------------------- writes ---------------------------- #
    def append(
        self, conversation_id: ConversationId, messages: Sequence[Message]
    ) -> RepositoryResult[int]:
        """Durable-first append. The cache is only extended after the
        repository confirms the write (write-through)."""
        written = self._repository.append_messages(conversation_id, messages)
        if not written.ok:
            # Unknown disk state (e.g. lock timeout mid-batch): drop the
            # cached list so the next read reloads the truth from disk.
            self._cache.invalidate_messages(conversation_id)
            return written
        self._cache.append_messages(conversation_id, list(messages))
        return written

    # ----------------------------- lifecycle --------------------------- #
    def invalidate(self, conversation_id: ConversationId) -> None:
        self._cache.invalidate_messages(conversation_id)

    def pin(self, conversation_id: ConversationId) -> None:
        self._cache.pin(conversation_id)

    def unpin(self, conversation_id: ConversationId) -> None:
        self._cache.unpin(conversation_id)

    def statistics(self) -> CacheStatistics:
        return self._cache.statistics()

    # ------------------------------ internals -------------------------- #
    def _working_set(self, conversation_id: ConversationId) -> RepositoryResult[list[Message]]:
        cached = self._cache.get_messages(conversation_id)
        if cached is not None:
            self._events.cache_hit(conversation_id, kind="messages")
            return ok(cached)
        self._events.cache_miss(conversation_id, kind="messages")
        # Lazy load: materialize the complete log once, in id (== append) order.
        loaded = self._repository.read_messages(conversation_id, limit=1_000_000)
        if not loaded.ok:
            return loaded
        messages = loaded.unwrap()
        # Canonical total order is id order (monotonic ULID mint order).
        # Concurrent appends can reach the log in a different interleaving
        # than their ids were minted in; sorting restores the true timeline.
        messages.sort(key=lambda m: m.id)
        self._cache.put_messages(conversation_id, messages)
        return ok(list(messages))
