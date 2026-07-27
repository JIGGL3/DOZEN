"""ConversationManager — THE public interface to conversation persistence.

Façade rule (SADD Phase 1.3): no component outside ``dozen/context`` touches
a repository directly. The manager owns and coordinates:

    repositories (via the PersistenceProvider port)
    ConversationCache        read-through / write-through / pinning / stats
    MessageStore             the in-memory working set of message logs
    ConversationEvents       lifecycle event publishing (EventBus port)
    ConversationRegistry     live workflow sessions
    ConversationFactory      id/timestamp minting (total message ordering)
    ConversationMetadataManager   namespaced metadata mutations

Thread model: many readers / one writer per conversation, provided by the
persistence layer's per-conversation locks; the manager's own structures
(cache, registry, bus) are individually thread-safe, and manifest mutations
go through optimistic-version retry (``mutate_manifest``) so concurrent
logical updates never silently overwrite each other. That combination is the
groundwork for future multi-agent execution.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, Sequence

from ..domain.enums import ConversationStatus, ErrorCode
from ..domain.models import Message, StorageManifest
from ..domain.results import (
    OperationResult,
    RepositoryResult,
    fail,
    ok,
)
from ..domain.types import ConversationId, MessageId, ProviderId, RunId, WorkflowId
from ..ports import PersistenceProvider
from .cache import CacheStatistics, ConversationCache
from .events import ConversationEvents, InProcessEventBus
from .factory import ConversationFactory, derive_title
from .message_store import MessageStore
from .metadata import ConversationMetadataManager
from .session import ConversationRegistry, ConversationSession

_MUTATE_ATTEMPTS = 10  # optimistic-conflict retries before giving up
_MUTATE_BACKOFF_MAX = 0.01  # jitter cap per retry (seconds)


class ConversationManager:
    def __init__(
        self,
        provider: PersistenceProvider,
        events: Optional[ConversationEvents] = None,
        cache: Optional[ConversationCache] = None,
        factory: Optional[ConversationFactory] = None,
        registry: Optional[ConversationRegistry] = None,
    ) -> None:
        self._provider = provider
        self._conversations = provider.conversations()
        self._summaries = provider.summaries()
        self.events = events or ConversationEvents(InProcessEventBus())
        self.cache = cache or ConversationCache()
        self.factory = factory or ConversationFactory()
        self.registry = registry or ConversationRegistry(
            ids=self.factory.ids, clock=self.factory.clock
        )
        self.messages = MessageStore(provider.messages(), self.cache, self.events)
        self.metadata = ConversationMetadataManager(self)
        # In-process serialization of manifest mutations, per conversation:
        # same-process writers queue here (no spurious CAS conflicts); the
        # optimistic version check still guards against OTHER processes.
        self._mutation_guard = threading.Lock()
        self._mutation_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def create_conversation(
        self,
        title: str = "",
        metadata: Optional[dict[str, object]] = None,
    ) -> RepositoryResult[StorageManifest]:
        manifest = self.factory.new_conversation(title=title, metadata=metadata)
        created = self._conversations.create_conversation(manifest)
        if not created.ok:
            return created
        stored = created.unwrap()
        self.cache.put_manifest(stored)  # write-through
        self.events.conversation_created(stored.conversation.id, stored.conversation.title)
        return created

    def get_conversation(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]:
        cached = self.cache.get_manifest(conversation_id)
        if cached is not None:
            self.events.cache_hit(conversation_id, kind="manifest")
            return ok(cached)
        self.events.cache_miss(conversation_id, kind="manifest")
        loaded = self._conversations.read_manifest(conversation_id)
        if not loaded.ok:
            return loaded
        manifest = loaded.unwrap()
        self.cache.put_manifest(manifest)  # read-through fill
        self.events.conversation_loaded(conversation_id, source="repository")
        return loaded

    def conversation_exists(self, conversation_id: ConversationId) -> bool:
        if self.cache.get_manifest(conversation_id) is not None:
            self.events.cache_hit(conversation_id, kind="manifest")
            return True
        return self._conversations.read_manifest(conversation_id).ok

    def list_conversations(
        self,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RepositoryResult[list[StorageManifest]]:
        return self._conversations.list_conversations(status=status, limit=limit, offset=offset)

    def rename_conversation(
        self, conversation_id: ConversationId, new_title: str
    ) -> RepositoryResult[StorageManifest]:
        if not isinstance(new_title, str) or not new_title.strip():
            return fail(ErrorCode.VALIDATION_FAILED, "title must be a non-empty string")
        old_title = ""

        def mutate(manifest: StorageManifest) -> None:
            nonlocal old_title
            old_title = manifest.conversation.title
            manifest.conversation.title = new_title.strip()

        result = self.mutate_manifest(conversation_id, mutate)
        if result.ok:
            self.events.conversation_renamed(conversation_id, old_title, new_title.strip())
        return result

    def archive_conversation(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]:
        def mutate(manifest: StorageManifest) -> None:
            manifest.conversation.status = ConversationStatus.ARCHIVED

        result = self.mutate_manifest(conversation_id, mutate)
        if result.ok:
            self.registry.end_sessions_for(conversation_id)
            self.events.conversation_archived(conversation_id)
        return result

    def delete_conversation(self, conversation_id: ConversationId) -> OperationResult:
        deleted = self._conversations.delete_conversation(conversation_id)
        # Invalidate regardless: even a failed delete leaves disk state in
        # doubt, and the cache must never outlive certainty.
        self.cache.invalidate(conversation_id)
        if deleted.ok:
            self.registry.end_sessions_for(conversation_id)
            self.events.conversation_deleted(conversation_id)
        return deleted

    # ------------------------------------------------------------------ #
    # Messages (ordering: monotonic ULIDs from one factory => append order
    # == id order == read order, across threads)
    # ------------------------------------------------------------------ #
    def append_user_message(
        self,
        conversation_id: ConversationId,
        content: str,
        run_id: Optional[RunId] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> RepositoryResult[Message]:
        if not isinstance(content, str) or not content.strip():
            return fail(ErrorCode.VALIDATION_FAILED, "message content must be non-empty")
        message = self.factory.new_user_message(
            conversation_id, content, run_id=run_id, metadata=metadata
        )
        return self._append(conversation_id, message)

    def append_assistant_message(
        self,
        conversation_id: ConversationId,
        content: str,
        run_id: Optional[RunId] = None,
        provider: Optional[ProviderId] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> RepositoryResult[Message]:
        if not isinstance(content, str) or not content.strip():
            return fail(ErrorCode.VALIDATION_FAILED, "message content must be non-empty")
        message = self.factory.new_assistant_message(
            conversation_id, content, run_id=run_id,
            provider=provider, agent_name=agent_name, metadata=metadata,
        )
        return self._append(conversation_id, message)

    def read_messages(
        self,
        conversation_id: ConversationId,
        limit: int = 200,
        after_id: Optional[MessageId] = None,
    ) -> RepositoryResult[list[Message]]:
        return self.messages.read(conversation_id, limit=limit, after_id=after_id)

    def count_messages(self, conversation_id: ConversationId) -> RepositoryResult[int]:
        return self.messages.count(conversation_id)

    def _append(
        self, conversation_id: ConversationId, message: Message
    ) -> RepositoryResult[Message]:
        problems = message.validate()
        if problems:
            return fail(ErrorCode.VALIDATION_FAILED, "invalid message", problems=problems)
        written = self.messages.append(conversation_id, [message])
        if not written.ok:
            return written
        # Appends move manifest statistics on disk; refresh the cached copy.
        refreshed = self._conversations.read_manifest(conversation_id)
        if refreshed.ok:
            self.cache.put_manifest(refreshed.unwrap())
        self.events.message_added(conversation_id, message.id, message.role.value)
        for session in self.registry.sessions_for(conversation_id):
            session.touch(self.factory.clock)
        return ok(message)

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def start_session(
        self,
        conversation_id: ConversationId,
        run_id: Optional[RunId] = None,
        workflow_id: Optional[WorkflowId] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> RepositoryResult[ConversationSession]:
        exists = self.get_conversation(conversation_id)
        if not exists.ok:
            return exists
        session = self.registry.start_session(
            conversation_id, run_id=run_id, workflow_id=workflow_id, metadata=metadata
        )
        self.cache.pin(conversation_id)  # active conversations never evict
        self.events.session_started(conversation_id, session.session_id, str(run_id or ""))
        return ok(session)

    def end_session(self, session_id: str) -> Optional[ConversationSession]:
        session = self.registry.end_session(session_id)
        if session is not None:
            if not self.registry.sessions_for(session.conversation_id):
                self.cache.unpin(session.conversation_id)
            self.events.session_ended(
                session.conversation_id, session.session_id, str(session.run_id or "")
            )
        return session

    # ------------------------------------------------------------------ #
    # Coordination helpers
    # ------------------------------------------------------------------ #
    def mutate_manifest(
        self,
        conversation_id: ConversationId,
        mutator: Callable[[StorageManifest], None],
        attempts: int = _MUTATE_ATTEMPTS,
    ) -> RepositoryResult[StorageManifest]:
        """Optimistic-concurrency update loop: fresh read -> mutate ->
        versioned write; on CONFLICT (another writer won) retry against the
        newer state. Concurrent modification thus degrades to retry, never to
        lost updates."""
        with self._mutation_guard:
            lock = self._mutation_locks.setdefault(conversation_id, threading.Lock())
        with lock:
            return self._mutate_manifest_locked(conversation_id, mutator, attempts)

    def _mutate_manifest_locked(
        self,
        conversation_id: ConversationId,
        mutator: Callable[[StorageManifest], None],
        attempts: int,
    ) -> RepositoryResult[StorageManifest]:
        last_failure: RepositoryResult[StorageManifest] = fail(
            ErrorCode.CONFLICT, "no update attempt made", conversation_id=conversation_id
        )
        for attempt in range(max(1, attempts)):
            fresh = self._conversations.read_manifest(conversation_id)
            if not fresh.ok:
                self.cache.invalidate(conversation_id)
                return fresh
            manifest = fresh.unwrap()
            mutator(manifest)
            updated = self._conversations.update_manifest(
                manifest, expected_version=manifest.conversation.version
            )
            if updated.ok:
                self.cache.put_manifest(updated.unwrap())
                return updated
            if updated.error.code is not ErrorCode.CONFLICT:
                return updated
            last_failure = updated  # stale read: loop with the newer version
            if attempt + 1 < attempts:
                # Jittered backoff de-syncs contending writers; without it,
                # N threads hammering one manifest can starve each other
                # through every retry and surface spurious conflicts.
                time.sleep(random.uniform(0, _MUTATE_BACKOFF_MAX))
        self.cache.invalidate(conversation_id)
        return last_failure

    def cache_statistics(self) -> CacheStatistics:
        return self.cache.statistics()

    def derive_title(self, prompt: str) -> str:
        return derive_title(prompt)

    def health_check(self) -> OperationResult:
        return self._provider.health_check()

    def close(self) -> None:
        self.cache.clear()
        self._provider.close()
