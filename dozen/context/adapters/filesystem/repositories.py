"""The three filesystem repositories — concrete adapters for the persistence
ports defined in Phase 1.1.

Shared rules:

* Every public method returns a Result; environmental problems (missing
  files, permissions, disk full, corrupt data) never raise.
* Reads take the in-process shared lock; writes take the exclusive
  in-process + cross-process lock. See ``locks.py`` for the model.
* ``conversation.version`` moves ONLY through ``update_manifest`` (optimistic
  concurrency for logical updates). Appends refresh the manifest's derived
  statistics (message_count, approx_tokens, summary_count, last_message_at,
  updated_at) without bumping the version — bookkeeping, not a logical edit.
* JSONL corruption is contained: damaged lines are skipped by readers and
  visible through ``integrity_report``; they are never rewritten or deleted.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...domain.enums import ConversationStatus, ErrorCode, MessageRole
from ...domain.models import Message, StorageManifest, Summary
from ...domain.results import (
    OperationResult,
    RepositoryResult,
    done,
    fail,
    ok,
)
from ...domain.types import ConversationId, MessageId, Timestamp
from ...utils import utc_now_iso
from .errors import io_failure
from .jsonl import JsonLineReader, JsonLineWriter, JsonlReadReport
from .layout import StorageManager
from .locks import LockTimeoutError
from .manifest import ManifestManager
from .serializer import StorageSerializer

# Storage-level token approximation for manifest statistics only (the real
# TokenEstimator arrives in a later phase and never reads this number).
_APPROX_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN) if text else 0


class _RepositoryBase:
    def __init__(
        self,
        storage: StorageManager,
        manifests: ManifestManager,
        serializer: Optional[StorageSerializer] = None,
    ) -> None:
        self._storage = storage
        self._manifests = manifests
        self._serializer = serializer or StorageSerializer()

    def _require_conversation(self, conversation_id: ConversationId) -> OperationResult:
        checked = self._storage.validate_id(conversation_id)
        if not checked.ok:
            return checked
        if not self._storage.conversation_exists(conversation_id):
            return fail(
                ErrorCode.NOT_FOUND, "no such conversation",
                conversation_id=conversation_id,
            )
        return done()

    def _refresh_stats(self, manifest: StorageManifest) -> None:
        """Persist derived statistics. Best-effort by contract: the appended
        data is already durable, and a stats write failure must not make the
        caller believe the append failed (a retry would duplicate records).
        A stale manifest self-heals on the next successful write."""
        manifest.updated_at = Timestamp(utc_now_iso())
        manifest.conversation.updated_at = manifest.updated_at
        self._manifests.write(manifest)


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
class FileSystemConversationRepository(_RepositoryBase):
    """CRUD + listing + search over conversation manifests."""

    def create_conversation(self, manifest: StorageManifest) -> RepositoryResult[StorageManifest]:
        conversation_id = manifest.conversation.id
        checked = self._storage.validate_id(conversation_id)
        if not checked.ok:
            return checked
        ensured = self._storage.ensure_root()
        if not ensured.ok:
            return ensured
        created = self._storage.create_conversation_dir(conversation_id)
        if not created.ok:
            return created
        try:
            with self._storage.locks.write(conversation_id, self._storage.lock_path(conversation_id)):
                if self._storage.manifest_path(conversation_id).is_file():
                    return fail(
                        ErrorCode.ALREADY_EXISTS, "conversation exists",
                        conversation_id=conversation_id,
                    )
                if manifest.conversation.version < 1:
                    manifest.conversation.version = 1
                written = self._manifests.write(manifest)
                if not written.ok:
                    return written
                return ok(manifest)
        except LockTimeoutError as exc:
            return io_failure(exc, "writer lock timeout", conversation_id=conversation_id)

    def read_manifest(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]:
        checked = self._storage.validate_id(conversation_id)
        if not checked.ok:
            return checked
        with self._storage.locks.read(conversation_id):
            return self._manifests.read(conversation_id)

    def update_manifest(
        self, manifest: StorageManifest, expected_version: int
    ) -> RepositoryResult[StorageManifest]:
        conversation_id = manifest.conversation.id
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        try:
            with self._storage.locks.write(conversation_id, self._storage.lock_path(conversation_id)):
                current = self._manifests.read(conversation_id)
                if not current.ok:
                    return current
                current_manifest = current.unwrap()
                actual = current_manifest.conversation.version
                if actual != expected_version:
                    return fail(
                        ErrorCode.CONFLICT, "version mismatch",
                        expected=expected_version, actual=actual,
                    )
                # Statistics are storage-owned bookkeeping, moved ONLY by
                # appends (which do not bump the version). A logical update
                # based on an earlier read must never write stale counters
                # back — always keep the freshest stats from disk.
                manifest.conversation.stats = current_manifest.conversation.stats
                manifest.conversation.version = expected_version + 1
                manifest.updated_at = Timestamp(utc_now_iso())
                manifest.conversation.updated_at = manifest.updated_at
                written = self._manifests.write(manifest)
                if not written.ok:
                    return written
                return ok(manifest)
        except LockTimeoutError as exc:
            return io_failure(exc, "writer lock timeout", conversation_id=conversation_id)

    def list_conversations(
        self,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RepositoryResult[list[StorageManifest]]:
        ids = self._storage.list_conversation_ids()
        if not ids.ok:
            return ids
        selected: list[StorageManifest] = []
        skipped = 0
        for conversation_id in ids.unwrap():
            if len(selected) >= limit:
                break
            found = self.read_manifest(ConversationId(conversation_id))
            if not found.ok:
                continue  # unreadable entry must not poison the listing
            manifest = found.unwrap()
            if status is not None and manifest.conversation.status != status:
                continue
            if skipped < offset:
                skipped += 1
                continue
            selected.append(manifest)
        return ok(selected)

    def delete_conversation(self, conversation_id: ConversationId) -> OperationResult:
        checked = self._storage.validate_id(conversation_id)
        if not checked.ok:
            return checked
        if not self._storage.conversation_exists(conversation_id):
            return done()  # idempotent, matching the port's reference behavior
        try:
            with self._storage.locks.write(conversation_id, self._storage.lock_path(conversation_id)):
                return self._storage.delete_conversation_dir(conversation_id)
        except LockTimeoutError as exc:
            return io_failure(exc, "writer lock timeout", conversation_id=conversation_id)

    # -------------------------- extensions ---------------------------- #
    def exists(self, conversation_id: ConversationId) -> bool:
        return self._storage.conversation_exists(conversation_id)

    def search(
        self,
        status: Optional[ConversationStatus] = None,
        title_contains: Optional[str] = None,
        metadata_filters: Optional[dict[str, object]] = None,
        created_from: Optional[Timestamp] = None,
        created_to: Optional[Timestamp] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RepositoryResult[list[StorageManifest]]:
        """Filtered listing. Timestamp bounds are inclusive; ISO-8601 UTC
        strings compare correctly as plain strings."""
        ids = self._storage.list_conversation_ids()
        if not ids.ok:
            return ids
        needle = title_contains.lower() if title_contains else None
        selected: list[StorageManifest] = []
        skipped = 0
        for conversation_id in ids.unwrap():
            if len(selected) >= limit:
                break
            found = self.read_manifest(ConversationId(conversation_id))
            if not found.ok:
                continue
            conv = found.unwrap().conversation
            if status is not None and conv.status != status:
                continue
            if needle is not None and needle not in conv.title.lower():
                continue
            if created_from is not None and conv.created_at < created_from:
                continue
            if created_to is not None and conv.created_at > created_to:
                continue
            if metadata_filters and any(
                conv.metadata.values.get(key) != value
                for key, value in metadata_filters.items()
            ):
                continue
            if skipped < offset:
                skipped += 1
                continue
            selected.append(found.unwrap())
        return ok(selected)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
class FileSystemMessageRepository(_RepositoryBase):
    """Append-only JSONL message log with streaming reads and search."""

    def append_messages(
        self, conversation_id: ConversationId, messages: Sequence[Message]
    ) -> RepositoryResult[int]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        for message in messages:
            if not message.id:
                return fail(ErrorCode.VALIDATION_FAILED, "message id must be non-empty")
            if message.conversation_id != conversation_id:
                return fail(
                    ErrorCode.VALIDATION_FAILED,
                    "message belongs to another conversation",
                    message_id=message.id, expected=conversation_id,
                    found=message.conversation_id,
                )
        if not messages:
            return ok(0)
        records = [self._serializer.to_dict(message) for message in messages]
        try:
            with self._storage.locks.write(conversation_id, self._storage.lock_path(conversation_id)):
                writer = JsonLineWriter(
                    self._storage.messages_path(conversation_id),
                    fsync=self._storage.config.fsync_appends,
                )
                appended = writer.append(records)
                if not appended.ok:
                    return appended
                manifest = self._manifests.read(conversation_id)
                if manifest.ok:
                    stats = manifest.unwrap().conversation.stats
                    stats.message_count += len(messages)
                    stats.approx_tokens += sum(_approx_tokens(m.content) for m in messages)
                    stats.last_message_at = messages[-1].created_at
                    self._refresh_stats(manifest.unwrap())
                return appended
        except LockTimeoutError as exc:
            return io_failure(exc, "writer lock timeout", conversation_id=conversation_id)

    def read_messages(
        self,
        conversation_id: ConversationId,
        limit: int = 200,
        after_id: Optional[MessageId] = None,
    ) -> RepositoryResult[list[Message]]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        reader = JsonLineReader(self._storage.messages_path(conversation_id))
        collected: list[Message] = []
        passed_anchor = after_id is None
        try:
            with self._storage.locks.read(conversation_id):
                for record, error in reader.scan():
                    if error is not None or record is None:
                        continue  # damaged line: contained, reported via integrity_report
                    if not passed_anchor:
                        if str(record.get("id", "")) == after_id:
                            passed_anchor = True
                        continue
                    if len(collected) >= limit:
                        break
                    parsed = self._serializer.from_dict(record, Message)
                    if parsed.ok:
                        collected.append(parsed.unwrap())
        except OSError as exc:
            return io_failure(exc, "cannot read messages", conversation_id=conversation_id)
        if after_id is not None and not passed_anchor:
            return ok([])  # unknown anchor -> empty page (reference behavior)
        return ok(collected)

    def count_messages(self, conversation_id: ConversationId) -> RepositoryResult[int]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        with self._storage.locks.read(conversation_id):
            return JsonLineReader(self._storage.messages_path(conversation_id)).count()

    # -------------------------- extensions ---------------------------- #
    def read_tail(self, conversation_id: ConversationId, count: int) -> RepositoryResult[list[Message]]:
        """Last N messages without scanning the whole log — the hot read of
        context assembly."""
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        with self._storage.locks.read(conversation_id):
            tail = JsonLineReader(self._storage.messages_path(conversation_id)).read_tail(count)
        if not tail.ok:
            return tail
        messages: list[Message] = []
        for record in tail.unwrap().records:
            parsed = self._serializer.from_dict(record, Message)
            if parsed.ok:
                messages.append(parsed.unwrap())
        return ok(messages)

    def get_message(
        self, conversation_id: ConversationId, message_id: MessageId
    ) -> RepositoryResult[Message]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        reader = JsonLineReader(self._storage.messages_path(conversation_id))
        try:
            with self._storage.locks.read(conversation_id):
                for record, error in reader.scan():
                    if error is None and record is not None and str(record.get("id", "")) == message_id:
                        return self._serializer.from_dict(record, Message)
        except OSError as exc:
            return io_failure(exc, "cannot read messages", conversation_id=conversation_id)
        return fail(
            ErrorCode.NOT_FOUND, "no such message",
            conversation_id=conversation_id, message_id=message_id,
        )

    def search_messages(
        self,
        conversation_id: ConversationId,
        role: Optional[MessageRole] = None,
        created_from: Optional[Timestamp] = None,
        created_to: Optional[Timestamp] = None,
        metadata_filters: Optional[dict[str, object]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> RepositoryResult[list[Message]]:
        """Streamed filter over the log: by role, timestamp window (inclusive,
        lexicographic ISO-8601 comparison) and metadata equality."""
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        reader = JsonLineReader(self._storage.messages_path(conversation_id))
        selected: list[Message] = []
        skipped = 0
        try:
            with self._storage.locks.read(conversation_id):
                for record, error in reader.scan():
                    if error is not None or record is None:
                        continue
                    if len(selected) >= limit:
                        break
                    parsed = self._serializer.from_dict(record, Message)
                    if not parsed.ok:
                        continue
                    message = parsed.unwrap()
                    if role is not None and message.role != role:
                        continue
                    if created_from is not None and message.created_at < created_from:
                        continue
                    if created_to is not None and message.created_at > created_to:
                        continue
                    if metadata_filters and any(
                        message.metadata.values.get(key) != value
                        for key, value in metadata_filters.items()
                    ):
                        continue
                    if skipped < offset:
                        skipped += 1
                        continue
                    selected.append(message)
        except OSError as exc:
            return io_failure(exc, "cannot search messages", conversation_id=conversation_id)
        return ok(selected)

    def integrity_report(self, conversation_id: ConversationId) -> RepositoryResult[JsonlReadReport]:
        """Corruption visibility: how many lines are damaged, and where."""
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        with self._storage.locks.read(conversation_id):
            return JsonLineReader(self._storage.messages_path(conversation_id)).read()


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
class FileSystemSummaryRepository(_RepositoryBase):
    """Append-only JSONL summary log."""

    def append_summary(
        self, conversation_id: ConversationId, summary: Summary
    ) -> RepositoryResult[Summary]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        if not summary.id:
            return fail(ErrorCode.VALIDATION_FAILED, "summary id must be non-empty")
        if summary.conversation_id != conversation_id:
            return fail(
                ErrorCode.VALIDATION_FAILED, "summary belongs to another conversation",
                summary_id=summary.id, expected=conversation_id,
                found=summary.conversation_id,
            )
        try:
            with self._storage.locks.write(conversation_id, self._storage.lock_path(conversation_id)):
                writer = JsonLineWriter(
                    self._storage.summaries_path(conversation_id),
                    fsync=self._storage.config.fsync_appends,
                )
                appended = writer.append([self._serializer.to_dict(summary)])
                if not appended.ok:
                    return appended
                manifest = self._manifests.read(conversation_id)
                if manifest.ok:
                    manifest.unwrap().conversation.stats.summary_count += 1
                    self._refresh_stats(manifest.unwrap())
                return ok(summary)
        except LockTimeoutError as exc:
            return io_failure(exc, "writer lock timeout", conversation_id=conversation_id)

    def read_summaries(self, conversation_id: ConversationId) -> RepositoryResult[list[Summary]]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        reader = JsonLineReader(self._storage.summaries_path(conversation_id))
        summaries: list[Summary] = []
        try:
            with self._storage.locks.read(conversation_id):
                for record, error in reader.scan():
                    if error is not None or record is None:
                        continue
                    parsed = self._serializer.from_dict(record, Summary)
                    if parsed.ok:
                        summaries.append(parsed.unwrap())
        except OSError as exc:
            return io_failure(exc, "cannot read summaries", conversation_id=conversation_id)
        return ok(summaries)

    # -------------------------- extensions ---------------------------- #
    def count_summaries(self, conversation_id: ConversationId) -> RepositoryResult[int]:
        exists = self._require_conversation(conversation_id)
        if not exists.ok:
            return exists
        with self._storage.locks.read(conversation_id):
            return JsonLineReader(self._storage.summaries_path(conversation_id)).count()
