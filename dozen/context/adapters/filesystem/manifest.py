"""ManifestManager — durable read/write of ``manifest.json`` with recovery.

Write protocol: the current manifest bytes (last known good) are first copied
to ``manifest.json.bak`` atomically, then the new manifest atomically replaces
``manifest.json``. Two atomic renames — at every instant the directory holds
at least one intact manifest.

Read protocol: parse ``manifest.json``; if it is corrupt (not missing —
missing means NOT_FOUND), fall back to the backup, and when the backup parses,
restore it as the live manifest so the directory self-heals.
"""

from __future__ import annotations

import json
from typing import Optional

from ...domain.enums import ErrorCode
from ...domain.models import StorageManifest
from ...domain.results import OperationResult, RepositoryResult, fail, ok
from ...domain.types import ConversationId
from .errors import io_failure
from .layout import StorageManager
from .serializer import StorageSerializer
from .versioning import StorageVersionManager


class ManifestManager:
    def __init__(
        self,
        storage: StorageManager,
        serializer: Optional[StorageSerializer] = None,
        versions: Optional[StorageVersionManager] = None,
    ) -> None:
        self.storage = storage
        self.serializer = serializer or StorageSerializer()
        self.versions = versions or StorageVersionManager()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def write(self, manifest: StorageManifest) -> OperationResult:
        """Caller must hold the conversation's write lock."""
        conversation_id = manifest.conversation.id
        writable = self.versions.check_writable(manifest.storage_format_version)
        if not writable.ok:
            return writable
        path = self.storage.manifest_path(conversation_id)
        backup = self.storage.manifest_backup_path(conversation_id)
        try:
            if path.is_file():
                previous = path.read_bytes()
                backed_up = self.storage.atomic.write_bytes(backup, previous)
                if not backed_up.ok:
                    return backed_up
        except OSError as exc:
            return io_failure(exc, "cannot back up manifest", path=str(path))
        payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
        return self.storage.atomic.write_text(path, payload)

    # ------------------------------------------------------------------ #
    # Read (with automatic recovery)
    # ------------------------------------------------------------------ #
    def read(self, conversation_id: ConversationId) -> RepositoryResult[StorageManifest]:
        path = self.storage.manifest_path(conversation_id)
        if not path.is_file():
            return fail(
                ErrorCode.NOT_FOUND, "no such conversation",
                conversation_id=conversation_id,
            )
        primary = self._read_file(conversation_id, str(path))
        if primary.ok:
            return primary
        if primary.error.code != ErrorCode.SERIALIZATION_FAILED:
            return primary  # real I/O fault — recovery cannot help
        return self._recover_from_backup(conversation_id, primary)

    def _read_file(self, conversation_id: ConversationId, path: str) -> RepositoryResult[StorageManifest]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = handle.read()
        except OSError as exc:
            return io_failure(exc, "cannot read manifest", path=path)
        parsed = self.serializer.deserialize(payload, StorageManifest)
        if not parsed.ok:
            return parsed
        manifest = parsed.unwrap()
        readable = self.versions.check_readable(manifest.storage_format_version)
        if not readable.ok:
            return readable
        if manifest.conversation.id != conversation_id:
            return fail(
                ErrorCode.SERIALIZATION_FAILED, "manifest belongs to another conversation",
                expected=conversation_id, found=manifest.conversation.id, path=path,
            )
        return ok(manifest)

    def _recover_from_backup(
        self, conversation_id: ConversationId, primary_failure: RepositoryResult[StorageManifest]
    ) -> RepositoryResult[StorageManifest]:
        backup = self.storage.manifest_backup_path(conversation_id)
        if not backup.is_file():
            return primary_failure
        recovered = self._read_file(conversation_id, str(backup))
        if not recovered.ok:
            return primary_failure  # both copies bad: report the primary fault
        # Self-heal: promote the backup to be the live manifest again. No lock
        # is taken here (ManifestManager is lock-free by contract — callers
        # own locking); the restore is a single atomic replace, so concurrent
        # readers always see an intact manifest either way.
        try:
            restore = self.storage.atomic.write_bytes(
                self.storage.manifest_path(conversation_id), backup.read_bytes()
            )
        except OSError as exc:
            return io_failure(exc, "cannot restore manifest backup", path=str(backup))
        if not restore.ok:
            return restore
        return recovered
