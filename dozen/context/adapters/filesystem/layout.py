"""StorageManager — owner of the on-disk layout and shared infrastructure.

Layout (SADD §10.1)::

    <root>/                        # StorageConfig.root_path
        <conversation-id>/
            manifest.json          # conversation record (atomic replace)
            manifest.json.bak      # previous good manifest (recovery)
            messages.jsonl         # append-only message log
            summaries.jsonl        # append-only summary log
            .writer.lock           # advisory cross-process writer lock

With ``StorageConfig.shard_directories`` enabled, conversation directories
are nested one level deep by id prefix (``<root>/<id[:2]>/<id>/``) so a root
with hundreds of thousands of conversations stays listable.

The manager also hosts the lock registry and the atomic writer, so every
repository shares one concurrency domain and one durability policy.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from ...config.models import StorageConfig
from ...domain.enums import ErrorCode
from ...domain.results import OperationResult, RepositoryResult, done, fail, ok
from .atomic import AtomicFileWriter
from .errors import io_failure
from .locks import ConversationLockRegistry

MANIFEST_NAME = "manifest.json"
MANIFEST_BACKUP_NAME = "manifest.json.bak"
MESSAGES_NAME = "messages.jsonl"
SUMMARIES_NAME = "summaries.jsonl"
LOCK_NAME = ".writer.lock"

# Path-safe conversation ids: ULIDs and similar tokens. Anything else could
# escape the storage root or collide with reserved file names.
_ID_SAFE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def is_path_safe_id(conversation_id: str) -> bool:
    return (
        bool(conversation_id)
        and len(conversation_id) <= 128
        and all(ch in _ID_SAFE_CHARS for ch in conversation_id)
    )


class StorageManager:
    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        atomic: Optional[AtomicFileWriter] = None,
        locks: Optional[ConversationLockRegistry] = None,
    ) -> None:
        self.config = config or StorageConfig()
        self.root = Path(self.config.root_path)
        self.atomic = atomic or AtomicFileWriter(fsync=self.config.fsync_appends)
        self.locks = locks or ConversationLockRegistry()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def conversation_dir(self, conversation_id: str) -> Path:
        if self.config.shard_directories:
            return self.root / conversation_id[:2].lower() / conversation_id
        return self.root / conversation_id

    def manifest_path(self, conversation_id: str) -> Path:
        return self.conversation_dir(conversation_id) / MANIFEST_NAME

    def manifest_backup_path(self, conversation_id: str) -> Path:
        return self.conversation_dir(conversation_id) / MANIFEST_BACKUP_NAME

    def messages_path(self, conversation_id: str) -> Path:
        return self.conversation_dir(conversation_id) / MESSAGES_NAME

    def summaries_path(self, conversation_id: str) -> Path:
        return self.conversation_dir(conversation_id) / SUMMARIES_NAME

    def lock_path(self, conversation_id: str) -> Path:
        return self.conversation_dir(conversation_id) / LOCK_NAME

    # ------------------------------------------------------------------ #
    # Directory lifecycle
    # ------------------------------------------------------------------ #
    def validate_id(self, conversation_id: str) -> OperationResult:
        if not is_path_safe_id(conversation_id):
            return fail(
                ErrorCode.VALIDATION_FAILED,
                "conversation id is not a path-safe token",
                conversation_id=conversation_id,
            )
        return done()

    def ensure_root(self) -> OperationResult:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            return done()
        except OSError as exc:
            return io_failure(exc, "cannot create storage root", path=str(self.root))

    def conversation_exists(self, conversation_id: str) -> bool:
        return (
            is_path_safe_id(conversation_id)
            and self.manifest_path(conversation_id).is_file()
        )

    def create_conversation_dir(self, conversation_id: str) -> RepositoryResult[Path]:
        checked = self.validate_id(conversation_id)
        if not checked.ok:
            return checked
        directory = self.conversation_dir(conversation_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Pre-create the logs so the layout is complete from birth.
            for name in (MESSAGES_NAME, SUMMARIES_NAME):
                log = directory / name
                if not log.exists():
                    log.touch()
            return ok(directory)
        except OSError as exc:
            return io_failure(exc, "cannot create conversation directory", path=str(directory))

    def delete_conversation_dir(self, conversation_id: str) -> OperationResult:
        checked = self.validate_id(conversation_id)
        if not checked.ok:
            return checked
        directory = self.conversation_dir(conversation_id)
        if not directory.exists():
            return done()  # idempotent
        try:
            self._rmtree_with_retry(directory)
            self.locks.forget(conversation_id)
            return done()
        except OSError as exc:
            return io_failure(exc, "cannot delete conversation directory", path=str(directory))

    @staticmethod
    def _rmtree_with_retry(directory: Path, attempts: int = 3) -> None:
        """Windows can transiently refuse deletes (antivirus/indexer handles);
        retry briefly before giving up."""
        for attempt in range(attempts):
            try:
                shutil.rmtree(directory)
                return
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def list_conversation_ids(self) -> RepositoryResult[list[str]]:
        """Every directory under the root that contains a manifest. Sorted by
        id — ULIDs make that creation order."""
        if not self.root.is_dir():
            return ok([])
        found: list[str] = []
        try:
            candidates = (
                (shard_child for shard in self.root.iterdir() if shard.is_dir()
                 for shard_child in shard.iterdir())
                if self.config.shard_directories
                else self.root.iterdir()
            )
            for entry in candidates:
                if entry.is_dir() and (entry / MANIFEST_NAME).is_file():
                    found.append(entry.name)
            return ok(sorted(found))
        except OSError as exc:
            return io_failure(exc, "cannot list storage root", path=str(self.root))

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def health_check(self) -> OperationResult:
        """Root exists (or can be created) and is writable — proven by a real
        create/delete round-trip, not just a permission-bit guess."""
        ensured = self.ensure_root()
        if not ensured.ok:
            return ensured
        try:
            fd, probe = tempfile.mkstemp(dir=str(self.root), prefix=".health.", suffix=".probe")
            os.close(fd)
            os.unlink(probe)
            return done()
        except OSError as exc:
            return io_failure(exc, "storage root is not writable", path=str(self.root))
