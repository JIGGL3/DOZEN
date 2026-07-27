"""Filesystem persistence backend (Phase 1.2).

Storage layout, durability and concurrency rules are documented per module:

* ``atomic``       — AtomicFileWriter (temp file -> fsync -> atomic rename)
* ``jsonl``        — JsonLineWriter / JsonLineReader (append log + recovery)
* ``serializer``   — StorageSerializer (model <-> JSON, error mapping)
* ``versioning``   — StorageVersionManager (compatibility + migration hooks)
* ``locks``        — readers-writer + cross-process lock files
* ``layout``       — StorageManager (paths, lifecycle, health)
* ``manifest``     — ManifestManager (backup + self-healing reads)
* ``repositories`` — the three repository adapters
* ``factory``      — RepositoryFactory / FileSystemPersistenceProvider
"""

from __future__ import annotations

from .atomic import AtomicFileWriter
from .factory import (
    FileSystemPersistenceProvider,
    RepositoryFactory,
    create_persistence,
)
from .jsonl import JsonLineReader, JsonLineWriter, JsonlLineError, JsonlReadReport
from .layout import StorageManager
from .locks import ConversationLockRegistry, FileLock, LockTimeoutError, ReadWriteLock
from .manifest import ManifestManager
from .repositories import (
    FileSystemConversationRepository,
    FileSystemMessageRepository,
    FileSystemSummaryRepository,
)
from .serializer import StorageSerializer
from .versioning import MigrationHook, StorageVersionManager, VersionCompatibility

__all__ = [
    "AtomicFileWriter",
    "ConversationLockRegistry",
    "FileLock",
    "FileSystemConversationRepository",
    "FileSystemMessageRepository",
    "FileSystemPersistenceProvider",
    "FileSystemSummaryRepository",
    "JsonLineReader",
    "JsonLineWriter",
    "JsonlLineError",
    "JsonlReadReport",
    "LockTimeoutError",
    "ManifestManager",
    "MigrationHook",
    "ReadWriteLock",
    "RepositoryFactory",
    "StorageManager",
    "StorageSerializer",
    "StorageVersionManager",
    "VersionCompatibility",
    "create_persistence",
]
