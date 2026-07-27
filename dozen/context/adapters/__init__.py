"""Adapters — concrete implementations of the context package's ports.

Import rule (see ``dozen/context/__init__.py``): adapters may import
``ports/``, ``domain/``, ``config/`` and ``utils/`` — never the orchestrator,
providers, or browser layers. Each storage medium lives in its own subpackage;
v1 ships the filesystem backend only.
"""

from __future__ import annotations

from .filesystem import (
    AtomicFileWriter,
    FileSystemConversationRepository,
    FileSystemMessageRepository,
    FileSystemPersistenceProvider,
    FileSystemSummaryRepository,
    JsonLineReader,
    JsonLineWriter,
    ManifestManager,
    RepositoryFactory,
    StorageManager,
    StorageSerializer,
    StorageVersionManager,
)

__all__ = [
    "AtomicFileWriter",
    "FileSystemConversationRepository",
    "FileSystemMessageRepository",
    "FileSystemPersistenceProvider",
    "FileSystemSummaryRepository",
    "JsonLineReader",
    "JsonLineWriter",
    "ManifestManager",
    "RepositoryFactory",
    "StorageManager",
    "StorageSerializer",
    "StorageVersionManager",
]
