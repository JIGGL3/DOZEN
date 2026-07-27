"""RepositoryFactory + FileSystemPersistenceProvider.

The provider is the composition root of the storage backend: it wires the
StorageManager, serializer, version manager and manifest manager once, and
hands out the three repositories that share that infrastructure (one lock
registry, one durability policy). ``RepositoryFactory`` is the only place
that knows which backend name maps to which provider class — later backends
register here without touching any caller.
"""

from __future__ import annotations

from typing import Optional, Union

from ...config.models import ContextConfig, PersistenceConfig
from ...domain.enums import ErrorCode
from ...domain.results import OperationResult, RepositoryResult, done, fail, ok
from .layout import StorageManager
from .manifest import ManifestManager
from .repositories import (
    FileSystemConversationRepository,
    FileSystemMessageRepository,
    FileSystemSummaryRepository,
)
from .serializer import StorageSerializer
from .versioning import StorageVersionManager


class FileSystemPersistenceProvider:
    """Implements the ``PersistenceProvider`` port for the filesystem backend."""

    def __init__(self, config: Optional[PersistenceConfig] = None) -> None:
        self.config = config or PersistenceConfig()
        self._storage = StorageManager(self.config.storage)
        self._serializer = StorageSerializer()
        self._versions = StorageVersionManager()
        self._manifests = ManifestManager(self._storage, self._serializer, self._versions)
        self._conversations = FileSystemConversationRepository(
            self._storage, self._manifests, self._serializer
        )
        self._messages = FileSystemMessageRepository(
            self._storage, self._manifests, self._serializer
        )
        self._summaries = FileSystemSummaryRepository(
            self._storage, self._manifests, self._serializer
        )
        self._closed = False

    # ----------------------- PersistenceProvider ---------------------- #
    def conversations(self) -> FileSystemConversationRepository:
        self._ensure_open()
        return self._conversations

    def messages(self) -> FileSystemMessageRepository:
        self._ensure_open()
        return self._messages

    def summaries(self) -> FileSystemSummaryRepository:
        self._ensure_open()
        return self._summaries

    def health_check(self) -> OperationResult:
        if self._closed:
            return fail(ErrorCode.IO_ERROR, "persistence provider is closed")
        return self._storage.health_check()

    def close(self) -> None:
        """Idempotent. The backend holds no open handles between operations,
        so closing only fences off further use."""
        self._closed = True

    # ----------------------------- misc ------------------------------- #
    @property
    def storage(self) -> StorageManager:
        return self._storage

    @property
    def version_manager(self) -> StorageVersionManager:
        return self._versions

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("persistence provider is closed")  # programmer error


class RepositoryFactory:
    """Backend-name -> provider dispatch (the only backend-aware component)."""

    _BACKENDS = {
        "filesystem": FileSystemPersistenceProvider,
    }

    @classmethod
    def create(
        cls, config: Optional[Union[PersistenceConfig, ContextConfig]] = None
    ) -> RepositoryResult[FileSystemPersistenceProvider]:
        if isinstance(config, ContextConfig):
            persistence = config.persistence
        elif isinstance(config, PersistenceConfig):
            persistence = config
        elif config is None:
            persistence = PersistenceConfig()
        else:
            return fail(
                ErrorCode.VALIDATION_FAILED, "unsupported configuration object",
                got=type(config).__name__,
            )
        problems = persistence.validate()
        if problems:
            return fail(
                ErrorCode.VALIDATION_FAILED, "invalid persistence configuration",
                problems=problems,
            )
        provider_cls = cls._BACKENDS.get(persistence.backend)
        if provider_cls is None:
            return fail(
                ErrorCode.VALIDATION_FAILED, "unknown persistence backend",
                backend=persistence.backend, supported=sorted(cls._BACKENDS),
            )
        return ok(provider_cls(persistence))

    @classmethod
    def supported_backends(cls) -> list[str]:
        return sorted(cls._BACKENDS)


def create_persistence(
    config: Optional[Union[PersistenceConfig, ContextConfig]] = None
) -> RepositoryResult[FileSystemPersistenceProvider]:
    """Module-level convenience over ``RepositoryFactory.create``."""
    return RepositoryFactory.create(config)
