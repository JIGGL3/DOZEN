"""ConversationMetadataManager — the namespaced metadata surface.

Metadata mutations ride the manager's optimistic-concurrency path (read ->
mutate -> versioned update -> retry on conflict), so concurrent writers can
never silently overwrite each other. Reads are served from the cache's
metadata snapshot when warm.

Namespace convention (from the SADD): keys are dot-namespaced; the reserved
prefixes (``workflow.`` ``planner.`` ``agent.`` ``user.``) are documented
homes for each subsystem. Unreserved prefixes are allowed — the reservation
exists so subsystems don't collide, not to forbid extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..domain.enums import ErrorCode
from ..domain.models import ConversationMetadata, StorageManifest
from ..domain.results import RepositoryResult, fail, ok
from ..domain.types import ConversationId

if TYPE_CHECKING:  # avoid a runtime circular import with manager.py
    from .manager import ConversationManager

RESERVED_NAMESPACES = ConversationMetadata.RESERVED_NAMESPACES


class ConversationMetadataManager:
    def __init__(self, manager: "ConversationManager") -> None:
        self._manager = manager

    # ------------------------------- writes ---------------------------- #
    def set_values(
        self, conversation_id: ConversationId, values: dict[str, object]
    ) -> RepositoryResult[StorageManifest]:
        problems = self._validate(values)
        if problems:
            return fail(
                ErrorCode.VALIDATION_FAILED, "invalid metadata keys",
                problems=problems,
            )

        def mutate(manifest: StorageManifest) -> None:
            manifest.conversation.metadata.values.update(values)

        result = self._manager.mutate_manifest(conversation_id, mutate)
        if result.ok:
            self._manager.events.conversation_updated(conversation_id, reason="metadata")
        return result

    def delete_keys(
        self, conversation_id: ConversationId, keys: list[str]
    ) -> RepositoryResult[StorageManifest]:
        def mutate(manifest: StorageManifest) -> None:
            for key in keys:
                manifest.conversation.metadata.values.pop(key, None)

        result = self._manager.mutate_manifest(conversation_id, mutate)
        if result.ok:
            self._manager.events.conversation_updated(conversation_id, reason="metadata")
        return result

    # -------------------------------- reads ---------------------------- #
    def get_all(self, conversation_id: ConversationId) -> RepositoryResult[dict[str, object]]:
        cached = self._manager.cache.get_metadata(conversation_id)
        if cached is not None:
            self._manager.events.cache_hit(conversation_id, kind="metadata")
            return ok(cached)
        self._manager.events.cache_miss(conversation_id, kind="metadata")
        manifest = self._manager.get_conversation(conversation_id)
        if not manifest.ok:
            return manifest
        return ok(dict(manifest.unwrap().conversation.metadata.values))

    def get(
        self, conversation_id: ConversationId, key: str, default: object = None
    ) -> RepositoryResult[object]:
        values = self.get_all(conversation_id)
        if not values.ok:
            return values
        return ok(values.unwrap().get(key, default))

    def namespace(
        self, conversation_id: ConversationId, prefix: str
    ) -> RepositoryResult[dict[str, object]]:
        values = self.get_all(conversation_id)
        if not values.ok:
            return values
        return ok({k: v for k, v in values.unwrap().items() if k.startswith(prefix)})

    # ------------------------------ internals -------------------------- #
    @staticmethod
    def _validate(values: dict[str, object]) -> list[str]:
        problems: list[str] = []
        for key in values:
            if not isinstance(key, str) or not key.strip():
                problems.append(f"metadata key must be a non-empty string, got {key!r}")
        return problems
