"""StorageVersionManager — storage-format compatibility policy + migration hooks.

Policy (SADD §11):

* CURRENT  — read and write freely.
* OLDER    — always readable; writable only through a registered migration
             path (hooks are registered here, *implemented* in later phases).
* NEWER    — readable in tolerance mode (the foundation models preserve every
             unknown field, so nothing is lost); never writable — a writer
             must not silently downgrade data produced by a newer schema.
* UNSUPPORTED — versions below the minimum readable version.

This module intentionally ships ZERO migrations: only the registry, the path
finder, and the runner architecture, per the Phase 1.2 scope.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from ...domain.enums import ErrorCode
from ...domain.results import OperationResult, RepositoryResult, done, fail, ok

# A migration hook transforms one manifest/record dict from version N to N+1.
MigrationHook = Callable[[dict], dict]


class VersionCompatibility(str, Enum):
    CURRENT = "current"
    OLDER = "older"
    NEWER = "newer"
    UNSUPPORTED = "unsupported"


class StorageVersionManager:
    CURRENT_VERSION = 1
    MIN_READABLE_VERSION = 1

    def __init__(self) -> None:
        # (from_version) -> hook producing from_version + 1
        self._migrations: dict[int, MigrationHook] = {}

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def classify(self, version: int) -> VersionCompatibility:
        if version < self.MIN_READABLE_VERSION:
            return VersionCompatibility.UNSUPPORTED
        if version < self.CURRENT_VERSION:
            return VersionCompatibility.OLDER
        if version == self.CURRENT_VERSION:
            return VersionCompatibility.CURRENT
        return VersionCompatibility.NEWER

    def check_readable(self, version: int) -> OperationResult:
        if self.classify(version) is VersionCompatibility.UNSUPPORTED:
            return fail(
                ErrorCode.VERSION_UNSUPPORTED, "storage version too old to read",
                found=version, min_readable=self.MIN_READABLE_VERSION,
            )
        return done()

    def check_writable(self, version: int) -> OperationResult:
        kind = self.classify(version)
        if kind is VersionCompatibility.CURRENT:
            return done()
        if kind is VersionCompatibility.OLDER and self.migration_path(version) is not None:
            return done()  # upgradeable before write
        return fail(
            ErrorCode.VERSION_UNSUPPORTED, "storage version not writable",
            found=version, current=self.CURRENT_VERSION, compatibility=kind.value,
        )

    # ------------------------------------------------------------------ #
    # Migration architecture (hooks only — no migrations exist in v1)
    # ------------------------------------------------------------------ #
    def register_migration(self, from_version: int, hook: MigrationHook) -> None:
        """Register the hook that upgrades ``from_version`` -> ``from_version + 1``."""
        self._migrations[from_version] = hook

    def migration_path(self, from_version: int, to_version: Optional[int] = None) -> Optional[list[MigrationHook]]:
        """The hook chain from one version to another, or None if incomplete."""
        target = self.CURRENT_VERSION if to_version is None else to_version
        if from_version > target:
            return None  # downgrades are not supported by design
        hooks: list[MigrationHook] = []
        for step in range(from_version, target):
            hook = self._migrations.get(step)
            if hook is None:
                return None
            hooks.append(hook)
        return hooks

    def migrate(self, data: dict, from_version: int) -> RepositoryResult[dict]:
        """Run the registered chain up to CURRENT_VERSION. With no registered
        hooks (v1 reality) an older version yields VERSION_UNSUPPORTED — the
        caller keeps the data untouched."""
        if from_version == self.CURRENT_VERSION:
            return ok(data)
        hooks = self.migration_path(from_version)
        if hooks is None:
            return fail(
                ErrorCode.VERSION_UNSUPPORTED, "no migration path registered",
                found=from_version, current=self.CURRENT_VERSION,
            )
        current = data
        for index, hook in enumerate(hooks):
            try:
                current = hook(current)
            except Exception as exc:  # a hook is third-party code: contain it
                return fail(
                    ErrorCode.VERSION_UNSUPPORTED, "migration hook failed",
                    step=from_version + index, exception=type(exc).__name__, detail=str(exc),
                )
        return ok(current)
