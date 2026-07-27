"""StorageVersionManager: compatibility classification and the migration-hook
architecture (no real migrations exist in v1 — that is asserted too)."""

from __future__ import annotations

import unittest

from dozen.context.adapters.filesystem import StorageVersionManager, VersionCompatibility
from dozen.context.domain.enums import ErrorCode


class _V3Manager(StorageVersionManager):
    """A future deployment (current=3) for exercising OLDER-version paths."""

    CURRENT_VERSION = 3
    MIN_READABLE_VERSION = 1


class TestClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.v1 = StorageVersionManager()
        self.v3 = _V3Manager()

    def test_current(self) -> None:
        self.assertIs(self.v1.classify(1), VersionCompatibility.CURRENT)
        self.assertIs(self.v3.classify(3), VersionCompatibility.CURRENT)

    def test_newer(self) -> None:
        self.assertIs(self.v1.classify(2), VersionCompatibility.NEWER)

    def test_older(self) -> None:
        self.assertIs(self.v3.classify(2), VersionCompatibility.OLDER)

    def test_unsupported(self) -> None:
        self.assertIs(self.v1.classify(0), VersionCompatibility.UNSUPPORTED)
        self.assertIs(self.v1.classify(-4), VersionCompatibility.UNSUPPORTED)

    def test_readability(self) -> None:
        self.assertTrue(self.v1.check_readable(1).ok)
        self.assertTrue(self.v1.check_readable(2).ok)   # newer: tolerant read
        self.assertTrue(self.v3.check_readable(1).ok)   # older: always readable
        rejected = self.v1.check_readable(0)
        self.assertFalse(rejected.ok)
        self.assertEqual(rejected.error.code, ErrorCode.VERSION_UNSUPPORTED)

    def test_writability(self) -> None:
        self.assertTrue(self.v1.check_writable(1).ok)
        self.assertFalse(self.v1.check_writable(2).ok)          # never downgrade newer data
        self.assertFalse(self.v3.check_writable(1).ok)          # older without migration
        self.assertFalse(self.v1.check_writable(0).ok)


class TestMigrationArchitecture(unittest.TestCase):
    def test_v1_ships_no_migrations(self) -> None:
        manager = StorageVersionManager()
        self.assertEqual(manager.migration_path(StorageVersionManager.CURRENT_VERSION), [])

    def test_missing_hooks_mean_no_path(self) -> None:
        manager = _V3Manager()
        self.assertIsNone(manager.migration_path(1))
        result = manager.migrate({"schema_version": 1}, from_version=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VERSION_UNSUPPORTED)

    def test_registered_hooks_form_a_chain(self) -> None:
        manager = _V3Manager()
        manager.register_migration(1, lambda d: {**d, "hop1": True})
        manager.register_migration(2, lambda d: {**d, "hop2": True})
        self.assertTrue(manager.check_writable(1).ok)  # now upgradeable
        migrated = manager.migrate({"seed": 1}, from_version=1).unwrap()
        self.assertEqual(migrated, {"seed": 1, "hop1": True, "hop2": True})

    def test_partial_chain_is_still_no_path(self) -> None:
        manager = _V3Manager()
        manager.register_migration(1, lambda d: d)  # 1 -> 2 only; 2 -> 3 missing
        self.assertIsNone(manager.migration_path(1))
        self.assertFalse(manager.check_writable(1).ok)

    def test_hook_failure_is_contained(self) -> None:
        manager = _V3Manager()

        def bad_hook(_data: dict) -> dict:
            raise RuntimeError("boom")

        manager.register_migration(1, bad_hook)
        manager.register_migration(2, lambda d: d)
        result = manager.migrate({}, from_version=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, ErrorCode.VERSION_UNSUPPORTED)
        self.assertEqual(result.error.details.get("step"), 1)

    def test_migrate_current_is_identity(self) -> None:
        manager = StorageVersionManager()
        data = {"schema_version": 1, "payload": [1, 2, 3]}
        self.assertIs(manager.migrate(data, from_version=1).unwrap(), data)

    def test_downgrade_has_no_path(self) -> None:
        manager = _V3Manager()
        self.assertIsNone(manager.migration_path(3, to_version=1))


if __name__ == "__main__":
    unittest.main()
