"""Runtime build identification (live-acceptance correction, Part 2).

Proves the running build is self-identifying and content-free, so a stale
workspace can never again be tested unknowingly.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

import dozen.build_info as bi
import webllm.server as srv


class BuildIdentityTestCase(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_stable(self) -> None:
        self.assertEqual(bi.build_fingerprint(), bi.build_fingerprint())
        self.assertRegex(bi.build_fingerprint(), r"^[0-9a-f]{16}$")

    def test_identity_carries_schema_and_baseline(self) -> None:
        identity = bi.build_identity(reveal_paths=False)
        self.assertEqual(
            identity["test_baseline"], "phase4f-live-fix-candidate"
        )
        self.assertEqual(
            identity["finalization_schema_version"],
            bi.FINALIZATION_SCHEMA_VERSION,
        )
        self.assertIn("build_fingerprint", identity)

    def test_paths_hidden_unless_explicitly_revealed(self) -> None:
        remote = bi.build_identity(reveal_paths=False)
        self.assertNotIn("source_root", remote)
        self.assertNotIn("dozen_module_path", remote)
        local = bi.build_identity(reveal_paths=True)
        self.assertIn("source_root", local)
        self.assertIn("dozen_module_path", local)

    def test_source_root_is_the_real_workspace(self) -> None:
        local = bi.build_identity(reveal_paths=True)
        module_path = Path(local["dozen_module_path"])
        self.assertEqual(module_path.name, "dozen")
        self.assertTrue((module_path / "finalization.py").exists())
        self.assertEqual(Path(local["source_root"]), module_path.parent)

    def test_startup_banner_mentions_baseline_and_fingerprint(self) -> None:
        banner = bi.startup_banner()
        self.assertIn("phase4f-live-fix-candidate", banner)
        self.assertIn(bi.build_fingerprint(), banner)
        self.assertIn("source_root=", banner)

    def test_debug_endpoint_gated_by_debug_mode(self) -> None:
        # When the reliability debug mode is disabled (the default), the build
        # endpoint returns 404 exactly like every other debug route — no path
        # disclosure to ordinary remote users.
        with self.assertRaises(HTTPException) as ctx:
            srv.debug_build()
        self.assertEqual(ctx.exception.status_code, 404)

    def test_remote_debug_identity_stays_path_free_when_enabled(self) -> None:
        with mock.patch("webllm.server._debug_api", return_value=(object(), Exception)):
            identity = srv.debug_build()
        self.assertNotIn("source_root", identity)
        self.assertNotIn("dozen_module_path", identity)

    def test_fingerprint_covers_live_fix_production_surfaces(self) -> None:
        relevant = (
            "dozen/models.py",
            "webllm/server.py",
            "webllm/static/index.html",
        )
        for changed_path in relevant:
            with self.subTest(changed_path=changed_path), tempfile.TemporaryDirectory(
                prefix="dozen-build-id-"
            ) as raw:
                root = Path(raw)
                for relative in bi._FINGERPRINT_SOURCES:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bi._SOURCE_ROOT / relative, destination)
                with mock.patch.object(bi, "_SOURCE_ROOT", root):
                    before = bi.build_fingerprint()
                    target = root / changed_path
                    target.write_bytes(target.read_bytes() + b"\n# fingerprint probe\n")
                    after = bi.build_fingerprint()
                self.assertNotEqual(before, after)

    def test_fingerprint_is_independent_of_absolute_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dozen-build-a-") as a, \
                tempfile.TemporaryDirectory(prefix="dozen-build-b-") as b:
            roots = (Path(a), Path(b))
            for root in roots:
                for relative in bi._FINGERPRINT_SOURCES:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bi._SOURCE_ROOT / relative, destination)
            with mock.patch.object(bi, "_SOURCE_ROOT", roots[0]):
                first = bi.build_fingerprint()
            with mock.patch.object(bi, "_SOURCE_ROOT", roots[1]):
                second = bi.build_fingerprint()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
