"""Phase 2 Part A — provider Chromium window geometry.

Proves the intended launch behavior (not just implementation details):
the geometry policy's output, its compatibility rules, and — through the
real ``_ensure_session`` code path with a fake Playwright — that first
startup AND worker-restart/tab-recovery launches both receive it, with the
persistent profile untouched. No real browser is launched.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

from webllm.browser_manager import (
    BrowserManager,
    _ANTIBOT_ARG,
    provider_window_launch_kwargs,
)
from webllm.providers import get_adapter


class TestGeometryPolicy(unittest.TestCase):
    def test_headed_uses_native_maximized_window(self) -> None:
        kwargs = provider_window_launch_kwargs(headless=False)
        self.assertIs(kwargs["no_viewport"], True)
        self.assertIn("--start-maximized", kwargs["args"])
        self.assertIn(_ANTIBOT_ARG, kwargs["args"])  # existing behavior preserved

    def test_headed_has_no_conflicting_geometry(self) -> None:
        kwargs = provider_window_launch_kwargs(headless=False)
        # No fixed viewport emulation may accompany the native window…
        self.assertNotIn("viewport", kwargs)
        joined = " ".join(kwargs["args"])
        # …and no fixed-size/fixed-position/kiosk/fullscreen flags may fight
        # the maximized native window.
        for forbidden in ("--window-size", "--window-position", "--kiosk",
                          "--start-fullscreen", "--app="):
            self.assertNotIn(forbidden, joined)

    def test_viewport_and_no_viewport_are_mutually_exclusive(self) -> None:
        # Playwright compatibility rule: emulation and native sizing must
        # never be requested together, in either mode.
        for headless in (False, True):
            kwargs = provider_window_launch_kwargs(headless)
            self.assertFalse(
                "viewport" in kwargs and kwargs.get("no_viewport"),
                f"headless={headless} mixes viewport emulation with no_viewport",
            )

    def test_headless_keeps_deterministic_viewport(self) -> None:
        kwargs = provider_window_launch_kwargs(headless=True)
        self.assertEqual(kwargs["viewport"], {"width": 1280, "height": 900})
        self.assertNotIn("no_viewport", kwargs)
        self.assertNotIn("--start-maximized", " ".join(kwargs["args"]))
        self.assertIn(_ANTIBOT_ARG, kwargs["args"])

    def test_policy_is_not_provider_specific(self) -> None:
        # The policy takes no provider identity: identical for every adapter.
        self.assertEqual(provider_window_launch_kwargs(False),
                         provider_window_launch_kwargs(False))


class _FakePage:
    pass


class _FakeContext:
    def __init__(self) -> None:
        self.pages = [_FakePage()]
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self) -> None:
        self.launches: list[dict] = []

    def launch_persistent_context(self, user_data_dir, **kwargs):
        record = dict(kwargs)
        record["user_data_dir"] = user_data_dir
        self.launches.append(record)
        return _FakeContext()


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()


class TestLaunchPathsUseThePolicy(unittest.TestCase):
    """Drives the real _ensure_session with a fake Playwright."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-geometry-"))
        self.bm = BrowserManager(profiles_dir=str(self.tmp), headless=False)
        self.fake_pw = _FakePlaywright()
        # Bypass only the worker-thread plumbing; the session logic under
        # test (profile dirs, launch kwargs, recovery relaunch) stays real.
        self.bm._ensure_playwright = lambda worker: self.fake_pw
        self.bm._current_worker = lambda key: None

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_startup_uses_the_geometry_policy(self) -> None:
        self.bm._ensure_session(get_adapter("openai"))
        self.assertEqual(len(self.fake_pw.chromium.launches), 1)
        launch = self.fake_pw.chromium.launches[0]
        expected = provider_window_launch_kwargs(headless=False)
        self.assertIs(launch["no_viewport"], True)
        self.assertEqual(launch["args"], expected["args"])
        self.assertNotIn("viewport", launch)

    def test_recovery_relaunch_uses_the_same_geometry(self) -> None:
        adapter = get_adapter("openai")
        self.bm._ensure_session(adapter)
        first_context = self.bm._sessions["openai"].context
        # force_new is exactly what remove/recover/restart paths pass.
        self.bm._ensure_session(adapter, force_new=True)
        self.assertEqual(len(self.fake_pw.chromium.launches), 2)
        self.assertTrue(first_context.closed)  # stale context torn down
        first, second = self.fake_pw.chromium.launches
        for key in ("no_viewport", "args", "headless"):
            self.assertEqual(first[key], second[key])

    def test_persistent_profile_behavior_is_preserved(self) -> None:
        self.bm._ensure_session(get_adapter("openai"))
        launch = self.fake_pw.chromium.launches[0]
        profile = Path(launch["user_data_dir"])
        self.assertEqual(profile, self.tmp / "openai")   # same on-disk profile
        self.assertTrue(profile.is_dir())                # created as before
        # Recovery reuses the SAME profile dir (cookies survive).
        self.bm._ensure_session(get_adapter("openai"), force_new=True)
        self.assertEqual(self.fake_pw.chromium.launches[1]["user_data_dir"],
                         str(self.tmp / "openai"))

    def test_geometry_identical_across_providers(self) -> None:
        self.bm._ensure_session(get_adapter("openai"))
        self.bm._ensure_session(get_adapter("anthropic"))
        a, b = self.fake_pw.chromium.launches
        self.assertEqual(a["args"], b["args"])
        self.assertEqual(a.get("no_viewport"), b.get("no_viewport"))

    def test_headless_manager_launches_with_viewport_fallback(self) -> None:
        bm = BrowserManager(profiles_dir=str(self.tmp / "hl"), headless=True)
        fake = _FakePlaywright()
        bm._ensure_playwright = lambda worker: fake
        bm._current_worker = lambda key: None
        bm._ensure_session(get_adapter("openai"))
        launch = fake.chromium.launches[0]
        self.assertEqual(launch["viewport"], {"width": 1280, "height": 900})
        self.assertNotIn("no_viewport", launch)


class TestSingleAuthoritativeLocation(unittest.TestCase):
    def test_one_launch_site_and_it_uses_the_policy(self) -> None:
        source = Path("webllm/browser_manager.py").read_text(encoding="utf-8")
        launch_sites = re.findall(r"launch_persistent_context\(", source)
        self.assertEqual(len(launch_sites), 1,
                         "geometry decisions must stay in one launch site")
        # And that one site takes its kwargs from the policy helper: inspect
        # the full call (up to the line that closes the argument list).
        site = source[source.index("launch_persistent_context("):]
        site = site[:site.index("\n        )")]
        self.assertIn("**provider_window_launch_kwargs(self.headless)", site)
        self.assertNotIn("viewport=", site)  # no inline geometry at the call


if __name__ == "__main__":
    unittest.main()
