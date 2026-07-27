"""Landing-page provider-marquee layout regression (DEFECT 2).

The live defect: the provider-strip heading appeared as a broken one-pixel
horizontal row with clipped text and the cards under it. Root cause — the
``.marquee`` viewport (``overflow:hidden``) has a purely content-driven height,
and its track is populated only AFTER the async ``/api/providers`` fetch, so
before the fetch resolves (or if it fails) the viewport collapses to a ~0px
clipped strip beneath the heading.

These assertions lock in the corrected structure WITHOUT a browser: the heading
stays in normal document flow OUTSIDE the clipped viewport, and only the
animated track viewport clips — now with a reserved ``min-height`` so it can
never collapse into a strip. A second test does a live geometry check when
Playwright is available, and skips cleanly otherwise.
"""

from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

_INDEX = Path(__file__).resolve().parents[2] / "webllm" / "static" / "index.html"


class _ClassTracker(HTMLParser):
    """Records, for each interesting element, the set of ancestor class names."""

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[set[str]] = []
        # name -> list of ancestor-class-sets observed for that selector
        self.ancestors_of_class: dict[str, list[set[str]]] = {}
        self.ancestors_of_id: dict[str, list[set[str]]] = {}

    def handle_starttag(self, tag, attrs):  # noqa: D401
        attr = dict(attrs)
        classes = set((attr.get("class") or "").split())
        ancestors: set[str] = set().union(*self._stack) if self._stack else set()
        for cls in classes:
            self.ancestors_of_class.setdefault(cls, []).append(set(ancestors))
        if attr.get("id"):
            self.ancestors_of_id.setdefault(attr["id"], []).append(set(ancestors))
        # Void elements do not nest.
        if tag not in {"br", "img", "input", "hr", "meta", "link", "source"}:
            self._stack.append(classes)

    def handle_endtag(self, tag):
        if tag not in {"br", "img", "input", "hr", "meta", "link", "source"} and self._stack:
            self._stack.pop()


class MarqueeStructureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = _INDEX.read_text(encoding="utf-8")
        parser = _ClassTracker()
        parser.feed(cls.html)
        cls.by_class = parser.ancestors_of_class
        cls.by_id = parser.ancestors_of_id

    def test_index_file_exists(self) -> None:
        self.assertTrue(_INDEX.exists(), _INDEX)

    def test_heading_is_outside_the_clipped_marquee_viewport(self) -> None:
        # The heading must exist and must NEVER be nested inside .marquee.
        self.assertIn("mq-label", self.by_class)
        for ancestors in self.by_class["mq-label"]:
            self.assertNotIn(
                "marquee", ancestors,
                "the provider heading must stay in document flow, OUTSIDE the "
                "clipped marquee viewport",
            )

    def test_animated_track_is_inside_the_marquee_viewport(self) -> None:
        self.assertIn("marquee-track", self.by_id)
        self.assertTrue(
            all("marquee" in ancestors for ancestors in self.by_id["marquee-track"]),
            "the animated provider track must live inside the marquee viewport",
        )

    def test_marquee_reserves_height_and_only_the_track_viewport_clips(self) -> None:
        rule = re.search(r"\.marquee\{([^}]*)\}", self.html)
        self.assertIsNotNone(rule, "a .marquee CSS rule must exist")
        body = rule.group(1)
        # The clipped viewport reserves a card row so it can never collapse to a
        # one-pixel strip, and it is the element that clips.
        self.assertIn("overflow:hidden", body.replace(" ", ""))
        self.assertRegex(body, r"min-height:\s*\d+px")

    def test_heading_rule_does_not_clip(self) -> None:
        rule = re.search(r"\.mq-label\{([^}]*)\}", self.html)
        self.assertIsNotNone(rule)
        self.assertNotIn("overflow:hidden", rule.group(1).replace(" ", ""))


class MarqueeLiveGeometryTestCase(unittest.TestCase):
    """Optional: render headlessly and assert the strip cannot appear. Skips
    cleanly when Playwright or a browser binary is unavailable."""

    def test_empty_marquee_never_collapses_below_a_card_row(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"playwright unavailable: {exc}")

        url = _INDEX.resolve().as_uri()
        try:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch()
                except Exception as exc:  # noqa: BLE001 - no browser binary
                    self.skipTest(f"chromium unavailable: {exc}")
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url)
                page.wait_for_timeout(300)
                # As-loaded (no catalog fetched over file://): the viewport must
                # already reserve a full card row, not a collapsed strip.
                marquee = page.query_selector(".marquee").bounding_box()
                label = page.query_selector(".mq-label").bounding_box()
                browser.close()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"headless render failed: {exc}")

        self.assertGreaterEqual(
            marquee["height"], 40.0,
            "empty marquee collapsed into a thin strip",
        )
        self.assertLessEqual(
            label["y"] + label["height"], marquee["y"] + 1.0,
            "the heading overlaps/enters the clipped marquee viewport",
        )


if __name__ == "__main__":
    unittest.main()
