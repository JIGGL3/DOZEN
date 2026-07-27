"""Provider-adapter interruption-seam tests against a fake page.

Exercises the REAL ``ProviderAdapter.interrupt_generation`` DOM logic — stop
detection, exactly-one click, bounded quiescence — with a lightweight fake page,
so no browser is required. Proves the seam never navigates, reloads, starts a new
chat or submits a prompt.
"""

from __future__ import annotations

import threading
import unittest

from webllm.browser_cancellation import InterruptionPolicy, StopActionStatus
from webllm.providers import ProviderAdapter, get_adapter

# One real adapter's stop selectors (ChatGPT). Any element whose selector is in
# ``stop_selectors`` is considered present on the fake page.
_OPENAI = get_adapter("openai")
_STOP_SEL = _OPENAI.stop_button_selectors[0]


class FakeLocator:
    def __init__(self, page: "FakePage", matches: bool, kind: str = "stop") -> None:
        self._page = page
        self._matches = matches
        self._kind = kind

    @property
    def first(self) -> "FakeLocator":
        return self

    def count(self) -> int:
        self._page.operation_idents.append(threading.get_ident())
        if not self._matches:
            return 0
        return self._page.match_count if self._kind == "stop" else 1

    def nth(self, index: int) -> "FakeLocator":
        self._page.operation_idents.append(threading.get_ident())
        return self

    def is_visible(self, timeout=None) -> bool:
        self._page.operation_idents.append(threading.get_ident())
        if timeout is not None:
            self._page.operation_timeouts.append(timeout)
        if self._kind == "stop":
            return self._matches and self._page.stop_visible
        return self._matches and self._page.composer_usable

    def is_enabled(self, timeout=None) -> bool:
        self._page.operation_idents.append(threading.get_ident())
        if timeout is not None:
            self._page.operation_timeouts.append(timeout)
        if self._kind == "stop":
            return self._matches and self._page.stop_enabled
        return self._matches and self._page.composer_usable

    def click(self, timeout=None, force=False) -> None:
        self._page.operation_idents.append(threading.get_ident())
        if self._page.click_raises:
            raise RuntimeError("synthetic click failure")
        self._page.clicks += 1
        if self._page.click_stops:
            self._page.stop_visible = False

    def evaluate(self, js, *args) -> object:
        if "click" in js:
            if self._page.click_raises:
                raise RuntimeError("synthetic click failure")
            self._page.clicks += 1
            if self._page.click_stops:
                self._page.stop_visible = False
        return None


class FakePage:
    """A minimal page: models a visible/absent stop control and quiescence."""

    def __init__(self, *, stop_visible=True, present=True, click_raises=False,
                 click_stops=True, quiescence_after=0, stop_enabled=True,
                 composer_usable=True, match_count=1, page_closed=False) -> None:
        self.stop_visible = stop_visible
        self.present = present            # do any stop selectors match at all?
        self.click_raises = click_raises
        self.click_stops = click_stops
        self.quiescence_after = quiescence_after
        self.stop_enabled = stop_enabled
        self.composer_usable = composer_usable
        self.match_count = match_count
        self.operation_timeouts = []
        self.operation_idents = []
        self.page_closed = page_closed
        self.clicks = 0
        self.waits = 0
        self.goto_calls = 0
        self.reload_calls = 0
        self._poll = 0

    def locator(self, sel):
        self.operation_idents.append(threading.get_ident())
        if sel in _OPENAI.stop_button_selectors:
            return FakeLocator(self, self.present, "stop")
        if sel in _OPENAI.input_selectors:
            return FakeLocator(self, True, "composer")
        return FakeLocator(self, False, "other")

    def wait_for_timeout(self, ms) -> None:
        self.operation_idents.append(threading.get_ident())
        self.waits += 1
        if self.quiescence_after:
            self._poll += 1
            if self._poll >= self.quiescence_after:
                self.stop_visible = False

    # Presence of these would be a contract violation for interrupt_generation.
    def goto(self, *a, **k) -> None:
        self.goto_calls += 1

    def reload(self, *a, **k) -> None:
        self.reload_calls += 1

    def is_closed(self) -> bool:
        self.operation_idents.append(threading.get_ident())
        return self.page_closed


_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class TestInterruptGeneration(unittest.TestCase):
    def test_supported_stop_action_clicks_once_and_confirms_quiescence(self) -> None:
        page = FakePage(stop_visible=True, click_stops=True)
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.STOPPED)
        self.assertTrue(result.quiescent)
        self.assertEqual(page.clicks, 1)

    def test_delayed_quiescence_is_confirmed_within_grace(self) -> None:
        page = FakePage(stop_visible=True, click_stops=False, quiescence_after=2)
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.STOPPED)
        self.assertTrue(result.quiescent)

    def test_quiescence_timeout_reports_not_confirmed(self) -> None:
        page = FakePage(stop_visible=True, click_stops=False, quiescence_after=0)
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.STOPPED)
        self.assertFalse(result.quiescent)
        self.assertEqual(result.diagnostic, "quiescence-not-confirmed")

    def test_hidden_stop_control_is_not_misclassified_as_idle(self) -> None:
        page = FakePage(stop_visible=False)
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.NO_CONTROL)
        self.assertEqual(page.clicks, 0)  # no stale click

    def test_missing_stop_control_is_not_proof_of_idle(self) -> None:
        page = FakePage(stop_visible=True, present=False)  # no selector matches
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.NO_CONTROL)
        self.assertEqual(page.clicks, 0)

    def test_click_failure_is_reported(self) -> None:
        page = FakePage(stop_visible=True, click_raises=True)
        result = _OPENAI.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.FAILED)

    def test_unsupported_when_adapter_has_no_stop_selectors(self) -> None:
        bare = ProviderAdapter(
            key="bare", name="Bare", login_url="x", new_chat_url="x",
            stop_button_selectors=[],
        )
        page = FakePage(stop_visible=True)
        result = bare.interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.UNSUPPORTED)
        self.assertEqual(page.clicks, 0)

    def test_never_navigates_reloads_or_reopens(self) -> None:
        for page in (
            FakePage(stop_visible=True, click_stops=True),
            FakePage(stop_visible=False),
            FakePage(stop_visible=True, click_raises=True),
        ):
            _OPENAI.interrupt_generation(page, None, _FAST)
            self.assertEqual(page.goto_calls, 0)
            self.assertEqual(page.reload_calls, 0)

    def test_runs_synchronously_on_caller_thread(self) -> None:
        page = FakePage(stop_visible=True, click_stops=True)
        seen = {}

        def run() -> None:
            _OPENAI.interrupt_generation(page, None, _FAST)
            seen["ident"] = threading.get_ident()

        t = threading.Thread(target=run)
        t.start()
        t.join()
        self.assertEqual(seen["ident"], t.ident)
        self.assertEqual(set(page.operation_idents), {t.ident})


if __name__ == "__main__":
    unittest.main()
