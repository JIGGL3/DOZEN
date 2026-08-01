"""Completion detection must fire from the stability window, not the timeout.

The in-page observer promise is recreated on EVERY slice with a fresh
``lastChange`` clock, so it can only ever measure ``observer_slice_ms`` of
stability. With the shipped tuning (``stability_ms=1200`` vs
``observer_slice_ms=250``) its ``done`` branch was unreachable: a finished
answer was only returned by the ``generation_timeout_s`` fallback 300s later,
which presented as "the model replied but DOZEN never picked it up".

The stability window is therefore measured by the Python caller, across slices.
These tests keep the real ``_await_response_via_observer`` logic and replace only
the page.
"""

from __future__ import annotations

import dataclasses
import time
import unittest

from dozen.cancellation import CancelledError
from webllm.providers import ProviderError, WaitTuning, get_adapter

# Fast but faithful: stability_ms (60) is still LONGER than observer_slice_ms (1),
# exactly the relationship that made the in-page `done` unreachable in production.
FAST = WaitTuning(
    first_token_timeout_s=5,
    generation_timeout_s=5,
    stability_ms=60,
    observer_slice_ms=1,
    stalled_generation_grace_s=0.4,
)

ANSWER = '{"analysis":"trivial","direct_answer":"Hey! How can I help you today?"}'


def adapter(**tuning_overrides):
    tuning = dataclasses.replace(FAST, **tuning_overrides) if tuning_overrides else FAST
    return dataclasses.replace(get_adapter("openai"), tuning=tuning)


class ScriptedPage:
    """Returns a scripted sequence of observer snapshots.

    Never reports ``status='done'`` unless told to — mirroring the real page,
    whose ``done`` branch cannot be reached with the shipped tuning.
    """

    def __init__(self, snapshots: list[dict], *, hold_last: bool = True) -> None:
        self.snapshots = list(snapshots)
        self.hold_last = hold_last
        self.calls = 0

    def evaluate(self, script, payload):
        self.calls += 1
        time.sleep(0.005)  # a slice costs real time, as in the browser
        if self.snapshots:
            snap = (
                self.snapshots.pop(0)
                if len(self.snapshots) > 1 or not self.hold_last
                else self.snapshots[0]
            )
        else:
            raise AssertionError("page exhausted; the call should have returned")
        return {
            "status": snap.get("status", "streaming"),
            "text": snap.get("text", ""),
            "count": snap.get("count", 1),
            "generating": snap.get("generating", False),
            "changed": snap.get("changed", False),
        }

    def wait_for_timeout(self, ms):
        time.sleep(min(ms, 20) / 1000.0)


def observe(ad, page, before=0, before_text="", cancel=lambda: False,
            stop_selectors=None):
    return ad._await_response_via_observer(
        page, ad.response_selectors[0], before, before_text, cancel,
        stop_selectors=stop_selectors,
    )


class TestStabilityCompletion(unittest.TestCase):
    def test_settled_answer_is_returned_without_waiting_for_the_timeout(self) -> None:
        """The regression: the page never says 'done', yet we must not hang."""
        ad = adapter()
        page = ScriptedPage([{"text": ANSWER, "count": 1, "generating": False}])
        started = time.time()
        self.assertEqual(observe(ad, page), ANSWER)
        elapsed = time.time() - started
        # Returned from the stability window, nowhere near generation_timeout_s.
        self.assertLess(elapsed, 1.0)
        self.assertGreaterEqual(elapsed, 0.06)  # did honour the window

    def test_streaming_text_is_not_returned_early(self) -> None:
        ad = adapter()
        page = ScriptedPage(
            [
                {"text": "{", "count": 1, "changed": True},
                {"text": '{"analysis"', "count": 1, "changed": True},
                {"text": ANSWER[:40], "count": 1, "changed": True},
                {"text": ANSWER, "count": 1, "generating": False},
            ]
        )
        self.assertEqual(observe(ad, page), ANSWER)

    def test_a_change_during_a_slice_restarts_the_window(self) -> None:
        """Identical text across slices is not stable if it mutated between."""
        ad = adapter()
        page = ScriptedPage(
            [
                # Same text each time, but the page reports it kept changing, so
                # the window must not accumulate...
                {"text": ANSWER, "count": 1, "changed": True},
                {"text": ANSWER, "count": 1, "changed": True},
                {"text": ANSWER, "count": 1, "changed": True},
                # ...until it finally settles.
                {"text": ANSWER, "count": 1, "changed": False},
            ],
            hold_last=True,
        )
        self.assertEqual(observe(ad, page), ANSWER)
        self.assertGreaterEqual(page.calls, 4)

    def test_in_page_done_fast_path_still_wins_immediately(self) -> None:
        ad = adapter()
        page = ScriptedPage([{"status": "done", "text": ANSWER, "count": 2}])
        started = time.time()
        self.assertEqual(observe(ad, page, before=1), ANSWER)
        self.assertLess(time.time() - started, 0.2)  # no stability wait needed


class TestStuckStopControl(unittest.TestCase):
    """A stop selector matching a permanent button must not cost 300 seconds."""

    def test_stable_text_is_accepted_despite_generating_still_true(self) -> None:
        ad = adapter()
        page = ScriptedPage([{"text": ANSWER, "count": 1, "generating": True}])
        started = time.time()
        self.assertEqual(observe(ad, page), ANSWER)
        elapsed = time.time() - started
        self.assertGreaterEqual(elapsed, 0.4)   # waited the stall grace
        self.assertLess(elapsed, 2.0)           # but not generation_timeout_s

    def test_genuinely_streaming_output_is_not_cut_off_by_the_grace(self) -> None:
        """The grace requires byte-identical text, so live streaming is safe."""
        ad = adapter(stalled_generation_grace_s=0.2)
        page = ScriptedPage(
            [
                {"text": "part 1", "count": 1, "generating": True, "changed": True},
                {"text": "part 1 part 2", "count": 1, "generating": True, "changed": True},
                {"text": "part 1 part 2 done", "count": 1, "generating": False},
            ]
        )
        self.assertEqual(observe(ad, page), "part 1 part 2 done")


class TestContaminationGuardsPreserved(unittest.TestCase):
    def test_a_mutating_old_answer_is_never_returned(self) -> None:
        """count must EXCEED the baseline; text mutation alone is not an answer."""
        ad = adapter(generation_timeout_s=0.4, first_token_timeout_s=0.3)
        page = ScriptedPage(
            [{"text": "OLD PARTIAL mutating", "count": 1, "generating": False}]
        )
        with self.assertRaises(ProviderError):
            observe(ad, page, before=1, before_text="OLD PARTIAL")

    def test_empty_answer_never_satisfies_the_window(self) -> None:
        ad = adapter(generation_timeout_s=0.4, first_token_timeout_s=0.3)
        page = ScriptedPage([{"text": "   ", "count": 1, "generating": False}])
        with self.assertRaises(ProviderError):
            observe(ad, page)

    def test_cancellation_unwinds_between_slices(self) -> None:
        ad = adapter()
        page = ScriptedPage([{"text": ANSWER, "count": 1, "generating": True}])
        polls = 0

        def cancel() -> bool:
            nonlocal polls
            polls += 1
            return polls > 2

        with self.assertRaises(CancelledError):
            observe(ad, page, cancel=cancel)


class TestThinkingPausesAreNotTruncated(unittest.TestCase):
    """Modern models think for a long time — before AND during an answer.

    A partial answer must never be accepted just because the model went quiet
    for a while; the stop control going away is the real end-of-generation
    signal, and only an implausibly long stall may override it.
    """

    def test_long_mid_answer_pause_while_generating_is_not_cut_off(self) -> None:
        ad = adapter(stalled_generation_grace_s=180.0)
        page = ScriptedPage(
            [
                {"text": "Here is the plan:", "count": 1, "generating": True},
                # ... a long think, text completely unchanged, stop still shown.
                *[{"text": "Here is the plan:", "count": 1, "generating": True}] * 30,
                # ... then it resumes and finishes.
                {"text": "Here is the plan: step one", "count": 1,
                 "generating": True, "changed": True},
                {"text": "Here is the plan: step one, step two", "count": 1,
                 "generating": False},
            ],
            hold_last=False,
        )
        self.assertEqual(
            observe(ad, page), "Here is the plan: step one, step two"
        )

    def test_thinking_before_any_text_never_starts_the_clock(self) -> None:
        """No answer yet is not a stalled answer — the 90s budget governs."""
        ad = adapter(first_token_timeout_s=0.35, generation_timeout_s=5)
        page = ScriptedPage([{"text": "", "count": 0, "generating": True}])
        with self.assertRaises(ProviderError) as ctx:
            observe(ad, page)
        self.assertIn("No response appeared", str(ctx.exception))

    def test_shipped_grace_exceeds_any_plausible_pause(self) -> None:
        t = WaitTuning()
        self.assertGreaterEqual(t.stalled_generation_grace_s, 120.0)


class TestStopSignalTrust(unittest.TestCase):
    """A stop selector that already matches on the IDLE page is a false signal."""

    class _IdlePage:
        def __init__(self, matching: list[str]) -> None:
            self.matching = matching

        def evaluate(self, script, selectors):
            return [s for s in selectors if s in self.matching]

    def test_selectors_matching_while_idle_are_dropped(self) -> None:
        ad = get_adapter("openai")
        stuck = ad.stop_button_selectors[-1]  # the generic aria-label matcher
        page = self._IdlePage([stuck])
        trusted = ad._trustworthy_stop_selectors(page)
        self.assertNotIn(stuck, trusted)
        self.assertEqual(trusted, [s for s in ad.stop_button_selectors if s != stuck])

    def test_clean_idle_page_keeps_every_selector(self) -> None:
        ad = get_adapter("openai")
        self.assertEqual(
            ad._trustworthy_stop_selectors(self._IdlePage([])),
            list(ad.stop_button_selectors),
        )

    def test_probe_failure_trusts_nothing(self) -> None:
        class Boom:
            def evaluate(self, *_a):
                raise RuntimeError("page gone")

        self.assertEqual(get_adapter("openai")._trustworthy_stop_selectors(Boom()), [])

    def test_without_a_trusted_stop_signal_a_long_quiet_window_is_required(self) -> None:
        """Text stability is the only evidence left, so it must be a long one."""
        ad = adapter(no_stop_signal_stability_s=0.5, stability_ms=60)
        page = ScriptedPage([{"text": ANSWER, "count": 1, "generating": False}])
        started = time.time()
        self.assertEqual(observe(ad, page, stop_selectors=[]), ANSWER)
        elapsed = time.time() - started
        # Not the short 60ms window — the long no-stop-signal window.
        self.assertGreaterEqual(elapsed, 0.5)

    def test_with_a_trusted_stop_signal_the_short_window_applies(self) -> None:
        ad = adapter(no_stop_signal_stability_s=5.0, stability_ms=60)
        page = ScriptedPage([{"text": ANSWER, "count": 1, "generating": False}])
        started = time.time()
        self.assertEqual(
            observe(ad, page, stop_selectors=["button[data-testid='stop-button']"]),
            ANSWER,
        )
        self.assertLess(time.time() - started, 2.0)


class TestShippedTuningIsCoherent(unittest.TestCase):
    def test_stability_window_exceeds_one_slice(self) -> None:
        """Documents WHY the window cannot be measured in-page.

        If this ever inverts, the in-page ``done`` branch becomes reachable
        again — which is fine, but the Python window must keep working either
        way, so both paths are tested above.
        """
        t = WaitTuning()
        self.assertGreater(t.stability_ms, t.observer_slice_ms)

    def test_stall_grace_is_shorter_than_the_generation_timeout(self) -> None:
        t = WaitTuning()
        self.assertLess(t.stalled_generation_grace_s, t.generation_timeout_s)
        self.assertGreater(t.stalled_generation_grace_s, t.stability_ms / 1000.0)


if __name__ == "__main__":
    unittest.main()
