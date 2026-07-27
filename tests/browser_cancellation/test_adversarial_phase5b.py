"""Adversarial regressions for independently reproduced Phase 5B defects."""

from __future__ import annotations

import dataclasses
import threading
import unittest

from dozen.cancellation import CancelledError
from webllm.browser_cancellation import (
    InterruptionActionResult,
    InterruptionOutcome,
    InterruptionPolicy,
    InterruptionReason,
    InterruptionRequest,
    InterruptionSnapshot,
    StopActionStatus,
)
from webllm.browser_jobs import BrowserJobId
from webllm.providers import PROVIDERS, WaitTuning, get_adapter

from ._util import CooperativeGate, Gate, InterruptibleBrowserManager, poll
from .test_adapter_seam import FakePage


_FAST = InterruptionPolicy(
    stop_action_timeout_s=0.05,
    post_stop_grace_s=0.05,
    quiescence_poll_interval_s=0.02,
)


class TestInterruptionStateHardening(unittest.TestCase):
    def test_deserialization_never_coerces_boolean_or_integer_fields(self) -> None:
        data = {
            "job_id": BrowserJobId.new().value,
            "provider": "openai",
            "requested": "false",
            "action_attempted": "false",
            "attempt_count": "1",
        }
        with self.assertRaises(TypeError):
            InterruptionSnapshot.from_dict(data)

    def test_impossible_request_timestamps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InterruptionSnapshot(
                job_id=BrowserJobId.new().value,
                provider="openai",
                requested=False,
                requested_at=10.0,
            )
        with self.assertRaises(ValueError):
            InterruptionSnapshot(
                job_id=BrowserJobId.new().value,
                provider="openai",
                requested=True,
                reason=InterruptionReason.CANCELLED.value,
                requested_at=20.0,
                completed_at=19.0,
                outcome=InterruptionOutcome.FAILED.value,
            )

    def test_first_terminal_interruption_outcome_is_stable(self) -> None:
        req = InterruptionRequest(
            job_id=BrowserJobId.new().value,
            provider="openai",
        )
        req.request(InterruptionReason.CANCELLED, "test", require_physical=True)
        self.assertTrue(req.begin_action())
        self.assertTrue(req.complete(InterruptionOutcome.STOPPED, "first"))
        self.assertFalse(req.complete(InterruptionOutcome.FAILED, "second"))
        snap = req.snapshot()
        self.assertEqual(snap.outcome, InterruptionOutcome.STOPPED.value)
        self.assertEqual(snap.diagnostic, "first")


class _AmbiguousLocator:
    def __init__(self, page: "_AmbiguousPage") -> None:
        self.page = page

    def count(self) -> int:
        return 1

    def nth(self, index: int) -> "_AmbiguousLocator":
        return self

    def is_visible(self, timeout=None) -> bool:
        return True

    def is_enabled(self, timeout=None) -> bool:
        return True

    def click(self, timeout=None) -> None:
        self.page.click_calls += 1
        raise RuntimeError("detached after dispatch")


class _AmbiguousPage:
    def __init__(self) -> None:
        self.click_calls = 0
        self.loc = _AmbiguousLocator(self)

    def locator(self, selector):
        return self.loc

    def wait_for_timeout(self, milliseconds) -> None:
        return None


class TestStopControlSafety(unittest.TestCase):
    def test_ambiguous_click_failure_is_never_retried(self) -> None:
        page = _AmbiguousPage()
        result = get_adapter("openai").interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.FAILED)
        self.assertEqual(page.click_calls, 1)

    def test_disabled_and_multiple_controls_are_no_control(self) -> None:
        disabled = get_adapter("openai").interrupt_generation(
            FakePage(stop_enabled=False), None, _FAST
        )
        multiple = get_adapter("openai").interrupt_generation(
            FakePage(match_count=2), None, _FAST
        )
        self.assertIs(disabled.status, StopActionStatus.NO_CONTROL)
        self.assertIs(multiple.status, StopActionStatus.NO_CONTROL)

    def test_closed_page_is_failed_not_idle_or_no_control(self) -> None:
        result = get_adapter("openai").interrupt_generation(
            FakePage(page_closed=True), None, _FAST
        )
        self.assertIs(result.status, StopActionStatus.FAILED)

    def test_ownership_loss_during_lookup_prevents_click(self) -> None:
        page = FakePage()
        calls = 0

        def owner() -> bool:
            nonlocal calls
            calls += 1
            return calls < 4

        result = get_adapter("openai").interrupt_generation(
            page, None, _FAST, owner_check=owner
        )
        self.assertIs(result.status, StopActionStatus.SETTLED_BEFORE_ACTION)
        self.assertEqual(page.clicks, 0)

    def test_only_provider_specific_stop_contracts_are_enabled(self) -> None:
        supported = {
            key for key, adapter in PROVIDERS.items()
            if adapter.supports_active_interruption
        }
        self.assertEqual(supported, {"openai", "anthropic", "google"})

    def test_one_idle_signal_does_not_confirm_quiescence(self) -> None:
        page = FakePage(click_stops=True, composer_usable=False)
        result = get_adapter("openai").interrupt_generation(page, None, _FAST)
        self.assertIs(result.status, StopActionStatus.STOPPED)
        self.assertFalse(result.quiescent)

    def test_each_playwright_probe_respects_the_remaining_bound(self) -> None:
        page = FakePage(click_stops=True)
        result = get_adapter("openai").interrupt_generation(page, None, _FAST)
        self.assertTrue(result.quiescent)
        self.assertTrue(page.operation_timeouts)
        self.assertLessEqual(max(page.operation_timeouts), 50)


class _ChangedOldResponsePage:
    def evaluate(self, script, payload):
        return {
            "status": "done",
            "text": "CANCELLED_PARTIAL_MUTATED",
            "count": 1,
        }


class _SuccessorResponsePage:
    def evaluate(self, script, payload):
        return {"status": "done", "text": "SUCCESSOR_COMPLETE", "count": 2}


class TestContaminationAndOwnership(unittest.TestCase):
    def test_mutated_old_response_with_same_count_is_never_returned(self) -> None:
        adapter = dataclasses.replace(
            get_adapter("openai"),
            tuning=WaitTuning(
                first_token_timeout_s=30,
                generation_timeout_s=30,
                observer_slice_ms=1,
            ),
        )
        polls = 0

        def cancel_after_first_observation() -> bool:
            nonlocal polls
            polls += 1
            return polls > 1

        with self.assertRaises(CancelledError):
            adapter._await_response_via_observer(
                _ChangedOldResponsePage(),
                adapter.response_selectors[0],
                1,
                "CANCELLED_PARTIAL_BASELINE",
                cancel_after_first_observation,
            )

    def test_new_response_count_returns_only_successor_body(self) -> None:
        adapter = get_adapter("openai")
        result = adapter._await_response_via_observer(
            _SuccessorResponsePage(),
            adapter.response_selectors[0],
            1,
            "CANCELLED_PARTIAL",
            lambda: False,
        )
        self.assertEqual(result, "SUCCESSOR_COMPLETE")

    def test_unconfirmed_stop_quarantines_provider_and_skips_successor(self) -> None:
        bm = InterruptibleBrowserManager(
            interruption_policy=_FAST,
            stop_result=InterruptionActionResult.stopped(
                quiescent=False,
                diagnostic="quiescence-not-confirmed",
            ),
        )
        try:
            gate = CooperativeGate()
            bm.gates["old"] = gate
            old = bm.submit_prompt("openai", "old")
            self.assertTrue(gate.started.wait(2))
            successor = bm.submit_prompt("openai", "successor")
            self.assertTrue(old.request_cancel())
            with self.assertRaises(CancelledError):
                old.result(timeout=1)
            self.assertTrue(poll(lambda: bm.stop_calls == 1))
            with self.assertRaises(TimeoutError):
                successor.result(timeout=2)
            self.assertNotIn("successor", bm.prompts_sent())
            self.assertEqual(bm.active_browser_job("openai"), old.job_id)
            self.assertFalse(old.snapshot().physical_settled)
            self.assertEqual(
                old.interruption_snapshot().outcome,
                InterruptionOutcome.FAILED.value,
            )
        finally:
            bm.shutdown()

    def test_default_response_slice_is_subsecond(self) -> None:
        self.assertLessEqual(WaitTuning().observer_slice_ms, 500)

    def test_shutdown_releases_waiters_while_running_wait_unwinds(self) -> None:
        bm = InterruptibleBrowserManager(interruption_policy=_FAST)
        gate = Gate(value="late")
        bm.gates["blocked"] = gate
        handle = bm.submit_prompt("openai", "blocked")
        self.assertTrue(gate.started.wait(2))
        shutdown = threading.Thread(target=bm.shutdown)
        shutdown.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "shut down"):
                handle.result(timeout=1)
            self.assertTrue(shutdown.is_alive())
        finally:
            gate.release.set()
            shutdown.join(timeout=3)
        self.assertFalse(shutdown.is_alive())
        self.assertLessEqual(bm.stop_calls, 1)
        self.assertEqual(
            handle.interruption_snapshot().outcome,
            InterruptionOutcome.SETTLED_BEFORE_ACTION.value,
        )


if __name__ == "__main__":
    unittest.main()
