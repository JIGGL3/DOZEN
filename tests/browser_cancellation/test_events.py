"""Interruption observability: content-free events, deterministic ordering,
isolated sink failures, no prompt/response/DOM in event payloads.
"""

from __future__ import annotations

import threading
import unittest

from webllm.browser_cancellation import InterruptionPolicy
from webllm.browser_jobs import BrowserJobEvent, JobState

from ._util import CooperativeGate, InterruptibleBrowserManager, poll_state

_FAST = InterruptionPolicy(post_stop_grace_s=0.05, quiescence_poll_interval_s=0.02,
                           stop_action_timeout_s=0.05)


class _Sink:
    def __init__(self, raise_on=None):
        self.events: list[BrowserJobEvent] = []
        self.lock = threading.Lock()
        self.raise_on = raise_on

    def __call__(self, event: BrowserJobEvent) -> None:
        with self.lock:
            self.events.append(event)
        if self.raise_on is not None and self.raise_on in (event.reason or ""):
            raise RuntimeError("observer boom")

    def reasons(self) -> list[str]:
        with self.lock:
            return [e.reason for e in self.events]


class TestInterruptionEvents(unittest.TestCase):
    def _mgr(self, sink) -> InterruptibleBrowserManager:
        return InterruptibleBrowserManager(
            job_event_sink=sink, interruption_policy=_FAST
        )

    def test_interruption_events_are_emitted(self) -> None:
        sink = _Sink()
        bm = self._mgr(sink)
        try:
            gate = CooperativeGate(value="never")
            bm.gates["coop"] = gate
            h = bm.submit_prompt("openai", "coop")
            self.assertTrue(gate.started.wait(2))
            self.assertTrue(h.request_cancel())
            self.assertTrue(poll_state(bm, h.job_id, JobState.ABANDONED))
            reasons = sink.reasons()
            self.assertIn("interruption-started", reasons)
            self.assertTrue(any(r and r.startswith("interruption-stopped") for r in reasons))
            # Ordering: started precedes its terminal interruption event.
            started = reasons.index("interruption-started")
            done = next(i for i, r in enumerate(reasons)
                        if r and r.startswith("interruption-stopped"))
            self.assertLess(started, done)
        finally:
            bm.shutdown()

    def test_events_are_content_free(self) -> None:
        sink = _Sink()
        bm = self._mgr(sink)
        try:
            gate = CooperativeGate(value="SECRET-RESPONSE-TEXT")
            bm.gates["my secret prompt"] = gate
            h = bm.submit_prompt("openai", "my secret prompt")
            self.assertTrue(gate.started.wait(2))
            self.assertTrue(h.request_cancel())
            self.assertTrue(poll_state(bm, h.job_id, JobState.ABANDONED))
            blob = "".join(str(e.to_dict()) for e in sink.events)
            self.assertNotIn("SECRET-RESPONSE-TEXT", blob)
            self.assertNotIn("my secret prompt", blob)
            # Interruption fields carry only the reason/outcome vocabulary.
            interruption_events = [
                e for e in sink.events if (e.reason or "").startswith("interruption")
            ]
            self.assertTrue(interruption_events)
            for e in interruption_events:
                self.assertIn(e.interruption_reason, (None, "cancelled", "caller_timeout"))
        finally:
            bm.shutdown()

    def test_failing_sink_never_breaks_interruption(self) -> None:
        sink = _Sink(raise_on="interruption-started")
        bm = self._mgr(sink)
        try:
            gate = CooperativeGate(value="never")
            bm.gates["coop"] = gate
            h = bm.submit_prompt("openai", "coop")
            self.assertTrue(gate.started.wait(2))
            self.assertTrue(h.request_cancel())
            # Despite the sink raising, the job still settles and the outcome is
            # recorded — an observer must never break the job.
            self.assertTrue(poll_state(bm, h.job_id, JobState.ABANDONED))
            self.assertEqual(h.interruption_snapshot().outcome, "stopped")
        finally:
            bm.shutdown()


class TestEventModel(unittest.TestCase):
    def test_interruption_fields_round_trip(self) -> None:
        from webllm.browser_jobs import BrowserJobId

        ev = BrowserJobEvent(
            job_id=BrowserJobId.new().value,
            provider="openai",
            old_state="running",
            new_state="running",
            timestamp=1.0,
            reason="interruption-stopped",
            interruption_reason="cancelled",
            interruption_outcome="stopped",
            interruption_attempted=True,
        )
        self.assertEqual(BrowserJobEvent.from_dict(ev.to_dict()), ev)

    def test_legacy_event_still_valid_without_interruption_fields(self) -> None:
        from webllm.browser_jobs import BrowserJobId

        legacy = {
            "job_id": BrowserJobId.new().value,
            "provider": "openai",
            "old_state": None,
            "new_state": "queued",
            "timestamp": 1.0,
            "reason": "queued",
        }
        ev = BrowserJobEvent.from_dict(legacy)
        self.assertFalse(ev.interruption_attempted)
        self.assertIsNone(ev.interruption_reason)


if __name__ == "__main__":
    unittest.main()
