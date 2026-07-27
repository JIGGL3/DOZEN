"""Admission-control tests at the endpoint layer (no real browser).

``run_task`` / ``stop_task`` are driven directly with a controllable fake
orchestrator. The conversation layer runs against a throwaway temp root, so
these exercise the real server code path end to end without Playwright.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi import HTTPException

import webllm.server as srv
from dozen.models import OrchestrationResult
from webllm import conversations as convmod
from webllm.run_slot import RunSlot


class FakeOrchestrator:
    """Blocks in ``run`` until its gate opens or the run is cancelled."""

    def __init__(self, behavior: str = "success") -> None:
        self.behavior = behavior          # success | fail | block
        self.entered = threading.Event()  # run() has begun
        self.gate = threading.Event()     # run() proceeds once set
        self.run_calls = 0
        self.observed_cancel = None
        self._lock = threading.Lock()

    def open_gate(self) -> None:
        self.gate.set()

    def run(self, task, cancel, on_event):
        with self._lock:
            self.run_calls += 1
        self.observed_cancel = cancel
        on_event({"phase": "plan", "status": "start", "message": "fake planning"})
        self.entered.set()
        # Proceed when released OR when the run is cancelled.
        while not self.gate.is_set():
            if cancel.cancelled:
                break
            time.sleep(0.005)
        if self.behavior == "fail":
            raise RuntimeError("injected worker failure")
        return OrchestrationResult(
            task_id=getattr(task, "id", "t"),
            final_answer="final answer",
            error="",
        )


def _wait(predicate, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class AdmissionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Isolated conversation storage so prepare_run/finish_run stay offline.
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-admission-"))
        self._old_env = os.environ.get(convmod._ENV_ROOT)
        os.environ[convmod._ENV_ROOT] = str(self.tmp / "conversations")
        convmod.reset_service()
        # Fresh slot per test (run_task/stop_task read srv.RUN_SLOT at call time).
        self._old_slot = srv.RUN_SLOT
        srv.RUN_SLOT = RunSlot()
        self._old_orch = srv.state.orchestrator
        self.fake = FakeOrchestrator()
        srv.state.orchestrator = self.fake
        self._run_ids: list[str] = []

    def tearDown(self) -> None:
        # Unblock anything still parked in the fake and let it release.
        self.fake.open_gate()
        _wait(lambda: not srv.RUN_SLOT.is_active(), timeout=3.0)
        for run_id in self._run_ids:
            srv.RUNS.discard(run_id)
        srv.RUN_SLOT = self._old_slot
        srv.state.orchestrator = self._old_orch
        convmod.reset_service()
        if self._old_env is None:
            os.environ.pop(convmod._ENV_ROOT, None)
        else:
            os.environ[convmod._ENV_ROOT] = self._old_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def start(self, prompt="hello"):
        resp = srv.run_task(srv.RunRequest(prompt=prompt))
        self._run_ids.append(resp["run_id"])
        return resp

    # ------------------------------------------------------------------ #
    # 1. Second run is rejected
    # ------------------------------------------------------------------ #
    def test_second_run_rejected_409(self) -> None:
        self.fake.behavior = "block"
        first = self.start("first")
        self.assertTrue(self.fake.entered.wait(3.0))

        with self.assertRaises(HTTPException) as ctx:
            srv.run_task(srv.RunRequest(prompt="second"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("active", ctx.exception.detail.lower())
        self.assertIn(first["run_id"], ctx.exception.detail)   # names the holder
        self.assertEqual(self.fake.run_calls, 1)               # second never started

    # ------------------------------------------------------------------ #
    # 2. Slot released after success
    # ------------------------------------------------------------------ #
    def test_slot_released_after_success(self) -> None:
        self.fake.open_gate()                                  # completes at once
        self.start("first")
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))
        self.start("second")                                   # accepted
        self.assertTrue(_wait(lambda: self.fake.run_calls == 2))

    # ------------------------------------------------------------------ #
    # 3. Slot released after failure
    # ------------------------------------------------------------------ #
    def test_slot_released_after_failure(self) -> None:
        self.fake.behavior = "fail"
        self.fake.open_gate()
        self.start("boom")
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))
        self.fake.behavior = "success"
        self.start("recovered")                                # slot was freed
        self.assertTrue(_wait(lambda: self.fake.run_calls == 2))

    # ------------------------------------------------------------------ #
    # 4. Slot released after cancellation
    # ------------------------------------------------------------------ #
    def test_slot_released_after_cancellation(self) -> None:
        self.fake.behavior = "block"
        self.start("long")
        self.assertTrue(self.fake.entered.wait(3.0))
        srv.stop_task()                                        # cancel active run
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))
        self.fake.behavior = "success"
        self.fake.gate.clear()
        self.fake.entered.clear()
        self.fake.open_gate()
        self.start("after-cancel")                             # accepted
        self.assertTrue(_wait(lambda: self.fake.run_calls == 2))

    # ------------------------------------------------------------------ #
    # 5. Stop targets the active run
    # ------------------------------------------------------------------ #
    def test_stop_targets_active_run(self) -> None:
        self.fake.behavior = "block"
        first = self.start("long")
        self.assertTrue(self.fake.entered.wait(3.0))
        result = srv.stop_task()
        self.assertTrue(result["stopped"])
        self.assertEqual(result["run_id"], first["run_id"])
        self.assertTrue(self.fake.observed_cancel.cancelled)   # active token tripped
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))

    def test_stop_with_no_active_run(self) -> None:
        result = srv.stop_task()
        self.assertFalse(result["stopped"])
        self.assertIn("No task", result["detail"])

    def test_repeated_stop_is_safe(self) -> None:
        self.fake.behavior = "block"
        self.start("long")
        self.assertTrue(self.fake.entered.wait(3.0))
        r1 = srv.stop_task()
        self.assertTrue(r1["stopped"])
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))
        r2 = srv.stop_task()                                   # after release
        self.assertFalse(r2["stopped"])                        # nothing to stop now

    # ------------------------------------------------------------------ #
    # 6. Cleanup is exception-safe (startup failure after acquiring the slot)
    # ------------------------------------------------------------------ #
    def test_startup_failure_after_acquire_releases_slot(self) -> None:
        original = srv.compose_task_context

        def boom(*args, **kwargs):
            raise RuntimeError("context build exploded during startup")

        srv.compose_task_context = boom
        try:
            with self.assertRaises(RuntimeError):
                srv.run_task(srv.RunRequest(prompt="doomed"))
        finally:
            srv.compose_task_context = original
        # Ownership not leaked despite the mid-startup exception.
        self.assertFalse(srv.RUN_SLOT.is_active())
        self.assertEqual(self.fake.run_calls, 0)               # workflow never began
        # And a later run can still start.
        self.fake.open_gate()
        self.start("healthy-again")
        self.assertTrue(_wait(lambda: self.fake.run_calls == 1))

    # ------------------------------------------------------------------ #
    # 7. Event channel correctness
    # ------------------------------------------------------------------ #
    def test_events_belong_to_accepted_run_only(self) -> None:
        self.fake.behavior = "block"
        first = self.start("first")
        self.assertTrue(self.fake.entered.wait(3.0))

        # Rejected second run must produce no channel and no workflow start.
        with self.assertRaises(HTTPException):
            srv.run_task(srv.RunRequest(prompt="rejected"))
        self.assertEqual(self.fake.run_calls, 1)

        # Let the accepted run finish, then inspect ITS channel.
        self.fake.open_gate()
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active()))
        channel = srv.RUNS.get(first["run_id"])
        self.assertIsNotNone(channel)
        events, _done = channel.wait_batch(0, 1.0)
        phases = [e.get("phase") for e in events]
        self.assertIn("plan", phases)                          # progress event
        self.assertIn("result", phases)                        # terminal event
        # The terminal event carries this run's own id, nobody else's.
        terminal = [e for e in events if e.get("phase") == "result"][0]
        self.assertEqual(
            terminal["result"]["conversation_id"],
            first["conversation_id"],
        )

    # ------------------------------------------------------------------ #
    # 8. Concurrency safety at the endpoint (genuine race)
    # ------------------------------------------------------------------ #
    def test_concurrent_run_requests_admit_exactly_one(self) -> None:
        self.fake.behavior = "block"
        barrier = threading.Barrier(2)
        outcomes: list[str] = []
        lock = threading.Lock()

        def submit() -> None:
            barrier.wait()
            try:
                resp = srv.run_task(srv.RunRequest(prompt="racer"))
                with lock:
                    outcomes.append(("ok", resp["run_id"]))
                    self._run_ids.append(resp["run_id"])
            except HTTPException as exc:
                with lock:
                    outcomes.append(("rejected", exc.status_code))

        threads = [threading.Thread(target=submit) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        oks = [o for o in outcomes if o[0] == "ok"]
        rejected = [o for o in outcomes if o[0] == "rejected"]
        self.assertEqual(len(oks), 1)                          # exactly one admitted
        self.assertEqual(len(rejected), 1)                     # exactly one refused
        self.assertEqual(rejected[0][1], 409)
        self.assertTrue(_wait(lambda: self.fake.run_calls == 1))  # only one started


if __name__ == "__main__":
    unittest.main()
