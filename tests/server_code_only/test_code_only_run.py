"""Real-server /api/run regression for the exact code-only failure.

These tests drive the ACTUAL FastAPI ``run_task`` handler — real Task
construction, real contract resolution, the real planner/executor/root
finalization, and the real terminal SSE ``result`` event — using only
deterministic scripted provider replies (no Playwright, no network). This is
the coverage the existing fake-client unit tests deliberately do NOT provide:
proof that the server path itself delivers the code-only contract.

The scripted client is the finalization harness's ``FinalizationClient`` wired
into a real ``Orchestrator`` installed as ``srv.state.orchestrator``; the server
constructs the Task and resolves the contract exactly as it does in production.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

import webllm.server as srv
from webllm import conversations as convmod
from webllm.run_slot import RunSlot

from ..deterministic_finalization.harness import (
    FAILED_PROMPT,
    FILES,
    FinalizationClient,
    REPORT_PROMPT,
    SIZE_REFUSAL_TEXT,
    envelope_for,
    make_orchestrator,
    make_report_client,
    plan_json,
    typed_entry,
    typed_envelope,
)

# A pure-prose "architecture specification" a worker might return instead of
# files. Deliberately contains no code fence and no real code.
ARCH_SPEC = (
    "The orchestrator architecture consists of a Router, a Verifier and a "
    "Synthesizer. The Router dispatches each subtask to the appropriate "
    "provider API, the Verifier scores every output, and the Synthesizer "
    "merges the accepted results. This is a high-level design overview only."
)

# A JSON envelope truncated mid-artifact — the truncated-verification failure.
TRUNCATED_ENVELOPE = (
    '{"summary": "s", "key_decisions": [], "artifacts": [{"artifact_id": '
    '"verifier.py", "path": "verifier.py", "content": "class Verifier'
)


def _wait(predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class CodeOnlyServerRunTestCase(unittest.TestCase):
    """Base harness: isolated conversation storage + fresh admission slot."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-codeonly-"))
        self._old_env = os.environ.get(convmod._ENV_ROOT)
        os.environ[convmod._ENV_ROOT] = str(self.tmp / "conversations")
        convmod.reset_service()
        self._old_slot = srv.RUN_SLOT
        srv.RUN_SLOT = RunSlot()
        self._old_orch = srv.state.orchestrator
        self._run_ids: list[str] = []

    def tearDown(self) -> None:
        _wait(lambda: not srv.RUN_SLOT.is_active(), timeout=5.0)
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

    def run_orchestrator(self, orchestrator, prompt: str = FAILED_PROMPT) -> dict:
        """Drive the REAL /api/run handler to completion; return its terminal
        ``result`` event payload."""
        srv.state.orchestrator = orchestrator
        resp = srv.run_task(srv.RunRequest(prompt=prompt))
        self._run_ids.append(resp["run_id"])
        # The worker runs on a daemon thread; the slot is released on every
        # terminal path, so its release marks completion.
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active(), timeout=5.0))
        channel = srv.RUNS.get(resp["run_id"])
        self.assertIsNotNone(channel)
        events, _done = channel.wait_batch(0, timeout=2.0)
        terminal = [e for e in events if e.get("phase") == "result"]
        self.assertEqual(len(terminal), 1, "exactly one terminal result event")
        return terminal[0]

    def run_prompt(self, client, prompt: str = FAILED_PROMPT) -> dict:
        return self.run_orchestrator(make_orchestrator(client), prompt)

    def run_prompt_asgi(self, client, prompt: str = FAILED_PROMPT) -> dict:
        """Drive request parsing and routing through the actual ASGI app."""
        srv.state.orchestrator = make_orchestrator(client)

        async def post():
            transport = httpx.ASGITransport(app=srv.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as http:
                return await http.post("/api/run", json={"prompt": prompt})

        response = asyncio.run(post())
        self.assertEqual(response.status_code, 200, response.text)
        resp = response.json()
        self._run_ids.append(resp["run_id"])
        self.assertTrue(_wait(lambda: not srv.RUN_SLOT.is_active(), timeout=5.0))
        channel = srv.RUNS.get(resp["run_id"])
        self.assertIsNotNone(channel)
        events, _done = channel.wait_batch(0, timeout=2.0)
        terminal = [e for e in events if e.get("phase") == "result"]
        self.assertEqual(len(terminal), 1)
        return terminal[0]

    def delivery(self, terminal: dict) -> dict:
        return terminal["result"]["summary"]["delivery"]

    # ------------------------------------------------------------------ #
    # Case 1 — a complete typed artifact project
    # ------------------------------------------------------------------ #
    def test_case1_complete_project_artifact_assembly_no_synthesis(self) -> None:
        client = FinalizationClient()
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertEqual(terminal["status"], "done")
        self.assertEqual(result["error"], "")
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["finalization_mode"], "artifact_assembly")
        self.assertEqual(delivery["intent"], "implement")
        self.assertTrue(delivery["code_only"])
        self.assertTrue(delivery["artifact_plan_present"])
        self.assertEqual(delivery["assembly_status"], "complete")
        self.assertFalse(delivery["model_synthesis_invoked"])
        self.assertEqual(client.synth_calls, 0)
        for path in FILES:
            self.assertEqual(
                result["final_answer"].count(f"### {path}"), 1, path
            )

    # ------------------------------------------------------------------ #
    # Case 2 — architecture-only plan without an artifact plan
    # ------------------------------------------------------------------ #
    def test_case2_architecture_only_plan_replans_then_clean_diagnostic(self) -> None:
        arch_plan = {
            "analysis": "describe the architecture",
            "delegations": [
                {"id": "s1", "title": "Describe architecture",
                 "instruction": "Describe the architecture and recommend "
                                "libraries for the orchestrator.",
                 "assigned_model": "alpha"},
            ],
            "synthesis_strategy": "combine the descriptions",
        }
        client = FinalizationClient(plan=arch_plan)
        terminal = self.run_prompt(client)
        result = terminal["result"]
        # Exactly one bounded corrective re-plan happened (two planner calls).
        self.assertEqual(len(client.planner_prompts), 2)
        # The run failed cleanly at planning; no prose was delivered.
        self.assertEqual(terminal["status"], "error")
        self.assertNotEqual(result["error"], "")
        self.assertNotIn("recommend", result["final_answer"].lower())
        self.assertNotIn("high-level design", result["final_answer"].lower())
        self.assertEqual(client.synth_calls, 0)
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["intent"], "implement")
        self.assertTrue(delivery["code_only"])
        self.assertFalse(delivery["artifact_plan_present"])
        self.assertFalse(delivery["model_synthesis_invoked"])

    # ------------------------------------------------------------------ #
    # Case 3 — a worker returns an architecture spec instead of files
    # ------------------------------------------------------------------ #
    def test_case3_worker_architecture_spec_is_not_delivered(self) -> None:
        client = FinalizationClient(worker_replies={
            "s1": ARCH_SPEC, "s2": ARCH_SPEC, "s3": ARCH_SPEC,
        })
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertEqual(terminal["status"], "error")
        self.assertNotEqual(result["error"], "")
        # No ordered-section fallback: code-only stays in artifact assembly.
        self.assertEqual(self.delivery(terminal)["finalization_mode"],
                         "artifact_assembly")
        self.assertNotIn(ARCH_SPEC, result["final_answer"])
        self.assertNotIn("high-level design", result["final_answer"].lower())
        self.assertIn("could not be completed", result["final_answer"])
        self.assertEqual(client.synth_calls, 0)

    # ------------------------------------------------------------------ #
    # Case 4 — a worker returns a size refusal
    # ------------------------------------------------------------------ #
    def test_case4_size_refusal_never_reaches_terminal_output(self) -> None:
        client = FinalizationClient(worker_replies={
            "s2": [SIZE_REFUSAL_TEXT, SIZE_REFUSAL_TEXT],
        })
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertNotEqual(result["error"], "")
        self.assertNotIn(SIZE_REFUSAL_TEXT, result["final_answer"])
        self.assertNotIn("split this into multiple",
                         result["final_answer"].lower())
        # The refusal text must not leak anywhere in the terminal event.
        self.assertNotIn(SIZE_REFUSAL_TEXT, str(terminal))

    # ------------------------------------------------------------------ #
    # Case 5 — missing provider files
    # ------------------------------------------------------------------ #
    def test_case5_missing_provider_files_clean_diagnostic_no_partial(self) -> None:
        client = FinalizationClient(worker_replies={
            "s2": typed_envelope(typed_entry("providers/openai.py")),
        })
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertNotEqual(result["error"], "")
        self.assertIn("could not be completed", result["final_answer"])
        # Names the missing file, but renders no file body for it.
        self.assertIn("providers/anthropic.py", result["final_answer"])
        self.assertNotIn("### providers/anthropic.py", result["final_answer"])
        self.assertEqual(client.synth_calls, 0)

    # ------------------------------------------------------------------ #
    # Case 6 — a truncated verification/synthesis artifact
    # ------------------------------------------------------------------ #
    def test_case6_truncated_artifact_integrity_rejected_no_raw_body(self) -> None:
        client = FinalizationClient(worker_replies={
            "s3": [TRUNCATED_ENVELOPE, TRUNCATED_ENVELOPE],
        })
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertNotEqual(result["error"], "")
        # The raw truncated JSON body never appears in the terminal output.
        self.assertNotIn('"artifacts"', result["final_answer"])
        self.assertNotIn('class Verifier"', result["final_answer"])
        self.assertNotIn(TRUNCATED_ENVELOPE, str(terminal))
        self.assertIn("could not be completed", result["final_answer"])

    # ------------------------------------------------------------------ #
    # Case 7 — contract-loss simulation (a stale/legacy resolver)
    # ------------------------------------------------------------------ #
    def test_case7_contract_loss_root_guard_fails_closed(self) -> None:
        from dozen.intent import DeliverableContract, RequestIntent

        # Simulate a stale workspace / dropped contract: the resolver used
        # inside the orchestrator loses the code-only obligation. The root guard
        # re-derives the demand from the user's OWN words and fails closed.
        lost = DeliverableContract(intent=RequestIntent.ARCHITECTURE)
        with mock.patch("dozen.orchestrator.resolve_contract",
                        return_value=lost):
            terminal = self.run_prompt(FinalizationClient())
        result = terminal["result"]
        self.assertEqual(terminal["status"], "error")
        self.assertIn("code-only delivery contract was not preserved",
                      result["error"])
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["finalization_mode"], "artifact_assembly")
        self.assertTrue(delivery["code_only"])
        self.assertIn("could not be completed", result["final_answer"])
        self.assertNotIn("architecture", result["final_answer"].lower())

    # ------------------------------------------------------------------ #
    # DEFECT 1 — the exact live mechanism: a planner direct architecture answer
    # ------------------------------------------------------------------ #
    def test_defect1_direct_architecture_answer_is_not_delivered(self) -> None:
        import json as _json

        from dozen.llm_client import LLMResponse

        class DirectAnswerClient(FinalizationClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "MANAGER" in messages[0].content:
                    plan = {
                        "analysis": "trivial",
                        "direct_answer": ARCH_SPEC,
                        "synthesis_strategy": "",
                    }
                    return LLMResponse(text=_json.dumps(plan),
                                       provider=provider, model=model)
                return super().complete(provider=provider, model=model,
                                        messages=messages, **kwargs)

        terminal = self.run_prompt(DirectAnswerClient())
        result = terminal["result"]
        self.assertEqual(terminal["status"], "error")
        # The finalization boundary rejected the direct prose and produced the
        # clean deterministic artifact non-delivery instead.
        self.assertEqual(self.delivery(terminal)["finalization_mode"],
                         "artifact_assembly")
        self.assertNotIn(ARCH_SPEC, result["final_answer"])
        self.assertNotIn("high-level design", result["final_answer"].lower())
        self.assertIn("could not be completed", result["final_answer"])

    def test_actual_asgi_route_preserves_code_only_delivery(self) -> None:
        client = FinalizationClient()
        terminal = self.run_prompt_asgi(client)
        self.assertEqual(terminal["status"], "done")
        result = terminal["result"]
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["finalization_mode"], "artifact_assembly")
        self.assertTrue(delivery["code_only"])
        self.assertTrue(delivery["artifact_plan_present"])
        self.assertEqual(delivery["assembly_status"], "complete")
        self.assertFalse(delivery["model_synthesis_invoked"])
        self.assertEqual(client.synth_calls, 0)
        # The real planner accepted only owned implementation packages; no
        # delegation asks a provider to merge upstream worker results.
        for delegation in client.plan["delegations"]:
            instruction = delegation["instruction"].lower()
            self.assertNotIn("merge the worker", instruction)
            self.assertNotIn("combine all", instruction)
        # Every worker receives at most its direct typed dependency context, not
        # all earlier raw worker responses.
        for prompt in client.worker_prompts:
            self.assertLessEqual(prompt.count("--- Output of prerequisite"), 1)
        answer = result["final_answer"]
        for token in ('"artifacts"', '"key_decisions"', '"confidence"'):
            self.assertNotIn(token, answer)
        for refusal in ("too large", "cannot fit", "split this into"):
            self.assertNotIn(refusal, answer.lower())
        self.assertEqual(result["error"], "")
        for path in FILES:
            self.assertEqual(answer.count(f"### {path}"), 1)

    def test_valid_debug_code_only_contract_reaches_planner(self) -> None:
        client = FinalizationClient()
        terminal = self.run_prompt(
            client, "Fix the crash in this endpoint. Give code only."
        )
        result = terminal["result"]
        self.assertGreaterEqual(len(client.planner_prompts), 1)
        self.assertNotIn("contract was not preserved", result["error"])
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["intent"], "debug")
        self.assertTrue(delivery["code_only"])

    def test_legitimate_non_code_direct_answer_remains_compatible(self) -> None:
        answer = "The function acquires the lock before changing shared state."
        client = FinalizationClient(plan={
            "analysis": "a direct explanation is sufficient",
            "direct_answer": answer,
            "synthesis_strategy": "",
        })
        terminal = self.run_prompt(client, "Explain the code only.")
        self.assertEqual(terminal["status"], "done")
        self.assertEqual(terminal["result"]["final_answer"], answer)
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["intent"], "explain")
        self.assertFalse(delivery["code_only"])
        self.assertEqual(delivery["finalization_mode"], "")

    def test_small_direct_code_obeys_artifact_contract(self) -> None:
        direct_code = "def add(a, b):\n    return a + b"
        client = FinalizationClient(plan={
            "analysis": "small direct implementation",
            "direct_answer": direct_code,
            "synthesis_strategy": "",
        })
        terminal = self.run_prompt(
            client, "Write a small add function. Give code only."
        )
        self.assertEqual(terminal["status"], "error")
        self.assertNotIn(direct_code, terminal["result"]["final_answer"])
        self.assertIn("could not be completed", terminal["result"]["final_answer"])
        self.assertEqual(
            self.delivery(terminal)["finalization_mode"], "artifact_assembly"
        )
        self.assertEqual(client.synth_calls, 0)

    def test_empty_artifact_envelope_is_clean_non_delivery(self) -> None:
        empty = typed_envelope()
        client = FinalizationClient(worker_replies={"s1": [empty, empty]})
        terminal = self.run_prompt(client)
        result = terminal["result"]
        self.assertEqual(terminal["status"], "error")
        self.assertIn("could not be completed", result["final_answer"])
        self.assertNotIn('"artifacts"', result["final_answer"])
        self.assertEqual(self.delivery(terminal)["assembly_status"], "failed")
        self.assertEqual(client.synth_calls, 0)

    def test_provider_failure_is_not_delivered_or_serialized(self) -> None:
        class ProviderFailureClient(FinalizationClient):
            def complete(self, *, provider, model, messages, **kwargs):
                system = messages[0].content
                if "MANAGER" not in system and "SYNTHESIZER" not in system:
                    raise RuntimeError("PROVIDER_SECRET provider payload")
                return super().complete(
                    provider=provider, model=model, messages=messages, **kwargs
                )

        terminal = self.run_prompt(ProviderFailureClient())
        self.assertEqual(terminal["status"], "error")
        self.assertNotIn("PROVIDER_SECRET", repr(terminal))
        self.assertIn(
            "could not be completed", terminal["result"]["final_answer"]
        )
        self.assertFalse(self.delivery(terminal)["model_synthesis_invoked"])

    def test_queue_rejection_uses_deterministic_server_fallback(self) -> None:
        class QueueRejectionError(RuntimeError):
            pass

        client = make_report_client(
            synth_reply=QueueRejectionError("run slot unavailable")
        )
        terminal = self.run_orchestrator(
            make_orchestrator(client, enable_model_polish=True), REPORT_PROMPT
        )
        self.assertEqual(terminal["status"], "done")
        result = terminal["result"]
        self.assertEqual(result["error"], "")
        self.assertNotIn("run slot unavailable", repr(terminal))
        self.assertIn("## Storage engines", result["final_answer"])
        # Phase 4G: model synthesis is eliminated, so the scripted queue
        # rejection never fires — delivery is deterministic ordered sections and
        # no synthesis is invoked even with the (now inert) polish flag on.
        self.assertEqual(client.synth_calls, 0)
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["finalization_mode"], "ordered_sections")
        self.assertFalse(delivery["model_synthesis_invoked"])

    def test_server_cancellation_keeps_delivery_telemetry(self) -> None:
        class CancellingClient(FinalizationClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "MANAGER" not in messages[0].content:
                    self.cancel_token.cancel()
                    self.cancel_token.check()
                return super().complete(
                    provider=provider, model=model, messages=messages, **kwargs
                )

        client = CancellingClient()
        terminal = self.run_prompt(client)
        self.assertEqual(terminal["status"], "cancelled")
        delivery = self.delivery(terminal)
        self.assertEqual(delivery["intent"], "implement")
        self.assertTrue(delivery["code_only"])
        self.assertTrue(delivery["artifact_plan_present"])
        self.assertEqual(delivery["assembly_status"], "none")
        self.assertFalse(delivery["model_synthesis_invoked"])

    def test_pre_cancelled_contract_loss_stays_cancelled_and_cleans_up(self) -> None:
        from dozen.cancellation import CancelToken, NULL_TOKEN
        from dozen.intent import DeliverableContract, RequestIntent

        lost = DeliverableContract(intent=RequestIntent.ARCHITECTURE)
        client = FinalizationClient()
        orchestrator = make_orchestrator(client)
        token = CancelToken()
        token.cancel()
        with mock.patch("dozen.orchestrator.resolve_contract", return_value=lost):
            result = orchestrator.run(FAILED_PROMPT, cancel=token)
        self.assertTrue(result.error.startswith("Cancelled:"))
        self.assertNotIn("contract was not preserved", result.error)
        self.assertIs(client.cancel_token, NULL_TOKEN)
        self.assertEqual(len(client.planner_prompts), 0)

    # ------------------------------------------------------------------ #
    # Terminal-event telemetry shape + content-free guarantee
    # ------------------------------------------------------------------ #
    def test_terminal_event_carries_content_free_delivery_telemetry(self) -> None:
        client = FinalizationClient()
        terminal = self.run_prompt(client)
        delivery = self.delivery(terminal)
        self.assertEqual(
            set(delivery),
            {"finalization_mode", "intent", "code_only",
             "artifact_plan_present", "assembly_status",
             "model_synthesis_invoked"},
        )
        # No prompt text and no provider output leaked into telemetry.
        blob = repr(delivery)
        self.assertNotIn("orchestrate between", blob)
        self.assertNotIn("class Orchestrator", blob)
        # Types are safe primitives only.
        self.assertIsInstance(delivery["finalization_mode"], str)
        self.assertIsInstance(delivery["intent"], str)
        self.assertIsInstance(delivery["code_only"], bool)
        self.assertIsInstance(delivery["artifact_plan_present"], bool)
        self.assertIsInstance(delivery["assembly_status"], str)
        self.assertIsInstance(delivery["model_synthesis_invoked"], bool)


if __name__ == "__main__":
    unittest.main()
