"""Phase 4G — model-based synthesis is ELIMINATED from every production path.

This file used to assert an OPTIONAL model-polish pass. Phase 4G removes model
synthesis from production entirely: neither an explicit "polished narrative"
request nor the trusted ``enable_model_polish`` config flag may cause a single
synthesizer provider call. Whatever reply is scripted for the (now dead)
synthesizer, the run always returns the deterministic ordered result and
``synth_calls`` stays zero.
"""

from __future__ import annotations

import json
import unittest

from dozen.cancellation import CancelledError, CancelToken

from .harness import make_report_client, REPORT_PROMPT, make_orchestrator


class TestDisabledByDefault(unittest.TestCase):
    def test_no_synthesis_call_for_a_default_report(self) -> None:
        client = make_report_client()
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode, "ordered_sections")
        self.assertFalse(result.model_synthesis_invoked)

    def test_typed_warning_and_evidence_survive_executor_to_final_answer(self) -> None:
        envelope = json.dumps({
            "task_id": "r1",
            "status": "complete",
            "content": "Storage findings with sufficient concrete detail.",
            "warnings": ["benchmark is preliminary"],
            "evidence_refs": ["benchmark[7]"],
        })
        result = make_orchestrator(
            make_report_client(worker_replies={"r1": envelope})
        ).run(REPORT_PROMPT)
        self.assertIn("**Warnings**", result.final_answer)
        self.assertIn("benchmark is preliminary", result.final_answer)
        self.assertIn("**Evidence**", result.final_answer)
        self.assertIn(r"benchmark\[7\]", result.final_answer)
        self.assertNotIn('"evidence_refs"', result.final_answer)


class TestPolishConfigIsInert(unittest.TestCase):
    """Requesting polish / enabling the config never causes a synthesis call."""

    def _deterministic_text(self) -> str:
        return make_orchestrator(make_report_client()).run(REPORT_PROMPT).final_answer

    def test_explicit_polish_request_stays_deterministic(self) -> None:
        client = make_report_client(
            synth_reply="A single polished narrative covering everything."
        )
        prompt = REPORT_PROMPT + " Present it as one polished narrative."
        result = make_orchestrator(client).run(prompt)
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode, "ordered_sections")
        self.assertFalse(result.model_synthesis_invoked)
        self.assertNotIn("A single polished narrative", result.final_answer)

    def test_trusted_config_flag_is_inert(self) -> None:
        client = make_report_client(synth_reply="Config-enabled polish result.")
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.final_answer, self._deterministic_text())
        self.assertNotIn("Config-enabled polish result.", result.final_answer)


class TestDeterministicRegardlessOfSynthReply(unittest.TestCase):
    """The scripted synthesizer reply is NEVER consumed — output is stable."""

    def _deterministic_text(self) -> str:
        return make_orchestrator(make_report_client()).run(REPORT_PROMPT).final_answer

    def test_good_synth_reply_is_ignored(self) -> None:
        deterministic = self._deterministic_text()
        client = make_report_client(synth_reply="Polished replacement text.")
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(result.final_answer, deterministic)
        self.assertIn("## Storage engines", result.final_answer)
        self.assertEqual(client.synth_calls, 0)

    def test_typed_warning_and_evidence_survive_without_a_model(self) -> None:
        envelope = json.dumps({
            "task_id": "r1", "status": "complete",
            "content": "Storage findings with sufficient concrete detail.",
            "warnings": ["benchmark is preliminary"],
            "evidence_refs": ["benchmark-7"],
        })
        client = make_report_client(
            worker_replies={"r1": envelope},
            synth_reply="Polished narrative without copied metadata.",
        )
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(client.synth_calls, 0)
        self.assertIn("benchmark is preliminary", result.final_answer)
        self.assertIn("benchmark-7", result.final_answer)

    def test_refusal_reply_never_reaches_the_answer(self) -> None:
        deterministic = self._deterministic_text()
        client = make_report_client(synth_reply="I cannot fulfill this request.")
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(result.final_answer, deterministic)
        self.assertNotIn("I cannot fulfill", result.final_answer)
        self.assertEqual(client.synth_calls, 0)

    def test_malformed_reply_never_reaches_the_answer(self) -> None:
        deterministic = self._deterministic_text()
        broken = '{"summary": "s", "key_decisions": ["a"], "artifacts": {"f": "def x('
        client = make_report_client(synth_reply=broken)
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(result.final_answer, deterministic)
        self.assertNotIn("def x(", result.final_answer)
        self.assertEqual(client.synth_calls, 0)

    def test_would_be_synth_error_is_never_raised(self) -> None:
        # A synth reply scripted to raise proves the synthesizer is never
        # called: the run completes normally with the deterministic result.
        deterministic = self._deterministic_text()
        client = make_report_client(
            synth_reply=RuntimeError("provider unavailable")
        )
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT
        )
        self.assertEqual(result.final_answer, deterministic)
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)

    def test_would_be_synth_cancellation_is_never_triggered(self) -> None:
        # The synth reply is a CancelledError, but the fresh token is NOT
        # tripped; because the synthesizer never runs, the error never fires and
        # the run finalizes deterministically rather than cancelling.
        client = make_report_client(synth_reply=CancelledError("stop pressed"))
        result = make_orchestrator(client, enable_model_polish=True).run(
            REPORT_PROMPT, cancel=CancelToken()
        )
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)
        delivery = result.summary()["delivery"]
        self.assertEqual(delivery["finalization_mode"], "ordered_sections")
        self.assertFalse(delivery["model_synthesis_invoked"])


if __name__ == "__main__":
    unittest.main()
