"""Phase 6 — terminal SSE transport carries only rendered, bounded payloads."""

from __future__ import annotations

import json
import unittest

from dozen.models import OrchestrationResult
from dozen.presentation import FinalPresentation, PresentationKind
from webllm.server import _terminal_result_event

from ..artifact_results.harness import CONTENT, assemble_dashboard


def result_with(**overrides) -> OrchestrationResult:
    base = dict(task_id="task_1", final_answer="A clean answer.")
    base.update(overrides)
    return OrchestrationResult(**base)


class TestTerminalEvent(unittest.TestCase):
    def test_final_answer_is_always_a_string(self) -> None:
        event = _terminal_result_event(
            result_with(final_answer=""), cancelled=False, conversation_id="c1"
        )
        self.assertIsInstance(event["result"]["final_answer"], str)
        event = _terminal_result_event(
            result_with(), cancelled=False, conversation_id="c1"
        )
        self.assertEqual(event["result"]["final_answer"], "A clean answer.")

    def test_no_raw_envelope_in_the_event(self) -> None:
        result = result_with(final_answer="The rendered answer.")
        event = _terminal_result_event(result, cancelled=False, conversation_id="c1")
        serialized = json.dumps(event)
        for token in ('"key_decisions"', '"confidence"', '"delegations"'):
            self.assertNotIn(token, serialized)

    def test_artifact_metadata_travels_separately_without_file_bodies(self) -> None:
        assembly = assemble_dashboard()
        result = result_with(
            final_answer="rendered deliverable text",
            artifact_assembly=assembly,
        )
        event = _terminal_result_event(result, cancelled=False, conversation_id="c1")
        artifacts = event["result"]["summary"]["artifacts"]
        self.assertEqual(artifacts["assembled"], len(assembly.artifacts))
        # Compact summary: counts only, never file contents.
        serialized = json.dumps(event["result"]["summary"])
        for content in CONTENT.values():
            self.assertNotIn(content.strip()[:40], serialized)

    def test_parser_failure_error_is_bounded_and_payload_free(self) -> None:
        noisy = "Worker call failed: " + json.dumps(
            {"summary": "s", "artifacts": {"f.py": "x" * 4000}}
        )
        event = _terminal_result_event(
            result_with(error=noisy), cancelled=False, conversation_id="c1"
        )
        error = event["result"]["error"]
        self.assertLessEqual(len(error), 500)
        self.assertNotIn("x" * 200, error)
        self.assertEqual(event["status"], "error")

    def test_summary_and_warnings_do_not_duplicate_raw_error_payloads(self) -> None:
        raw = "Authorization: Bearer secret-token\n" + ("Q" * 800)
        event = _terminal_result_event(
            result_with(error=raw, warnings=[raw]),
            cancelled=False, conversation_id="c1",
        )
        payload = event["result"]
        self.assertNotIn("secret-token", payload["error"])
        self.assertNotIn("secret-token", payload["summary"]["error"])
        self.assertNotIn("secret-token", payload["warnings"][0])
        self.assertLessEqual(len(payload["summary"]["error"]), 500)

    def test_non_string_final_answer_is_not_stringified(self) -> None:
        event = _terminal_result_event(
            result_with(final_answer={"artifacts": {"secret": "body"}}),
            cancelled=False, conversation_id="c1",
        )
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["result"]["final_answer"], "")
        self.assertNotIn("secret", event["result"]["error"])

    def test_answer_kind_is_additive_and_optional(self) -> None:
        legacy = _terminal_result_event(
            result_with(), cancelled=False, conversation_id="c1"
        )
        self.assertNotIn("answer_kind", legacy["result"])
        typed = _terminal_result_event(
            result_with(presentation=FinalPresentation(
                kind=PresentationKind.PROSE, text="A clean answer."
            )),
            cancelled=False, conversation_id="c1",
        )
        self.assertEqual(typed["result"]["answer_kind"], "prose")

    def test_existing_consumers_keep_their_fields(self) -> None:
        event = _terminal_result_event(
            result_with(), cancelled=False, conversation_id="c1"
        )
        payload = event["result"]
        for key in ("final_answer", "error", "warnings", "cancelled",
                    "summary", "conversation_id"):
            self.assertIn(key, payload)
        self.assertEqual(event["phase"], "result")
        self.assertEqual(event["status"], "done")

    def test_statuses_still_agree_with_payload(self) -> None:
        cancelled = _terminal_result_event(
            result_with(error="Cancelled: user stop"), cancelled=True,
            conversation_id="c1",
        )
        self.assertEqual(cancelled["status"], "cancelled")
        failed = _terminal_result_event(
            result_with(error="boom"), cancelled=False, conversation_id="c1"
        )
        self.assertEqual(failed["status"], "error")


if __name__ == "__main__":
    unittest.main()
