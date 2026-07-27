"""Phase 4F — SSE/frontend compatibility and cross-cutting integration checks
that are not already covered by the failed-prompt regression scenarios."""

from __future__ import annotations

import unittest

from .harness import FinalizationClient, run_failed_prompt


class TestSSECompatibility(unittest.TestCase):
    def test_terminal_result_event_carries_the_finalization_mode(self) -> None:
        from webllm.server import _terminal_result_event

        result = run_failed_prompt(FinalizationClient())
        event = _terminal_result_event(
            result, cancelled=False, conversation_id="c1"
        )
        summary = event["result"]["summary"]
        self.assertEqual(summary.get("finalization_mode"), "artifact_assembly")
        self.assertIsInstance(event["result"]["final_answer"], str)
        self.assertIn("answer_kind", event["result"])

    def test_final_answer_is_always_plain_text_never_a_raw_object(self) -> None:
        from webllm.server import _terminal_result_event

        result = run_failed_prompt(FinalizationClient())
        event = _terminal_result_event(
            result, cancelled=False, conversation_id="c1"
        )
        self.assertIsInstance(event["result"]["final_answer"], str)


class TestFrontendContract(unittest.TestCase):
    def test_frontend_still_only_reads_final_answer_as_a_string(self) -> None:
        import pathlib
        src = pathlib.Path("webllm/static/index.html").read_text(encoding="utf-8")
        self.assertIn('typeof r.final_answer === "string"', src)

    def test_literal_protocol_docs_bypass_frontend_envelope_unwrap(self) -> None:
        import pathlib
        from dozen.presentation import (
            PresentationKind,
            render_final_presentation,
        )

        fixture = (
            '```json\n{"summary":"documented","key_decisions":[],'
            '"artifacts":{},"confidence":1}\n```'
        )
        presentation = render_final_presentation(
            fixture, protocol_content_requested=True
        )
        self.assertEqual(presentation.kind, PresentationKind.LITERAL)
        self.assertEqual(presentation.text, fixture)
        src = pathlib.Path("webllm/static/index.html").read_text(encoding="utf-8")
        self.assertIn("isJsonData || isLiteral", src)


if __name__ == "__main__":
    unittest.main()
