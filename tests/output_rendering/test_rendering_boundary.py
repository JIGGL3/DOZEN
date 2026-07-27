"""Phase 6 — the typed presentation model and the single rendering boundary."""

from __future__ import annotations

import json
import unittest

from dozen.intent import DeliverableContract, RequestIntent
from dozen.models import SubTaskResult, TaskStatus
from dozen.presentation import (
    FinalPresentation,
    PresentationKind,
    render_final_presentation,
    render_results_fallback,
    sanitize_diagnostic,
    user_requested_json,
)

from .harness import envelope


def completed(sid: str, title: str, output: str) -> SubTaskResult:
    return SubTaskResult(subtask_id=sid, title=title,
                         status=TaskStatus.COMPLETED, output=output)


class TestPresentationModel(unittest.TestCase):
    def test_presentation_is_typed_immutable_and_serializable(self) -> None:
        presentation = FinalPresentation(
            kind=PresentationKind.PROSE, text="Answer.",
            diagnostics=("note one",),
        )
        with self.assertRaises(Exception):
            presentation.text = "changed"  # type: ignore[misc]
        payload = json.dumps(presentation.to_dict())
        self.assertIn('"prose"', payload)
        self.assertEqual(presentation.schema_version, 1)

    def test_presentation_round_trip_and_none_handling(self) -> None:
        original = FinalPresentation(
            kind=PresentationKind.JSON_DATA,
            text=None,  # type: ignore[arg-type]
            diagnostics=["safe note"],  # type: ignore[arg-type]
        )
        restored = FinalPresentation(**original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.text, "")
        self.assertNotIn("None", json.dumps(restored.to_dict()))
        with self.assertRaises(ValueError):
            FinalPresentation(kind="future_kind", text="x")  # type: ignore[arg-type]

    def test_diagnostics_are_bounded_and_payload_free(self) -> None:
        huge = "failure: " + json.dumps({"artifacts": {"f": "x" * 5000}})
        presentation = FinalPresentation(
            kind=PresentationKind.DIAGNOSTIC, text="clean", diagnostics=(huge,)
        )
        stored = presentation.diagnostics[0]
        self.assertLessEqual(len(stored), 500)
        self.assertNotIn("x" * 200, stored)

    def test_sanitize_diagnostic_elides_embedded_payloads(self) -> None:
        message = "call failed: " + json.dumps({"summary": "s", "body": "y" * 400})
        cleaned = sanitize_diagnostic(message)
        self.assertIn("payload elided", cleaned)
        self.assertNotIn("y" * 100, cleaned)
        # Deterministic / idempotent on already-clean text.
        self.assertEqual(sanitize_diagnostic("plain reason"), "plain reason")

    def test_diagnostic_redacts_credentials_and_control_bytes(self) -> None:
        cleaned = sanitize_diagnostic(
            "failed\nAuthorization: Bearer abc123\nCOOKIE=session-secret\x00tail"
        )
        self.assertNotIn("abc123", cleaned)
        self.assertNotIn("session-secret", cleaned)
        self.assertNotIn("\x00", cleaned)


class TestRenderingBoundary(unittest.TestCase):
    def test_prose_passes_through_unchanged(self) -> None:
        presentation = render_final_presentation("A clear answer.", [])
        self.assertEqual(presentation.kind, PresentationKind.PROSE)
        self.assertEqual(presentation.text, "A clear answer.")
        self.assertEqual(presentation.diagnostics, ())

    def test_source_assignment_with_protocol_keys_passes_byte_identical(self) -> None:
        code = 'const data = {"summary":"product","artifacts":{"x":1}};'
        presentation = render_final_presentation(code, [])
        self.assertEqual(presentation.kind, PresentationKind.PROSE)
        self.assertEqual(presentation.text, code)

    def test_source_fixture_with_planner_keys_passes_byte_identical(self) -> None:
        code = (
            'const plan = {"analysis":"x","delegations":[],'
            '"synthesis_strategy":"y"};'
        )
        presentation = render_final_presentation(code, [])
        self.assertEqual(presentation.kind, PresentationKind.PROSE)
        self.assertEqual(presentation.text, code)

    def test_envelope_final_answer_is_flattened(self) -> None:
        presentation = render_final_presentation(
            envelope({"answer.md": "The real content."}), []
        )
        self.assertEqual(presentation.text, "The real content.")
        self.assertEqual(presentation.kind, PresentationKind.PROSE)

    def test_deeply_nested_envelope_fails_closed_after_decode_bound(self) -> None:
        value = "SAFE BODY"
        for _ in range(5):
            value = envelope({"answer.md": value})
        presentation = render_final_presentation(value, [])
        self.assertEqual(presentation.kind, PresentationKind.DIAGNOSTIC)
        self.assertNotIn('"artifacts"', presentation.text)

    def test_malformed_envelope_falls_back_to_worker_outputs(self) -> None:
        broken = '{"summary": "s", "key_decisions": ["a"], "artifacts": {"f": "unclosed'
        results = [completed("s1", "Research", "Useful research findings here.")]
        presentation = render_final_presentation(broken, results)
        self.assertEqual(presentation.text, "Useful research findings here.")
        self.assertNotIn("unclosed", presentation.text)

    def test_malformed_envelope_without_outputs_is_a_clean_diagnostic(self) -> None:
        broken = '{"summary": "s", "key_decisions": ["a"], "artifacts": {"f": "unclosed'
        presentation = render_final_presentation(broken, [])
        self.assertEqual(presentation.kind, PresentationKind.DIAGNOSTIC)
        self.assertNotIn("unclosed", presentation.text)
        self.assertTrue(presentation.diagnostics)

    def test_all_failed_subtask_summary_is_bounded_without_status_change(self) -> None:
        raw = "[No subtasks produced a usable result] " + ("worker failed; " * 200)
        presentation = render_final_presentation(raw, [])
        self.assertEqual(presentation.kind, PresentationKind.PROSE)
        self.assertLessEqual(len(presentation.text), 500)

    def test_orchestration_json_is_never_the_answer(self) -> None:
        control = json.dumps({
            "analysis": "x", "delegations": [{"id": "s1"}],
            "synthesis_strategy": "y",
        })
        presentation = render_final_presentation(control, [])
        self.assertEqual(presentation.kind, PresentationKind.DIAGNOSTIC)
        self.assertIn("internal planning data", presentation.text)
        self.assertNotIn('"delegations"', presentation.text)

    def test_json_only_when_requested(self) -> None:
        data = '{"metric": 42, "unit": "ms"}'
        requested = render_final_presentation(data, [], json_requested=True)
        self.assertEqual(requested.kind, PresentationKind.JSON_DATA)
        self.assertEqual(requested.text, data)
        unrequested = render_final_presentation(data, [], json_requested=False)
        self.assertEqual(unrequested.kind, PresentationKind.PROSE)
        self.assertEqual(unrequested.text, data)

    def test_requested_json_with_protocol_looking_keys_is_byte_identical(self) -> None:
        data = '{"summary":"inventory","artifacts":{"count":3}}'
        presentation = render_final_presentation(data, [], json_requested=True)
        self.assertEqual(presentation.kind, PresentationKind.JSON_DATA)
        self.assertEqual(presentation.text, data)

    def test_requested_fenced_json_with_protocol_keys_is_byte_identical(self) -> None:
        data = '```json\n{"summary":"inventory","artifacts":{"count":3}}\n```'
        presentation = render_final_presentation(data, [], json_requested=True)
        self.assertEqual(presentation.kind, PresentationKind.JSON_DATA)
        self.assertEqual(presentation.text, data)

    def test_requested_prefixed_json_fixture_is_byte_identical(self) -> None:
        data = 'Here is the example:\n{"summary":"inventory","artifacts":{"count":3}}'
        presentation = render_final_presentation(data, [], json_requested=True)
        self.assertEqual(presentation.kind, PresentationKind.JSON_DATA)
        self.assertEqual(presentation.text, data)

    def test_long_prefixed_malformed_envelope_fails_closed(self) -> None:
        data = ("Here is background context. " * 100) + '{"artifacts":"SECRET'
        presentation = render_final_presentation(data, [])
        self.assertEqual(presentation.kind, PresentationKind.DIAGNOSTIC)
        self.assertNotIn("SECRET", presentation.text)

    def test_rendering_is_deterministic(self) -> None:
        results = [completed("s1", "A", "Alpha output."),
                   completed("s2", "B", "Beta output.")]
        one = render_final_presentation("Answer.", results)
        two = render_final_presentation("Answer.", results)
        self.assertEqual(one, two)


class TestResultsFallback(unittest.TestCase):
    def test_fallback_orders_and_headlines_sections(self) -> None:
        results = [completed("s1", "Research", "Alpha findings."),
                   completed("s2", "Draft", "Beta prose.")]
        text = render_results_fallback(results)
        self.assertLess(text.index("Research"), text.index("Draft"))
        self.assertIn("## Research", text)
        self.assertIn("Alpha findings.", text)

    def test_fallback_excludes_protocol_outputs(self) -> None:
        results = [
            completed("s1", "Good", "Real content that helps."),
            completed("s2", "Echo", json.dumps({
                "analysis": "x", "delegations": [{"id": "s"}],
                "synthesis_strategy": "y",
            })),
        ]
        text = render_results_fallback(results)
        self.assertIn("Real content that helps.", text)
        self.assertNotIn('"delegations"', text)

    def test_fallback_decodes_envelope_outputs(self) -> None:
        results = [completed("s1", "Only", envelope({"a.md": "Inner body."}))]
        self.assertEqual(render_results_fallback(results), "Inner body.")

    def test_empty_fallback_sentinel(self) -> None:
        self.assertEqual(
            render_results_fallback([]), "[No usable subtask outputs produced.]"
        )


class TestJsonRequestDetector(unittest.TestCase):
    def test_explicit_json_requests_detected(self) -> None:
        self.assertTrue(user_requested_json(prompt="Return the result as JSON."))
        self.assertTrue(user_requested_json(prompt="Output raw JSON only."))
        self.assertTrue(user_requested_json(prompt="Respond in JSON format."))
        self.assertTrue(user_requested_json(prompt="Write it to data.json please."))
        self.assertTrue(user_requested_json(desired_output="JSON"))

    def test_mentions_are_not_requests(self) -> None:
        self.assertFalse(user_requested_json(prompt="Why is my JSON parser slow?"))
        self.assertFalse(user_requested_json(prompt="Explain what JSON is."))
        self.assertFalse(user_requested_json(prompt="Fix the bug in the tokenizer."))

    def test_negated_json_and_code_contract_do_not_enable_raw_json_mode(self) -> None:
        self.assertFalse(user_requested_json(prompt="Do not return JSON."))
        contract = DeliverableContract(
            intent=RequestIntent.IMPLEMENT,
            deliverable="a working API implementation",
            code_required=True,
        )
        self.assertFalse(user_requested_json(
            prompt="Build an API that returns JSON.", contract=contract,
        ))


if __name__ == "__main__":
    unittest.main()
