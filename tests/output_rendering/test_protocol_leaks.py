"""Phase 6 — protocol JSON must never leak into user-facing output."""

from __future__ import annotations

import json
import unittest

from dozen.llm_client import LLMError, _extract_json
from dozen.presentation import (
    PresentationKind,
    render_final_presentation,
)
from dozen.validation import (
    decode_nested_envelope,
    is_malformed_protocol_envelope,
    looks_like_protocol_envelope,
)

from .harness import ScriptedClient, envelope, make_orchestrator, two_step_plan

RESEARCH = "Collect the relevant facts"
DRAFT = "Write the final explanation"

_PROTOCOL_TOKENS = ('"summary"', '"key_decisions"', '"artifacts"', '"confidence"')


class TestWorkerEnvelopeNeverRaw(unittest.TestCase):
    def test_valid_worker_envelopes_do_not_appear_raw(self) -> None:
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: envelope({"facts.md": "Mutexes serialize access to shared state."}),
            DRAFT: envelope({"answer.md": "A mutex guards a critical section."}),
        }, synth_reply="A mutex guards a critical section, serializing access.")
        result = make_orchestrator(client).run("Explain what a mutex does.")
        self.assertEqual(result.error, "")
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, result.final_answer)
        for r in result.subtask_results:
            for token in _PROTOCOL_TOKENS:
                self.assertNotIn(token, r.output)

    def test_internal_confidence_and_key_decisions_stay_hidden(self) -> None:
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: envelope({"facts.md": "Fact one about the scheduler design."},
                               decisions=["SECRET-DECISION-TOKEN"], confidence=0.42),
            DRAFT: envelope({"answer.md": "The scheduler uses a priority queue."}),
        }, synth_reply="The scheduler uses a priority queue as designed.")
        result = make_orchestrator(client).run("Explain the scheduler design.")
        self.assertNotIn("0.42", result.final_answer)
        self.assertNotIn('"confidence"', result.final_answer)
        # The decisions are captured as bounded metadata, never displayed raw.
        self.assertNotIn('"key_decisions"', result.final_answer)

    def test_synth_envelope_never_reaches_output_synthesis_eliminated(self) -> None:
        # Phase 4G: model synthesis is eliminated. Even with the (now inert)
        # polish flag on, no synthesizer runs, so a scripted synth envelope is
        # never flattened into the answer — the deterministic ordered sections
        # of the real worker prose are returned instead.
        synth_envelope = envelope({"final.md": "The complete stitched answer."})
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: "Some clear research findings about the topic today.",
            DRAFT: "Some clear draft prose describing the topic in detail.",
        }, synth_reply=synth_envelope)
        result = make_orchestrator(client, enable_model_polish=True).run(
            "Explain the topic."
        )
        self.assertNotIn("The complete stitched answer.", result.final_answer)
        self.assertNotIn('"artifacts"', result.final_answer)
        self.assertIn("research findings", result.final_answer)

    def test_malformed_worker_envelope_produces_clean_diagnostic(self) -> None:
        broken = '{"summary": "Did the work", "key_decisions": ["a"], "artifacts": {"f.py": "def x('
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: broken,
            DRAFT: broken,
        })
        result = make_orchestrator(client).run("Explain the topic.")
        # The broken payload never reaches the user or the stored outputs.
        self.assertNotIn("def x(", result.final_answer)
        self.assertNotIn('"artifacts"', result.final_answer)
        for r in result.subtask_results:
            self.assertNotIn("def x(", r.output)
            self.assertNotIn(broken, r.error)
        research = result.subtask_results[0]
        self.assertEqual(research.status.value, "failed")
        self.assertIn("malformed worker envelope", research.error)

    def test_nested_and_double_encoded_envelopes_are_decoded(self) -> None:
        inner = envelope({"core.md": "The actual inner content of the answer."})
        outer = envelope({"wrapped.json": inner})
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: outer,
            DRAFT: envelope({"answer.md": "Plain drafted answer content here."}),
        }, synth_reply="Final answer built from the decoded content.")
        result = make_orchestrator(client).run("Explain the topic.")
        research = result.subtask_results[0]
        self.assertIn("The actual inner content", research.output)
        self.assertNotIn('"key_decisions"', research.output)

    def test_code_containing_json_survives_byte_identical(self) -> None:
        code = (
            'import json\n\n'
            'PAYLOAD = {"summary": "demo", "artifacts": {"x": 1}}\n\n'
            'def dump():\n    return json.dumps(PAYLOAD)\n'
        )
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: envelope({"tool.py": code}),
            DRAFT: envelope({"answer.md": "Use the dump helper for serialization."}),
        }, synth_reply="Use the dump helper as shown in tool.py.")
        result = make_orchestrator(client).run("Explain the serialization helper.")
        self.assertIn(code, result.subtask_results[0].output)

    def test_json_code_fence_stays_intact(self) -> None:
        prose = (
            "Configure it like this:\n\n```json\n"
            '{"retries": 3, "timeout_s": 30}\n```\n\nThen restart the service.'
        )
        client = ScriptedClient(two_step_plan(), {
            RESEARCH: prose,
            DRAFT: "The restart procedure is documented above in detail.",
        }, synth_reply=prose)
        result = make_orchestrator(client).run("Explain the configuration.")
        self.assertIn('```json\n{"retries": 3, "timeout_s": 30}\n```',
                      result.final_answer)

    def test_user_requested_json_is_delivered_as_json(self) -> None:
        data = json.dumps({"users": [{"id": 1, "name": "Ada"}]}, indent=2)
        plan = {
            "analysis": "direct", "delegations": [],
            "direct_answer": data, "synthesis_strategy": "",
        }
        client = ScriptedClient(plan, {})
        result = make_orchestrator(client).run(
            "Give me the user list as raw JSON only, no prose."
        )
        self.assertEqual(result.error, "")
        self.assertEqual(result.final_answer, data)
        self.assertIsNotNone(result.presentation)
        self.assertEqual(result.presentation.kind.value, "json_data")

    def test_raw_parser_exception_carries_no_payload(self) -> None:
        payload = "not json " + ("x" * 50000)
        with self.assertRaises(LLMError) as ctx:
            _extract_json(payload)
        message = str(ctx.exception)
        self.assertLess(len(message), 600)
        self.assertNotIn("x" * 1000, message)
        self.assertIn("50009 chars", message)

    def test_worker_provider_exception_is_sanitized_end_to_end(self) -> None:
        class FailingWorkerClient(ScriptedClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "MANAGER" in messages[0].content:
                    return super().complete(
                        provider=provider, model=model, messages=messages, **kwargs
                    )
                raise RuntimeError("Authorization: Bearer worker-secret")

        result = make_orchestrator(
            FailingWorkerClient(two_step_plan(), {})
        ).run("Explain the topic.")
        serialized = json.dumps(result.summary()) + result.final_answer + result.error
        self.assertNotIn("worker-secret", serialized)

    def test_planner_exception_is_sanitized(self) -> None:
        class FailingPlannerClient(ScriptedClient):
            def complete(self, **kwargs):
                raise RuntimeError("COOKIE=planner-secret")

        result = make_orchestrator(
            FailingPlannerClient(two_step_plan(), {})
        ).run("Explain the topic.")
        self.assertNotIn("planner-secret", result.error)

    def test_parser_exception_contains_no_secret_payload_head(self) -> None:
        secret = "COOKIE=session-super-secret"
        with self.assertRaises(LLMError) as ctx:
            _extract_json(secret + " not-json")
        self.assertNotIn(secret, str(ctx.exception))


class TestEnvelopePredicates(unittest.TestCase):
    def test_prose_and_code_are_never_flagged(self) -> None:
        self.assertFalse(looks_like_protocol_envelope("Plain answer text."))
        self.assertFalse(looks_like_protocol_envelope(
            'def f():\n    return {"summary": 1, "artifacts": 2}\n'
        ))
        self.assertFalse(is_malformed_protocol_envelope(
            'The keys "summary" and "artifacts" are part of the schema.'
        ))

    def test_broken_envelope_is_flagged(self) -> None:
        broken = '{"summary": "s", "artifacts": {"a.py": "def f('
        self.assertTrue(looks_like_protocol_envelope(broken))
        self.assertTrue(is_malformed_protocol_envelope(broken))

    def test_one_key_and_short_prose_prefix_are_fail_closed(self) -> None:
        one_key = '{"artifacts": {"secret.txt": "TOP-SECRET'
        prefixed = 'Here is the result: {"summary":"s","artifacts":{"x":"SECRET'
        for text in (one_key, prefixed):
            self.assertTrue(is_malformed_protocol_envelope(text))

    def test_source_assignment_with_protocol_keys_is_not_an_envelope(self) -> None:
        source = 'PAYLOAD = {"summary":"s","artifacts":{"x":1}'
        self.assertFalse(is_malformed_protocol_envelope(source))

    def test_large_and_truncated_control_protocols_fail_closed(self) -> None:
        large_verdict = json.dumps({
            "passed": False, "score": 0, "feedback": "SECRET" * 200,
        })
        truncated_router = '{"agent":"alpha","reason":"SECRET'
        for value in (large_verdict, truncated_router):
            presentation = render_final_presentation(value, [])
            self.assertEqual(presentation.kind, PresentationKind.DIAGNOSTIC)
            self.assertNotIn("SECRET", presentation.text)

    def test_valid_envelope_is_not_malformed(self) -> None:
        text = envelope({"a.md": "content body"})
        self.assertTrue(looks_like_protocol_envelope(text))
        self.assertFalse(is_malformed_protocol_envelope(text))

    def test_fenced_envelope_is_recognized(self) -> None:
        fenced = "```json\n" + envelope({"a.md": "content body"}) + "\n```"
        self.assertTrue(looks_like_protocol_envelope(fenced))
        self.assertFalse(is_malformed_protocol_envelope(fenced))

    def test_nested_decode_is_bounded_and_idempotent_on_plain_text(self) -> None:
        plain = "Just an ordinary answer."
        self.assertEqual(decode_nested_envelope(plain), plain)
        inner = envelope({"core.md": "innermost value"})
        middle = envelope({"mid.json": inner})
        outer = envelope({"outer.json": middle})
        # Depth 2: outer and middle decode; the inner envelope remains as text
        # rather than recursing unboundedly.
        decoded = decode_nested_envelope(
            json.loads(outer)["artifacts"]["outer.json"], max_depth=2
        )
        self.assertIn("innermost value", decoded)


if __name__ == "__main__":
    unittest.main()
