"""Phase 4F — the deterministic ordered-section stitcher and prose envelope."""

from __future__ import annotations

import json
import unittest

from dozen.finalization import (
    NO_USABLE_RESULT_SENTINEL,
    FinalizationPolicy,
    build_ordered_sections,
    is_malformed_prose_envelope,
    neutralize_embedded_envelopes,
    parse_prose_envelope,
    render_ordered_sections,
    user_requested_protocol_content,
)
from dozen.models import Plan, SubTask, SubTaskResult, TaskStatus

_PROTOCOL_TOKENS = ('"summary"', '"key_decisions"', '"artifacts"', '"confidence"')


def plan_of(*titles: str) -> Plan:
    return Plan(analysis="a", synthesis_strategy="s", subtasks=[
        SubTask(title=title, instruction=f"Do {title}.", id=f"s{index}")
        for index, title in enumerate(titles, 1)
    ])


def completed(sid: str, title: str, body: str, **kwargs) -> SubTaskResult:
    return SubTaskResult(subtask_id=sid, title=title,
                         status=TaskStatus.COMPLETED, output=body, **kwargs)


def failed(sid: str, title: str, error: str) -> SubTaskResult:
    return SubTaskResult(subtask_id=sid, title=title,
                         status=TaskStatus.FAILED, error=error)


class TestSectionStitching(unittest.TestCase):
    def test_successful_sections_render_with_plan_titles(self) -> None:
        plan = plan_of("Core architecture", "Scheduler and execution",
                       "Verification")
        results = [
            completed("s1", "Core architecture", "Core body."),
            completed("s2", "Scheduler and execution", "Scheduler body."),
            completed("s3", "Verification", "Verification body."),
        ]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertEqual(
            text,
            "## Core architecture\n\nCore body.\n\n"
            "## Scheduler and execution\n\nScheduler body.\n\n"
            "## Verification\n\nVerification body.",
        )

    def test_single_successful_section_renders_content_only(self) -> None:
        plan = plan_of("Only task")
        results = [completed("s1", "Only task", "The concise answer.")]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertEqual(text, "The concise answer.")

    def test_failed_section_renders_a_bounded_readable_note(self) -> None:
        plan = plan_of("Works", "Breaks")
        results = [
            completed("s1", "Works", "Good body."),
            failed("s2", "Breaks", "provider timed out " + "x" * 900),
        ]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertIn("## Breaks", text)
        self.assertIn("was not completed", text)
        self.assertNotIn("x" * 400, text)  # bounded diagnostic

    def test_failed_sections_can_be_hidden_by_policy(self) -> None:
        plan = plan_of("Works", "Breaks")
        results = [
            completed("s1", "Works", "Good body."),
            failed("s2", "Breaks", "boom"),
        ]
        policy = FinalizationPolicy(render_failed_sections=False)
        text = render_ordered_sections(
            build_ordered_sections(plan, results, policy=policy), policy=policy
        )
        self.assertEqual(text, "Good body.")

    def test_all_failed_returns_the_legacy_sentinel(self) -> None:
        plan = plan_of("One", "Two")
        results = [failed("s1", "One", "refused"), failed("s2", "Two", "empty")]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertTrue(text.startswith(NO_USABLE_RESULT_SENTINEL))
        self.assertIn("One: refused", text)
        self.assertIn("Two: empty", text)

    def test_empty_completed_output_becomes_a_diagnostic_section(self) -> None:
        plan = plan_of("Works", "Silent")
        results = [
            completed("s1", "Works", "Good body."),
            completed("s2", "Silent", "   "),
        ]
        sections = build_ordered_sections(plan, results)
        self.assertFalse(sections[1].usable)
        text = render_ordered_sections(sections)
        self.assertIn("Good body.", text)

    def test_intentionally_repeated_content_is_preserved_in_each_section(self) -> None:
        plan = plan_of("First", "Second")
        body = "The identical body produced twice."
        results = [
            completed("s1", "First", body),
            completed("s2", "Second", body),
        ]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertEqual(text.count(body), 2)
        self.assertNotIn("content not repeated", text)

    def test_internal_metadata_never_appears(self) -> None:
        plan = plan_of("Envelope")
        envelope = json.dumps({
            "summary": "hidden summary", "key_decisions": ["SECRET-DECISION"],
            "artifacts": {"a.md": "Visible body of the answer."},
            "confidence": 0.42,
        })
        results = [completed("s1", "Envelope", envelope)]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertIn("Visible body of the answer.", text)
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, text)
        self.assertNotIn("SECRET-DECISION", text)
        self.assertNotIn("0.42", text)

    def test_embedded_envelope_inside_a_section_is_neutralized(self) -> None:
        plan = plan_of("Report")
        embedded = (
            "Scheduler and execution engine\n"
            + json.dumps({
                "summary": "…", "key_decisions": [],
                "artifacts": {"s.py": "class S:\n    pass"},
                "confidence": 0.95,
            })
            + "\nProvider execution layer"
        )
        results = [completed("s1", "Report", embedded)]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertNotIn('"confidence"', text)
        self.assertIn("class S:", text)
        self.assertIn("Provider execution layer", text)

    def test_malformed_envelope_section_fails_closed(self) -> None:
        plan = plan_of("Good", "Broken")
        broken = '{"summary": "s", "key_decisions": ["k"], "artifacts": {"f.py": "def x('
        results = [
            completed("s1", "Good", "Real content here."),
            completed("s2", "Broken", broken),
        ]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertNotIn("def x(", text)
        self.assertNotIn('"artifacts"', text)
        self.assertIn("Real content here.", text)

    def test_orchestration_json_section_fails_closed(self) -> None:
        plan = plan_of("Echo")
        echo = json.dumps({
            "analysis": "internal", "delegations": [{"id": "s1"}],
            "synthesis_strategy": "hidden",
        })
        results = [completed("s1", "Echo", echo)]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertNotIn('"delegations"', text)
        self.assertTrue(text.startswith(NO_USABLE_RESULT_SENTINEL))

    def test_requested_json_content_is_left_untouched(self) -> None:
        # Explicitly requested JSON is user data: the embedded-envelope scanner
        # must not run over it. (The complete canonical four-field envelope
        # remains internal platform-wide; requested data is ordinary JSON.)
        plan = plan_of("Data")
        data = json.dumps({"users": [{"id": 1, "name": "Ada"}], "count": 1})
        body = "Here is the requested inventory data:\n" + data
        results = [completed("s1", "Data", body)]
        sections = build_ordered_sections(plan, results, json_requested=True)
        self.assertIn(data, sections[0].content)
        self.assertIn("requested inventory data", sections[0].content)

    def test_worker_warnings_surface_as_section_metadata(self) -> None:
        plan = plan_of("Warned")
        results = [completed("s1", "Warned", "Body.",
                             worker_warnings=["numbers unverified"])]
        sections = build_ordered_sections(plan, results)
        self.assertIn("numbers unverified", sections[0].warnings)
        # Metadata only — never rendered into the visible answer.
        text = render_ordered_sections(sections)
        self.assertIn("**Warnings**", text)
        self.assertIn("numbers unverified", text)

    def test_worker_evidence_refs_render_through_typed_channel(self) -> None:
        plan = plan_of("Evidence")
        results = [completed(
            "s1", "Evidence", "Body.",
            worker_evidence_refs=["report[4]", "<internal-tag>"],
        )]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertIn("**Evidence**", text)
        self.assertIn(r"report\[4\]", text)
        self.assertIn(r"\<internal-tag\>", text)

    def test_no_model_is_ever_involved(self) -> None:
        # Pure functions: no client, no provider, no I/O handles anywhere in
        # the signature — determinism is structural.
        import inspect

        for fn in (build_ordered_sections, render_ordered_sections):
            parameters = inspect.signature(fn).parameters
            self.assertNotIn("client", parameters)
            self.assertNotIn("provider", parameters)

    def test_section_output_is_byte_stable(self) -> None:
        plan = plan_of("One", "Two", "Three")
        results = [
            completed("s1", "One", "Alpha."),
            failed("s2", "Two", "boom"),
            completed("s3", "Three", "Gamma."),
        ]
        renders = {
            render_ordered_sections(build_ordered_sections(plan, results))
            for _ in range(5)
        }
        self.assertEqual(len(renders), 1)


class TestProseEnvelope(unittest.TestCase):
    def envelope(self, **overrides) -> str:
        payload = {
            "task_id": "architecture-review",
            "status": "complete",
            "content": "User-facing section content.",
            "warnings": [],
            "evidence_refs": [],
        }
        payload.update(overrides)
        return json.dumps(payload)

    def test_valid_envelope_parses_and_content_is_the_body(self) -> None:
        parsed = parse_prose_envelope(self.envelope(warnings=["w1"],
                                                    evidence_refs=["ref-1"]))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.content, "User-facing section content.")
        self.assertEqual(parsed.warnings, ("w1",))
        self.assertEqual(parsed.evidence_refs, ("ref-1",))

    def test_unknown_fields_are_ignored_by_policy(self) -> None:
        parsed = parse_prose_envelope(self.envelope(surprise=123, order=99))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.content, "User-facing section content.")

    def test_worker_task_id_is_checked_but_not_trusted(self) -> None:
        plan = plan_of("Trusted title")
        results = [SubTaskResult(
            subtask_id="s1", title="Trusted title",
            status=TaskStatus.COMPLETED,
            output=self.envelope(task_id="some-other-task"),
        )]
        sections = build_ordered_sections(plan, results)
        self.assertEqual(sections[0].subtask_id, "s1")
        self.assertTrue(any("task id" in w for w in sections[0].internal_notes))
        self.assertFalse(sections[0].warnings)
        self.assertEqual(sections[0].content, "User-facing section content.")

    def test_malformed_prose_envelope_fails_closed(self) -> None:
        truncated = '{"task_id": "t1", "status": "complete", "content": "cut'
        self.assertTrue(is_malformed_prose_envelope(truncated))
        plan = plan_of("Broken")
        results = [completed("s1", "Broken", truncated)]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        self.assertNotIn('"task_id"', text)
        self.assertNotIn("cut", text)

    def test_missing_content_prose_envelope_fails_closed(self) -> None:
        missing = json.dumps({
            "task_id": "s1", "status": "complete", "warnings": []
        })
        self.assertTrue(is_malformed_prose_envelope(missing))
        text = render_ordered_sections(build_ordered_sections(
            plan_of("Broken"), [completed("s1", "Broken", missing)]
        ))
        self.assertNotIn('"task_id"', text)
        self.assertTrue(text.startswith(NO_USABLE_RESULT_SENTINEL))

    def test_double_encoded_prose_envelope_is_unwrapped(self) -> None:
        encoded = json.dumps(self.envelope(task_id="s1", content="Exact body."))
        parsed = parse_prose_envelope(encoded)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.content, "Exact body.")
        text = render_ordered_sections(build_ordered_sections(
            plan_of("Decoded"), [completed("s1", "Decoded", encoded)]
        ))
        self.assertEqual(text, "Exact body.")

    def test_requested_prose_shaped_json_is_exact_data(self) -> None:
        payload = self.envelope(task_id="fixture", content="fixture content")
        sections = build_ordered_sections(
            plan_of("JSON fixture"),
            [completed("s1", "JSON fixture", payload)],
            json_requested=True,
        )
        self.assertEqual(sections[0].content, payload)

    def test_requested_protocol_documentation_is_exact(self) -> None:
        fixture = "```json\n" + json.dumps({
            "summary": "documented", "key_decisions": [],
            "artifacts": {}, "confidence": 1,
        }) + "\n```"
        sections = build_ordered_sections(
            plan_of("Protocol docs"),
            [completed("s1", "Protocol docs", fixture)],
            protocol_content_requested=True,
        )
        self.assertEqual(sections[0].content, fixture)

    def test_requested_source_using_protocol_field_names_is_trusted(self) -> None:
        self.assertTrue(user_requested_protocol_content(
            "Write Python source defining a JSON fixture with summary, "
            "artifacts, key_decisions, and confidence fields."
        ))
        self.assertFalse(user_requested_protocol_content(
            "Summarize the artifact choices with confidence."
        ))

    def test_non_envelope_json_is_not_misparsed(self) -> None:
        # Shape-matching but structurally invalid (wrong "content" type) IS a
        # malformed envelope by the same strictness as the artifact envelope.
        wrong_type = '{"task_id": 4, "content": {"nested": true}}'
        self.assertIsNone(parse_prose_envelope(wrong_type))
        self.assertTrue(is_malformed_prose_envelope(wrong_type))
        # Genuinely unrelated JSON (neither key present) is untouched.
        unrelated = '{"id": 4, "value": {"nested": true}}'
        self.assertIsNone(parse_prose_envelope(unrelated))
        self.assertFalse(is_malformed_prose_envelope(unrelated))

    def test_prose_mentioning_the_keys_is_untouched(self) -> None:
        prose = 'The fields "task_id" and "content" identify each reply.'
        self.assertIsNone(parse_prose_envelope(prose))
        self.assertFalse(is_malformed_prose_envelope(prose))


class TestEmbeddedEnvelopeProtection(unittest.TestCase):
    def test_source_code_with_protocol_keys_survives_byte_identical(self) -> None:
        code = (
            'PAYLOAD = {"summary": "demo", "artifacts": {"x": 1}}\n'
            "def dump():\n    return PAYLOAD\n"
        )
        cleaned, notes = neutralize_embedded_envelopes(code)
        self.assertEqual(cleaned, code)
        self.assertEqual(notes, ())

    def test_truncated_embedded_envelope_tail_is_removed(self) -> None:
        text = ('Heading\n{"summary": "s", "key_decisions": [], '
                '"artifacts": {"f.py": "def broken(')
        cleaned, notes = neutralize_embedded_envelopes(text)
        self.assertNotIn("def broken(", cleaned)
        self.assertTrue(cleaned.startswith("Heading"))
        self.assertTrue(notes)

    def test_benign_json_between_envelopes_is_preserved(self) -> None:
        benign = json.dumps({"retries": 3, "timeout_s": 30})
        envelope = json.dumps({
            "summary": "s", "key_decisions": [],
            "artifacts": {"a.md": "Flattened body."}, "confidence": 0.9,
        })
        text = f"Intro\n{benign}\nMiddle\n{envelope}\nOutro"
        cleaned, _notes = neutralize_embedded_envelopes(text)
        self.assertIn(benign, cleaned)
        self.assertIn("Flattened body.", cleaned)
        self.assertNotIn('"confidence"', cleaned)

    def test_scan_limit_never_leaks_a_later_envelope(self) -> None:
        envelopes = [json.dumps({
            "summary": f"s{i}", "key_decisions": [],
            "artifacts": {f"a{i}.md": f"body-{i}"}, "confidence": 0.9,
        }) for i in range(7)]
        text = "Intro\n" + "\n".join(envelopes) + "\nOutro"
        cleaned, notes = neutralize_embedded_envelopes(text, max_scans=6)
        self.assertNotIn('"confidence"', cleaned)
        self.assertNotIn("body-6", cleaned)
        self.assertTrue(any("scan limit" in note for note in notes))
        rendered = render_ordered_sections(build_ordered_sections(
            plan_of("Bounded"), [completed("s1", "Bounded", text)]
        ))
        self.assertTrue(rendered.startswith(NO_USABLE_RESULT_SENTINEL))


if __name__ == "__main__":
    unittest.main()
