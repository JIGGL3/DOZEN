"""Phase 4F — end-to-end regression using the EXACT live failed prompt.

The user requested:

    I want you to make me a orchestrator model than can orchestrate between
    different llm apis, divide tasks among them, verify outputs and
    synthesise back the results and give ti to user. (give code only)

and got back raw JSON envelopes, truncated files, worker refusals, repeated
headings, missing modules, explanatory prose, a synthesizer refusal, and no
coherent runnable project. Every scenario below is a regression fixture for
that exact failure.
"""

from __future__ import annotations

import json
import unittest

from dozen.finalization import FinalizationMode

from .harness import (
    CORE_MARKER,
    FAILED_PROMPT,
    FILES,
    PROVIDERS_MARKER,
    QUALITY_MARKER,
    FinalizationClient,
    SIZE_REFUSAL_TEXT,
    envelope_for,
    make_orchestrator,
    plan_json,
    run_failed_prompt,
    typed_entry,
    typed_envelope,
)

_PROTOCOL_TOKENS = ('"summary"', '"key_decisions"', '"artifacts"', '"confidence"')


class TestScenario1CompleteValidProject(unittest.TestCase):
    def test_no_synthesizer_call_complete_deterministic_rendering(self) -> None:
        client = FinalizationClient()
        result = run_failed_prompt(client)
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode,
                         FinalizationMode.ARTIFACT_ASSEMBLY.value)
        for path in FILES:
            self.assertEqual(result.final_answer.count(f"### {path}"), 1, path)
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, result.final_answer)

    def test_manifest_order_is_stable(self) -> None:
        client = FinalizationClient()
        result = run_failed_prompt(client)
        positions = [result.final_answer.index(f"### {path}") for path in FILES]
        self.assertEqual(positions, sorted(positions))


class TestScenario2OneSizeRefusal(unittest.TestCase):
    def test_refusal_classified_and_never_reaches_output(self) -> None:
        client = FinalizationClient(worker_replies={
            "s2": [SIZE_REFUSAL_TEXT, envelope_for("s2")],
        })
        result = run_failed_prompt(client)
        self.assertEqual(result.error, "")
        self.assertNotIn(SIZE_REFUSAL_TEXT, result.final_answer)
        self.assertNotIn("split this into multiple", result.final_answer.lower())
        for path in FILES:
            self.assertIn(path, result.final_answer)

    def test_repeated_refusal_never_succeeds_with_missing_files(self) -> None:
        client = FinalizationClient(worker_replies={
            "s2": [SIZE_REFUSAL_TEXT, SIZE_REFUSAL_TEXT],
        })
        result = run_failed_prompt(client)
        self.assertNotEqual(result.error, "")
        self.assertNotIn(SIZE_REFUSAL_TEXT, result.final_answer)
        # The clean diagnostic NAMES the missing path — that's the required
        # behavior — but never renders its (nonexistent) file content.
        self.assertNotIn("### providers/openai.py", result.final_answer)
        self.assertIn("providers/openai.py", result.final_answer)


class TestScenario3HugeTypedWorkerEnvelope(unittest.TestCase):
    def test_large_valid_envelope_parses_and_raw_json_never_shows(self) -> None:
        huge_body = FILES["orchestrator.py"] + ("\n# padding line\n" * 500)
        client = FinalizationClient(worker_replies={
            "s1": typed_envelope(
                typed_entry("orchestrator.py", content=huge_body),
                typed_entry("config.json"),
            ),
        })
        result = run_failed_prompt(client)
        self.assertEqual(result.error, "")
        self.assertIn("class Orchestrator:", result.final_answer)
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, result.final_answer)


class TestScenario4TruncatedEnvelope(unittest.TestCase):
    def test_truncated_worker_envelope_fails_closed(self) -> None:
        truncated = '{"summary": "s", "key_decisions": [], "artifacts": [{"artifact_id": "orchestrator.py", "path": "orchestrator.py", "content": "class Orch'
        client = FinalizationClient(worker_replies={
            "s1": [truncated, envelope_for("s1")],
        })
        result = run_failed_prompt(client)
        self.assertEqual(result.error, "")
        self.assertNotIn("class Orch\"", result.final_answer)
        self.assertIn("class Orchestrator:", result.final_answer)

    def test_permanently_truncated_envelope_never_marks_success(self) -> None:
        truncated = '{"summary": "s", "key_decisions": [], "artifacts": [{"artifact_id": "orchestrator.py", "path": "orchestrator.py", "content": "class Orch'
        client = FinalizationClient(worker_replies={"s1": [truncated, truncated]})
        result = run_failed_prompt(client)
        self.assertNotEqual(result.error, "")
        self.assertNotIn('"artifacts"', result.final_answer)


class TestScenario5MissingProviderModules(unittest.TestCase):
    def test_missing_files_produce_a_clean_diagnostic_no_synthesizer(self) -> None:
        client = FinalizationClient(worker_replies={
            "s2": typed_envelope(typed_entry("providers/openai.py")),
        })
        result = run_failed_prompt(client)
        self.assertEqual(client.synth_calls, 0)
        self.assertNotEqual(result.error, "")
        self.assertIn("providers/anthropic.py", result.final_answer)
        self.assertIn("could not be completed", result.final_answer)
        self.assertNotIn("### providers/anthropic.py", result.final_answer)


class TestScenario6ReverseCompletionOrder(unittest.TestCase):
    def test_output_stays_in_manifest_order_regardless_of_completion(self) -> None:
        forward = run_failed_prompt(FinalizationClient()).final_answer

        class ReverseClient(FinalizationClient):
            """Same replies; the executor's own scheduling still governs
            dispatch, so this asserts stability against parallel completion
            timing by running with max_parallelism > 1."""

        client = ReverseClient()
        result = make_orchestrator(client, max_parallelism=3).run(FAILED_PROMPT)
        self.assertEqual(result.error, "")
        positions_forward = [forward.index(f"### {p}") for p in FILES]
        positions_result = [
            result.final_answer.index(f"### {p}") for p in FILES
        ]
        self.assertEqual(positions_forward, sorted(positions_forward))
        self.assertEqual(positions_result, sorted(positions_result))
        self.assertEqual(result.final_answer, forward)


class TestScenario7RecursiveScopeLossAttempt(unittest.TestCase):
    def test_recursive_producer_cannot_bypass_artifact_obligation(self) -> None:
        # The "providers" delegation is recursive; its designated producer
        # returns ordinary prose instead of the typed artifact envelope. The
        # scope obligation is judged regardless — prose cannot substitute.
        # (The child's own instruction is deliberately distinct from
        # PROVIDERS_MARKER so the scripted reply below — not the compliant
        # fixture envelope — is what the producer actually returns.)
        prose_marker = "Describe the provider integration approach in prose."
        child_plan = {
            "analysis": "prose instead of artifacts",
            "delegations": [{
                "id": "c1", "title": "Provider notes",
                "instruction": prose_marker,
                "assigned_model": "alpha",
                "produces_parent_artifacts": True,
            }],
            "synthesis_strategy": "return the notes",
        }
        client = FinalizationClient(
            plan=plan_json(providers_complex=True),
            child_plan=child_plan,
            extra_worker_markers={
                prose_marker: "The provider clients should wrap the two "
                              "vendor SDKs in a common interface with retries.",
            },
        )
        result = make_orchestrator(client).run(FAILED_PROMPT)
        self.assertNotEqual(result.error, "")
        self.assertNotIn("wrap the two vendor SDKs", result.final_answer)
        self.assertNotIn("### providers/openai.py", result.final_answer)
        self.assertIn("providers/openai.py", result.final_answer)
        self.assertIn("providers/anthropic.py", result.final_answer)


class TestScenario8OptionalProseReport(unittest.TestCase):
    def test_four_subtask_report_renders_in_plan_order_no_synthesis(self) -> None:
        from .harness import make_report_client, REPORT_PROMPT, REPORT_SECTIONS
        client = make_report_client()
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)
        titles = [title for title, _marker in REPORT_SECTIONS.values()]
        positions = [result.final_answer.index(f"## {t}") for t in titles]
        self.assertEqual(positions, sorted(positions))


class TestScenario9ExplicitPolishedReport(unittest.TestCase):
    def test_polish_request_stays_deterministic_no_synthesis(self) -> None:
        # Phase 4G: "present it as one polished narrative" no longer triggers a
        # model synthesis pass. The deterministic ordered report is returned and
        # the scripted synth reply is never consumed.
        from .harness import make_report_client
        deterministic = make_orchestrator(make_report_client()).run(
            "Research and produce a report comparing four database systems "
            "for our analytics workload."
        ).final_answer
        client = make_report_client(synth_reply="The polished final report.")
        prompt = ("Research and produce a report comparing four database "
                  "systems for our analytics workload. Present it as one "
                  "polished narrative.")
        result = make_orchestrator(client).run(prompt)
        self.assertEqual(result.final_answer, deterministic)
        self.assertEqual(client.synth_calls, 0)

    def test_model_refusal_returns_the_ordered_result_unchanged(self) -> None:
        from .harness import make_report_client
        deterministic = make_orchestrator(make_report_client()).run(
            "Research and produce a report comparing four database systems "
            "for our analytics workload."
        ).final_answer
        client = make_report_client(synth_reply="I cannot fulfill this request.")
        prompt = ("Research and produce a report comparing four database "
                  "systems for our analytics workload. Present it as one "
                  "polished narrative.")
        result = make_orchestrator(client).run(prompt)
        self.assertEqual(result.final_answer, deterministic)


class TestScenario10EmbeddedRawEnvelope(unittest.TestCase):
    def test_worker_envelope_under_a_heading_is_flattened_not_shown_raw(self) -> None:
        # The exact failed-run structure: a raw protocol envelope appeared
        # UNDER one subtask's heading, sandwiched between the adjacent
        # sections' headings ("Scheduler and execution engine" / "Provider
        # execution layer" in the live failure). End to end, the worker's
        # envelope reply is parsed and flattened before section construction
        # (Part E/K), so the stitched multi-section answer never shows it raw.
        from .harness import make_report_client
        worker_envelope = json.dumps({
            "summary": "Implemented the scheduler.", "key_decisions": [],
            "artifacts": {"scheduler.py": "class Scheduler:\n    pass"},
            "confidence": 0.95,
        })
        client = make_report_client(worker_replies={"r2": worker_envelope})
        result = make_orchestrator(client).run(
            "Research and produce a report comparing four database systems "
            "for our analytics workload."
        )
        self.assertEqual(result.error, "")
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, result.final_answer)
        self.assertIn("class Scheduler:", result.final_answer)
        # The adjacent sections' trusted headings remain intact around it.
        self.assertIn("## Storage engines", result.final_answer)
        self.assertIn("## Operational cost", result.final_answer)

    def test_embedded_json_mid_section_is_flattened_in_place(self) -> None:
        # Variant: the envelope is embedded MID-TEXT within one section's
        # already-flattened content (Part K's own scanner, exercised at the
        # finalization layer directly with the exact failed-run text shape).
        from dozen.finalization import build_ordered_sections, render_ordered_sections
        from dozen.models import Plan, SubTask, SubTaskResult, TaskStatus

        embedded = (
            "Scheduler and execution engine\n"
            + json.dumps({
                "summary": "…", "key_decisions": [],
                "artifacts": {"scheduler.py": "class Scheduler:\n    pass"},
                "confidence": 0.95,
            })
            + "\nProvider execution layer"
        )
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Report", instruction="x", id="r1"),
        ])
        results = [SubTaskResult(subtask_id="r1", title="Report",
                                 status=TaskStatus.COMPLETED, output=embedded)]
        text = render_ordered_sections(build_ordered_sections(plan, results))
        for token in _PROTOCOL_TOKENS:
            self.assertNotIn(token, text)
        self.assertIn("class Scheduler:", text)
        self.assertIn("Scheduler and execution engine", text)
        self.assertIn("Provider execution layer", text)


if __name__ == "__main__":
    unittest.main()
