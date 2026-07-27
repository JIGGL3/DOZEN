"""Phase 4G — end-to-end: no run ever performs model synthesis.

Every scenario from the phase brief is exercised through the real orchestrator
and asserts: zero synthesizer calls, no raw internal JSON, no worker refusal
text in a successful answer, no partial project reported as success, and stable
plan/manifest-ordered output regardless of worker behaviour.
"""

from __future__ import annotations

import json
import unittest

from dozen import LLMResponse
from dozen.cancellation import CancelToken
from dozen.finalization import (
    DEFAULT_FINALIZATION_POLICY,
    build_ordered_sections,
    render_ordered_sections,
)
from dozen.models import Plan, SubTask, SubTaskResult, TaskStatus

from ..deterministic_finalization.harness import (
    FILES,
    FinalizationClient,
    REPORT_PROMPT,
    REPORT_SECTIONS,
    ReportClient,
    SIZE_REFUSAL_TEXT,
    make_orchestrator,
    make_report_client,
    report_plan_json,
    run_failed_prompt,
    typed_entry,
    typed_envelope,
)

_RAW_JSON_TOKENS = ('"key_decisions"', '"summary"', '"confidence"', '"artifacts"')
_REFUSAL_MARKERS = ("too large", "split this into", "cannot fit", "part 1 of",
                    "multiple messages", "I cannot", "I must decline")


def _assert_clean(test: unittest.TestCase, text: str) -> None:
    for token in _RAW_JSON_TOKENS:
        test.assertNotIn(token, text)
    lowered = text.lower()
    for marker in _REFUSAL_MARKERS:
        test.assertNotIn(marker.lower(), lowered)


class TestCodeOnlyDelivery(unittest.TestCase):
    def test_exact_failed_prompt_delivers_deterministically(self) -> None:
        client = FinalizationClient()
        result = run_failed_prompt(client)
        self.assertEqual(result.error, "")
        self.assertEqual(client.synth_calls, 0)
        self.assertFalse(result.model_synthesis_invoked)
        self.assertEqual(result.finalization_mode, "artifact_assembly")
        self.assertIn("class Orchestrator", result.final_answer)
        _assert_clean(self, result.final_answer)
        # Each accepted file appears exactly once, in manifest order.
        positions = [result.final_answer.find(f"### {path}") for path in FILES]
        self.assertTrue(all(p >= 0 for p in positions))
        self.assertEqual(positions, sorted(positions))

    def test_missing_artifacts_yield_a_clean_failure_not_a_partial(self) -> None:
        # s3 (quality package) returns only verifier.py, never synthesis.py.
        partial = typed_envelope(typed_entry("verifier.py"))
        client = FinalizationClient(worker_replies={"s3": partial})
        result = run_failed_prompt(client)
        self.assertEqual(client.synth_calls, 0)
        self.assertNotEqual(result.error, "")            # not reported as success
        self.assertIn("could not be completed", result.final_answer)
        self.assertIn("synthesis.py", result.final_answer)  # named as missing
        # No partial source, no raw JSON, no refusal.
        self.assertNotIn("class Verifier", result.final_answer)
        _assert_clean(self, result.final_answer)

    def test_worker_size_refusal_is_quarantined(self) -> None:
        client = FinalizationClient(worker_replies={"s3": SIZE_REFUSAL_TEXT})
        result = run_failed_prompt(client)
        self.assertEqual(client.synth_calls, 0)
        self.assertNotEqual(result.error, "")
        _assert_clean(self, result.final_answer)
        # The literal refusal text never survives into the answer.
        self.assertNotIn("split this", result.final_answer.lower())

    def test_truncated_envelope_never_prints_raw(self) -> None:
        truncated = '{"summary": "s", "key_decisions": ["a"], "artifacts": [{"artifact_id'
        client = FinalizationClient(worker_replies={"s3": truncated})
        result = run_failed_prompt(client)
        self.assertEqual(client.synth_calls, 0)
        self.assertNotIn('"artifact_id', result.final_answer)
        _assert_clean(self, result.final_answer)


class TestProseDelivery(unittest.TestCase):
    def test_four_prose_workers_render_in_plan_order(self) -> None:
        client = make_report_client()
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode, "ordered_sections")
        titles = [title for title, _m in REPORT_SECTIONS.values()]
        positions = [result.final_answer.index(f"## {t}") for t in titles]
        self.assertEqual(positions, sorted(positions))
        _assert_clean(self, result.final_answer)

    def test_user_words_synthesise_the_results_do_not_enable_synthesis(self) -> None:
        client = make_report_client()
        prompt = REPORT_PROMPT + " Please synthesise the results into one answer."
        result = make_orchestrator(client).run(prompt)
        self.assertEqual(client.synth_calls, 0)
        self.assertFalse(result.model_synthesis_invoked)
        self.assertEqual(result.finalization_mode, "ordered_sections")

    def test_polish_config_enabled_still_never_synthesizes(self) -> None:
        client = make_report_client(synth_reply="Would-be polished narrative.")
        result = make_orchestrator(client, enable_model_polish=True).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        self.assertNotIn("Would-be polished narrative.", result.final_answer)

    def test_identical_prose_from_different_tasks_is_preserved(self) -> None:
        shared = "Both sections reached the same conclusion for the workload."
        client = make_report_client(worker_replies={"r1": shared, "r2": shared})
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        # Distinct task ownership is preserved — the section is not deduped away.
        self.assertIn("## Storage engines", result.final_answer)
        self.assertIn("## Query performance", result.final_answer)
        self.assertEqual(result.final_answer.count(shared), 2)


class _FailingReportClient(ReportClient):
    """Raises a scripted exception on ONE report worker call."""

    def __init__(self, failing_marker: str, exc: Exception, **kw) -> None:
        super().__init__(**kw)
        self._failing_marker = failing_marker
        self._exc = exc

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        if "MANAGER" not in system and self._failing_marker in messages[-1].content:
            raise self._exc
        return super().complete(provider=provider, model=model,
                                messages=messages, **kwargs)


class TestProviderFailureModes(unittest.TestCase):
    def test_provider_failure_stays_deterministic(self) -> None:
        marker = REPORT_SECTIONS["r2"][1]
        client = _FailingReportClient(
            marker, RuntimeError("provider unavailable"),
            plan=report_plan_json(),
        )
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode, "ordered_sections")
        # Other sections still delivered; the failed one is a bounded note.
        self.assertIn("## Storage engines", result.final_answer)
        _assert_clean(self, result.final_answer)

    def test_queue_rejection_stays_deterministic(self) -> None:
        class QueueRejectionError(RuntimeError):
            pass

        marker = REPORT_SECTIONS["r3"][1]
        client = _FailingReportClient(
            marker, QueueRejectionError("run slot unavailable"),
            plan=report_plan_json(),
        )
        result = make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        self.assertEqual(result.finalization_mode, "ordered_sections")
        # Surviving sections are delivered; the rejected one becomes a bounded
        # "not completed" note, never a synthesis fallback or raw JSON.
        self.assertIn("## Storage engines", result.final_answer)
        self.assertIn("not completed", result.final_answer)
        _assert_clean(self, result.final_answer)

    def test_cancellation_never_triggers_synthesis(self) -> None:
        class CancellingReportClient(ReportClient):
            def complete(self, *, provider, model, messages, **kwargs):
                if "MANAGER" not in messages[0].content:
                    self.cancel_token.cancel()
                    self.cancel_token.check()
                return super().complete(provider=provider, model=model,
                                        messages=messages, **kwargs)

        client = CancellingReportClient(plan=report_plan_json())
        result = make_orchestrator(client).run(REPORT_PROMPT, cancel=CancelToken())
        self.assertTrue(result.error.startswith("Cancelled:"))
        self.assertEqual(client.synth_calls, 0)
        self.assertFalse(result.model_synthesis_invoked)


class TestOrderingIsPlanNotCompletion(unittest.TestCase):
    """Ordered finalization uses TRUSTED plan order, never completion order."""

    def _plan(self) -> Plan:
        return Plan(
            analysis="a", synthesis_strategy="s",
            subtasks=[
                SubTask(title=f"Section {i}", instruction=f"Write section {i}.",
                        id=f"s{i}")
                for i in range(1, 5)
            ],
        )

    def _results(self, order) -> list[SubTaskResult]:
        made = {
            f"s{i}": SubTaskResult(
                subtask_id=f"s{i}", title=f"Section {i}",
                status=TaskStatus.COMPLETED,
                output=f"Body of section {i} with enough words to be usable.",
            )
            for i in range(1, 5)
        }
        return [made[sid] for sid in order]

    def _rendered_order(self, completion_order) -> list[int]:
        sections = build_ordered_sections(
            self._plan(), self._results(completion_order),
            policy=DEFAULT_FINALIZATION_POLICY,
        )
        text = render_ordered_sections(sections)
        return [text.index(f"## Section {i}") for i in range(1, 5)]

    def test_reverse_completion_order_renders_in_plan_order(self) -> None:
        positions = self._rendered_order(["s4", "s3", "s2", "s1"])
        self.assertEqual(positions, sorted(positions))

    def test_random_completion_order_renders_in_plan_order(self) -> None:
        positions = self._rendered_order(["s3", "s1", "s4", "s2"])
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
