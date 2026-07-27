"""Phase 4G Part F — no ACTUAL provider call carries a synthesis-role contract.

The failure this prevents is a provider prompt that instructs a model to act as
the final synthesizer / QA reviewer and merge every other worker's output. This
inspects every system AND user prompt sent during representative runs (the exact
failed code-only prompt, a prose report, and a polish-enabled report) and
asserts none contains an active synthesis-role contract. It also pins the
planner and worker SYSTEM prompts directly.
"""

from __future__ import annotations

import unittest

from dozen import LLMResponse
from dozen.prompts import PLANNER_SYSTEM, WORKER_SYSTEM

from ..deterministic_finalization.harness import (
    FAILED_PROMPT,
    FinalizationClient,
    REPORT_PROMPT,
    make_orchestrator,
    report_plan_json,
)

# Phrases that only ever appear in an ACTIVE synthesis-role contract (the old
# planner "authorize the Synthesizer" rule, or a synthesizer/group/chunk prompt).
# The planner's PROHIBITION wording ("never create a subtask that merges…") is
# deliberately NOT in this list.
_ACTIVE_SYNTHESIS_CONTRACT_PHRASES = (
    "act as a final qa reviewer",
    "authorize and require the synthesizer",
    "you are the synthesizer",
    "synthesizer and final qa reviewer",
    "stitch them into one coherent",
    "combining consecutive sections",
    "condensing one part",
    "merge the given sections",
    "produce the merged, condensed part",
    "merge all worker",
    "integrate every supplied section",
    "consolidate the complete codebase",
)


class _CapturingClient(FinalizationClient):
    """Records the (system, user) text of EVERY provider call."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        self.calls.append((messages[0].content, messages[-1].content))
        return super().complete(provider=provider, model=model,
                                messages=messages, **kwargs)


def _assert_no_active_contract(test: unittest.TestCase, calls) -> None:
    for system, user in calls:
        blob = (system + "\n" + user).lower()
        for phrase in _ACTIVE_SYNTHESIS_CONTRACT_PHRASES:
            test.assertNotIn(
                phrase, blob,
                msg=f"an active synthesis-role contract reached a provider: {phrase!r}",
            )


class TestNoSynthesisContractInProviderCalls(unittest.TestCase):
    def test_exact_failed_code_only_prompt(self) -> None:
        client = _CapturingClient()
        make_orchestrator(client).run(FAILED_PROMPT)
        self.assertTrue(client.calls)                 # provider calls happened
        self.assertEqual(client.synth_calls, 0)
        _assert_no_active_contract(self, client.calls)

    def test_prose_report_run(self) -> None:
        client = _CapturingClient(plan=report_plan_json())
        make_orchestrator(client).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        _assert_no_active_contract(self, client.calls)

    def test_polish_enabled_report_run(self) -> None:
        client = _CapturingClient(plan=report_plan_json(),
                                  synth_reply="would-be polish")
        make_orchestrator(client, enable_model_polish=True).run(REPORT_PROMPT)
        self.assertEqual(client.synth_calls, 0)
        _assert_no_active_contract(self, client.calls)


class TestSystemPromptsAreClean(unittest.TestCase):
    def test_planner_system_has_no_active_synthesis_contract(self) -> None:
        lowered = PLANNER_SYSTEM.lower()
        self.assertNotIn("act as a final qa reviewer", lowered)
        self.assertNotIn("authorize and require the synthesizer", lowered)

    def test_planner_system_prohibits_synthesis_delegations(self) -> None:
        lowered = PLANNER_SYSTEM.lower()
        self.assertIn("never create a subtask", lowered)
        self.assertIn("deterministic", lowered)

    def test_worker_system_never_asks_to_merge_other_outputs(self) -> None:
        lowered = WORKER_SYSTEM.lower()
        for phrase in ("merge", "synthesize", "combine the outputs",
                       "integrate the outputs"):
            self.assertNotIn(phrase, lowered)


if __name__ == "__main__":
    unittest.main()
