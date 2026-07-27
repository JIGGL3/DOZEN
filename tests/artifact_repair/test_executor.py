"""Phase 4E Parts I/J — targeted repair through the EXISTING executor attempt
loop. No second retry loop, no attempt-budget reset, cancellation intact,
the semantic verifier untouched, the repair verifier scoped to the targets.
"""

from __future__ import annotations

import unittest

from dozen.cancellation import CancelToken, CancelledError
from dozen.models import TaskStatus
from dozen.verifier import Verdict

from ..artifact_results.test_executor_collection import (
    EnvelopeClient,
    all_envelopes,
    make_executor,
    run,
)
from .harness import (
    APP,
    CONTENT,
    LAYOUT,
    ROUTER,
    entry,
    envelope,
    envelope_with,
    worker_envelope_for,
)

TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"
TRUNCATED_ROUTER = 'import { DashboardPage } from "../features/dashboard/DashboardPage";\nimport { Suspense } from'


def broken_shell(**overrides):
    return envelope_with("shell", **(overrides or {LAYOUT: TRUNCATED_LAYOUT}))


def shell_prompts(client):
    return [p for p in client.worker_prompts if "Do the shell work." in p]


def other_envelopes():
    return {sid: worker_envelope_for(sid)
            for sid in ("foundation", "shared-ui", "dashboard", "tests", "validation")}


class TestTargetedSecondAttempt(unittest.TestCase):
    def test_scenario_1_only_the_damaged_file_is_re_requested(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertTrue(shell.artifact_collection.satisfied)
        # App and router are byte-identical to attempt 1.
        by_id = shell.artifact_collection.accepted_by_artifact()
        self.assertEqual(by_id[APP][0].content, CONTENT[APP])
        self.assertEqual(by_id[APP][0].attempt, 1)
        self.assertEqual(by_id[ROUTER][0].attempt, 1)
        # The repaired layout comes from attempt 2.
        self.assertEqual(by_id[LAYOUT][0].attempt, 2)
        self.assertEqual(by_id[LAYOUT][0].content, CONTENT[LAYOUT])
        # The second prompt asks ONLY for the layout.
        retry = shell_prompts(client)[1]
        self.assertIn("TARGETED ARTIFACT REPAIR", retry)
        self.assertIn(LAYOUT, retry)
        # The repair target LIST names only the layout — no other package's file.
        target_list = retry.split("Return ONLY these repair artifacts")[1].split(
            "Return one typed artifact envelope"
        )[0]
        self.assertNotIn("package.json", target_list)
        self.assertNotIn("Card.tsx", target_list)

    def test_the_preserved_files_are_not_in_the_repair_targets(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        run(make_executor(client, repairs=1))
        retry = shell_prompts(client)[1]
        targets = retry.split("Return ONLY these repair artifacts")[1]
        self.assertNotIn(APP, targets)
        self.assertNotIn(ROUTER, targets)

    def test_scenario_2_one_missing_and_one_damaged(self) -> None:
        # Attempt 1: valid App, missing router, invalid layout.
        client = EnvelopeClient({
            "shell": [
                envelope(entry(APP), entry(LAYOUT, content=TRUNCATED_LAYOUT)),
                envelope(entry(ROUTER), entry(LAYOUT)),
            ],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertTrue(shell.artifact_collection.satisfied)
        retry = shell_prompts(client)[1]
        block = retry.split("Return ONLY these repair artifacts")[1]
        self.assertIn(ROUTER, block)
        self.assertIn(LAYOUT, block)
        self.assertNotIn(APP, block)

    def test_a_corrected_second_attempt_succeeds(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertTrue(shell.artifacts_satisfied)

    def test_a_partial_second_attempt_narrows_the_third(self) -> None:
        client = EnvelopeClient({
            "shell": [
                envelope(entry(APP)),               # router + layout missing
                envelope(entry(ROUTER)),            # only router fixed
                envelope(entry(LAYOUT)),            # layout fixed
            ],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=2))["shell"]
        self.assertEqual(shell.attempts, 3)
        self.assertTrue(shell.artifact_collection.satisfied)
        third = shell_prompts(client)[2]
        block = third.split("Return ONLY these repair artifacts")[1]
        self.assertIn(LAYOUT, block)
        self.assertNotIn(ROUTER, block)


class TestBudgetAndProgress(unittest.TestCase):
    def test_a_first_attempt_success_never_repairs(self) -> None:
        client = EnvelopeClient(all_envelopes())
        run(make_executor(client, repairs=2))
        self.assertEqual(client.worker_calls, 6)

    def test_a_repeated_identical_response_stops_safely(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), broken_shell(), broken_shell()],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=2))["shell"]
        self.assertFalse(shell.artifacts_satisfied)
        # App and router remain preserved; only the layout is missing.
        self.assertEqual(
            {c.artifact_id for c in shell.artifact_collection.accepted},
            {APP, ROUTER},
        )
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        # No-progress means the whole budget is NOT spent hammering the same file.
        self.assertLessEqual(len(shell_prompts(client)), 3)

    def test_the_existing_maximum_attempts_is_respected(self) -> None:
        # Each attempt makes real progress (a new file), so no-progress never
        # fires — yet the run still stops at 1 initial + 2 repairs, leaving the
        # last file outstanding. The budget, not repair, is the ceiling.
        client = EnvelopeClient({
            "shell": [
                envelope(entry(APP)),                    # router + layout missing
                envelope(entry(ROUTER)),                 # layout still missing
                envelope(entry(LAYOUT, content=TRUNCATED_LAYOUT)),  # never valid
            ],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=2))["shell"]
        self.assertEqual(len(shell_prompts(client)), 3)
        self.assertEqual(shell.attempts, 3)
        self.assertFalse(shell.artifacts_satisfied)
        self.assertEqual(
            {c.artifact_id for c in shell.artifact_collection.accepted},
            {APP, ROUTER},
        )

    def test_no_new_retry_loop_at_zero_repairs(self) -> None:
        client = EnvelopeClient({"shell": broken_shell(), **other_envelopes()})
        run(make_executor(client, repairs=0))
        self.assertEqual(len(shell_prompts(client)), 1)

    def test_a_repair_report_is_attached(self) -> None:
        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        report = shell.artifact_repair
        self.assertIsNotNone(report)
        self.assertTrue(report.attempted)
        self.assertEqual(report.attempts_used, 1)
        self.assertIn(LAYOUT, report.repaired_artifact_ids)
        self.assertIn(APP, report.preserved_artifact_ids)

    def test_a_healthy_run_reports_no_repair(self) -> None:
        client = EnvelopeClient(all_envelopes())
        shell = run(make_executor(client, repairs=2))["shell"]
        self.assertFalse(shell.artifact_repair.attempted)


class TestUnchangedBehavior(unittest.TestCase):
    def test_structurally_unusable_collection_skips_the_verifier(self) -> None:
        calls = []

        class CountingVerifier:
            def verify(self, subtask, output, dependency_outputs, **kwargs):
                calls.append(subtask.id)
                return Verdict(True, 1.0, "")

        client = EnvelopeClient({
            "shell": envelope(entry(
                "unknown-artifact", path="unknown.tsx",
                content="export const unknownValue = 1;\n",
            )),
            **other_envelopes(),
        })
        run(make_executor(
            client, repairs=0, verify=True, verifier=CountingVerifier()
        ))
        self.assertNotIn("shell", calls)

    def test_cancellation_is_unchanged(self) -> None:
        token = CancelToken()

        class CancellingClient(EnvelopeClient):
            def complete(self, *, provider, model, messages, **kwargs):
                token.cancel()
                token.check()
                return super().complete(provider=provider, model=model,
                                        messages=messages, **kwargs)

        with self.assertRaises(CancelledError):
            make_executor(
                CancellingClient({"shell": broken_shell(), **other_envelopes()}),
                repairs=2, cancel=token,
            ).run(*_run_args())

    def test_the_unscoped_worker_is_unchanged(self) -> None:
        from dozen.models import Plan, SubTask
        client = EnvelopeClient()
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Write the app", instruction="Write app.py.", id="s1"),
        ])
        results = run(make_executor(client, repairs=2), plan)
        self.assertIsNone(results["s1"].artifact_collection)
        self.assertIsNone(results["s1"].artifact_repair)
        self.assertNotIn("TARGETED ARTIFACT REPAIR", client.worker_prompts[0])

    def test_the_semantic_verifier_is_still_distinct_from_repair(self) -> None:
        # A verified-but-artifact-incomplete first attempt still reuses the same
        # bounded attempt for repair, exactly as in Phase 4C/4D.
        class PassVerifier:
            def verify(self, subtask, output, dependency_outputs, **kwargs):
                return Verdict(True, 1.0, "")

        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        shell = run(make_executor(client, repairs=1, verify=True,
                                  verifier=PassVerifier()))["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertTrue(shell.artifact_collection.satisfied)

    def test_the_repair_verifier_receives_the_narrow_scope(self) -> None:
        captured = []

        class ScopeCapturingVerifier:
            def verify(self, subtask, output, dependency_outputs, contract=None,
                       scope=None):
                if subtask.id == "shell" and scope is not None:
                    captured.append(tuple(scope.output_artifact_ids))
                return Verdict(True, 1.0, "")

        client = EnvelopeClient({
            "shell": [broken_shell(), envelope(entry(LAYOUT))],
            **other_envelopes(),
        })
        run(make_executor(client, repairs=1, verify=True,
                          verifier=ScopeCapturingVerifier()))
        # Attempt 1: full package scope. Attempt 2: only the repair target.
        self.assertEqual(captured[0], (APP, ROUTER, LAYOUT))
        self.assertEqual(captured[1], (LAYOUT,))


def _run_args():
    from dozen.intent import resolve_contract
    from dozen.models import Task
    from ..artifact_decomposition.harness import react_plan
    return (
        Task(prompt="Build me a React dashboard.",
             contract=resolve_contract("Build me a React dashboard with tests.")),
        react_plan(), 0,
    )


if __name__ == "__main__":
    unittest.main()
