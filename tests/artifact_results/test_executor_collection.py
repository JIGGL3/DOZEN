"""Phase 4C — executor integration: collection, the existing repair loop, and
everything that must NOT change (cancellation, verification, unscoped workers).
"""

from __future__ import annotations

import json
import unittest

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse
from dozen.artifact_results import CollectionErrorCode
from dozen.cancellation import CancelToken, CancelledError
from dozen.config import OrchestratorConfig
from dozen.executor import Executor
from dozen.intent import resolve_contract
from dozen.models import Plan, SubTask, Task, TaskStatus
from dozen.router import Router
from dozen.verifier import Verdict, Verifier

from ..artifact_decomposition.harness import react_plan
from .harness import CONTENT, entry, envelope, worker_envelope_for

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")

REAL_CODE = """```tsx
export default function App() {
  return <div>ready</div>;
}
```"""


class EnvelopeClient(LLMClient):
    """Returns a scripted worker envelope per subtask; records prompts."""

    def __init__(self, per_subtask=None, default=None) -> None:
        super().__init__(mock=True)
        self.per_subtask = per_subtask or {}
        self.default = default
        self.worker_prompts: list[str] = []
        self.worker_calls = 0

    def _payload_for(self, prompt: str):
        for marker, payloads in self.per_subtask.items():
            if f"Do the {marker} work." in prompt:
                if isinstance(payloads, list):
                    index = min(
                        sum(1 for p in self.worker_prompts[:-1]
                            if f"Do the {marker} work." in p),
                        len(payloads) - 1,
                    )
                    return payloads[index]
                return payloads
        return self.default

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        if "Worker" in system:
            self.worker_calls += 1
            self.worker_prompts.append(messages[1].content)
            payload = self._payload_for(messages[1].content)
            text = REAL_CODE if payload is None else json.dumps(payload)
            return LLMResponse(text=text, provider=provider, model=model)
        if "Verifier" in system:
            return LLMResponse(
                text=json.dumps({"passed": True, "score": 0.9, "feedback": ""}),
                provider=provider, model=model,
            )
        return LLMResponse(text=REAL_CODE, provider=provider, model=model)


def make_executor(client, *, repairs=0, verify=False, verifier=None,
                  recurse_fn=None, cancel=None) -> Executor:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9}, tier=4)
    pool = AgentPool([agent])
    config = OrchestratorConfig(max_parallelism=1, max_repair_attempts=repairs,
                                verify_outputs=verify, use_llm_router=False,
                                escalate_on_failure=False)
    return Executor(
        client=client, pool=pool,
        router=Router(client, pool, router_agent=agent, use_llm_router=False),
        verifier=verifier or Verifier(client, agent),
        config=config,
        recurse_fn=recurse_fn or (lambda *_a: REAL_CODE),
        log=lambda _m: None,
        cancel=cancel,
    )


def run(executor, plan=None) -> dict:
    results = executor.run(
        Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
        plan or react_plan(), depth=0,
    )
    return {result.subtask_id: result for result in results}


def all_envelopes() -> dict:
    return {sid: worker_envelope_for(sid)
            for sid in ("foundation", "shell", "shared-ui", "dashboard",
                        "tests", "validation")}


class TestScopedCollection(unittest.TestCase):
    def test_a_successful_scoped_worker_attaches_its_collection(self) -> None:
        results = run(make_executor(EnvelopeClient(all_envelopes())))
        shell = results["shell"]
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(shell.artifact_collection)
        self.assertTrue(shell.artifact_collection.satisfied)
        self.assertTrue(shell.artifacts_satisfied)
        self.assertEqual(len(shell.artifact_collection.accepted), 3)
        self.assertEqual(shell.artifact_collection.package_ids, ("application-shell",))

    def test_worker_text_remains_available(self) -> None:
        results = run(make_executor(EnvelopeClient(all_envelopes())))
        shell = results["shell"]
        self.assertIn("### src/app/App.tsx", shell.output)
        self.assertIn(CONTENT["src/app/App.tsx"], shell.output)

    def test_scoped_workers_receive_the_typed_contract(self) -> None:
        client = EnvelopeClient(all_envelopes())
        run(make_executor(client))
        shell_prompt = next(
            p for p in client.worker_prompts if "Do the shell work." in p
        )
        self.assertIn('"artifact_id"', shell_prompt)
        self.assertIn('"key_decisions"', shell_prompt)

    def test_missing_artifacts_are_marked_without_faking_success(self) -> None:
        envelopes = all_envelopes()
        envelopes["shell"] = envelope(entry("src/app/App.tsx"))
        results = run(make_executor(EnvelopeClient(envelopes)))
        shell = results["shell"]
        # Semantically a real answer...
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        # ...but its artifact obligations were NOT met, and that stays visible.
        self.assertFalse(shell.artifacts_satisfied)
        self.assertEqual(
            shell.artifact_collection.missing_required_artifact_ids,
            ("src/app/router.tsx", "src/components/layout/DashboardLayout.tsx"),
        )

    def test_out_of_scope_submissions_are_rejected_not_accepted(self) -> None:
        envelopes = all_envelopes()
        envelopes["dashboard"] = envelope(
            entry("src/features/dashboard/DashboardPage.tsx"),
            entry("package.json"),
        )
        results = run(make_executor(EnvelopeClient(envelopes)))
        collection = results["dashboard"].artifact_collection
        self.assertEqual(
            [r.code for r in collection.rejected], [CollectionErrorCode.OUT_OF_SCOPE]
        )
        self.assertEqual(len(collection.accepted), 1)


class TestExistingRepairLoop(unittest.TestCase):
    def test_missing_artifacts_enter_the_existing_repair_feedback(self) -> None:
        client = EnvelopeClient({
            "shell": [
                envelope(entry("src/app/App.tsx")),          # incomplete
                worker_envelope_for("shell"),                # corrected
            ],
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        results = run(make_executor(client, repairs=1))
        shell = results["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertTrue(shell.artifact_collection.satisfied)
        self.assertEqual(len(shell.artifact_collection.accepted), 3)
        retry_prompt = [p for p in client.worker_prompts if "Do the shell work." in p][1]
        self.assertIn("REVISION REQUIRED", retry_prompt)
        self.assertIn("src/app/router.tsx", retry_prompt)

    def test_a_second_incomplete_attempt_fails_deterministically(self) -> None:
        incomplete = envelope(entry("src/app/App.tsx"))
        client = EnvelopeClient({
            "shell": [incomplete, incomplete],
            **{sid: worker_envelope_for(sid)
               for sid in ("foundation", "shared-ui", "dashboard", "tests",
                           "validation")},
        })
        shell = run(make_executor(client, repairs=1))["shell"]
        self.assertEqual(shell.attempts, 2)
        self.assertFalse(shell.artifacts_satisfied)
        self.assertEqual(len(shell.artifact_collection.accepted), 1)
        # The genuine answer is still available; only the artifact contract failed.
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertTrue(shell.output.strip())

    def test_no_new_retry_loop_is_added(self) -> None:
        envelopes = all_envelopes()
        envelopes["shell"] = envelope(entry("src/app/App.tsx"))
        client = EnvelopeClient(envelopes)
        run(make_executor(client, repairs=0))
        shell_calls = [p for p in client.worker_prompts if "Do the shell work." in p]
        self.assertEqual(len(shell_calls), 1)  # bounded by max_repair_attempts only

    def test_a_satisfied_worker_is_never_retried_for_artifacts(self) -> None:
        client = EnvelopeClient(all_envelopes())
        run(make_executor(client, repairs=2))
        self.assertEqual(client.worker_calls, 6)

    def test_verifier_feedback_and_artifact_feedback_combine(self) -> None:
        class FailingVerifier:
            def verify(self, subtask, output, dependency_outputs, **kwargs):
                return Verdict(False, 0.2, "the shell needs a route table")

        envelopes = all_envelopes()
        envelopes["shell"] = envelope(entry("src/app/App.tsx"))
        client = EnvelopeClient(envelopes)
        run(make_executor(client, repairs=1, verify=True,
                          verifier=FailingVerifier()))
        retry_prompt = [p for p in client.worker_prompts if "Do the shell work." in p][1]
        self.assertIn("route table", retry_prompt)
        self.assertIn("src/app/router.tsx", retry_prompt)


class TestUnchangedBehavior(unittest.TestCase):
    def test_unscoped_workers_are_unchanged(self) -> None:
        client = EnvelopeClient()
        plan = Plan(analysis="a", synthesis_strategy="s", subtasks=[
            SubTask(title="Write the app", instruction="Write app.py.", id="s1"),
        ])
        results = client and run(make_executor(client), plan)
        self.assertIsNone(results["s1"].artifact_collection)
        self.assertTrue(results["s1"].artifacts_satisfied)
        self.assertEqual(results["s1"].status, TaskStatus.COMPLETED)
        self.assertNotIn('"artifact_id"', client.worker_prompts[0])

    def test_a_legacy_envelope_from_a_scoped_worker_does_not_crash(self) -> None:
        results = run(make_executor(EnvelopeClient()))  # every worker: raw markdown
        shell = results["shell"]
        self.assertEqual(shell.status, TaskStatus.COMPLETED)
        self.assertFalse(shell.artifact_collection.engaged)
        self.assertFalse(shell.artifact_collection.satisfied)

    def test_verification_behavior_is_unchanged(self) -> None:
        captured: list = []

        class ScopeCapturingVerifier:
            def verify(self, subtask, output, dependency_outputs, contract=None,
                       scope=None):
                captured.append((contract, scope))
                return Verdict(True, 1.0, "")

        results = run(make_executor(EnvelopeClient(all_envelopes()), verify=True,
                                    verifier=ScopeCapturingVerifier()))
        self.assertEqual(len(captured), 6)
        self.assertTrue(all(contract is IMPLEMENT for contract, _ in captured))
        self.assertTrue(all(scope is not None for _, scope in captured))
        self.assertTrue(all(r.status == TaskStatus.COMPLETED for r in results.values()))

    def test_cancellation_behavior_is_unchanged(self) -> None:
        token = CancelToken()

        class CancellingClient(EnvelopeClient):
            def complete(self, *, provider, model, messages, **kwargs):
                token.cancel()
                token.check()
                return super().complete(provider=provider, model=model,
                                        messages=messages, **kwargs)

        with self.assertRaises(CancelledError):
            make_executor(CancellingClient(all_envelopes()), cancel=token).run(
                Task(prompt="Build me a React dashboard.", contract=IMPLEMENT),
                react_plan(), depth=0,
            )


if __name__ == "__main__":
    unittest.main()
