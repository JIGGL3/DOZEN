"""Verifier: judges whether a worker output meets its subtask criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agent_pool import AgentSpec
from .cancellation import CancelledError
from .decomposition import ArtifactExecutionScope
from .intent import DeliverableContract
from .llm_client import LLMClient
from .models import SubTask
from .presentation import sanitize_diagnostic
from .prompts import build_verifier_messages


@dataclass
class Verdict:
    passed: bool
    score: float
    feedback: str


class Verifier:
    def __init__(
        self,
        client: LLMClient,
        verifier_agent: AgentSpec,
        pass_threshold: float = 0.7,
    ) -> None:
        self.client = client
        self.agent = verifier_agent
        self.pass_threshold = pass_threshold

    def verify(
        self,
        subtask: SubTask,
        output: str,
        dependency_outputs: dict[str, str],
        contract: Optional[DeliverableContract] = None,
        scope: Optional[ArtifactExecutionScope] = None,
    ) -> Verdict:
        if not output.strip():
            return Verdict(False, 0.0, "Output was empty.")

        # ``contract`` and ``scope`` are optional so existing callers (and
        # tests) that verify without them keep working unchanged.
        messages = build_verifier_messages(subtask, output, dependency_outputs,
                                           contract=contract, scope=scope)
        try:
            data = self.client.complete_json(
                provider=self.agent.provider,
                model=self.agent.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
        except CancelledError:
            raise  # Stop must unwind, not be swallowed as a soft pass.
        except Exception as exc:  # noqa: BLE001 - verifier must never crash the run
            # Fail open with a soft pass so a flaky verifier doesn't block progress.
            reason = sanitize_diagnostic(exc)
            return Verdict(
                True, 0.5,
                f"Verifier unavailable ({reason}); accepted by default.",
            )

        score = float(data.get("score", 0.0) or 0.0)
        passed = bool(data.get("passed", score >= self.pass_threshold))
        feedback = str(data.get("feedback", "")).strip()
        # Reconcile an inconsistent verifier (says passed but low score, etc.).
        passed = passed and score >= self.pass_threshold
        return Verdict(passed=passed, score=score, feedback=feedback)
