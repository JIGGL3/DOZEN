"""Shared scripted clients for the output-rendering (Phase 6) tests."""

from __future__ import annotations

import json

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.config import OrchestratorConfig


def two_step_plan() -> dict:
    return {
        "analysis": "split into research and drafting",
        "direct_answer": None,
        "delegations": [
            {
                "id": "s1",
                "title": "Research",
                "instruction": "Collect the relevant facts about the topic.",
                "assigned_model": "alpha",
                "depends_on": [],
            },
            {
                "id": "s2",
                "title": "Draft",
                "instruction": "Write the final explanation using the facts.",
                "assigned_model": "alpha",
                "depends_on": ["s1"],
            },
        ],
        "synthesis_strategy": "Combine research and draft into one answer.",
    }


class ScriptedClient(LLMClient):
    """Planner/worker/synthesizer scripting keyed on the system prompt."""

    def __init__(self, plan: dict, worker_replies: dict[str, str],
                 synth_reply: str = "Combined final explanation.") -> None:
        super().__init__(mock=True)
        self.plan = plan
        self.worker_replies = worker_replies
        self.synth_reply = synth_reply
        self.synth_inputs: list[str] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[-1].content
        if "MANAGER" in system:
            text = json.dumps(self.plan)
        elif "SYNTHESIZER" in system or "combining consecutive sections" in system \
                or "condensing one part" in system:
            self.synth_inputs.append(user)
            text = self.synth_reply
        else:
            text = "unmatched worker reply with enough words to pass validation"
            for marker, reply in self.worker_replies.items():
                if marker in user:
                    text = reply
                    break
        return LLMResponse(text=text, provider=provider, model=model)


def make_orchestrator(client: LLMClient, **config_overrides) -> Orchestrator:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "writing": 0.9}, tier=4)
    config = OrchestratorConfig(
        max_parallelism=1, max_repair_attempts=1, verify_outputs=False,
        use_llm_router=False, verbose=False, **config_overrides,
    )
    return Orchestrator(client=client, pool=AgentPool([agent]), config=config)


def envelope(artifacts: dict[str, str], summary: str = "Did the work.",
             decisions: list[str] | None = None, confidence: float = 0.9) -> str:
    return json.dumps({
        "summary": summary,
        "key_decisions": decisions if decisions is not None else ["kept it simple"],
        "artifacts": artifacts,
        "confidence": confidence,
    })
