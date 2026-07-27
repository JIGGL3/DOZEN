"""The swappable agent pool.

Each ``AgentSpec`` describes one model the orchestrator can delegate to: which
provider/model string to call, what it is good at (capabilities + strengths),
and cost/latency hints the router can weigh.

This is the "swappable pool" idea: add, remove, or re-tag agents here
and the router adapts automatically. Wire the actual API calls in
``llm_client.py``; the ``provider``/``model`` fields below are passed straight
through to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Capability(str, Enum):
    """Common capability tags. The planner emits these per subtask and the
    router matches them against each agent's declared strengths. You are free to
    use arbitrary strings too; these are just convenient, well-known ones."""

    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    WRITING = "writing"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    PLANNING = "planning"
    VISION = "vision"
    LONG_CONTEXT = "long_context"
    TOOL_USE = "tool_use"
    FAST_CHEAP = "fast_cheap"


@dataclass
class AgentSpec:
    """One model in the pool."""

    name: str
    provider: str  # passed to LLMClient (e.g. "openai", "anthropic", "google", "local")
    model: str  # model id string for that provider

    # What this model is good at. Higher weight => stronger preference when a
    # subtask requires that capability. Map of capability -> 0..1 score.
    strengths: dict[str, float] = field(default_factory=dict)

    # Human-readable capability blurb shown to the Manager LLM for intelligent
    # routing, e.g. "Best for logic, coding, and architecture." If empty, one is
    # auto-derived from ``strengths`` (see ``profile_text``).
    profile: str = ""

    # Relative quality on hard problems (1-5). The router biases difficult
    # subtasks toward higher-tier agents.
    tier: int = 3

    # Hints used for tie-breaking / budgeting.
    cost_per_1k_tokens: float = 0.0
    avg_latency_s: float = 2.0
    max_context_tokens: int = 128_000

    # Generation defaults for this model.
    temperature: float = 0.2
    max_tokens: int = 4096

    # Set False to temporarily exclude an agent from routing (the "opt-out").
    enabled: bool = True

    def capability_score(self, capability: str) -> float:
        return self.strengths.get(capability, 0.0)

    def profile_text(self) -> str:
        """A human-readable capability blurb for the Manager's model catalog.

        Uses the explicit ``profile`` if set; otherwise synthesizes one from the
        top strengths so every agent always has a usable description.
        """
        if self.profile.strip():
            return self.profile.strip()
        top = sorted(self.strengths.items(), key=lambda kv: -kv[1])[:4]
        if not top:
            return "General-purpose assistant."
        caps = ", ".join(k.replace("_", " ") for k, _ in top)
        return f"Best for {caps}."


class AgentPool:
    """A registry of agents the orchestrator can route to."""

    def __init__(self, agents: Optional[list[AgentSpec]] = None) -> None:
        self._agents: dict[str, AgentSpec] = {}
        for a in agents or []:
            self.add(a)

    def add(self, agent: AgentSpec) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Duplicate agent name: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentSpec:
        return self._agents[name]

    def enabled_agents(self) -> list[AgentSpec]:
        return [a for a in self._agents.values() if a.enabled]

    def all_agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def disable(self, name: str) -> None:
        self._agents[name].enabled = False

    def enable(self, name: str) -> None:
        self._agents[name].enabled = True

    def describe_for_prompt(self) -> str:
        """Human-readable catalog injected into the router prompt."""
        lines = []
        for a in self.enabled_agents():
            strengths = ", ".join(
                f"{k}:{v:.1f}" for k, v in sorted(a.strengths.items(), key=lambda x: -x[1])
            ) or "general"
            lines.append(
                f"- {a.name} (tier {a.tier}, ~${a.cost_per_1k_tokens:.4f}/1k tok, "
                f"~{a.avg_latency_s:.1f}s) | strengths: {strengths}"
            )
        return "\n".join(lines)

    def describe_for_planner(self) -> str:
        """The "Available Models" block the Manager LLM uses to route subtasks.

        Each line is: the EXACT name the Manager must echo in ``assigned_model``,
        followed by a plain-language capability profile and a quality tier. This
        is what enables intelligent (not round-robin) delegation.

        Example output:
            - chatgpt: Best for logic, coding, and architecture. (tier 5/5)
            - gemini: Best for rapid research, creative writing, and summarization. (tier 4/5)
        """
        lines = []
        for a in self.enabled_agents():
            lines.append(
                f"- {a.name}: {a.profile_text()} (tier {a.tier}/5; "
                f"output cap {a.max_tokens} tokens)"
            )
        return "\n".join(lines)

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def resolve(self, name: str) -> Optional[AgentSpec]:
        """Best-effort match of a Manager-provided model name to a pool agent.

        Tolerant of case and surrounding punctuation/quotes so a slightly noisy
        ``assigned_model`` ("ChatGPT", "`gemini`") still resolves. Returns None
        if there is no enabled match.
        """
        if not name:
            return None
        needle = name.strip().strip("`'\"").lower()
        for a in self.enabled_agents():
            if a.name.lower() == needle:
                return a
        # Loose contains-match as a final fallback (e.g. "gpt" -> a "gpt-..." name).
        for a in self.enabled_agents():
            if needle and (needle in a.name.lower() or a.name.lower() in needle):
                return a
        return None

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: object) -> bool:
        return name in self._agents


def default_pool() -> AgentPool:
    """A reasonable starter pool.

    NOTE: ``provider``/``model`` strings are placeholders. Update them to match
    whatever you wire up in ``LLMClient._call_provider``. Capability scores are
    editable knobs that directly shape routing decisions.
    """
    return AgentPool(
        [
            AgentSpec(
                name="deep-reasoner",
                provider="anthropic",          # <-- adjust to your wiring
                model="REPLACE_WITH_MODEL_ID",  # e.g. a strong reasoning model
                tier=5,
                strengths={
                    Capability.REASONING: 0.95,
                    Capability.MATH: 0.9,
                    Capability.PLANNING: 0.9,
                    Capability.CODING: 0.85,
                    Capability.RESEARCH: 0.8,
                },
                cost_per_1k_tokens=0.015,
                avg_latency_s=6.0,
                temperature=0.1,
            ),
            AgentSpec(
                name="code-specialist",
                provider="openai",
                model="REPLACE_WITH_MODEL_ID",
                tier=5,
                strengths={
                    Capability.CODING: 0.97,
                    Capability.REASONING: 0.85,
                    Capability.TOOL_USE: 0.85,
                    Capability.MATH: 0.8,
                },
                cost_per_1k_tokens=0.01,
                avg_latency_s=5.0,
                temperature=0.0,
            ),
            AgentSpec(
                name="long-context-analyst",
                provider="google",
                model="REPLACE_WITH_MODEL_ID",
                tier=4,
                strengths={
                    Capability.LONG_CONTEXT: 0.95,
                    Capability.RESEARCH: 0.88,
                    Capability.SUMMARIZATION: 0.9,
                    Capability.EXTRACTION: 0.88,
                    Capability.REASONING: 0.8,
                },
                cost_per_1k_tokens=0.007,
                avg_latency_s=4.0,
                max_context_tokens=1_000_000,
            ),
            AgentSpec(
                name="fast-generalist",
                provider="openai",
                model="REPLACE_WITH_MODEL_ID",
                tier=3,
                strengths={
                    Capability.WRITING: 0.85,
                    Capability.SUMMARIZATION: 0.85,
                    Capability.FAST_CHEAP: 0.95,
                    Capability.REASONING: 0.7,
                    Capability.EXTRACTION: 0.8,
                },
                cost_per_1k_tokens=0.0008,
                avg_latency_s=1.5,
                temperature=0.3,
            ),
        ]
    )
