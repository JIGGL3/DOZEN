"""Router: selects the best agent in the pool for a given subtask.

Two complementary strategies:

* A fast, deterministic **capability score** (default) — no extra LLM call. It
  matches the subtask's required capabilities against each agent's strengths and
  biases by difficulty/tier, cost and latency.
* An optional **LLM router** for nuanced cases, which reads the agent catalog and
  picks. If it returns an unknown agent, we fall back to the heuristic.
"""

from __future__ import annotations

from .agent_pool import AgentPool, AgentSpec
from .cancellation import CancelledError
from .llm_client import LLMClient
from .models import SubTask
from .prompts import build_router_messages


class RouterError(RuntimeError):
    pass


class Router:
    def __init__(
        self,
        client: LLMClient,
        pool: AgentPool,
        router_agent: AgentSpec | None = None,
        use_llm_router: bool = False,
    ) -> None:
        self.client = client
        self.pool = pool
        self.router_agent = router_agent
        self.use_llm_router = use_llm_router

    def route(self, subtask: SubTask, context_chars: int = 0) -> AgentSpec:
        candidates = self.pool.enabled_agents()
        if not candidates:
            raise RouterError("No enabled agents available to route to.")

        if self.use_llm_router and self.router_agent is not None:
            chosen = self._llm_route(subtask, context_chars)
            if chosen is not None:
                return chosen

        return self._heuristic_route(subtask, context_chars, candidates)

    # ------------------------------------------------------------------ #
    def _heuristic_route(
        self, subtask: SubTask, context_chars: int, candidates: list[AgentSpec]
    ) -> AgentSpec:
        # Roughly 4 chars per token for the context-fit check.
        approx_tokens = context_chars // 4

        best: tuple[float, AgentSpec] | None = None
        for agent in candidates:
            score = self._score(agent, subtask, approx_tokens)
            if best is None or score > best[0]:
                best = (score, agent)
        assert best is not None
        return best[1]

    @staticmethod
    def _score(agent: AgentSpec, subtask: SubTask, approx_tokens: int) -> float:
        caps = subtask.required_capabilities or ["reasoning"]

        # 1) Capability match (the dominant term).
        cap_score = sum(agent.capability_score(c) for c in caps) / len(caps)

        # 2) Difficulty/tier alignment: hard subtasks favor higher tiers.
        #    difficulty 1..5 -> weight 0..1; tier 1..5 -> 0..1.
        diff_w = (subtask.difficulty - 1) / 4.0
        tier_w = (agent.tier - 1) / 4.0
        tier_score = diff_w * tier_w + (1 - diff_w) * (1 - abs(tier_w - 0.5))

        # 3) Cost/latency penalty (small; only matters as a tie-breaker).
        cost_penalty = min(agent.cost_per_1k_tokens * 5.0, 0.3)
        latency_penalty = min(agent.avg_latency_s / 100.0, 0.1)

        # 4) Hard constraint: context must fit. Heavily penalize if it doesn't.
        fit_penalty = 0.0
        if approx_tokens > agent.max_context_tokens:
            fit_penalty = 5.0

        return (
            cap_score * 1.0
            + tier_score * 0.35
            - cost_penalty
            - latency_penalty
            - fit_penalty
        )

    # ------------------------------------------------------------------ #
    def _llm_route(self, subtask: SubTask, context_chars: int) -> AgentSpec | None:
        assert self.router_agent is not None
        messages = build_router_messages(
            subtask, self.pool.describe_for_prompt(), context_chars
        )
        try:
            data = self.client.complete_json(
                provider=self.router_agent.provider,
                model=self.router_agent.model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
        except CancelledError:
            raise
        except Exception:
            return None

        name = str(data.get("agent", "")).strip()
        if name in self.pool and self.pool.get(name).enabled:
            return self.pool.get(name)
        return None
