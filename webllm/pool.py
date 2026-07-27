"""Build an agent pool + orchestrator wired to the web-automation client.

The default ``dozen.agent_pool.default_pool`` ships with placeholder model ids.
Here we build a pool dynamically from *only* the providers the user actually
selected and logged into, so the router can never pick an account that isn't
available.
"""

from __future__ import annotations

from typing import Optional

from dozen import AgentPool, AgentSpec, Capability, Orchestrator
from dozen.config import OrchestratorConfig

from .browser_manager import BrowserManager
from .client import WebAutomationLLMClient
from .providers import get_adapter

# Per-provider capability profiles + a default UI model label. The ``model``
# string is informational for web automation (we use whatever model the logged
# in account currently has selected), but it flows through to logging.
_PROVIDER_AGENTS = {
    "openai": dict(
        name="chatgpt",
        model="gpt (web)",
        tier=5,
        strengths={
            Capability.REASONING: 0.9,
            Capability.CODING: 0.92,
            Capability.WRITING: 0.88,
            Capability.TOOL_USE: 0.85,
            Capability.MATH: 0.85,
            Capability.RESEARCH: 0.8,
        },
        profile="Best for logic, step-by-step reasoning, coding, software "
        "architecture, math, and structured problem-solving.",
        temperature=0.2,
    ),
    "anthropic": dict(
        name="claude",
        model="claude (web)",
        tier=5,
        strengths={
            Capability.REASONING: 0.95,
            Capability.WRITING: 0.95,
            Capability.CODING: 0.9,
            Capability.LONG_CONTEXT: 0.9,
            Capability.RESEARCH: 0.85,
            Capability.SUMMARIZATION: 0.9,
        },
        profile="Best for nuanced reasoning, long-form & high-quality writing, "
        "careful analysis of large documents, and faithful summarization.",
        temperature=0.2,
    ),
    "google": dict(
        name="gemini",
        model="gemini (web)",
        tier=4,
        strengths={
            Capability.LONG_CONTEXT: 0.95,
            Capability.RESEARCH: 0.9,
            Capability.SUMMARIZATION: 0.88,
            Capability.EXTRACTION: 0.85,
            Capability.REASONING: 0.82,
            Capability.VISION: 0.85,
        },
        profile="Best for rapid research, creative writing, summarization, "
        "huge-context document analysis, data extraction, and multimodal/vision.",
        temperature=0.3,
    ),
    "copilot": dict(
        name="copilot", model="copilot (web)", tier=4,
        strengths={
            Capability.REASONING: 0.82, Capability.CODING: 0.85,
            Capability.RESEARCH: 0.85, Capability.WRITING: 0.8,
        },
        temperature=0.2,
    ),
    "grok": dict(
        name="grok", model="grok (web)", tier=4,
        strengths={
            Capability.REASONING: 0.83, Capability.RESEARCH: 0.85,
            Capability.WRITING: 0.8, Capability.CODING: 0.8,
        },
        temperature=0.3,
    ),
    "meta": dict(
        name="meta-ai", model="meta-ai (web)", tier=3,
        strengths={
            Capability.WRITING: 0.8, Capability.REASONING: 0.72,
            Capability.FAST_CHEAP: 0.85, Capability.VISION: 0.75,
        },
        temperature=0.3,
    ),
    "perplexity": dict(
        name="perplexity", model="perplexity (web)", tier=4,
        strengths={
            Capability.RESEARCH: 0.96, Capability.SUMMARIZATION: 0.9,
            Capability.EXTRACTION: 0.85, Capability.REASONING: 0.78,
        },
        temperature=0.2,
    ),
    "mistral": dict(
        name="le-chat", model="mistral (web)", tier=3,
        strengths={
            Capability.FAST_CHEAP: 0.92, Capability.CODING: 0.82,
            Capability.REASONING: 0.75, Capability.WRITING: 0.8,
        },
        temperature=0.3,
    ),
    "huggingface": dict(
        name="huggingchat", model="huggingchat (web)", tier=3,
        strengths={
            Capability.FAST_CHEAP: 0.9, Capability.CODING: 0.78,
            Capability.REASONING: 0.72, Capability.WRITING: 0.75,
        },
        temperature=0.3,
    ),
    "groq": dict(
        name="groq", model="groq (web)", tier=3,
        strengths={
            Capability.FAST_CHEAP: 0.98, Capability.SUMMARIZATION: 0.82,
            Capability.REASONING: 0.72, Capability.EXTRACTION: 0.8,
        },
        temperature=0.3,
    ),
    "deepseek": dict(
        name="deepseek", model="deepseek (web)", tier=4,
        strengths={
            Capability.REASONING: 0.9, Capability.CODING: 0.9,
            Capability.MATH: 0.88, Capability.FAST_CHEAP: 0.85,
        },
        temperature=0.2,
    ),
    "poe": dict(
        name="poe", model="poe (web)", tier=3,
        strengths={
            Capability.WRITING: 0.8, Capability.REASONING: 0.75,
            Capability.FAST_CHEAP: 0.8,
        },
        temperature=0.3,
    ),
    "pi": dict(
        name="pi", model="pi (web)", tier=2,
        strengths={
            Capability.WRITING: 0.82, Capability.SUMMARIZATION: 0.78,
            Capability.FAST_CHEAP: 0.85,
        },
        temperature=0.4,
    ),
}


def web_pool(provider_keys: list[str]) -> AgentPool:
    """One agent per logged-in provider."""
    agents: list[AgentSpec] = []
    for key in provider_keys:
        get_adapter(key)  # validates the provider is supported
        spec = _PROVIDER_AGENTS.get(key)
        if spec is None:
            continue
        agents.append(
            AgentSpec(
                name=spec["name"],
                provider=key,
                model=spec["model"],
                tier=spec["tier"],
                strengths=spec["strengths"],
                profile=spec.get("profile", ""),
                temperature=spec["temperature"],
                max_tokens=4096,
            )
        )
    if not agents:
        raise ValueError("No supported providers given to web_pool().")
    return AgentPool(agents)


def build_orchestrator(
    browser: BrowserManager,
    provider_keys: list[str],
    config: Optional[OrchestratorConfig] = None,
) -> Orchestrator:
    """Wire the web client + pool into a ready-to-run Orchestrator."""
    from dozen.reliability.client import wrap_client

    # Phase 2.1.2: passive execution recording. The decorator is transparent —
    # results, exceptions and attribute access (incl. cancel_token) pass
    # through untouched; every complete() call lands in the in-memory ring.
    client = wrap_client(WebAutomationLLMClient(browser))
    pool = web_pool(provider_keys)
    if config is None:
        config = OrchestratorConfig(
            # Web UIs are slow; keep parallelism modest. Calls serialize on the
            # browser thread anyway, so this mainly bounds queued work.
            max_parallelism=2,
            max_repair_attempts=1,
            # Verification doubles the number of (slow) web calls; opt-in.
            verify_outputs=False,
            use_llm_router=False,
        )
    return Orchestrator(client=client, pool=pool, config=config)
