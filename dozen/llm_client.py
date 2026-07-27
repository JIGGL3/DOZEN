"""LLM client abstraction.

This is the ONE place you need to wire in real model APIs. Every model in the
agent pool is reached through a single ``LLMClient.complete()`` call, so the
orchestrator never has to know provider-specific details.

HOW TO USE
----------
1. Implement the provider call(s) inside ``_call_provider`` below. Reference
   implementations for OpenAI, Anthropic and Google are provided as comments.
2. Each ``AgentSpec`` in the pool carries a ``provider`` and a ``model`` string;
   they are passed straight through to ``_call_provider`` so you can branch on
   them.
3. Until you add real calls, the client runs in ``mock`` mode so you can
   dry-run and inspect the orchestration logic end-to-end.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .cancellation import NULL_TOKEN, CancelToken, CancelledError
from .presentation import sanitize_diagnostic


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str = ""
    model: str = ""
    raw: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Single integration point for all model providers.

    Parameters
    ----------
    mock:
        When True (default until you add real API calls), responses are
        synthesized locally so you can test the orchestration flow without keys.
    mock_handler:
        Optional callable ``(provider, model, messages, **kw) -> str`` to make
        the mock smarter in tests.
    """

    def __init__(
        self,
        mock: bool = True,
        mock_handler: Optional[Callable[..., str]] = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
    ) -> None:
        self.mock = mock
        self.mock_handler = mock_handler
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        # Cooperative cancellation. The orchestrator swaps in a live token at
        # the start of each run; until then this inert token is never tripped.
        self.cancel_token: CancelToken = NULL_TOKEN

    # ------------------------------------------------------------------ #
    # Public API used by the rest of the orchestrator
    # ------------------------------------------------------------------ #
    def measure_input_chars(self, messages: list[LLMMessage]) -> int:
        """Characters this client knows it will submit for one completion."""
        return sum(len(message.content) for message in messages)

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run one chat completion against a single model, with retries."""
        start = time.time()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            # Bail out immediately if the user pressed Stop.
            self.cancel_token.check()
            try:
                if self.mock:
                    text = self._mock_response(provider, model, messages, **kwargs)
                else:
                    text = self._call_provider(
                        provider=provider,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                return LLMResponse(
                    text=text,
                    provider=provider,
                    model=model,
                    latency_s=round(time.time() - start, 3),
                )
            except CancelledError:
                # Cancellation must never be retried or swallowed.
                raise
            except Exception as exc:  # noqa: BLE001 - we re-raise after retries
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
                    self.cancel_token.check()

        reason = sanitize_diagnostic(last_err)
        raise LLMError(
            f"LLM call failed after {self.max_retries} attempts "
            f"(provider={provider}, model={model}): {reason}"
        )

    def complete_json(
        self,
        *,
        provider: str,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Like ``complete`` but parses a JSON object out of the response.

        Tolerant of models that wrap JSON in prose or ```json fences. If the
        reply cannot be parsed as JSON, re-prompt (up to ``max_retries``) with a
        corrective instruction, since transport-level retries in ``complete`` do
        not cover a well-delivered-but-malformed payload.
        """
        attempt_messages = list(messages)
        last_err: Optional[Exception] = None

        for _ in range(self.max_retries):
            resp = self.complete(
                provider=provider,
                model=model,
                messages=attempt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            try:
                return _extract_json(resp.text)
            except LLMError as exc:
                last_err = exc
                # Show the model its bad reply and demand strict JSON next time.
                attempt_messages = list(messages) + [
                    LLMMessage("assistant", resp.text[:2000]),
                    LLMMessage(
                        "user",
                        "Your previous reply could not be parsed as JSON. Reply "
                        "with ONLY a single valid JSON object — no prose, no "
                        "markdown, no code fences.",
                    ),
                ]

        raise last_err or LLMError("complete_json failed to produce valid JSON.")

    # ------------------------------------------------------------------ #
    # >>> WIRE YOUR REAL PROVIDER CALLS IN HERE <<<
    # ------------------------------------------------------------------ #
    def _call_provider(
        self,
        *,
        provider: str,
        model: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Dispatch to a concrete provider and return the assistant text.

        Replace the body below with real calls. Branch on ``provider``.
        The reference snippets are intentionally commented out.
        """

        # ---- OpenAI (and OpenAI-compatible endpoints, incl. Sakana API) -----
        # if provider in ("openai", "openai_compatible", "sakana"):
        #     from openai import OpenAI
        #     client = OpenAI(
        #         api_key=os.environ["OPENAI_API_KEY"],
        #         # base_url="https://api.sakana.ai/v1",  # for Sakana / others
        #     )
        #     resp = client.chat.completions.create(
        #         model=model,
        #         messages=[{"role": m.role, "content": m.content} for m in messages],
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #     )
        #     return resp.choices[0].message.content or ""

        # ---- Anthropic ------------------------------------------------------
        # if provider == "anthropic":
        #     import anthropic
        #     client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        #     system = "\n".join(m.content for m in messages if m.role == "system")
        #     chat = [
        #         {"role": m.role, "content": m.content}
        #         for m in messages if m.role != "system"
        #     ]
        #     resp = client.messages.create(
        #         model=model,
        #         system=system or None,
        #         messages=chat,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #     )
        #     return "".join(block.text for block in resp.content if block.type == "text")

        # ---- Google Gemini --------------------------------------------------
        # if provider == "google":
        #     from google import genai
        #     client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        #     prompt = "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)
        #     resp = client.models.generate_content(model=model, contents=prompt)
        #     return resp.text or ""

        # ---- Local (Ollama / vLLM / LM Studio, OpenAI-compatible) -----------
        # if provider == "local":
        #     from openai import OpenAI
        #     client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        #     resp = client.chat.completions.create(
        #         model=model,
        #         messages=[{"role": m.role, "content": m.content} for m in messages],
        #         temperature=temperature,
        #     )
        #     return resp.choices[0].message.content or ""

        raise NotImplementedError(
            f"No real API wired for provider={provider!r}. "
            "Implement it in LLMClient._call_provider, or run with mock=True."
        )

    # ------------------------------------------------------------------ #
    # Mock mode: lets you exercise the orchestration flow with no keys
    # ------------------------------------------------------------------ #
    def _mock_response(
        self, provider: str, model: str, messages: list[LLMMessage], **kwargs: Any
    ) -> str:
        if self.mock_handler is not None:
            return self.mock_handler(provider, model, messages, **kwargs)

        system = " ".join(m.content for m in messages if m.role == "system").lower()
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        # The planner/manager asks for a JSON plan.
        if "you are the manager" in system or "you are the planner" in system:
            return json.dumps(
                {
                    "analysis": "[mock] Broke the task into research, then drafting.",
                    "direct_answer": None,
                    "delegations": [
                        {
                            "id": "s1",
                            "title": "Research key points",
                            "instruction": "Gather the core facts needed to answer the task.",
                            "assigned_model": "",
                            "model_selection_reasoning": "[mock] left to automatic routing.",
                            "required_capabilities": ["reasoning", "research"],
                            "depends_on": [],
                            "success_criteria": "Covers the main relevant points.",
                            "expected_output": "A bullet list of findings.",
                            "complex": False,
                            "difficulty": 2,
                        },
                        {
                            "id": "s2",
                            "title": "Draft the answer",
                            "instruction": "Write a complete first draft using the research.",
                            "assigned_model": "",
                            "model_selection_reasoning": "[mock] left to automatic routing.",
                            "required_capabilities": ["writing", "reasoning"],
                            "depends_on": ["s1"],
                            "success_criteria": "Directly answers the objective.",
                            "expected_output": "A structured draft.",
                            "complex": False,
                            "difficulty": 3,
                        },
                    ],
                    "synthesis_strategy": "Use the draft as the spine; fold in any "
                    "missing research points and resolve contradictions.",
                }
            )

        # The router asks for a model choice.
        if "you are the router" in system:
            return json.dumps({"agent": "__FIRST__", "reason": "[mock] best capability match."})

        # The verifier asks for a score.
        if "you are the verifier" in system:
            return json.dumps(
                {
                    "passed": True,
                    "score": 0.9,
                    "feedback": "[mock] Meets the success criteria.",
                }
            )

        # The synthesizer assembles a final answer.
        if "you are the synthesizer" in system:
            return (
                "[mock-synthesis] Final answer assembled from subtask outputs.\n\n"
                + user[-600:]
            )

        # Default worker behavior.
        return f"[mock-{provider}:{model}] {user[:300]}"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    """Remove a ```json ... ``` (or plain ```) fenced block's fences if present.

    Returns the fenced content when a fence exists, else the original text.
    """
    fence = re.search(r"```[ \t]*([a-zA-Z0-9_-]*)\r?\n?(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(2).strip()
    return text


def _repair_jsonish(text: str) -> str:
    """Best-effort cleanup of *almost*-JSON so ``json.loads`` can accept it.

    Handles the malformations LLMs commonly emit:
      * ``//`` line comments and ``/* ... */`` block comments,
      * trailing commas before ``}`` or ``]``,
      * smart/curly quotes used instead of straight quotes.
    String contents are preserved (comment stripping is quote-aware).
    """
    # Normalize curly quotes -> straight quotes.
    text = (
        text.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )

    out: list[str] = []
    in_str = False
    quote = ""
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            i += 1
            continue
        # Not in a string: look for comments to strip.
        if ch in '"\'':
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1

    cleaned = "".join(out)
    # Remove trailing commas:  {..., }  or  [..., ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first complete, brace-balanced ``{...}`` span in ``text``.

    String-aware (ignores braces inside quoted strings and escapes), so it does
    not get fooled by ``{`` / ``}`` that appear inside values. This is far more
    robust than a naive first-``{`` / last-``}`` slice, which can swallow trailing
    prose or stop short on nested objects.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            continue
        if ch in '"\'':
            in_str = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract a single JSON object from (possibly messy) model text.

    Strategy, cheapest first:
      1. straight parse,
      2. parse after stripping code fences,
      3. parse the first brace-balanced ``{...}`` span,
      4. parse that span after repairing comments / trailing commas / quotes.
    Raises :class:`LLMError` only if every strategy fails.
    """
    if not text or not text.strip():
        raise LLMError("Empty model output; no JSON to parse.")

    candidates: list[str] = []
    raw = text.strip()
    candidates.append(raw)

    unfenced = _strip_code_fences(raw)
    if unfenced != raw:
        candidates.append(unfenced)

    for base in (unfenced, raw):
        span = _first_balanced_object(base)
        if span:
            candidates.append(span)
            break

    # Try each candidate as-is, then repaired.
    last_err: Optional[Exception] = None
    seen: set[str] = set()
    for cand in candidates:
        for attempt in (cand, _repair_jsonish(cand)):
            if not attempt or attempt in seen:
                continue
            seen.add(attempt)
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_err = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            last_err = LLMError("Parsed JSON was not an object.")

    # The exception message travels into subtask errors, SSE terminal events
    # and logs, so it must NEVER carry the raw payload — only a bounded,
    # whitespace-collapsed head for diagnosis.
    raise LLMError(
        f"Could not parse a JSON object from model output: {last_err} "
        f"(payload {len(text)} chars)"
    )
