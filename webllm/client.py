"""WebAutomationLLMClient — a drop-in replacement for ``dozen.LLMClient``.

It speaks the exact same interface the orchestrator already depends on
(``complete`` and ``complete_json``), but instead of calling REST APIs it drives
logged-in browser sessions through :class:`~webllm.browser_manager.BrowserManager`.

Because it subclasses ``LLMClient``:
* ``complete_json`` (JSON extraction + corrective re-prompting) is inherited.
* The retry/backoff contract of ``complete`` is preserved (we re-implement the
  loop here so the *messages* match the web flow, but the semantics are identical).

The orchestrator never knows the difference — Planner, Router, Verifier,
Synthesizer and Workers all keep calling ``client.complete(...)``.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from dozen.cancellation import CancelledError
from dozen.llm_client import LLMClient, LLMError, LLMMessage, LLMResponse
from dozen.presentation import sanitize_diagnostic
from dozen.prompts import (
    SCOPED_WORKER_ARTIFACT_CONTRACT,
    WORKER_ARTIFACT_CONTRACT,
)
from dozen.validation import sanitize_prompt_for_web_ui

from .browser_manager import BrowserManager
from .browser_queue import BrowserQueueRejectedError
from .providers import GeminiSubmissionProviderError, ProviderError


def render_messages(messages: list[LLMMessage]) -> str:
    """Flatten a chat message list into a single prompt for a web textbox.

    Each ``complete`` call is stateless (the orchestrator resends the full
    context every time and we start a fresh chat), so we serialize the whole
    conversation into one prompt. System content is hoisted to the top as
    explicit instructions; any multi-turn content is labelled so the model can
    still follow it.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    convo = [m for m in messages if m.role != "system"]

    blocks: list[str] = []
    if system_parts:
        blocks.append(
            "## SYSTEM INSTRUCTIONS\n" + "\n\n".join(p.strip() for p in system_parts)
        )

    if len(convo) == 1 and convo[0].role == "user":
        blocks.append(convo[0].content.strip())
    else:
        for m in convo:
            label = "USER" if m.role == "user" else "ASSISTANT"
            blocks.append(f"## {label}\n{m.content.strip()}")

    return "\n\n".join(blocks).strip()


# Roles whose prompts are STRUCTURED (they require JSON schema, braces, and
# explicit "You are the …" framing to work). These are routed to capable models
# and must NOT be humanized/sanitized, or we'd destroy their schema.
# The SYNTHESIZER is here too: its prompt embeds raw worker outputs (code,
# YAML, JSON) that the web-UI sanitizer would shred, and the humanized path
# would re-attach the worker artifact contract whenever an output mentions it —
# making the synthesizer answer in artifact JSON instead of a final deliverable.
_STRUCTURED_ROLE_MARKERS = (
    "you are the manager",
    "you are the planner",
    "you are the router",
    "you are the verifier",
    "you are the synthesizer",
)


def _is_structured_role(messages: list[LLMMessage]) -> bool:
    system = " ".join(m.content for m in messages if m.role == "system").lower()
    return any(marker in system for marker in _STRUCTURED_ROLE_MARKERS)


def render_messages_for_web(messages: list[LLMMessage]) -> str:
    """Render a worker/synth call as a clean, human-like chat message.

    Drops the framework's system roleplay (e.g. "You are an expert Worker agent
    in a multi-agent system"), keeps only the conversational content, then runs
    it through :func:`sanitize_prompt_for_web_ui` so what we paste reads like a
    person typed it — the fix for Gemini's injection-style refusals.
    """
    convo = [m for m in messages if m.role != "system"] or list(messages)
    if len(convo) == 1:
        body = convo[0].content
    else:
        # Preserve a light USER/ASSISTANT structure for multi-turn repairs, but
        # without machine-y headers (sanitizer strips any that slip through).
        parts = []
        for m in convo:
            parts.append(m.content if m.role == "user" else f"Earlier reply:\n{m.content}")
        body = "\n\n".join(parts)

    # Worker prompts carry the artifact JSON contract. The sanitizer strips JSON
    # (it would otherwise look like injection), so detect a worker call and
    # re-attach the canonical contract AFTER sanitizing — guaranteeing the worker
    # still gets a clean, intact output contract.
    # The canonical worker contract is always the FINAL prompt block. Detect
    # that trusted position instead of a marker that user/dependency text can
    # spoof, then remove exactly one suffix before sanitization.
    contract = None
    if body.endswith(SCOPED_WORKER_ARTIFACT_CONTRACT):
        contract = SCOPED_WORKER_ARTIFACT_CONTRACT
    elif body.endswith(WORKER_ARTIFACT_CONTRACT):
        contract = WORKER_ARTIFACT_CONTRACT
    if contract is not None:
        body = body[:-len(contract)].rstrip()
        # Dependency/user text is untrusted and may quote either canonical
        # block. Remove those quoted copies so only the selected final contract
        # is reattached after sanitization.
        body = body.replace(SCOPED_WORKER_ARTIFACT_CONTRACT, "")
        body = body.replace(WORKER_ARTIFACT_CONTRACT, "")
    cleaned = sanitize_prompt_for_web_ui(body)
    if contract is not None:
        cleaned = f"{cleaned}\n\n{contract}"
    return cleaned


class WebAutomationLLMClient(LLMClient):
    """LLMClient backed by Playwright-driven web UIs."""

    def __init__(
        self,
        browser: BrowserManager,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ) -> None:
        # mock=False: we never use the mock path. The parent stores retry config.
        super().__init__(
            mock=False, max_retries=max_retries, retry_backoff_s=retry_backoff_s
        )
        self.browser = browser

    # ------------------------------------------------------------------ #
    # The single overridden seam. Same signature & return type as the base.
    # ------------------------------------------------------------------ #
    def measure_input_chars(self, messages: list[LLMMessage]) -> int:
        """Exact serialized textbox length, including known web wrappers."""
        prompt = (
            render_messages(messages)
            if _is_structured_role(messages)
            else render_messages_for_web(messages)
        )
        return len(prompt)

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
        # Structured roles (planner/router/verifier) keep their full JSON-bearing
        # prompt. Everything else (workers, synthesizer) is rewritten to look
        # like a normal human message so web UIs don't flag it as injection.
        if _is_structured_role(messages):
            prompt = render_messages(messages)
        else:
            prompt = render_messages_for_web(messages)
        start = time.time()
        last_err: Optional[Exception] = None
        # The browser thread polls this between waits to abort generation fast.
        should_cancel = lambda: self.cancel_token.cancelled

        for attempt in range(1, self.max_retries + 1):
            self.cancel_token.check()
            try:
                text = self.browser.send_prompt(provider, prompt, should_cancel)
                return LLMResponse(
                    text=text,
                    provider=provider,
                    model=model,
                    latency_s=round(time.time() - start, 3),
                )
            except CancelledError:
                # Stop pressed: do not retry, propagate so the run unwinds.
                raise
            except BrowserQueueRejectedError:
                # Bounded-queue backpressure is a deterministic admission result,
                # not a transient provider/UI failure. Preserve its typed reason
                # and fail fast instead of retrying and fabricating more load.
                raise
            except GeminiSubmissionProviderError:
                # Submission failures carry the exact typed category and whether
                # a physical send was attempted.  Retrying here could duplicate
                # an ambiguously acknowledged prompt, and wrapping would erase
                # the category from the public client path.
                raise
            except ProviderError as exc:
                # UI/selector/login issues: worth a retry (page may settle).
                last_err = exc
            except TimeoutError as exc:
                last_err = exc
            except Exception as exc:  # noqa: BLE001 - re-raised after retries
                last_err = exc

            if attempt < self.max_retries:
                # Linear backoff, matching the base client's behaviour, with a
                # little extra room for rate-limit cool-downs on the web UIs.
                time.sleep(self.retry_backoff_s * attempt)
                self.cancel_token.check()

        reason = sanitize_diagnostic(last_err)
        raise LLMError(
            f"Web automation call failed after {self.max_retries} attempts "
            f"(provider={provider}, model={model}): {reason}"
        )

    # ``complete_json`` is inherited unchanged from LLMClient: it calls
    # ``self.complete`` (this method) and reuses the JSON extraction / repair
    # loop, so JSON-emitting roles (planner, router, verifier) work as-is.
