"""ContextBuilder — conversation history in, workflow-ready context out.

    Conversation History -> Message Selection -> Formatting -> Final Context

Depends ONLY on the ConversationManager public API (never repositories) and
never raises for environmental problems: a missing conversation, a corrupt
history, or a broken disk all yield a stateless-fallback ``BuiltContext``
(empty history, ``fallback=True``) so the workflow always runs.
"""

from __future__ import annotations

from typing import Optional

from ..domain.types import ConversationId
from ..manager import ConversationManager
from ..utils import is_ulid
from .budget import DEFAULT_TOKEN_BUDGET, TokenBudgetEstimator, TokenEstimator
from .formatter import ContextFormatter
from .models import BuiltContext
from .selector import MessageSelector

_HISTORY_LOAD_LIMIT = 100_000  # effectively "the full conversation"


class ContextBuilder:
    def __init__(
        self,
        manager: ConversationManager,
        budget_tokens: int = DEFAULT_TOKEN_BUDGET,
        estimator: Optional[TokenEstimator] = None,
        formatter: Optional[ContextFormatter] = None,
        selector: Optional[MessageSelector] = None,
    ) -> None:
        self.manager = manager
        self.budget_tokens = budget_tokens
        self.estimator = estimator or TokenBudgetEstimator()
        self.formatter = formatter or ContextFormatter()
        self.selector = selector or MessageSelector(self.formatter, self.estimator)

    def build_context(
        self,
        conversation_id: Optional[str],
        current_prompt: str = "",
        budget_tokens: Optional[int] = None,
    ) -> BuiltContext:
        budget = self.budget_tokens if budget_tokens is None else budget_tokens
        candidate = (conversation_id or "").strip()
        if not candidate or not is_ulid(candidate):
            return self._fallback(candidate, current_prompt, budget, "invalid or missing conversation id")
        cid = ConversationId(candidate)

        try:
            loaded = self.manager.read_messages(cid, limit=_HISTORY_LOAD_LIMIT)
        except Exception as exc:  # noqa: BLE001 — context must never kill a run
            return self._fallback(candidate, current_prompt, budget, f"history load crashed: {exc}")
        if not loaded.ok:
            return self._fallback(
                candidate, current_prompt, budget,
                f"{loaded.error.code.value}: {loaded.error.message}",
            )

        selection = self.selector.select(loaded.unwrap(), current_prompt, budget)
        formatted = self.formatter.format(selection.selected)
        estimated = selection.history_tokens + selection.prompt_tokens
        return BuiltContext(
            conversation_id=candidate,
            formatted_history=formatted,
            estimated_tokens=estimated,
            message_count=len(selection.selected),
            truncated=selection.truncated,
            selected_messages=selection.selected,
            budget_remaining=max(0, budget - estimated),
            budget=budget,
        )

    def _fallback(
        self, conversation_id: str, current_prompt: str, budget: int, reason: str
    ) -> BuiltContext:
        """Stateless execution: no history, workflow proceeds untouched."""
        prompt_tokens = self.estimator.estimate(current_prompt)
        return BuiltContext(
            conversation_id=conversation_id,
            formatted_history="",
            estimated_tokens=prompt_tokens,
            message_count=0,
            truncated=False,
            selected_messages=[],
            budget_remaining=max(0, budget - prompt_tokens),
            budget=budget,
            fallback=True,
            fallback_reason=reason,
        )
