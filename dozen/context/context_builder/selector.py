"""MessageSelector — which history fits the budget.

Phase 1.4 policy (exactly as specified, nothing smarter):

* Full conversation, chronological order preserved in the result.
* No semantic retrieval, no embeddings, no summarization.
* Over budget -> keep the NEWEST messages, drop the OLDEST first.
* The current prompt's cost is reserved up front and is never evictable.
"""

from __future__ import annotations

from typing import Sequence

from ..domain.models import Message
from .budget import TokenEstimator
from .formatter import ContextFormatter
from .models import SelectionResult


class MessageSelector:
    def __init__(self, formatter: ContextFormatter, estimator: TokenEstimator) -> None:
        self.formatter = formatter
        self.estimator = estimator

    def select(
        self,
        messages: Sequence[Message],
        current_prompt: str,
        budget_tokens: int,
    ) -> SelectionResult:
        prompt_tokens = self.estimator.estimate(current_prompt)
        # The current prompt always ships, even when it alone exceeds the
        # budget — history then simply gets nothing.
        available = max(0, budget_tokens - prompt_tokens)

        kept_reversed: list[Message] = []
        history_tokens = 0
        truncated = False
        for message in reversed(messages):  # newest first
            cost = self.estimator.estimate(self.formatter.block_cost_text(message))
            if history_tokens + cost > available:
                truncated = True
                break  # everything older is larger history — drop the rest
            kept_reversed.append(message)
            history_tokens += cost

        selected = list(reversed(kept_reversed))  # back to chronological
        return SelectionResult(
            selected=selected,
            truncated=truncated,
            dropped_count=len(messages) - len(selected),
            history_tokens=history_tokens,
            prompt_tokens=prompt_tokens,
        )
