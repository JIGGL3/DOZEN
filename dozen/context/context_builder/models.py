"""Result models of the Context Builder layer (Phase 1.4).

Plain dataclasses — this layer has no persistence of its own and never
serializes; it produces an in-memory answer to one question: "given this
conversation and this new prompt, what context should the workflow see?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..domain.models import Message


@dataclass(frozen=True)
class SelectionResult:
    """What the MessageSelector kept, in chronological order."""

    selected: list[Message]
    truncated: bool
    dropped_count: int
    history_tokens: int          # estimated tokens of the kept history blocks
    prompt_tokens: int           # estimated tokens reserved for the current prompt


@dataclass
class BuiltContext:
    """The complete output of ``ContextBuilder.build_context``."""

    conversation_id: str
    formatted_history: str                 # "User: …\n\nAssistant: …" blocks; "" when none
    estimated_tokens: int                  # history + current prompt, estimated
    message_count: int                     # messages included in formatted_history
    truncated: bool                        # True when older messages were dropped
    selected_messages: list[Message] = field(default_factory=list)
    budget_remaining: int = 0              # max(0, budget - estimated_tokens)
    budget: int = 0                        # the budget this build ran under
    # Stateless-fallback bookkeeping: when context loading fails for ANY
    # reason, the workflow proceeds without history and these say why.
    fallback: bool = False
    fallback_reason: Optional[str] = None

    @property
    def has_history(self) -> bool:
        return bool(self.formatted_history)

    def summary_line(self) -> str:
        """One log-friendly line describing this build."""
        if self.fallback:
            return f"context fallback (stateless): {self.fallback_reason}"
        state = "truncated" if self.truncated else "complete"
        return (
            f"{self.message_count} past message(s), ~{self.estimated_tokens} tokens "
            f"({state}; {self.budget_remaining} of {self.budget} budget left)"
        )
