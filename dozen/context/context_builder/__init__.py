"""Context Builder layer (Phase 1.4).

Turns stored conversation history into the context block the workflow sees:

    Conversation History -> Message Selection -> Formatting -> Final Context

Depends only on the ConversationManager public API. Failure of any kind
degrades to stateless execution — never to a failed workflow.
"""

from __future__ import annotations

from .budget import DEFAULT_TOKEN_BUDGET, TokenBudgetEstimator, TokenEstimator
from .builder import ContextBuilder
from .formatter import ContextFormatter
from .models import BuiltContext, SelectionResult
from .selector import MessageSelector

__all__ = [
    "BuiltContext",
    "ContextBuilder",
    "ContextFormatter",
    "DEFAULT_TOKEN_BUDGET",
    "MessageSelector",
    "SelectionResult",
    "TokenBudgetEstimator",
    "TokenEstimator",
]
