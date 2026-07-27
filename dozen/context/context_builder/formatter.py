"""ContextFormatter — renders selected messages into the prompt block.

Contract (Phase 1.4, fixed): each message becomes

    User: <content>

    Assistant: <content>

Message contents are NEVER modified — no trimming, no escaping, no
rewriting. Only the role label and the blank-line separator are added.
"""

from __future__ import annotations

from ..domain.enums import MessageRole
from ..domain.models import Message

_SEPARATOR = "\n\n"

_ROLE_LABELS = {
    MessageRole.USER: "User",
    MessageRole.ASSISTANT: "Assistant",
    MessageRole.SYSTEM: "System",
    MessageRole.TOOL: "Tool",
    MessageRole.SUMMARY: "Summary",
    MessageRole.UNKNOWN: "Unknown",
}


class ContextFormatter:
    """Stateless; both the selector (for budget math) and the builder (for
    the final block) use the same instance so sizes always agree."""

    def label(self, role: MessageRole) -> str:
        return _ROLE_LABELS.get(role, "Unknown")

    def format_message(self, message: Message) -> str:
        return f"{self.label(message.role)}: {message.content}"

    def format(self, messages: list[Message]) -> str:
        return _SEPARATOR.join(self.format_message(m) for m in messages)

    def block_cost_text(self, message: Message) -> str:
        """The exact text a message contributes to the joined output
        (block + separator), so budget estimates match reality."""
        return self.format_message(message) + _SEPARATOR
