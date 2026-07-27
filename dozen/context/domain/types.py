"""Branded identifier and unit types (no primitive obsession).

``NewType`` gives static-checker strength with zero runtime cost: passing a
``MessageId`` where a ``ConversationId`` is expected is a type error, while the
serialized form stays a plain JSON string/number.
"""

from __future__ import annotations

from typing import NewType

# --- identifiers (ULID strings — time-sortable, see utils/ulid.py) --------- #
ConversationId = NewType("ConversationId", str)
MessageId = NewType("MessageId", str)
SummaryId = NewType("SummaryId", str)
RunId = NewType("RunId", str)
WorkflowId = NewType("WorkflowId", str)
AgentId = NewType("AgentId", str)
ProviderId = NewType("ProviderId", str)
EventId = NewType("EventId", str)
# Stable reference to a ContextSection inside one build (window audit trail).
SectionRef = NewType("SectionRef", str)

# --- units ------------------------------------------------------------------ #
TokenCount = NewType("TokenCount", int)
# ISO-8601 UTC with millisecond precision, e.g. "2026-07-08T09:15:32.123Z".
Timestamp = NewType("Timestamp", str)
SchemaVersion = NewType("SchemaVersion", int)
