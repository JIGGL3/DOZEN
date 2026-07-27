"""ConversationSession + ConversationRegistry (Phase 1.3).

A session is the live record of ONE workflow run against one conversation:
which run, which workflow, which providers/agents have been touched, when it
was last active. Phase 1.4's context injection will read the active session
to know *what* it is building context for — this module is that foundation.

The registry is the thread-safe owner of every open session and the index
from conversation id -> active sessions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from ..domain.types import ConversationId, RunId, Timestamp, WorkflowId
from ..ports import Clock, IdGenerator
from .factory import SystemClock, UlidIdGenerator


@dataclass
class ConversationSession:
    """Mutable, in-memory only — sessions are not persisted (a restart ends
    every session by definition; conversations persist, sessions do not)."""

    session_id: str
    conversation_id: ConversationId
    run_id: Optional[RunId]
    workflow_id: Optional[WorkflowId]
    created_at: Timestamp
    last_activity: Timestamp
    provider_history: list[str] = field(default_factory=list)
    current_provider: Optional[str] = None
    current_agent: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def touch(self, clock: Clock) -> None:
        self.last_activity = clock.now()

    def record_provider(self, provider: str, clock: Clock) -> None:
        self.current_provider = provider
        if not self.provider_history or self.provider_history[-1] != provider:
            self.provider_history.append(provider)
        self.touch(clock)

    def record_agent(self, agent: str, clock: Clock) -> None:
        self.current_agent = agent
        self.touch(clock)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "provider_history": list(self.provider_history),
            "current_provider": self.current_provider,
            "current_agent": self.current_agent,
            "metadata": dict(self.metadata),
        }


class ConversationRegistry:
    """Thread-safe registry of active sessions, indexed both ways."""

    def __init__(
        self,
        ids: Optional[IdGenerator] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._ids = ids or UlidIdGenerator()
        self._clock = clock or SystemClock()
        self._lock = threading.RLock()
        self._sessions: dict[str, ConversationSession] = {}
        self._by_conversation: dict[str, set[str]] = {}

    def start_session(
        self,
        conversation_id: ConversationId,
        run_id: Optional[RunId] = None,
        workflow_id: Optional[WorkflowId] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> ConversationSession:
        now = self._clock.now()
        session = ConversationSession(
            session_id=self._ids.new_id(),
            conversation_id=conversation_id,
            run_id=run_id,
            workflow_id=workflow_id,
            created_at=now,
            last_activity=now,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._sessions[session.session_id] = session
            self._by_conversation.setdefault(conversation_id, set()).add(session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[ConversationSession]:
        """Remove and return the session (None if unknown — idempotent)."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                bucket = self._by_conversation.get(session.conversation_id)
                if bucket is not None:
                    bucket.discard(session_id)
                    if not bucket:
                        del self._by_conversation[session.conversation_id]
        return session

    def sessions_for(self, conversation_id: ConversationId) -> list[ConversationSession]:
        with self._lock:
            ids = self._by_conversation.get(conversation_id, set())
            return [self._sessions[sid] for sid in sorted(ids) if sid in self._sessions]

    def active_sessions(self) -> list[ConversationSession]:
        with self._lock:
            return [self._sessions[sid] for sid in sorted(self._sessions)]

    def end_sessions_for(self, conversation_id: ConversationId) -> list[ConversationSession]:
        """End every session of one conversation (used by delete/archive)."""
        with self._lock:
            ids = list(self._by_conversation.get(conversation_id, set()))
        return [ended for sid in ids if (ended := self.end_session(sid)) is not None]

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def touch(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.touch(self._clock)
