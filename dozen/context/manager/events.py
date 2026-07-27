"""Conversation event system (Phase 1.3).

``InProcessEventBus`` is the default implementation of the ``EventBus`` port:
synchronous, thread-safe fan-out inside this process. ``ConversationEvents``
is the typed publisher the manager layer uses — one method per lifecycle
moment, so call sites never hand-build ``Event`` models.

Event payloads carry identifiers and counters only — never message content.
Handlers are third-party code: a raising handler is contained and must never
break the persistence operation that triggered the event.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from ..domain.enums import EventType
from ..domain.models import Event
from ..domain.types import ConversationId, EventId, Timestamp
from ..ports import Clock, IdGenerator
from .factory import SystemClock, UlidIdGenerator

EventHandler = Callable[[Event], None]

_RECENT_LIMIT = 256


class InProcessEventBus:
    """Synchronous pub/sub implementing the ``EventBus`` port.

    * ``subscribe(handler)`` — every event.
    * ``subscribe(handler, event_type=...)`` — one event type.
    * Returned callable unsubscribes; calling it twice is harmless.
    * ``recent()`` — bounded ring of the latest events, for tests/debug UIs.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_token = 0
        # token -> (event_type or None, handler)
        self._handlers: dict[int, tuple[Optional[EventType], EventHandler]] = {}
        self._recent: deque[Event] = deque(maxlen=_RECENT_LIMIT)

    def publish(self, event: Event) -> None:
        with self._lock:
            self._recent.append(event)
            targets = [
                handler for wanted, handler in self._handlers.values()
                if wanted is None or wanted is event.type
            ]
        for handler in targets:
            try:
                handler(event)
            except Exception:
                # A misbehaving subscriber must never fail the publisher.
                pass

    def subscribe(
        self, handler: EventHandler, event_type: Optional[EventType] = None
    ) -> Callable[[], None]:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._handlers[token] = (event_type, handler)

        def unsubscribe() -> None:
            with self._lock:
                self._handlers.pop(token, None)

        return unsubscribe

    def recent(self, event_type: Optional[EventType] = None) -> list[Event]:
        with self._lock:
            events = list(self._recent)
        if event_type is None:
            return events
        return [e for e in events if e.type is event_type]


class ConversationEvents:
    """Typed event publisher for the conversation layer."""

    def __init__(
        self,
        bus: Optional[InProcessEventBus] = None,
        ids: Optional[IdGenerator] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.bus = bus or InProcessEventBus()
        self._ids = ids or UlidIdGenerator()
        self._clock = clock or SystemClock()

    def _emit(
        self,
        event_type: EventType,
        conversation_id: Optional[ConversationId] = None,
        **payload: object,
    ) -> Event:
        event = Event(
            id=EventId(self._ids.new_id()),
            type=event_type,
            created_at=Timestamp(self._clock.now()),
            conversation_id=conversation_id,
            payload=dict(payload),
        )
        self.bus.publish(event)
        return event

    # ------------------------- conversation lifecycle ------------------ #
    def conversation_created(self, conversation_id: ConversationId, title: str) -> Event:
        return self._emit(EventType.CONVERSATION_CREATED, conversation_id, title=title)

    def conversation_loaded(self, conversation_id: ConversationId, source: str) -> Event:
        return self._emit(EventType.CONVERSATION_LOADED, conversation_id, source=source)

    def conversation_renamed(
        self, conversation_id: ConversationId, old_title: str, new_title: str
    ) -> Event:
        return self._emit(
            EventType.CONVERSATION_RENAMED, conversation_id,
            old_title=old_title, new_title=new_title,
        )

    def conversation_archived(self, conversation_id: ConversationId) -> Event:
        return self._emit(EventType.CONVERSATION_ARCHIVED, conversation_id)

    def conversation_deleted(self, conversation_id: ConversationId) -> Event:
        return self._emit(EventType.CONVERSATION_DELETED, conversation_id)

    def conversation_updated(self, conversation_id: ConversationId, reason: str) -> Event:
        return self._emit(EventType.CONVERSATION_UPDATED, conversation_id, reason=reason)

    # ------------------------------ messages --------------------------- #
    def message_added(
        self, conversation_id: ConversationId, message_id: str, role: str
    ) -> Event:
        return self._emit(
            EventType.MESSAGE_APPENDED, conversation_id,
            message_id=message_id, role=role,
        )

    def message_updated(self, conversation_id: ConversationId, message_id: str) -> Event:
        return self._emit(EventType.MESSAGE_UPDATED, conversation_id, message_id=message_id)

    def summary_added(self, conversation_id: ConversationId, summary_id: str) -> Event:
        return self._emit(EventType.SUMMARY_ADDED, conversation_id, summary_id=summary_id)

    # ------------------------------ sessions --------------------------- #
    def session_started(
        self, conversation_id: ConversationId, session_id: str, run_id: str
    ) -> Event:
        return self._emit(
            EventType.SESSION_STARTED, conversation_id,
            session_id=session_id, run_id=run_id,
        )

    def session_ended(
        self, conversation_id: ConversationId, session_id: str, run_id: str
    ) -> Event:
        return self._emit(
            EventType.SESSION_ENDED, conversation_id,
            session_id=session_id, run_id=run_id,
        )

    # ------------------------------- cache ----------------------------- #
    def cache_hit(self, conversation_id: ConversationId, kind: str) -> Event:
        return self._emit(EventType.CACHE_HIT, conversation_id, kind=kind)

    def cache_miss(self, conversation_id: ConversationId, kind: str) -> Event:
        return self._emit(EventType.CACHE_MISS, conversation_id, kind=kind)
