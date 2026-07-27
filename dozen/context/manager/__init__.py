"""Conversation Management layer (Phase 1.3).

The ONLY public interface for conversation operations. Components above this
package (server, workflow glue) talk to ``ConversationService`` /
``ConversationManager``; nothing else touches repositories directly.
"""

from __future__ import annotations

from .cache import CacheStatistics, ConversationCache
from .events import ConversationEvents, InProcessEventBus
from .factory import ConversationFactory, SystemClock, UlidIdGenerator, derive_title
from .manager import ConversationManager
from .message_store import MessageStore
from .metadata import ConversationMetadataManager
from .service import ConversationService
from .session import ConversationRegistry, ConversationSession

__all__ = [
    "CacheStatistics",
    "ConversationCache",
    "ConversationEvents",
    "ConversationFactory",
    "ConversationManager",
    "ConversationMetadataManager",
    "ConversationRegistry",
    "ConversationService",
    "ConversationSession",
    "InProcessEventBus",
    "MessageStore",
    "SystemClock",
    "UlidIdGenerator",
    "derive_title",
]
