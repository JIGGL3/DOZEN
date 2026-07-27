"""Immutable metadata helpers (Phase 2.1.2).

Small frozen value objects that group the fields an ExecutionAttempt is
assembled from. Pure data — construction and ``to_dict`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..context.domain.types import ConversationId, ProviderId, RunId, WorkflowId, Timestamp


@dataclass(frozen=True)
class ProviderMetadata:
    """Who is being called."""

    provider: ProviderId
    model: str = ""
    agent_name: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "agent_name": self.agent_name,
        }


@dataclass(frozen=True)
class TimingMetadata:
    """When it happened and how long it took."""

    started_at: Timestamp
    finished_at: Optional[Timestamp] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class WorkflowMetadata:
    """Which run/conversation/workflow/task the call belongs to.
    Every field is optional: recording degrades gracefully when the caller
    has no identity to offer (all-None == anonymous call)."""

    run_id: Optional[RunId] = None
    conversation_id: Optional[ConversationId] = None
    workflow_id: Optional[WorkflowId] = None
    task_id: Optional[str] = None
    subtask_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "WorkflowMetadata":
        """Mine known identity keys out of a mapping (e.g. a context-provider
        payload). Never mutates the input; unknown keys are ignored."""

        def opt(key: str) -> Optional[str]:
            value = data.get(key)
            return str(value) if isinstance(value, str) and value else None

        run = opt("run_id")
        conv = opt("conversation_id")
        wf = opt("workflow_id")
        return cls(
            run_id=RunId(run) if run else None,
            conversation_id=ConversationId(conv) if conv else None,
            workflow_id=WorkflowId(wf) if wf else None,
            task_id=opt("task_id"),
            subtask_id=opt("subtask_id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
        }
