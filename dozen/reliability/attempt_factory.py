"""AttemptFactory — the single place ExecutionAttempt objects are minted
(Phase 2.1.2).

Guarantees: every attempt id comes from one monotonic ULID generator (ids
sort in creation order, process-wide) and every timestamp from the injected
clock. No business logic — construction only.
"""

from __future__ import annotations

from typing import Optional

from ..context.manager.factory import UlidIdGenerator
from ..context.ports import IdGenerator
from .clock import ReliabilityClock, SystemReliabilityClock
from .metadata import ProviderMetadata, WorkflowMetadata
from .models import ExecutionAttempt
from .types import AttemptId, AttemptStatus, ExecutionStage


class AttemptFactory:
    def __init__(
        self,
        ids: Optional[IdGenerator] = None,
        clock: Optional[ReliabilityClock] = None,
    ) -> None:
        self.ids = ids or UlidIdGenerator()
        self.clock = clock or SystemReliabilityClock()

    def create(
        self,
        provider_meta: ProviderMetadata,
        workflow: Optional[WorkflowMetadata] = None,
        attempt_number: int = 1,
        stage: ExecutionStage = ExecutionStage.UNKNOWN,
        prompt_character_count: int = 0,
        retry_number: int = 0,
        parent_attempt_id: Optional[AttemptId] = None,
    ) -> ExecutionAttempt:
        wf = workflow or WorkflowMetadata()
        return ExecutionAttempt(
            attempt_id=AttemptId(self.ids.new_id()),
            provider=provider_meta.provider,
            started_at=self.clock.now(),
            run_id=wf.run_id,
            conversation_id=wf.conversation_id,
            agent_name=provider_meta.agent_name,
            task_id=wf.task_id,
            subtask_id=wf.subtask_id,
            attempt_number=attempt_number,
            status=AttemptStatus.PENDING,
            result_metadata={"model": provider_meta.model} if provider_meta.model else {},
            extra={"workflow_id": wf.workflow_id} if wf.workflow_id else {},
            execution_stage=stage,
            model_name=provider_meta.model,
            prompt_character_count=max(0, prompt_character_count),
            retry_number=max(0, retry_number),
            parent_attempt_id=parent_attempt_id,
        )
