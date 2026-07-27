"""ConversationService — the application-facing entry point (what the server
uses; nothing above this module ever sees a repository or even the manager's
full surface).

Design constraints from Phase 1.3:

* **Graceful always**: a persistence hiccup must never break a workflow run.
  Every method returns a Result or a safe default; nothing raises for
  environmental problems.
* **Stateless LLM**: this service only *records* history. It never reads
  history into prompts — that is Phase 1.4's ContextBuilder.
"""

from __future__ import annotations

from typing import Optional

from ..config.models import PersistenceConfig, StorageConfig
from ..domain.enums import ErrorCode
from ..domain.models import Message, StorageManifest
from ..domain.results import RepositoryResult, fail, ok
from ..domain.types import ConversationId, ProviderId, RunId
from ..adapters.filesystem import RepositoryFactory
from ..utils import is_ulid
from .manager import ConversationManager
from .session import ConversationSession


class ConversationService:
    def __init__(
        self,
        root_path: str,
        manager: Optional[ConversationManager] = None,
        fsync: bool = True,
    ) -> None:
        if manager is not None:
            self.manager = manager
        else:
            config = PersistenceConfig(
                storage=StorageConfig(root_path=root_path, fsync_appends=fsync)
            )
            provider = RepositoryFactory.create(config).unwrap()
            self.manager = ConversationManager(provider)

    # ------------------------------------------------------------------ #
    # Conversation resolution (the server's "optional conversation id")
    # ------------------------------------------------------------------ #
    def ensure_conversation(
        self, conversation_id: Optional[str], title_hint: str = ""
    ) -> tuple[ConversationId, bool]:
        """Resolve the id a run should record into.

        * A valid, existing id -> reuse it (``created=False``).
        * Missing / malformed / unknown id -> create a fresh conversation
          titled from the prompt (``created=True``). Malformed ids are treated
          as "start new", never as an error — the client may hold a stale or
          foreign id and must always end up with a working conversation.
        """
        candidate = (conversation_id or "").strip()
        if candidate and is_ulid(candidate):
            cid = ConversationId(candidate)
            if self.manager.conversation_exists(cid):
                return cid, False
        created = self.manager.create_conversation(
            title=self.manager.derive_title(title_hint)
        )
        if created.ok:
            return created.unwrap().conversation.id, True
        # Persistence down: fall back to an ephemeral id so the run itself
        # still proceeds; recording ops against it will fail gracefully.
        return ConversationId(self.manager.factory.ids.new_id()), True

    # ------------------------------------------------------------------ #
    # Recording (store-only; no history is ever injected)
    # ------------------------------------------------------------------ #
    def record_user_message(
        self,
        conversation_id: ConversationId,
        prompt: str,
        run_id: Optional[str] = None,
        context: str = "",
        desired_output: str = "",
    ) -> RepositoryResult[Message]:
        metadata: dict[str, object] = {}
        if context.strip():
            metadata["workflow.context"] = context
        if desired_output.strip():
            metadata["workflow.desired_output"] = desired_output
        return self.manager.append_user_message(
            conversation_id, prompt,
            run_id=RunId(run_id) if run_id else None,
            metadata=metadata or None,
        )

    def record_assistant_message(
        self,
        conversation_id: ConversationId,
        content: str,
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
        error: Optional[str] = None,
        cancelled: bool = False,
    ) -> RepositoryResult[Message]:
        metadata: dict[str, object] = {}
        if error:
            metadata["workflow.error"] = error
        if cancelled:
            metadata["workflow.cancelled"] = True
        body = (content or "").strip()
        if not body:
            # Nothing usable came back (hard failure / stop before synthesis):
            # keep the log truthful with an explicit marker message.
            body = f"[no answer produced{': ' + error if error else ''}]"
        return self.manager.append_assistant_message(
            conversation_id, body,
            run_id=RunId(run_id) if run_id else None,
            provider=ProviderId(provider) if provider else None,
            metadata=metadata or None,
        )

    # ------------------------------------------------------------------ #
    # Run session bracketing
    # ------------------------------------------------------------------ #
    def begin_run(
        self,
        conversation_id: ConversationId,
        run_id: str,
        workflow_id: Optional[str] = None,
    ) -> Optional[ConversationSession]:
        started = self.manager.start_session(
            conversation_id,
            run_id=RunId(run_id),
            workflow_id=workflow_id,  # type: ignore[arg-type]
        )
        return started.unwrap() if started.ok else None

    def finish_run(self, session: Optional[ConversationSession]) -> None:
        if session is not None:
            self.manager.end_session(session.session_id)

    # ------------------------------------------------------------------ #
    # Introspection (debug endpoint / future history UI)
    # ------------------------------------------------------------------ #
    def get_history(
        self, conversation_id: str, limit: int = 500
    ) -> RepositoryResult[dict[str, object]]:
        if not is_ulid((conversation_id or "").strip()):
            return fail(
                ErrorCode.VALIDATION_FAILED, "not a valid conversation id",
                conversation_id=conversation_id,
            )
        cid = ConversationId(conversation_id.strip())
        manifest = self.manager.get_conversation(cid)
        if not manifest.ok:
            return manifest
        messages = self.manager.read_messages(cid, limit=limit)
        if not messages.ok:
            return messages
        m: StorageManifest = manifest.unwrap()
        return ok({
            "conversation_id": m.conversation.id,
            "title": m.conversation.title,
            "status": m.conversation.status.value,
            "created_at": m.conversation.created_at,
            "updated_at": m.conversation.updated_at,
            "message_count": m.conversation.stats.message_count,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "created_at": msg.created_at,
                }
                for msg in messages.unwrap()
            ],
        })

    def close(self) -> None:
        self.manager.close()
