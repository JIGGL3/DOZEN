"""ReliabilityClientDecorator — the transparent observation seam (Phase 2.1.2).

Wraps any ``LLMClient``-shaped object and records one ExecutionAttempt per
``complete()`` call. Transparency guarantees, each enforced by tests:

* The wrapped call's RESULT OBJECT is returned untouched — identity, not copy.
* Exceptions re-raise unmodified (same object, same traceback); recording of
  the failure happens on the way through, never instead.
* Attribute reads AND writes forward to the wrapped client — the orchestrator
  does ``client.cancel_token = token`` at run start (orchestrator.py:97) and
  that must keep reaching the real client.
* ``complete_json`` (used by Planner and Verifier) is executed via the wrapped
  CLASS's implementation **bound to this decorator**, so its internal
  ``self.complete(...)`` calls flow back through the decorator and get
  recorded too. The logic executed is byte-for-byte the original — only the
  ``self`` binding differs.
* Recording itself is fail-open: an exception inside the recorder must never
  break a model call.

NO retries. NO interception. NO provider switching. Observability only.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Mapping, Optional, Sequence

from ..cancellation import CancelledError
from ..context.domain.types import ProviderId
from .clock import ReliabilityClock, SystemReliabilityClock
from .metadata import ProviderMetadata, WorkflowMetadata
from .recorder import InMemoryExecutionRecorder, default_recorder
from .types import AttemptStatus, ExecutionStage

# Stage inference (Phase 2.1.3): purely observational stack inspection.
# Maps the calling pipeline module to its ExecutionStage — no Planner/
# Executor/Verifier/Synthesizer code is modified or wrapped; we only LOOK at
# who is calling. Worker calls that carry non-empty repair feedback are the
# executor's repair attempts (frame-local read, still read-only).
_STAGE_BY_MODULE = {
    "planner.py": ExecutionStage.PLANNER,
    "executor.py": ExecutionStage.WORKER,
    "verifier.py": ExecutionStage.VERIFIER,
    "synthesizer.py": ExecutionStage.SYNTHESIZER,
}
_MAX_FRAMES = 25


def infer_execution_stage() -> ExecutionStage:
    """Which pipeline component is calling, judged by the nearest recognized
    dozen module on the stack. UNKNOWN when nothing matches (direct calls,
    tests, third-party callers)."""
    try:
        frame = sys._getframe(1)
    except Exception:
        return ExecutionStage.UNKNOWN
    depth = 0
    while frame is not None and depth < _MAX_FRAMES:
        filename = frame.f_code.co_filename.replace("\\", "/")
        for module, stage in _STAGE_BY_MODULE.items():
            if filename.endswith("/dozen/" + module):
                if stage is ExecutionStage.WORKER:
                    feedback = frame.f_locals.get("repair_feedback")
                    if isinstance(feedback, str) and feedback.strip():
                        return ExecutionStage.REPAIR
                return stage
        frame = frame.f_back
        depth += 1
    return ExecutionStage.UNKNOWN

# Attributes owned by the decorator itself; everything else forwards through.
_OWN_ATTRS = frozenset({"_wrapped", "_recorder", "_clock", "_context_provider"})

ContextProvider = Callable[[], Mapping[str, object]]


class ReliabilityClientDecorator:
    def __init__(
        self,
        wrapped: object,
        recorder: Optional[InMemoryExecutionRecorder] = None,
        clock: Optional[ReliabilityClock] = None,
        context_provider: Optional[ContextProvider] = None,
    ) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_recorder", recorder or default_recorder())
        object.__setattr__(self, "_clock", clock or SystemReliabilityClock())
        object.__setattr__(self, "_context_provider", context_provider)

    # ------------------------------------------------------------------ #
    # The recorded seam
    # ------------------------------------------------------------------ #
    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[object],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> object:
        attempt = None
        started = self._clock.monotonic()
        try:
            prompt_chars = sum(
                len(getattr(m, "content", "") or "") for m in messages
            )
            attempt = self._recorder.begin(
                ProviderMetadata(provider=ProviderId(provider), model=model),
                self._current_workflow(),
                stage=infer_execution_stage(),
                prompt_character_count=prompt_chars,
            )
        except Exception:
            pass  # recording must never block a model call

        try:
            result = self._wrapped.complete(
                provider=provider,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except CancelledError:
            self._finish_safely(
                attempt, AttemptStatus.CANCELLED, started,
                {"cancelled": True},
            )
            raise  # user intent: rethrow untouched
        except BaseException as exc:
            self._finish_safely(
                attempt, AttemptStatus.FAILED, started,
                {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise  # identical behavior: same exception, same traceback
        response_chars = len(getattr(result, "text", "") or "")
        if attempt is not None:
            attempt = attempt.with_response(response_chars)
        self._finish_safely(
            attempt, AttemptStatus.SUCCEEDED, started,
            {"response_chars": response_chars},
        )
        return result  # the EXACT object the wrapped client produced

    def complete_json(self, *args: Any, **kwargs: Any) -> object:
        """Run the wrapped class's own complete_json with ``self`` bound to
        the decorator, so its internal self.complete() calls are recorded.
        Same code path as before — only the binding changes."""
        method = type(self._wrapped).complete_json
        return method(self, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Full transparency for everything else
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str) -> Any:
        # Only called when the attribute is not on the decorator itself.
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _current_workflow(self) -> WorkflowMetadata:
        provider = self._context_provider
        if provider is None:
            return WorkflowMetadata()
        try:
            return WorkflowMetadata.from_mapping(provider() or {})
        except Exception:
            return WorkflowMetadata()  # a broken context source is not fatal

    def _finish_safely(
        self,
        attempt,
        status: AttemptStatus,
        started_monotonic: float,
        result_metadata: dict[str, object],
    ) -> None:
        if attempt is None:
            return
        try:
            latency_ms = max(0.0, (self._clock.monotonic() - started_monotonic) * 1000.0)
            self._recorder.finish(
                attempt, status, latency_ms=latency_ms, result_metadata=result_metadata
            )
        except Exception:
            pass  # fail-open by contract


def wrap_client(
    client: object,
    recorder: Optional[InMemoryExecutionRecorder] = None,
    context_provider: Optional[ContextProvider] = None,
) -> ReliabilityClientDecorator:
    """The one-liner build_orchestrator uses to install passive recording."""
    return ReliabilityClientDecorator(
        client, recorder=recorder, context_provider=context_provider
    )
