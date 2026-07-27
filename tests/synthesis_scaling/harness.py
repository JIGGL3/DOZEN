"""Shared fakes for the synthesis-scaling (Phase 6) tests. Fake clients only."""

from __future__ import annotations

from typing import Callable, Optional

from dozen import AgentSpec, LLMClient, LLMResponse
from dozen.models import Plan, SubTaskResult, Task, TaskStatus
from dozen.synthesizer import Synthesizer, _measure  # noqa: F401 (re-exported)
from dozen.prompts import build_synthesizer_messages  # noqa: F401 (re-exported)
from dozen.synthesis_scaling import SynthesisBudgetPolicy

STRATEGY = "Combine every section into one deliverable."


class RecordingClient(LLMClient):
    """Records every synthesis call (kind, input chars) and scripts replies.

    ``fail`` is an optional hook ``(kind, per_kind_index) -> Exception | None``
    used to inject provider failures/cancellation at precise points.
    """

    def __init__(
        self,
        final_reply: str = "FINAL SYNTHESIS ANSWER.",
        merge_reply: str = "MERGED PART.",
        chunk_reply: str = "CHUNK DIGEST.",
        fail: Optional[Callable[[str, int], Optional[Exception]]] = None,
    ) -> None:
        super().__init__(mock=True)
        self.final_reply = final_reply
        self.merge_reply = merge_reply
        self.chunk_reply = chunk_reply
        self.fail = fail
        self.calls: list[tuple[str, int]] = []
        self.inputs: list[tuple[str, str]] = []  # (kind, full user text)

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        size = sum(len(m.content) for m in messages)
        if "SYNTHESIZER" in system:
            kind = "final"
        elif "combining consecutive sections" in system:
            kind = "merge"
        elif "condensing one part" in system:
            kind = "chunk"
        else:
            kind = "other"
        index = sum(1 for k, _ in self.calls if k == kind)
        self.calls.append((kind, size))
        self.inputs.append((kind, messages[-1].content))
        if self.fail is not None:
            exc = self.fail(kind, index)
            if exc is not None:
                raise exc
        reply = {"final": self.final_reply, "merge": self.merge_reply,
                 "chunk": self.chunk_reply}.get(kind, "OTHER.")
        return LLMResponse(text=reply, provider=provider, model=model)

    def sizes(self, kind: Optional[str] = None) -> list[int]:
        return [s for k, s in self.calls if kind is None or k == kind]


def make_synthesizer(
    client: LLMClient,
    max_input_chars: int = 24000,
    policy: Optional[SynthesisBudgetPolicy] = None,
) -> Synthesizer:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"writing": 0.9}, tier=3)
    return Synthesizer(client, agent, max_input_chars=max_input_chars,
                       policy=policy)


def result(sid: str, title: str, output: str, *, error: str = "",
           feedback: str = "") -> SubTaskResult:
    return SubTaskResult(
        subtask_id=sid, title=title, status=TaskStatus.COMPLETED,
        output=output, error=error, verifier_feedback=feedback,
    )


def results_of(sizes: list[int], prefix: str = "S") -> list[SubTaskResult]:
    """Deterministic distinct outputs of the requested sizes, tail-marked."""
    out = []
    for i, size in enumerate(sizes):
        tail = f"TAIL-{i}-END"
        body = f"{prefix}{i} " + ("word " * max(0, (size - len(tail)) // 5))
        body = body[: max(0, size - len(tail))] + tail
        out.append(result(f"s{i}", f"{prefix}{i}", body))
    return out


def task_and_plan() -> tuple[Task, Plan]:
    return (
        Task(prompt="Assemble the combined deliverable."),
        Plan(analysis="a", subtasks=[], synthesis_strategy=STRATEGY),
    )


def measure_final_frame(contents: list[str], titles: list[str]) -> int:
    """Measured size of the REAL final-synthesis prompt for these sections."""
    task, plan = task_and_plan()
    sections = list(zip(titles, contents))
    return _measure(build_synthesizer_messages(task, plan.synthesis_strategy,
                                               sections))
