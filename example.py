"""Runnable example / smoke test for the Dozen orchestrator.

By default this runs in MOCK mode (no API keys, no real model calls) so you can
watch the full orchestration flow: planning -> DAG execution -> verification ->
synthesis. Once you wire real APIs in ``dozen/llm_client.py`` and set real model
ids in the pool, flip ``mock=False``.

Run:
    python example.py
"""

from __future__ import annotations

import json

from dozen import LLMClient, Orchestrator, Task
from dozen.agent_pool import default_pool
from dozen.config import OrchestratorConfig


def main() -> None:
    # 1) Build the LLM client.
    #    mock=True  -> synthesized responses, exercises the control flow only.
    #    mock=False -> calls LLMClient._call_provider (wire your APIs there).
    client = LLMClient(mock=True)

    # 2) Build the agent pool. Edit dozen/agent_pool.py:default_pool() to set real
    #    provider/model strings and capability scores for your models.
    pool = default_pool()

    # 3) Configure the orchestrator.
    config = OrchestratorConfig(
        max_depth=2,
        max_parallelism=4,
        max_repair_attempts=2,
        verify_outputs=True,
        use_llm_router=False,  # heuristic router by default
        verbose=True,
    )

    orchestrator = Orchestrator(client=client, pool=pool, config=config)

    # 4) Run a task. You can pass a plain string or a structured Task.
    task = Task(
        prompt=(
            "Design and outline a small CLI tool that converts CSV files to JSON, "
            "including the architecture, the core code, and a short usage guide."
        ),
        constraints=["Python 3.10+", "standard library only"],
        desired_output="A concise design doc with code and usage instructions.",
    )

    result = orchestrator.run(task)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.final_answer)

    print("\n" + "=" * 70)
    print("ORCHESTRATION TRACE")
    print("=" * 70)
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
