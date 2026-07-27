"""Shared Phase 4F fixtures: the exact failed live prompt and the scripted
LLM-orchestrator project it should have produced.

No real provider is ever contacted and nothing is written to disk.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from dozen import AgentPool, AgentSpec, LLMClient, LLMResponse, Orchestrator
from dozen.config import OrchestratorConfig

# The EXACT prompt from the failed live run — byte for byte.
FAILED_PROMPT = (
    "I want you to make me a orchestrator model than can orchestrate between "
    "different llm apis, divide tasks among them, verify outputs and "
    "synthesise back the results and give ti to user. (give code only)"
)

# ---------------------------------------------------------------------------
# The project the failed request should have produced. Contents are complete,
# balanced source bodies so Phase 4D integrity accepts them.
# ---------------------------------------------------------------------------
FILES: dict[str, str] = {
    "orchestrator.py": (
        "from providers.openai import OpenAIProvider\n"
        "from providers.anthropic import AnthropicProvider\n"
        "from verifier import Verifier\n"
        "from synthesis import Synthesizer\n\n\n"
        "class Orchestrator:\n"
        "    def __init__(self):\n"
        "        self.providers = [OpenAIProvider(), AnthropicProvider()]\n"
        "        self.verifier = Verifier()\n"
        "        self.synthesizer = Synthesizer()\n\n"
        "    def run(self, task):\n"
        "        parts = self.divide(task)\n"
        "        outputs = [p.complete(t) for p, t in zip(self.providers, parts)]\n"
        "        checked = [o for o in outputs if self.verifier.verify(o)]\n"
        "        return self.synthesizer.combine(checked)\n\n"
        "    def divide(self, task):\n"
        "        return [task for _ in self.providers]\n"
    ),
    "providers/openai.py": (
        "import json\n\n\n"
        "class OpenAIProvider:\n"
        "    name = 'openai'\n\n"
        "    def complete(self, prompt):\n"
        "        payload = {'model': 'gpt', 'prompt': prompt}\n"
        "        return json.dumps(payload)\n"
    ),
    "providers/anthropic.py": (
        "import json\n\n\n"
        "class AnthropicProvider:\n"
        "    name = 'anthropic'\n\n"
        "    def complete(self, prompt):\n"
        "        payload = {'model': 'claude', 'prompt': prompt}\n"
        "        return json.dumps(payload)\n"
    ),
    "verifier.py": (
        "class Verifier:\n"
        "    def verify(self, output):\n"
        "        return bool(output and output.strip())\n"
    ),
    "synthesis.py": (
        "class Synthesizer:\n"
        "    def combine(self, outputs):\n"
        "        return '\\n\\n'.join(outputs)\n"
    ),
    "config.json": '{\n  "max_parallel": 2,\n  "verify": true\n}\n',
}

# package id -> (subtask planner-id, owned artifact ids)
PACKAGES = {
    "core": ("s1", ("orchestrator.py", "config.json")),
    "providers": ("s2", ("providers/openai.py", "providers/anthropic.py")),
    "quality": ("s3", ("verifier.py", "synthesis.py")),
}

CORE_MARKER = "Write the core orchestrator module and configuration."
PROVIDERS_MARKER = "Write the provider client modules."
QUALITY_MARKER = "Write the verifier and synthesis modules."
MARKERS = {"s1": CORE_MARKER, "s2": PROVIDERS_MARKER, "s3": QUALITY_MARKER}


def artifact_block() -> dict:
    kinds = {"config.json": "config_file"}
    return {
        "title": "LLM API orchestrator",
        "artifacts": [
            {"path": path, "kind": kinds.get(path, "source_file")}
            for path in FILES
        ],
        "packages": [
            {
                "id": package_id,
                "title": package_id,
                "objective": f"Produce the {package_id} files.",
                "kind": "implementation",
                "owns": list(owned),
                "depends_on": (
                    [] if package_id == "core"
                    else ["core"] if package_id == "providers"
                    else ["providers"]
                ),
                "subtask_id": subtask_id,
                "completion": [f"the {package_id} files are complete and import cleanly"],
            }
            for package_id, (subtask_id, owned) in PACKAGES.items()
        ],
        "validations": [],
    }


def plan_json(*, providers_complex: bool = False,
              include_artifact_plan: bool = True) -> dict:
    delegations = [
        {"id": "s1", "title": "Core orchestrator",
         "instruction": CORE_MARKER, "assigned_model": "alpha"},
        {"id": "s2", "title": "Provider clients",
         "instruction": PROVIDERS_MARKER, "assigned_model": "alpha",
         "depends_on": ["s1"], "complex": providers_complex},
        {"id": "s3", "title": "Verification and synthesis",
         "instruction": QUALITY_MARKER, "assigned_model": "alpha",
         "depends_on": ["s2"]},
    ]
    data = {
        "analysis": "decompose the orchestrator project into bounded packages",
        "delegations": delegations,
        "synthesis_strategy": "assemble the files in manifest order",
    }
    if include_artifact_plan:
        data["artifact_plan"] = artifact_block()
    return data


def typed_entry(artifact_id: str, *, content: Optional[str] = None,
                **extra: Any) -> dict:
    item = {
        "artifact_id": artifact_id,
        "path": artifact_id,
        "content": FILES[artifact_id] if content is None else content,
        "language": "json" if artifact_id.endswith(".json") else "python",
        "complete": True,
    }
    item.update(extra)
    return item


def typed_envelope(*entries: Any, **extra: Any) -> str:
    payload = {
        "summary": "Implemented the assigned package.",
        "key_decisions": ["kept modules deterministic"],
        "artifacts": list(entries),
        "confidence": 0.93,
    }
    payload.update(extra)
    return json.dumps(payload)


def envelope_for(subtask_id: str) -> str:
    package_id = {sid: pid for pid, (sid, _o) in PACKAGES.items()}[subtask_id]
    _, owned = PACKAGES[package_id]
    return typed_envelope(*(typed_entry(aid) for aid in owned))


SIZE_REFUSAL_TEXT = (
    "The response is too large to provide in one message. I can split this "
    "into multiple messages if you would like me to continue."
)


class FinalizationClient(LLMClient):
    """Scripted planner/worker/synthesizer for the failed-prompt scenarios.

    ``worker_replies`` maps a subtask planner id to one reply string or a list
    of replies (consumed per attempt, last one repeating). Unmapped subtasks
    return the compliant typed envelope for their package.
    """

    def __init__(self, worker_replies: Optional[dict] = None,
                 plan: Optional[dict] = None,
                 child_plan: Optional[dict] = None,
                 synth_reply: object = "unused polished synthesis reply",
                 extra_worker_markers: Optional[dict] = None) -> None:
        super().__init__(mock=True)
        self.plan = plan or plan_json()
        self.child_plan = child_plan
        self.worker_replies = dict(worker_replies or {})
        self.synth_reply = synth_reply
        self.synth_calls = 0
        self.planner_prompts: list[str] = []
        self.worker_prompts: list[str] = []
        self._attempts: dict[str, int] = {}
        # Marker substring -> reply, checked BEFORE the standard s1/s2/s3
        # routing. Lets a recursive child's own delegation use an instruction
        # distinct from its parent's marker text (avoiding a collision where
        # both would otherwise route through the same MARKERS entry).
        self.extra_worker_markers = dict(extra_worker_markers or {})

    def _worker_reply(self, user: str) -> str:
        for marker, reply in self.extra_worker_markers.items():
            if marker in user:
                return reply
        for subtask_id, marker in MARKERS.items():
            if marker in user:
                scripted = self.worker_replies.get(subtask_id)
                if scripted is None:
                    return envelope_for(subtask_id)
                if isinstance(scripted, list):
                    index = min(self._attempts.get(subtask_id, 0),
                                len(scripted) - 1)
                    self._attempts[subtask_id] = index + 1
                    return scripted[index]
                return scripted
        return "generic worker output with enough words to pass validation"

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        system = messages[0].content
        user = messages[-1].content
        if "MANAGER" in system:
            self.planner_prompts.append(user)
            plan = self.plan
            if self.child_plan is not None and any(
                marker in user for marker in MARKERS.values()
            ):
                plan = self.child_plan
            return LLMResponse(text=json.dumps(plan), provider=provider,
                               model=model)
        if ("SYNTHESIZER" in system
                or "combining consecutive sections" in system
                or "condensing one part" in system):
            self.synth_calls += 1
            reply = self.synth_reply
            if isinstance(reply, Exception):
                raise reply
            return LLMResponse(text=str(reply), provider=provider, model=model)
        self.worker_prompts.append(user)
        return LLMResponse(text=self._worker_reply(user), provider=provider,
                           model=model)


def make_orchestrator(client: LLMClient, **overrides) -> Orchestrator:
    agent = AgentSpec(name="alpha", provider="openai", model="gpt",
                      strengths={"reasoning": 0.9, "coding": 0.9,
                                 "writing": 0.9}, tier=4)
    defaults = dict(
        max_parallelism=1, max_repair_attempts=1, verify_outputs=False,
        use_llm_router=False, verbose=False,
    )
    defaults.update(overrides)
    config = OrchestratorConfig(**defaults)
    return Orchestrator(client=client, pool=AgentPool([agent]), config=config)


def run_failed_prompt(client: LLMClient, **overrides):
    return make_orchestrator(client, **overrides).run(FAILED_PROMPT)


# --------------------------------------------------------------------------- #
# Prose-report fixtures (scenarios 8-10 and the ordered-section tests)
# --------------------------------------------------------------------------- #
REPORT_PROMPT = (
    "Research and produce a report comparing four database systems for our "
    "analytics workload."
)

REPORT_SECTIONS = {
    "r1": ("Storage engines", "Collect the storage engine facts."),
    "r2": ("Query performance", "Collect the query performance facts."),
    "r3": ("Operational cost", "Collect the operational cost facts."),
    "r4": ("Recommendation", "Write the final recommendation."),
}


def report_plan_json() -> dict:
    return {
        "analysis": "four report sections",
        "delegations": [
            {"id": sid, "title": title, "instruction": marker,
             "assigned_model": "alpha"}
            for sid, (title, marker) in REPORT_SECTIONS.items()
        ],
        "synthesis_strategy": "merge the sections in order",
    }


# Report markers resolve through the report plan's instructions, so the
# scripted client needs the report vocabulary instead of the project one.
REPORT_MARKERS = {sid: marker for sid, (_t, marker) in REPORT_SECTIONS.items()}


class ReportClient(FinalizationClient):
    def _worker_reply(self, user: str) -> str:
        for subtask_id, marker in REPORT_MARKERS.items():
            if marker in user:
                scripted = self.worker_replies.get(subtask_id)
                if scripted is None:
                    title = REPORT_SECTIONS[subtask_id][0]
                    return f"{title} findings stated plainly for the report."
                if isinstance(scripted, list):
                    index = min(self._attempts.get(subtask_id, 0),
                                len(scripted) - 1)
                    self._attempts[subtask_id] = index + 1
                    return scripted[index]
                return scripted
        return "generic worker output with enough words to pass validation"


def make_report_client(worker_replies: Optional[dict] = None,
                       synth_reply: object = "Polished unified report.") -> ReportClient:
    return ReportClient(worker_replies=worker_replies or {},
                        plan=report_plan_json(), synth_reply=synth_reply)
