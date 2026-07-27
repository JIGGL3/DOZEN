# Dozen — a local, transparent model orchestrator

A from-scratch, readable implementation of an **orchestration control flow**
(inspired by Sakana AI's "Fugu"): a single entry point that behaves like one
model, but under the hood **decomposes a task into a dependency graph of
subtasks, routes each subtask to the best model in a swappable pool, verifies the
outputs, and synthesizes them into one final answer** — recursively calling
itself for complex subtasks.

> The actual LLM API calls are intentionally left as **commented placeholders**
> in `dozen/llm_client.py`. Until you wire them, everything runs in **mock mode**
> so you can watch the full orchestration logic with no keys.

## What this gives you

- **Expert task decomposition** — a Planner LLM breaks a task into a minimal,
  non-overlapping subtask **DAG** with explicit dependencies, per-subtask
  capability tags, success criteria, and a concrete **synthesis strategy**.
- **Capability-based routing** — a Router picks the best agent per subtask
  (deterministic heuristic by default, optional LLM router) using capability
  match, difficulty/tier alignment, cost, latency, and context-window fit.
- **Parallel DAG execution** — independent subtasks run concurrently; dependent
  ones wait for and receive their prerequisites' outputs.
- **Verification + repair loop** — a Verifier scores each output against its
  criteria; failures trigger targeted retries with feedback, and can **escalate**
  to the strongest agent on the final attempt.
- **Recursive orchestration** — subtasks marked `complex` are re-orchestrated
  (Dozen calling itself) up to a configurable depth.
- **Synthesis** — a Synthesizer stitches all outputs into one coherent answer,
  following the planner's strategy and resolving conflicts.

## Architecture

```
                 ┌──────────────┐
   task ───────► │   Planner    │  task -> subtask DAG + synthesis strategy
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐   per subtask, in dependency order (parallel):
                 │   Executor   │ ───► Router  -> pick best agent
                 │  (DAG sched) │ ───► Worker  -> run the subtask
                 │              │ ───► Verifier-> score / repair / escalate
                 │              │ ───► (recurse if `complex`)
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │ Synthesizer  │  stitch outputs -> final answer
                 └──────┬───────┘
                        ▼
                  final answer
```

| File | Role |
|------|------|
| `dozen/orchestrator.py` | Entry point; wires roles and runs plan→execute→synthesize (with recursion). |
| `dozen/planner.py` | Decomposes a task into a validated subtask DAG. |
| `dozen/router.py` | Selects the best agent per subtask. |
| `dozen/executor.py` | Schedules the DAG, runs workers, verify/repair/escalate loop. |
| `dozen/verifier.py` | Scores outputs against success criteria. |
| `dozen/synthesizer.py` | Combines subtask outputs into the final answer. |
| `dozen/agent_pool.py` | The swappable pool of models (`AgentSpec`s) + capability scores. |
| `dozen/llm_client.py` | **The single place to wire real model APIs.** |
| `dozen/prompts.py` | All role prompts (where most of the "expertise" lives). |
| `dozen/models.py` | Core dataclasses (Task, SubTask, Plan, results). |
| `dozen/config.py` | Tunable knobs (depth, parallelism, repair attempts, thresholds). |

## Quick start (mock mode, no keys)

```bash
python example.py
```

You'll see the planner decompose the task, the executor route/run/verify each
subtask, and the synthesizer stitch the result — all with synthesized responses.

## Going live (3 steps)

1. **Wire your APIs.** Open `dozen/llm_client.py` and implement `_call_provider`.
   Reference snippets for OpenAI/OpenAI-compatible (incl. Sakana & local
   Ollama/vLLM), Anthropic, and Google Gemini are included as comments.
2. **Set real models in the pool.** Edit `default_pool()` in
   `dozen/agent_pool.py`: replace each `model="REPLACE_WITH_MODEL_ID"` and tune
   the `provider`, `strengths`, `tier`, `cost_per_1k_tokens`, etc. Add/remove
   agents freely — routing adapts automatically.
3. **Disable mock mode.** In `example.py`, use `LLMClient(mock=False)`.

```python
from dozen import Orchestrator, Task, LLMClient
from dozen.agent_pool import default_pool

orch = Orchestrator(client=LLMClient(mock=False), pool=default_pool())
result = orch.run("Write and test a Python function that parses ISO timestamps.")
print(result.final_answer)
```

## Configuration

See `dozen/config.py`:

- `max_depth` — recursion depth for `complex` subtasks.
- `max_parallelism` — concurrent subtasks.
- `max_repair_attempts` — retries after a failed verification.
- `verify_outputs` / `pass_threshold` — verification gate.
- `escalate_on_failure` — final retry on the strongest agent.
- `use_llm_router` — LLM-based routing vs. the deterministic heuristic.
- `planner_agent` / `router_agent` / `verifier_agent` / `synthesizer_agent` —
  pin specific pool agents to roles (otherwise auto-selected).

## Notes / honest limitations

This reproduces the **orchestration architecture**, not Sakana's *trained*
coordinator. Their quality comes from coordinator models optimized via
evolution/RL (the TRINITY and Conductor papers) plus an undisclosed technical
report. Here the "intelligence" lives in the prompts + control flow and in
whatever frontier models you put in the pool. It's a strong, extensible base you
can later improve by fine-tuning your own planner/router on traces this system
produces.

## Run concurrency — single active workflow (current-stage invariant)

DOZEN currently supports **one active workflow per server process**. The
production composition shares mutable execution state across a run — a
process-global orchestrator, one `BrowserManager` pool, and one client
cancellation token — so overlapping runs would corrupt each other's state.

This is an **explicit, enforced invariant, not an accidental limitation**:

- Admission control lives in the server composition layer (`webllm/run_slot.py`
  → `RunSlot`), the single authoritative source of truth for whether a run is
  active, which run it is, and which cancellation token belongs to it.
- A second `POST /api/run` while a run is active is rejected deterministically
  with **HTTP 409**, naming the run that holds the slot. The rejected request
  starts no workflow and leaves no side effects.
- The active slot is released on every terminal path — success, failure,
  cancellation, unexpected exception, or a startup failure after the slot was
  acquired — via `finally`-style cleanup, so ownership can never leak.
- `POST /api/stop` cancels **only** the currently active run's token; it is
  safe when no run is active, when called repeatedly, and when racing a run's
  completion (a finished or replaced run is never cancelled).

Run-scoped concurrent execution (multiple simultaneous workflows with isolated
browser ownership and per-run state) remains a **later roadmap item**; it is
deliberately out of scope at this stage.

## Provider window geometry & prompt keyboard

**Provider Chromium windows** open as normal, resizable, **natively maximized**
windows sized by Windows to the visible desktop **work area** (so they respect
the taskbar, display scaling, and the monitor they open on). Playwright
viewport emulation is disabled in headed mode (`no_viewport=True`), so page
content — including each provider's chat composer — always tracks the real
window instead of being letterboxed into a fixed region. The single
authoritative geometry policy is `provider_window_launch_kwargs()` in
`webllm/browser_manager.py`; every launch path (first login, worker restart,
closed-tab recovery) flows through it. Headless runs keep a fixed 1280×900
viewport for deterministic scraping.

**Orchestrator prompt box**: **Enter submits** the prompt (same path as the
Run button, once, never while a run is active or the prompt is blank);
**Shift+Enter inserts a newline** for multiline editing. IME composition and
Ctrl/Alt/Meta+Enter are left untouched. The decision logic is the pure
function in `webllm/static/prompt_keys.js`.

## Request intent and the deliverable contract

DOZEN infers **what kind of deliverable your request owes you** and enforces it
end to end. "Build me a React dashboard" is an *implementation* request: it must
produce working code, and an architecture essay is not an acceptable substitute.

**Intents** (`dozen/intent.py` — the one authoritative resolver; pure,
deterministic, no extra model call): `IMPLEMENT`, `MODIFY`, `DEBUG`, `REVIEW`,
`ARCHITECTURE`, `EXPLAIN`, `RESEARCH`, `CONTENT`, plus `UNKNOWN` (permissive
defaults). Resolution is verb-anchored, not keyword-soup: nouns like "code" or
"architecture" never decide the mode.

**Explicit instructions always beat inferred defaults.** The stronger,
action-oriented requirement is never discarded:

| Request | Resolves to |
|---|---|
| "Build me a React dashboard." | IMPLEMENT — code required, prose invalid |
| "Design and implement the dashboard, including tests." | IMPLEMENT — architecture is supporting material; code + tests required |
| "Build a dashboard, but only the architecture — do not write code." | ARCHITECTURE — the explicit no-code constraint wins |
| "Explain why this fails and fix it." | DEBUG — the fix is required, not just the explanation |
| "Review this implementation, but do not change anything." | REVIEW — repository changes forbidden |

**The contract propagates through the whole workflow.** It is resolved once at
the orchestration boundary (`Orchestrator.run`, so library callers and the web
server behave identically), stored on `Task.contract`, inherited by recursive
child tasks, and consumed by the planner, workers, verifier and synthesizer.

**Two guards make substitution impossible to ship silently:**

- *Plan guard* — for a code-bearing contract, a plan made only of "describe /
  recommend / explain" work is rejected and the planner is re-asked once with
  the specific problem as feedback. (A missing test step is advisory, not fatal.)
- *Final guard* — a code-required run that returns only prose (or openly says
  "no application code is included") is **never reported as success**: the result
  carries an explicit error and the answer is prefixed with what is missing.

Current limitations (deliberate, later-phase work): the final guard checks the
*mode* of the answer, not the correctness, completeness or truncation of the
code within it; multi-file artifact decomposition and output-truncation recovery
are not implemented.
