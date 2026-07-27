# Software Architecture Design Document
## Context Management System — Dozen AI Workflow Orchestrator

| | |
|---|---|
| **Document** | SADD-001: Context Management System |
| **Status** | Proposed — awaiting approval for Phase 1 |
| **Scope** | Architecture only. No implementation. |
| **Author** | Principal Architecture Review |
| **Date** | 2026-07-08 |

> **Grounding note (read first).** The brief describes a Node.js + TypeScript project. The
> workspace was inspected in full: **no Node/TS code exists**. The existing, functional
> system is the **Python** orchestrator (`dozen/` core library + `webllm/` web-automation
> layer + FastAPI server + static frontend). Per the brief's own rule — *inspect, don't
> assume* — Part I analyzes the real Python system. Part II specifies the Context
> Management System **language-agnostically** with TypeScript-flavored naming, so it can be
> implemented either as the foundation of the planned Node/TS orchestrator or adopted
> incrementally inside the current Python codebase. Nothing in the design depends on
> language-specific features; every contract is expressed as ports (interfaces) and data
> models.

---

# Part I — The Current System

## 1. Component Inventory (as inspected)

| Layer | Component | File | Role |
|---|---|---|---|
| Entry | FastAPI server | `webllm/server.py` | HTTP + SSE API; holds global `_State` (browser, orchestrator, cancel token) |
| Entry | Run channel | `webllm/progress.py` | In-memory SSE event queues per run (`RUNS`) |
| UI | Single-page app | `webllm/static/index.html` | Landing + console; renders live log and final answer |
| Core | `Orchestrator` | `dozen/orchestrator.py` | Pipeline: plan → execute DAG → verify → synthesize; final-answer guards |
| Core | `Planner` | `dozen/planner.py` | LLM → strict-JSON plan; DAG validation (cycle check); model-assignment resolution |
| Core | `Executor` | `dozen/executor.py` | ThreadPool DAG scheduler; route → work → validate → verify → repair → escalate; recursion for `complex` subtasks |
| Core | `Router` | `dozen/router.py` | Deterministic capability/tier scoring (LLM routing optional) |
| Core | `Verifier` | `dozen/verifier.py` | LLM pass/score/feedback gate |
| Core | `Synthesizer` | `dozen/synthesizer.py` | Hierarchical map-reduce stitching under a char budget |
| Core | `AgentPool` / `AgentSpec` | `dozen/agent_pool.py` | Swappable model registry: strengths, tier, cost/latency hints |
| Core | `LLMClient` | `dozen/llm_client.py` | Single completion seam; retries; tolerant JSON extraction/repair |
| Core | Models | `dozen/models.py` | `Task`, `SubTask`, `Plan`, `SubTaskResult`, `OrchestrationResult` |
| Core | Validation | `dozen/validation.py` | Refusal detection, plan-echo guard, prompt sanitizers, artifact parsing, `clip_text` |
| Core | Prompts | `dozen/prompts.py` | Role prompt builders (planner/router/worker/verifier/synthesizer) + artifact contract |
| Core | Config | `dozen/config.py` | `OrchestratorConfig` dataclass (depth, parallelism, char budgets, roles) |
| Core | Cancellation | `dozen/cancellation.py` | Cooperative `CancelToken` threaded through every call |
| Web | `WebAutomationLLMClient` | `webllm/client.py` | `LLMClient` subclass; renders messages for web UIs; sanitize/humanize worker prompts |
| Web | `BrowserManager` | `webllm/browser_manager.py` | One Playwright thread **per provider**; persistent profiles; job marshalling |
| Web | Provider adapters | `webllm/providers.py` | Selector-driven UI driving; DOM→Markdown response scraping; paste strategies; per-provider prompt caps |
| Web | Pool builder | `webllm/pool.py` | Builds `AgentPool` from the providers the user actually logged into |

## 2. Execution Path (traced, not assumed)

```
User Prompt
  │  POST /api/run {prompt}
  ▼
FastAPI server (server.py)
  │  creates CancelToken + RunChannel (SSE); spawns worker thread
  ▼
Orchestrator.run(Task)                       ← Task built fresh; NO history attached
  │  plan: Planner → LLMClient.complete_json → plan JSON → DAG validation
  ▼
Executor.run(task, plan)                     ← ThreadPoolExecutor, max_parallelism 4
  │  per subtask: Router (or planner assignment) → build_worker_messages(
  │      task.prompt + instruction + CLIPPED dependency outputs + artifact contract)
  ▼
WebAutomationLLMClient.complete              ← stateless: full context resent every call
  │  render (structured vs humanized+sanitized) → BrowserManager.send_prompt
  ▼
ProviderAdapter.send (per-provider thread)   ← fresh chat per call; paste; observe;
  │  scrape response DOM → Markdown             per-provider max_prompt_chars clip
  ▼
Executor: artifact parse → semantic validation → verifier → repair loop
  ▼
Synthesizer.synthesize                       ← hierarchical merge under char budget
  ▼
Orchestrator guards (artifact flatten, plan-echo) → OrchestrationResult
  ▼
SSE "result" event → frontend renders markdown
```

**The load-bearing observation:** every run is an island. `Task` carries `prompt`,
`context`, `desired_output` — whatever the caller passes *this time*. There is no
conversation identity, no message history, no persistence of results (`RUNS` is an
in-memory dict, lost on restart), and every notion of "how much fits" is a **character
count**, implemented three separate times (executor dep-clipping, synthesizer budget,
provider `max_prompt_chars`).

## 3. Current Strengths (keep these)

1. **Single completion seam.** Everything flows through `LLMClient.complete/complete_json`.
   One interface to intercept for context injection — the cleanest possible integration
   point for the new layer.
2. **Role-decomposed pipeline.** Planner/Router/Executor/Verifier/Synthesizer are separate
   classes with narrow contracts. The context layer can serve each role differently
   without touching their internals.
3. **Swappable `AgentPool`.** Capability/tier metadata already exists per model —
   `max_context_tokens` is already declared on `AgentSpec` (and currently unused!). The
   context layer gets its per-model window limits for free.
4. **Cooperative cancellation.** A single `CancelToken` threads through every blocking
   point. The context pipeline must (and can) adopt the same token.
5. **Defense-in-depth validation.** Refusal detection, plan-echo guards, artifact
   parsing, DOM→Markdown scraping. These protect whatever context we persist from being
   polluted by junk outputs.
6. **True per-provider parallelism** (post-refactor) with per-provider serialization —
   the concurrency model the context layer must be safe under.
7. **Structured progress events.** `on_event` sink → SSE. An event backbone already
   exists to extend toward context lifecycle events.

## 4. Current Weaknesses

1. **No conversation model at all.** The orchestrator is request/response. Nothing links
   run N to run N+1. "Persistent conversations" — a core project goal — has no seam to
   attach to.
2. **No durable storage.** `RUNS` (progress + results) is process memory; a restart wipes
   history. Browser profiles are the *only* persisted state in the entire system.
3. **Char-based budgeting, tripled.** `max_dep_output_chars` (executor),
   `max_synthesis_input_chars` (synthesizer), `max_prompt_chars` (provider adapter) are
   three independent truncation systems in units (characters) that don't correspond to
   any model's real constraint (tokens). They can disagree; none knows about the others.
4. **Lossy, unrecoverable trimming.** `clip_text` throws the middle away permanently.
   Nothing records *what* was cut, so no later stage can recover or summarize it.
5. **Context assembly welded into prompt builders.** `build_worker_messages` /
   `build_synthesizer_messages` decide inclusion, ordering, and truncation inline. There
   is no reusable "build me a context that fits" capability.
6. **Stateless provider calls by design.** `webllm` opens a fresh chat per call and
   resends everything. Correct for scrape reliability — but it makes token waste
   invisible and makes conversation continuity impossible without a context layer.
7. **No token estimation anywhere.** `AgentSpec.max_context_tokens` exists but nothing
   consumes it.
8. **Global mutable server state.** `_State` singleton; one active run (single global
   cancel token). Fine for local-first single-user; a wall for multi-agent/distributed.

## 5. Tight Coupling (specific)

| Coupling | Where | Why it hurts |
|---|---|---|
| Prompt builders ↔ truncation policy | `prompts.py` + call sites pre-clip | Changing "what fits" means editing every builder and call site |
| Executor ↔ dependency-context format | `build_worker_messages(dep_outputs)` dict of title→clipped text | No way to swap in summaries, memory, or retrieved context |
| Orchestrator ↔ Synthesizer construction | budget passed in constructor | Budget policy is frozen at wiring time, not per-request |
| Server ↔ run lifecycle | `_State.cancel`, `RUNS` global | Exactly one concurrent run; no run history |
| `Task` ↔ caller-supplied context | `Task.context: str` free text | Context is an untyped blob; nothing can reason about it |

## 6. Future Scalability Problems

1. **Multi-agent execution** needs *shared, versioned* conversation state — impossible
   against ad-hoc strings.
2. **Distributed execution** needs storage with atomic appends + optimistic concurrency —
   impossible against process memory.
3. **Provider failover mid-conversation** needs replayable, provider-neutral history —
   impossible when history only exists inside a provider's chat page.
4. **Long-running workflows** will overflow any window without summarization
   checkpoints; today overflow = silent middle-deletion.
5. **Intelligent prompt optimization** requires measuring tokens per section — today
   nothing measures anything.

## 7. Keep / Redesign / Debt-to-pay-now

**Unchanged (stable foundations):** `LLMClient` seam and retry/JSON machinery; role
pipeline (Planner/Router/Executor/Verifier/Synthesizer contracts); `AgentPool`/`AgentSpec`;
`BrowserManager` threading model; provider adapters + scraping; cancellation; validation
suite; SSE progress channel; the frontend.

**Redesigned / superseded:**
- The three char-budget systems → one token-aware `ContextWindowManager` (providers keep
  `max_prompt_chars` only as a final transport safety net).
- `Task.context: str` → structured `ContextRequest` / conversation reference.
- Inline dep-output embedding in prompt builders → `ContextBuilder` output rendered by
  builders.
- `RUNS` result amnesia → runs recorded as conversations via the `PersistenceLayer`.

**Technical debt to pay NOW (it compounds):**
1. Unify truncation into one policy engine (every future subsystem will otherwise grow
   its own fourth/fifth clipping system).
2. Introduce IDs + persistence for runs/messages (planner, validator, repair, analytics
   all need durable references; retrofitting IDs later touches everything).
3. Route all size decisions through token estimation (char heuristics baked any deeper
   become load-bearing lies).
4. Extract context assembly out of `prompts.py` (each new role prompt currently re-invents
   inclusion logic).

---
# Part II — The Context Management System

## 8. Design Goals & Non-Goals

**Goals.** Conversation + message lifecycle; deterministic context building under a token
budget; lossless-by-default trimming (evicted content becomes summaries, never vapor);
filesystem persistence today with a clean port to databases tomorrow; explicit seams for
memory, planner, validator, repair, multi-agent, and distributed execution; every
component testable in isolation via dependency inversion.

**Non-goals (this phase).** No vector search, no embeddings, no cloud sync
implementation, no database engine — only the *ports* they will plug into.

**Architectural style.** Hexagonal (ports & adapters) around a staged pipeline.
Domain logic (building, estimating, windowing, summarizing) never imports I/O; I/O
(filesystem, future DB, future network) implements ports. This is the single highest-
leverage decision: it is what makes "future database migration" a new adapter instead of
a rewrite.

---

## 9. Component Designs

Notation: method signatures are *design notation*, not code. `→` denotes the return
value; all mutating operations are asynchronous and cancellable.

### 9.1 ConversationManager — the façade

- **Purpose.** The single entry point every subsystem (workflow, planner, agents, UI)
  uses to interact with conversations. Nobody else touches storage or context assembly.
- **Responsibilities.** Conversation lifecycle (create/open/fork/archive/delete);
  appending messages through validation; requesting built context; emitting lifecycle
  events; enforcing per-conversation single-writer ordering.
- **Public methods.**
  - `createConversation(metadata?) → Conversation`
  - `getConversation(id) → Conversation`
  - `listConversations(filter?, page?) → ConversationSummary[]`
  - `appendMessage(conversationId, MessageDraft) → Message`
  - `buildContext(ContextRequest) → ContextResult`
  - `forkConversation(id, atMessageId?) → Conversation` — multi-agent branch point
  - `archiveConversation(id)` / `deleteConversation(id)`
  - `onEvent(listener)` — lifecycle event subscription
- **Dependencies.** `ConversationRepository` (port), `MessageStore`, `ContextPipeline`,
  `Clock/IdGenerator` utilities, `ContextConfig`.
- **Data flow.** UI/workflow → manager → (store, repository) for state; manager →
  pipeline for context; manager → event emitter for observers.
- **Why it exists.** Without a façade, every subsystem wires storage + pipeline itself —
  the exact coupling Part I documents. One narrow API also makes the later
  Python→TS (or local→distributed) boundary a network seam instead of a refactor.
- **Interactions.** *Everything* above the domain calls the manager; the manager is the
  only caller of the repository's write path; it feeds the pipeline and re-publishes
  pipeline telemetry as events.

### 9.2 MessageStore — the working set

- **Purpose.** Fast in-memory access to the active conversation's messages with
  write-through to the repository.
- **Responsibilities.** Ordered message list per open conversation; append/read/slice by
  index, id, or token-window; cache invalidation on external change; holds per-message
  cached `TokenEstimate`s.
- **Public methods.** `load(conversationId)`, `append(message)`,
  `getRange(fromId?, toId?) → Message[]`, `tail(maxCount|maxTokens) → Message[]`,
  `replaceRangeWithSummary(range, summaryMessage)`, `evict(conversationId)`.
- **Dependencies.** `ConversationRepository` (read/append), `TokenEstimator` (lazy,
  cached).
- **Why it exists.** The pipeline re-reads history on every request; hitting the
  filesystem each time is wasteful and makes token-window slicing O(file). Separating
  "working set" from "system of record" is also what lets a future distributed node hold
  a lease on a conversation.
- **Interactions.** Manager loads/evicts it; `ContextBuilder` reads from it exclusively
  (never from the repository directly); `SummaryManager` swaps ranges in it.

### 9.3 ConversationRepository — the storage port

- **Purpose.** The persistence *interface* (port). Defines what storage must do, not how.
- **Responsibilities.** Durable CRUD for conversation manifests; append-only message log
  per conversation; summary storage; atomic manifest updates with optimistic version
  check; pagination.
- **Public methods.** `createConversation(manifest)`, `readManifest(id)`,
  `updateManifest(manifest, expectedVersion)`, `appendMessages(id, messages[])`,
  `readMessages(id, page?) → Message[]`, `appendSummary(id, summary)`,
  `readSummaries(id)`, `list(filter?, page?)`, `delete(id)`.
- **Dependencies.** None (it is a port). Implementations depend on their medium.
- **Why it exists.** Dependency inversion: domain depends on this interface;
  `FileSystemRepository` (now), `SqliteRepository` / `PostgresRepository` /
  `RemoteRepository` (later) are drop-in adapters — the "future database migration"
  requirement reduced to writing one class.
- **Interactions.** Implemented by the `PersistenceLayer`; consumed by `MessageStore`
  and `ConversationManager` only.

### 9.4 ContextBuilder — assembly

- **Purpose.** Deterministically assemble the *candidate* context for one model call
  from conversation state + request intent, before fitting.
- **Responsibilities.** Select sections in priority order: system/role instructions →
  pinned messages → memory injections (port, empty today) → summaries covering evicted
  spans → recent verbatim tail → the current request (+ role-specific payloads:
  dependency outputs for workers, subtask outputs for synthesis). Tags every section
  with `priority`, `pinned`, `origin`.
- **Public methods.** `build(ContextRequest, ConversationView) → CandidateContext`
  (an ordered list of `ContextSection`s, not yet fitted).
- **Dependencies.** `MessageStore` (read), `MemoryProvider` port (future, no-op today),
  `ContextConfig` (section priorities).
- **Why it exists.** Part I shows assembly logic smeared across prompt builders. One
  builder = one place where planner/validator/repair/memory integration lands (each is
  just a new section source with a priority).
- **Interactions.** Stage 1 of the `ContextPipeline`; consumes store + memory port;
  hands `CandidateContext` to the window manager.

### 9.5 TokenEstimator — measurement

- **Purpose.** Convert text/sections/messages into token counts for a given provider —
  the system's only source of size truth.
- **Responsibilities.** Per-provider estimation strategies; conservative safety margin;
  memoization keyed by (contentHash, providerFamily); estimate whole `CandidateContext`s
  section-by-section.
- **Public methods.** `estimateText(text, providerFamily) → TokenEstimate`,
  `estimateMessages(messages[], providerFamily) → TokenEstimate`,
  `estimateContext(candidate, providerFamily) → TokenEstimate` (per-section breakdown).
- **Dependencies.** None on domain (leaf component); optional tokenizer adapters.
- **Options considered.**
  1. *Exact tokenizers per provider* — precise, but web-driven providers expose no
     tokenizer contract, adds heavy deps, breaks local-first simplicity.
  2. *Pure chars/4 heuristic* — free, but wrong enough (code vs prose vs CJK) to cause
     overflow bounces.
  3. **Chosen: pluggable strategy with calibrated heuristic default** — per-family
     chars-per-token ratios (prose ≈ 4.0, code ≈ 3.3, dense JSON ≈ 2.8) + a 10% safety
     margin, with an optional exact-tokenizer adapter slot per provider. Rationale:
     estimates only ever need to be *safely conservative*; exactness is an optimization,
     not a correctness requirement, and the port lets us add it where it pays.
- **Interactions.** Used by `MessageStore` (cached per message), `ContextWindowManager`
  (fitting), `SummaryManager` (compression targets), analytics (future).

### 9.6 ContextWindowManager — fitting

- **Purpose.** Fit a `CandidateContext` into a provider's real window, deciding what
  stays verbatim, what is summarized, and what is dropped — under explicit policy.
- **Responsibilities.** Resolve the budget: `min(AgentSpec.max_context_tokens,
  provider transport cap) − reservedForResponse`. Apply the window policy. Produce a
  `ContextWindow` with a full **eviction record** (nothing disappears silently).
- **Public methods.** `fit(candidate, TokenBudget, policy?) → ContextWindow`,
  `wouldFit(candidate, budget) → TokenEstimate`.
- **Window policy — options considered.**
  1. *Hard tail window* (keep last N tokens) — simple, but loses goals stated early;
     exactly today's failure mode with better units.
  2. *Importance-scored eviction* — flexible, but non-deterministic, hard to test, and
     premature without usage data.
  3. **Chosen: structured window** — pinned head (system + conversation goals) +
     summary layer (placeholders covering evicted middle spans) + verbatim tail (recency)
     + always-included current request; sections evicted middle-out by priority.
     Deterministic, unit-testable, mirrors how humans keep context. Policy is a value
     object so strategy 2 can be added later without touching the manager.
- **Dependencies.** `TokenEstimator`, `ContextConfig` (policy defaults), `SummaryManager`
  (requests placeholders for evicted spans — via port, so no cycle).
- **Why it exists.** Replaces all three char-clipping systems with one token-aware,
  provider-aware, *recorded* decision.
- **Interactions.** Stage 2 of the pipeline; emits eviction records that the
  `SummaryManager` consumes asynchronously.

### 9.7 SummaryManager — compression

- **Purpose.** Own summary lifecycle: create placeholders synchronously, produce real
  summaries asynchronously, maintain summary coverage maps.
- **Responsibilities.** For each evicted span: insert a placeholder summary message
  (`"[Summary pending for messages m12–m48]"`) immediately; queue an LLM summarization
  job; on completion, replace placeholder content and persist a `Summary` record with
  `sourceRange`, `level` (summaries of summaries), and token size; expose coverage
  (which spans are summarized at which level).
- **Public methods.** `placeholderFor(range) → Message`,
  `requestSummary(conversationId, range, priority?)`, `getCoverage(conversationId) →
  SummaryCoverage`, `resummarize(range, level)`.
- **Options considered.** *Blocking inline summarization* (context waits for the
  summary LLM call — adds seconds to every overflowing request) vs **async placeholder
  (chosen)** — the pipeline never blocks; the next request that needs the span picks up
  the completed summary; placeholders make the gap explicit to the model. Rationale:
  latency predictability beats first-request completeness, and repair loops tolerate a
  pending placeholder far better than every user tolerates a slow prompt.
- **Dependencies.** `LLMClient` seam (via a `Summarizer` port so the domain doesn't
  import provider code), `ConversationRepository` (persist summaries), `TokenEstimator`.
- **Interactions.** Fed by window-manager eviction records; read by `ContextBuilder`
  (summary layer); its jobs run on the existing executor/cancellation infrastructure.

### 9.8 ContextPipeline — orchestration of the above

- **Purpose.** The ordered, extensible sequence that turns a `ContextRequest` into a
  `ContextResult`.
- **Stages (v1).** `LoadStage` (store/view) → `AssembleStage` (ContextBuilder) →
  `MemoryStage` (no-op port today) → `EstimateStage` (TokenEstimator) → `FitStage`
  (ContextWindowManager) → `RenderStage` (provider-neutral `ContextResult`).
- **Public methods.** `run(ContextRequest) → ContextResult`, `use(stage, position?)` —
  registration for future stages (planner context, validator transcripts, repair
  history, retrieval).
- **Stage contract.** Every stage: `name`, `execute(PipelineState) → PipelineState`,
  honors the `CancelToken`, appends to a `PipelineTrace` (timings, token deltas,
  evictions) — the future observability hook.
- **Options considered.** *Monolithic builder method* (simpler now, but every future
  integration edits the core — the current system's exact disease) vs **middleware
  pipeline (chosen)** — planner/validator/repair/memory integrate as stages without
  touching existing ones (open/closed principle). Rationale: the brief lists six future
  integrations; the pipeline turns each into an additive change.
- **Dependencies.** The stages; `ContextConfig`.
- **Interactions.** Invoked only by `ConversationManager.buildContext`; its trace feeds
  the event system.

### 9.9 ContextModels — the shared vocabulary

- **Purpose.** The typed data contracts every component exchanges (§11). Pure data, zero
  behavior, zero I/O — importable by everything without creating cycles.

### 9.10 PersistenceLayer — the filesystem adapter (v1)

- **Purpose.** The `ConversationRepository` implementation for local-first operation.
- **Layout.** One directory per conversation:
  - `manifest.json` — Conversation record + `version` counter (optimistic concurrency)
  - `messages.jsonl` — append-only, one Message per line
  - `summaries.jsonl` — append-only Summary records
- **Responsibilities.** Crash-safe appends (write + fsync, line-atomic); manifest
  updates via temp-file + atomic rename guarded by `expectedVersion`; corruption
  recovery (skip torn tail lines, quarantine + warn); ULID-named directories.
- **Storage format — options considered.**
  1. *One JSON file per conversation* — every append rewrites the whole file: O(n) write
     amplification and torn-file corruption risk on crash. Rejected.
  2. *SQLite* — transactional and queryable, but adds a native dependency, WAL/locking
     ceremony across our per-provider threads, and buys little for append-mostly data.
     Deferred to an adapter when querying needs arrive.
  3. **Chosen: JSONL log + JSON manifest** — appends are O(1) and crash-safe, files are
     human-readable/diffable (local-first ethos), streaming reads are trivial, and the
     format maps 1:1 onto any future DB table. Rationale: the write pattern is
     append-dominated; the read pattern is tail-dominated; JSONL is optimal for both.
- **Interactions.** Behind the repository port only. Nothing else may touch these files.

### 9.11 Interfaces (ports) — the seams

`ConversationRepository` (§9.3) · `TokenizerAdapter` (exact tokenizers, optional) ·
`MemoryProvider` (`inject(request, view) → ContextSection[]`; no-op v1 — the future
memory system implements it) · `Summarizer` (LLM summarization behind the seam) ·
`ContextStage` (§9.8) · `EventSink` (lifecycle/telemetry consumers) · `Clock` /
`IdGenerator` (deterministic tests). **Why:** each port is precisely one of the brief's
"future" requirements turned into a compile-time seam today.

### 9.12 Utilities

ULID generation (time-sortable ids → natural file ordering and pagination);
content hashing (estimate caching, dedup); async mutex / per-conversation lock;
JSONL codec with torn-line recovery; markdown-safe section renderers. Pure functions,
leaf dependencies, no domain imports.

### 9.13 Types

Branded/opaque id types (`ConversationId`, `MessageId`, `SummaryId`), enums
(`MessageRole` incl. `summary`; `SectionOrigin`; `TrimAction`), `Result`-style error
envelopes for repository operations. **Why:** id mix-ups are the classic silent bug in
multi-store systems; branding makes them type errors.

### 9.14 Configuration

`ContextConfig`: per-role token reservations (system/response/tail minimums), window
policy defaults, summary trigger thresholds (e.g. summarize when evicted span >
N tokens), estimator ratios + safety margin, storage root path, feature flags for each
port (memory on/off, exact tokenizers on/off). Layered: defaults → file → per-request
overrides. **Why:** every tunable the current system hard-codes (three char budgets)
becomes one declared, observable policy surface.

---
## 10. Folder Structure (and why each folder exists)

```
src/
├─ context/                     # THE bounded context this SADD defines
│  ├─ domain/                   # Pure logic. No I/O imports allowed — ever.
│  │  ├─ models/                # §11 data contracts (ContextModels)
│  │  ├─ builder/               # ContextBuilder + section sources
│  │  ├─ window/                # ContextWindowManager + window policies
│  │  ├─ tokens/                # TokenEstimator + ratio tables
│  │  └─ summary/               # SummaryManager domain logic + coverage maps
│  ├─ pipeline/                 # ContextPipeline, stage contract, built-in stages
│  ├─ ports/                    # All interfaces (§9.11) — the dependency-inversion line
│  ├─ adapters/                 # Implementations of ports (I/O lives here only)
│  │  ├─ fs/                    # PersistenceLayer: manifest/JSONL repository
│  │  ├─ tokenizers/            # Optional exact TokenizerAdapters
│  │  └─ memory-noop/           # v1 MemoryProvider stub
│  ├─ manager/                  # ConversationManager façade + MessageStore
│  ├─ events/                   # Event types + EventSink fan-out
│  └─ config/                   # ContextConfig schema, defaults, loader
├─ orchestrator/                # Existing pipeline (planner/executor/… unchanged)
├─ providers/                   # Existing provider layer (unchanged)
└─ shared/                      # Utilities & Types used across contexts
   ├─ ids/  ├─ hashing/  ├─ concurrency/  └─ jsonl/
tests/
├─ context/unit/                # Domain tested with fakes (no filesystem)
├─ context/adapters/            # Repository contract tests run against EVERY adapter
└─ context/pipeline/            # Golden-file context assembly tests
docs/
└─ adr/                         # Architecture Decision Records (this SADD = ADR-001)
```

**Why this shape.** `domain/` vs `adapters/` enforces the hexagon *by import rule* —
reviewable and lintable, not aspirational. `ports/` is its own folder so the
dependency-inversion line is visible in every import path. `manager/` is separate from
`domain/` because the façade coordinates I/O and domain (it may import both; domain may
import neither). `adapters/fs` isolated per medium means the future `adapters/sqlite`
lands beside it and passes the same contract tests in `tests/context/adapters/`.
`shared/` holds only leaf utilities to keep the context package extractable as its own
library — which is precisely what a future TS port or service split needs.

## 11. Data Models

All records carry `schemaVersion` (int) — the migration lever for every future change.

**Conversation** — `id (ConversationId, ULID)`, `title`, `createdAt`, `updatedAt`,
`status (active|archived)`, `parentId? + forkedAtMessageId?` (multi-agent forks),
`version` (optimistic concurrency), `metadata (Metadata)`, `stats {messageCount,
approxTokens}` (denormalized for lists).

**Message** — `id (MessageId, ULID — time-sortable ⇒ ordering is intrinsic)`,
`conversationId`, `role (system|user|assistant|tool|summary)`, `content`,
`origin {kind: user|worker|synthesizer|planner|repair|memory, agentName?, provider?,
runId?, subtaskId?}`, `tokenEstimateCache? {family → TokenEstimate}`, `pinned (bool)`,
`supersededBy? (SummaryId)`, `createdAt`, `metadata`. *Extensibility:* `origin.kind` is
an open enum — planner/validator/repair transcripts become first-class history without
schema change.

**Summary** — `id (SummaryId)`, `conversationId`, `sourceRange {fromMessageId,
toMessageId, count}`, `level (1 = of messages, 2 = of summaries, …)`, `content`,
`status (pending|ready|failed)`, `tokens`, `createdAt`, `model?`.

**Metadata** — open string-keyed map with reserved namespaces: `workflow.*`,
`planner.*`, `agent.*`, `user.*`. Reserved namespaces are the documented extension
points; anything else is ignored by the core.

**ContextRequest** — `conversationId`, `purpose (worker|synthesizer|planner|verifier|
repair|chat)`, `currentInput {content, role}`, `targetProvider + targetModel`,
`tokenBudgetOverride?`, `policyOverride?`, `extraSections? (ContextSection[])`
(how executor passes dependency outputs today), `cancelToken`.

**ContextSection** — `origin (system|pinned|memory|summary|history|payload|request)`,
`priority (int; lower evicts first)`, `content`, `sourceIds? (MessageId[]|SummaryId[])`.

**ContextResult** — `sections (ContextSection[] final order)`, `renderedMessages`
(provider-neutral `{role, content}[]` — feeds the existing `LLMMessage` shape 1:1),
`window (ContextWindow)`, `trace (PipelineTrace)`.

**TokenEstimate** — `tokens (int)`, `method (heuristic|exact)`, `family`,
`confidence (0..1)`, `perSection? (map)`.

**ContextWindow** — `budget {modelMax, transportMax, reservedForResponse, usable}`,
`used (TokenEstimate)`, `kept (sectionRefs)`, `evicted [{sectionRef, action
(dropped|summarized|clipped), replacementSummaryId?}]`, `policyName`. The eviction
record is the design's audit trail: **nothing leaves context without a trace.**

## 12. Execution Flow (end-to-end, happy path)

1. **User sends a prompt** (existing `POST /api/run` gains optional `conversationId`).
2. Workflow asks `ConversationManager.getConversation(id)` — or `createConversation`
   when absent. Manager acquires the per-conversation writer lock.
3. Manager appends the **user Message** (validated, ULID-stamped) → `MessageStore`
   (memory) → write-through `Repository.appendMessages` (fsync'd JSONL line).
4. Workflow (executor, per model call) issues `buildContext(ContextRequest{purpose:
   worker, targetProvider, extraSections: dependencyOutputs})`.
5. `ContextPipeline` runs: **Load** (store view incl. summaries) → **Assemble**
   (sections: system → pinned → summaries → tail → payload → request) → **Memory**
   (no-op v1) → **Estimate** (cached per message; conservative) → **Fit** (structured
   window vs `min(model, transport) − reserved`; middle-out eviction; placeholders
   requested for large evicted spans) → **Render** (`ContextResult`).
6. Provider request: `renderedMessages` map onto the existing `LLMMessage[]`;
   `WebAutomationLLMClient` sends as today (provider `max_prompt_chars` remains as a
   last-resort transport guard that should now never trigger).
7. **Response received** → existing validation gauntlet (refusal/echo/artifact) →
   manager appends the **assistant Message** with full `origin` provenance
   (provider, agent, runId, subtaskId) → store + repository.
8. **History updated:** manifest `stats`/`updatedAt`/`version` bumped atomically;
   eviction records that crossed the summary threshold are queued to `SummaryManager`;
   completed summaries land as `role: summary` messages superseding their range;
   events (`message.appended`, `context.built`, `summary.ready`) fan out to sinks
   (SSE console today; analytics later). Lock released.

## 13. Sequence Diagram

```
User      Workflow      ConvManager      Repository      ContextBuilder*      Provider      Filesystem
 │            │              │                │                 │                 │              │
 │ prompt     │              │                │                 │                 │              │
 ├───────────►│ open(convId) │                │                 │                 │              │
 │            ├─────────────►│ readManifest   │                 │                 │              │
 │            │              ├───────────────►│ read manifest.json                │              │
 │            │              │                ├─────────────────────────────────────────────────►│
 │            │              │◄───────────────┤ Conversation    │                 │              │
 │            │ append(userMsg)               │                 │                 │              │
 │            ├─────────────►│ appendMessages │                 │                 │              │
 │            │              ├───────────────►│ append JSONL (fsync)              │              │
 │            │              │                ├─────────────────────────────────────────────────►│
 │            │ buildContext(request)         │                 │                 │              │
 │            ├─────────────►│ pipeline.run ────────────────── ►│ assemble        │              │
 │            │              │                │                 │ estimate → fit  │              │
 │            │              │◄─────────────────────────────────┤ ContextResult   │              │
 │            │◄─────────────┤ ContextResult  │                 │                 │              │
 │            │ complete(renderedMessages)    │                 │                 │              │
 │            ├──────────────────────────────────────────────────────────────────►│ drive web UI │
 │            │◄──────────────────────────────────────────────────────────────────┤ response     │
 │            │ append(assistantMsg + origin) │                 │                 │              │
 │            ├─────────────►│ appendMessages + manifest v++    │                 │              │
 │            │              ├───────────────►│ append / atomic rename            │              │
 │            │              │                ├────────────────────────────────────────────────►│
 │◄───────────┤ result       │ events: message.appended, context.built, summary.queued          │
 │            │              │                                                                  │
   *ContextBuilder column stands for the whole pipeline (load→assemble→memory→estimate→fit→render)
```

## 14. Class Diagram

```
┌──────────────────────┐        uses         ┌─────────────────────┐
│ ConversationManager  │────────────────────►│   ContextPipeline   │
│──────────────────────│                     │─────────────────────│
│ create/get/list/fork │                     │ run(req)            │
│ appendMessage        │                     │ use(stage)          │
│ buildContext         │                     └──────────┬──────────┘
│ onEvent              │                                │ stages (ContextStage port)
└───┬─────────┬────────┘        ┌───────────────────────┼─────────────────────────┐
    │         │                 ▼                       ▼                         ▼
    │         │        ┌────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
    │         │        │ ContextBuilder │   │ ContextWindowManager│   │   RenderStage      │
    │         │        │ build(req,view)│   │ fit(candidate,      │   │ → ContextResult    │
    │         │        └──────┬─────────┘   │     budget, policy) │   └────────────────────┘
    │         │               │ reads       └──────┬───────┬──────┘
    │         ▼               ▼                    │       │ eviction records
    │  ┌──────────────┐  ┌──────────────┐          │       ▼
    │  │ MessageStore │  │MemoryProvider│          │  ┌────────────────┐   Summarizer port
    │  │ load/append  │  │ (port, noop) │          │  │ SummaryManager │──────► LLM seam
    │  │ tail/replace │  └──────────────┘          │  │ placeholder/   │
    │  └──────┬───────┘                            │  │ request/cover  │
    │         │ write-through          TokenEstimator◄─┴───────┬───────┘
    │         ▼                        ┌──────────────┐        │ persists
    │  ┌──────────────────────────┐    │ estimateText │        │
    └─►│ ConversationRepository   │    │ estimateCtx  │        │
       │ (PORT)                   │    └──────────────┘        │
       │ manifests/messages/      │◄──────────────────────────-┘
       │ summaries, versions      │
       └──────────┬───────────────┘
                  │ implemented by
                  ▼
       ┌──────────────────────────┐     future: SqliteRepository, RemoteRepository
       │ FileSystemRepository     │
       │ manifest.json + *.jsonl  │
       └──────────────────────────┘
```

## 15. Dependency Graph

```
            shared utils / Types / ContextModels        (leaves — imported by all)
                             ▲
        ┌────────────────────┼──────────────────────┐
        │                    │                      │
  TokenEstimator      ContextBuilder         SummaryManager(domain)
        ▲                    ▲                      ▲
        └──────────┬─────────┘                      │
                   │                                │
          ContextWindowManager ─────────────────────┘        DOMAIN (no I/O imports)
                   ▲
                   │
             ContextPipeline ──── ports: ContextStage, MemoryProvider, Summarizer
                   ▲
                   │
           ConversationManager ── MessageStore
                   ▲                        │ depends on PORT only
                   │                        ▼
             (workflow / UI)       ConversationRepository (PORT)
                                            ▲
                                            │ implements
                                   FileSystemRepository            ADAPTERS (I/O)
```

Arrows point from dependent → dependency. **All arrows cross the port line pointing at
interfaces, never at adapters** — dependency inversion enforced structurally. Each box
has one reason to change (SRP); high cohesion inside `domain/`; the only shared state is
behind `MessageStore` + repository versioning (low coupling).

## 16. Design-Principles Support Matrix

| Requirement | Mechanism in this design |
|---|---|
| Easy testing | Pure domain + `Clock`/`IdGenerator` ports ⇒ deterministic units; repository contract tests shared across adapters; golden-file pipeline tests |
| Dependency injection | Manager and pipeline receive all collaborators via constructor ports; zero global state |
| Future plugins | `ContextStage` registration (`pipeline.use`) + `MemoryProvider`/`Summarizer` ports |
| Future providers | Providers only consume `ContextResult.renderedMessages`; estimator gains a family entry |
| Future databases | New `ConversationRepository` adapter; same contract tests prove parity |
| Future cloud sync | Append-only JSONL + version counters ⇒ log shipping / CRDT-friendly; `RemoteRepository` adapter |
| Future distributed execution | Per-conversation writer lock generalizes to a lease; optimistic `version` detects conflicts; ULIDs merge-sort across nodes |
| Future event system | `EventSink` port already carries lifecycle + pipeline telemetry |
| Future caching | Estimate memoization by content hash; `MessageStore` is the read-cache seam |
| Future analytics/observability | `PipelineTrace` per request: timings, token deltas, evictions — attach a sink |

## 17. Migration Plan (incremental, zero breaking changes)

**Phase 0 — Scaffold (no behavior change).** Land `context/` package: models, ports,
FileSystemRepository, estimator, unit tests. Nothing imports it yet. *Risk: none.*

**Phase 1 — Record (write-only shadow).** Server run-handler creates a conversation per
run and appends user prompt + final answer + per-subtask outputs (with `origin`) via the
manager. Existing flow untouched — persistence is additive. UI gains a read-only history
list. *Rollback: stop writing.*

**Phase 2 — Measure (shadow context).** Executor/synthesizer call `buildContext` in
shadow mode: pipeline runs, trace logged, result *discarded*; live path still uses
current clipping. Compare shadow windows vs today's char-clips on real runs to calibrate
estimator ratios. *Rollback: disable flag.*

**Phase 3 — Adopt (flip the seam).** Worker/synthesis prompt assembly consumes
`ContextResult.renderedMessages`; the three char-budget systems demote to transport
guards (`max_prompt_chars` stays as final safety net). Old clip paths deleted only after
a soak period behind a feature flag. *Rollback: flag back to legacy assembly.*

**Phase 4 — Converse.** `POST /api/run` accepts `conversationId`; UI offers "continue
conversation"; summaries activate for overflowing histories. First user-visible feature
— built on three phases of proven plumbing.

**Phase 5 — Extend.** Memory/planner/validator/repair integrations arrive as pipeline
stages + `origin` kinds; DB adapter lands when query patterns justify it.

Each phase is independently shippable, feature-flagged, and reversible; the TS rewrite
(if pursued) implements the same ports and inherits the same tests translated.

## 18. Risks & Mitigations

| Risk | Class | Mitigation baked into the design |
|---|---|---|
| Token estimates wrong → overflow bounces or wasted budget | performance/correctness | Conservative margins + Phase-2 shadow calibration against real provider rejections; per-family ratios; optional exact adapters |
| JSONL corruption on crash | data | Append-only + fsync, torn-tail recovery, manifest via atomic rename + version check; contract tests simulate torn writes |
| Summary drift (summaries misrepresent evicted content) | quality | Summaries are *additive* records pointing at still-persisted source messages — always re-derivable; levels bounded; eviction record keeps provenance |
| Pipeline latency added to every call | performance | Store working-set + estimate memoization ⇒ assembly is in-memory; only summarization calls LLMs, and asynchronously |
| Over-engineering vs current needs | maintainability | Ports ship with exactly one adapter each; no speculative implementations — the *seams* are the deliverable, kept cheap |
| Concurrent writers (multi-agent) corrupt ordering | scalability | Per-conversation writer lock now; optimistic `version` + fork model designed in from day one |
| Migration stalls mid-way (two context systems forever) | tech debt | Phases 2–3 include explicit deletion criteria and soak gates; legacy clip paths are flagged for removal, not left coexisting |
| Filesystem limits (thousands of conversations) | scalability | ULID sharded directories if needed; repository port means SQLite/Postgres adapter is the pressure-relief valve |

## 19. Architecture Readiness Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Current system fully traced; keep/redesign inventory explicit | ✅ (Part I) |
| 2 | All 14 required components specified with purpose/methods/dependencies/interactions | ✅ (§9) |
| 3 | Data models defined with versioning + extensibility points | ✅ (§11) |
| 4 | End-to-end flow, sequence, class, dependency diagrams | ✅ (§12–15) |
| 5 | Every major decision has compared options + rationale | ✅ (§9.5, 9.6, 9.7, 9.8, 9.10) |
| 6 | SOLID/DI enforced structurally (domain/ports/adapters import rule) | ✅ (§10, §15) |
| 7 | Incremental, reversible, non-breaking migration plan | ✅ (§17) |
| 8 | Risks identified with design-level mitigations | ✅ (§18) |
| 9 | Testing strategy per layer (unit/contract/golden) | ✅ (§10, §16) |
| 10 | Open items before coding | ⚠️ Two decisions for the team: (a) confirm target runtime for Phase 0 (extend current Python vs start TS package — ports are identical either way); (b) approve estimator ratio defaults to seed Phase-2 calibration |

**Verdict: READY FOR IMPLEMENTATION** — pending the two sign-offs in item 10. Phase 0
can start immediately; nothing in it blocks on those decisions.
