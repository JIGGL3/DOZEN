# SADD-002 — Reliability Layer Architecture

**System:** DOZEN — browser-automation LLM orchestrator
**Document:** Software Architecture Design Document (architecture only — no implementation)
**Status:** PROPOSED (awaiting sign-off)
**Depends on:** SADD-001 (Context Management System), Phases 1.1–1.4 (shipped)
**Prime constraint:** every provider is a live Playwright browser tab. There are **no provider APIs** and never will be in this design. All reliability mechanics must work through the browser.

---

## 1. Executive summary

Today a single provider failure fails the whole workflow: `ProviderError` (or a Playwright
`TargetClosedError`, or a silent stall) travels up from the browser tab through
`BrowserManager.send_prompt` → `WebAutomationLLMClient.complete` → `Executor`, burns the
subtask's repair attempts against the *same broken tab*, and the run degrades or dies. The
operator has to notice, diagnose, and re-run by hand.

The Reliability Layer makes provider failure a *routine, classified, recoverable event*:

```
provider signal (exception / DOM state / silence)
        ↓  Failure Detection Engine        — turn raw signals into a FailureEvent
        ↓  Failure Classifier              — assign one FailureType with confidence
        ↓  Recovery Engine                 — run that type's recovery ladder on the SAME provider
        ↓  Failover Engine (if unrecovered)— transfer the attempt to the best OTHER provider
        ↓  Checkpoint Engine               — nothing already computed is ever recomputed
        ↓  execution continues; the user never intervenes
```

The design is **wrapper-based and additive**. Not one line of Planner, Executor, Router,
Synthesizer, BrowserManager, or provider adapters changes. The layer inserts at three
existing seams:

1. **The client seam** — every LLM call in the system already funnels through one method:
   `LLMClient.complete()` (`dozen/llm_client.py:82`), implemented by
   `WebAutomationLLMClient.complete` (`webllm/client.py:127`). A `ReliabilityClient`
   *decorator* implements the same interface and wraps the real client per agent. The
   Executor keeps calling `agent.client.complete(...)` and gets reliability for free.
2. **The event seam** — SADD-001's `EventBus` port and `Event` model (Phase 1.1/1.3) carry
   health and failure telemetry; `ConversationSession` already tracks
   `provider_history` / `current_provider` for exactly this purpose.
3. **The persistence seam** — checkpoints reuse the Phase 1.2 storage engine (append-only
   JSONL + atomic manifest) under a new `.runs/` root. No new storage technology.

---

## 2. Goals and non-goals

### Goals
- G1. No single-provider failure fails a workflow while ≥1 healthy provider remains.
- G2. Every failure is detected within a bounded time (no infinite waits) and classified.
- G3. Recovery is policy-driven per failure type: deterministic, budgeted, auditable.
- G4. Failover preserves execution context (subtask instruction + dependency outputs +
  conversation context) — the receiving provider gets everything the failed one had.
- G5. A crashed orchestrator process resumes a run from its last checkpoint instead of
  restarting from zero.
- G6. Humans are pulled in *only* for what browsers cannot self-serve (captcha, credential
  re-entry) — and the run keeps moving on other providers meanwhile.
- G7. Extensible without redesign: new failure types, detectors, recovery actions and
  scoring signals are registry entries, not architecture changes.

### Non-goals
- Provider APIs, headless-detection evasion, or captcha *solving* (only detection + human
  handoff). Automating captcha bypass is explicitly out of scope.
- Multi-machine distribution (design leaves seams; v1 is single host).
- Semantic verification of answer *quality* — that remains the Verifier's job.
- Modifying planning/synthesis logic.

---

## 3. Current-state analysis (what exists, what breaks)

### 3.1 The seams the layer will use

| Existing element | Location | Reliability role |
|---|---|---|
| `LLMClient.complete()` | `dozen/llm_client.py` | **The interception point.** All Planner/Executor/Synthesizer traffic. |
| `AgentSpec.client` | `dozen/agent_pool.py` | Per-agent client handle — where the decorator is installed (in `webllm/pool.py:build_orchestrator`). |
| `BrowserManager._Worker` per provider | `webllm/browser_manager.py` | One thread + one Playwright driver per provider: the *unit of isolation and restart*. |
| `BrowserManager._is_closed_error` | `webllm/browser_manager.py:275` | Existing heuristic for `TAB_CLOSED` / `BROWSER_CRASH` classification. |
| `BrowserManager._session_alive` | `webllm/browser_manager.py:264` | Existing liveness probe for the Health Monitor. |
| `ProviderAdapter.logged_in_selectors` | `webllm/providers.py:163` | Existing auth-state probe for `LOGIN_REQUIRED` detection. |
| `ProviderAdapter._stop_button_visible` | `webllm/providers.py:492` | Existing "generation in progress" probe for `GENERATION_STALLED`. |
| `ProviderError` / `CancelledError` | `webllm/providers.py` | Raw failure signals (currently unclassified strings). |
| `Executor` retry loop (`max_repair_attempts`, escalate-on-final) | `dozen/executor.py:216-349` | Stays untouched: it handles *quality* retries; the reliability layer handles *transport* retries beneath it. |
| `Router._score` | `dozen/router.py:66` | Base capability scoring reused by the Failover Engine. |
| `EventBus`, `Event`, `EventType` | `dozen/context` (Phase 1.1/1.3) | Telemetry backbone. |
| `ExecutionContext` model | `dozen/context/domain/models.py` | The already-designed carrier of run/provider/task identity for checkpoints. |
| Phase 1.2 storage engine | `dozen/context/adapters/filesystem` | Checkpoint persistence (append log + atomic manifest + locks). |
| SSE `RunChannel.emit` | `webllm/progress.py` | User-visible reliability narration (`phase: "recovery"`, `"failover"`). |

### 3.2 Observed failure evidence (from live operation of this exact system)

- **Playwright driver death**: an unhandled `EPIPE` in the Node driver killed the whole
  server mid-session (2026-07-09). Today nothing supervises driver processes.
- **`Planning failed` marker** recorded in production conversation history: a planner call
  failed and the run produced no answer — no retry on another provider.
- **Copilot "message too long" bounce**: adapter-level mitigation exists (`max_prompt_chars`)
  but a bounce mid-run previously double-pasted and burned attempts.
- **Stale scrapes / empty responses**: `ProviderError("Scraped an empty response")` treated
  identically to a timeout — no differentiated handling.
- **Login expiry / captcha**: detected only at login-confirm time; a session that expires
  mid-run fails with a generic timeout after the full wait.

These five incidents map to nine of the twelve failure types in §7 — the taxonomy is
grounded, not speculative.

---

## 4. Architecture overview

### 4.1 Placement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATOR (unchanged)                     │
│   Planner ── Router ── Executor ── Verifier ── Synthesizer              │
│                              │ agent.client.complete()                  │
├──────────────────────────────▼──────────────────────────────────────────┤
│                     R E L I A B I L I T Y   L A Y E R  (new)            │
│                                                                         │
│  ┌────────────────────  ReliabilityClient (decorator) ───────────────┐  │
│  │  wraps one provider's client; same LLMClient interface            │  │
│  │                                                                    │  │
│  │   attempt ──► Failure Detection Engine ──► Classifier              │  │
│  │                     │                          │                    │  │
│  │                     ▼                          ▼                    │  │
│  │             Provider Health Manager ◄── Recovery Engine            │  │
│  │                     ▲                          │                    │  │
│  │                     │                          ▼                    │  │
│  │             Provider Registry  ◄────── Failover Engine ──► other   │  │
│  │                     ▲                                     provider  │  │
│  └─────────────────────┼────────────────────────────────────────────┘  │
│                        │                                                │
│   Health Monitor (background probes)      Checkpoint Engine (persist)  │
├─────────────────────────────────────────────────────────────────────────┤
│                     WebAutomationLLMClient (unchanged)                  │
│                     BrowserManager + per-provider workers (unchanged)   │
│                     Provider adapters + live browser tabs (unchanged)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Package layout (all new, all additive)

```
dozen/reliability/
    __init__.py
    types.py           failure taxonomy, health states, confidence types
    models.py          FailureEvent, HealthRecord, RecoveryPlan, RecoveryOutcome,
                       Checkpoint, ProbeResult, FailoverDecision (pure data)
    registry.py        ProviderRegistry
    health.py          ProviderHealthManager
    detection.py       FailureDetectionEngine + detector registry
    classify.py        FailureClassifier (signal → FailureType + confidence)
    recovery.py        RecoveryEngine + per-type policy table
    failover.py        FailoverEngine (health-aware scoring on top of Router._score)
    checkpoint.py      CheckpointEngine (event-sourced run log on Phase 1.2 storage)
    monitor.py         HealthMonitor (background probing via existing workers)
    client.py          ReliabilityClient (the LLMClient decorator)
    config.py          ReliabilityConfig (budgets, cadences, thresholds, flags)
webllm/
    reliability_probes.py   browser-side probe helpers (wrap, never modify, adapters)
```

Dependency rule (same discipline as SADD-001): `dozen/reliability` depends on
`dozen.llm_client` types, `dozen.agent_pool`, and `dozen.context` (events, storage,
models). It never imports Playwright directly — all browser interaction goes through
`BrowserManager`'s public job API and `webllm/reliability_probes.py`.

---

## 5. Component designs

### 5.1 Provider Health Manager (`health.py`)

**Responsibility:** the single writer of provider health state. Consumes attempt outcomes
(from ReliabilityClient) and probe results (from Health Monitor); publishes state
transitions on the EventBus; answers "how healthy is provider X right now?" for the
Failover Engine and Router-adjacent scoring.

**Health state machine (per provider):**

```
                +------------------ recovery verified ------------------+
                ▼                                                        │
 UNKNOWN ──► HEALTHY ──failures──► DEGRADED ──more──► SUSPECT ──► QUARANTINED
    │            ▲                     │                 │              │
  first        success             success            recovery      cooldown
  probe        streak              streak             attempt       expires
    │            │                     │                 │              │
    +────────────+─────────────────────+                 ▼              ▼
                                                    RECOVERING ──► probation
                                                         │        (HEALTHY on
                    NEEDS_HUMAN (captcha / login) ◄──────+         success streak)
                         │  operator resolves via UI
                         ▼
                    RECOVERING
```

- `HEALTHY` — full routing weight.
- `DEGRADED` — recent failures below threshold; still routable at reduced weight.
- `SUSPECT` — failure rate above threshold; routable only if no better option.
- `QUARANTINED` — removed from routing for a cooldown (exponential: 30s → 60s → 120s →
  cap 10min); Recovery Engine may work on it in the background.
- `NEEDS_HUMAN` — captcha/login wall; excluded from routing; UI banner raised; a pending
  probe re-checks after the operator acts.
- `RECOVERING` — a recovery plan is executing; not routable.
- `OFFLINE` — operator removed it or repeated quarantine exhaustion; ignored until re-added.

**Metrics kept per provider (sliding windows + EWMA):**

| Metric | Definition | Used by |
|---|---|---|
| `success_rate_1h / _24h` | successes ÷ attempts in window | failover scoring, state transitions |
| `latency_ewma_ms` | EWMA (α = 0.2) of send→response | failover scoring |
| `latency_p95_ms` | windowed p95 | stall thresholds (adaptive timeout) |
| `consecutive_failures` | reset on success | state transitions |
| `failure_histogram` | count per FailureType (windowed) | recovery policy tuning, registry display |
| `last_success_at / last_failure_at` | timestamps | staleness, UI |
| `quarantine_count` | lifetime + windowed | cooldown ladder, OFFLINE promotion |
| `confidence` | composite ∈ [0,1], §5.6 | routing multiplier |

**Interface (signatures only):**

```
ProviderHealthManager
  record_attempt(provider, outcome: AttemptOutcome) -> HealthTransition | None
  record_probe(provider, probe: ProbeResult) -> HealthTransition | None
  state(provider) -> HealthState
  snapshot(provider) -> HealthRecord            # metrics copy for scoring/UI
  routable_providers() -> list[str]             # HEALTHY/DEGRADED (+SUSPECT if flagged)
  begin_recovery(provider) -> None              # → RECOVERING
  end_recovery(provider, success: bool) -> None
  mark_needs_human(provider, reason) -> None
  human_resolved(provider) -> None              # operator clicked "I fixed it"
```

Thread-safety: one internal lock per provider record (same pattern as
`ConversationLockRegistry`); transitions are atomic; all mutation funnels through
`record_*`. Every transition publishes `EventType` `provider.health_changed` (new enum
members, additive as in Phase 1.3).

### 5.2 Failure Detection Engine (`detection.py`)

**Responsibility:** convert raw signals into structured `FailureEvent`s. Three detector
families, all registered in a detector registry (new detectors = new entries):

**A. Exception detectors** (synchronous, in the ReliabilityClient call path)
Interpret what the existing stack already throws:

| Signal source | Detector output |
|---|---|
| `ProviderError` message patterns ("Scraped an empty response", "Timed out waiting", "message too long", "input box not found") | candidate types with pattern-match confidence |
| `_is_closed_error(exc)` (existing heuristic) | `TAB_CLOSED` / `BROWSER_CRASH` |
| Playwright `TimeoutError` | `TIMEOUT` |
| Worker thread dead / driver `EPIPE` (job never completes; worker loop exited) | `BROWSER_CRASH` |
| `CancelledError` | **not a failure** — user intent; bypasses the layer entirely |

**B. DOM-state detectors** (on-demand probes, run *after* an exception or stall to refine
classification; executed on the provider's worker thread via `BrowserManager._enqueue`)

| Probe | Evidence for |
|---|---|
| `adapter.is_logged_in(page)` false / login selectors present | `LOGIN_REQUIRED` |
| captcha iframe/selector census (`iframe[src*="captcha"]`, `[class*="cf-turnstile"]`, provider-specific) | `CAPTCHA` |
| rate-limit banner census (per-provider selector/text list: "You've reached your limit", "Too many requests", regenerate-later buttons) | `RATE_LIMIT` / `MODEL_BUSY` |
| `page.url` off-domain / error page / interstitial | `UNEXPECTED_UI` / `NETWORK_ERROR` |
| input box resolvable? response container resolvable? (`_resolve`-style census) | `DOM_CHANGED` |
| `navigator.onLine`, failed `fetch` ping to provider origin | `NETWORK_ERROR` |

**C. Liveness detectors** (time-based, in the ReliabilityClient watchdog and Health Monitor)

| Signal | Evidence for |
|---|---|
| stop button visible but response text unchanged for `stall_window` (default 45s, adaptive to `latency_p95`) | `GENERATION_STALLED` |
| observer fired but scraped text empty/whitespace | `OUTPUT_CORRUPTED` |
| artifact-JSON contract expected but unparseable (existing `parse_worker_artifact` failure) | `OUTPUT_CORRUPTED` |
| no DOM mutation events at all for full timeout | `TIMEOUT` (with `silent=true` detail) |

**Interface:**

```
FailureDetectionEngine
  detect_from_exception(provider, exc, call_ctx) -> FailureEvent
  probe_dom(provider) -> list[DomEvidence]                  # runs B-family probes
  watch_liveness(provider, call_handle) -> LivenessVerdict  # C-family, during a call
  register_detector(family, detector) -> None               # extension point
```

`FailureEvent` (model): provider, raw signal, detector evidence list, timestamps,
call context (subtask id, run id, attempt number), screenshot/DOM snapshot reference
(optional, for diagnosis), and the classifier's verdict once assigned.

### 5.3 Failure Classifier (`classify.py`)

**Responsibility:** one `FailureEvent` in → one `FailureType` + confidence out.
Deterministic rule cascade (ordered; first confident match wins), combining exception
evidence with DOM probe evidence:

```
1. Cancelled?                        → not a failure (never enters the layer)
2. Worker/driver dead?               → BROWSER_CRASH        (confidence 0.95)
3. _is_closed_error?                 → TAB_CLOSED           (0.9)
4. DOM probe: captcha present?       → CAPTCHA              (0.95)
5. DOM probe: login wall?            → LOGIN_REQUIRED       (0.9)
6. DOM probe: rate-limit banner?     → RATE_LIMIT           (0.85)
7. exception: "message too long"?    → PROMPT_REJECTED      (0.9)
8. exception: input/selector miss + DOM probe confirms missing containers
                                     → DOM_CHANGED          (0.8)
9. liveness: stalled generation?     → GENERATION_STALLED   (0.8)
10. empty/unparseable output?        → OUTPUT_CORRUPTED     (0.75)
11. offline / origin unreachable?    → NETWORK_ERROR        (0.85)
12. busy indicator / queue message?  → MODEL_BUSY           (0.7)
13. plain timeout, DOM healthy?      → TIMEOUT              (0.7)
14. anything else                    → UNKNOWN              (0.3)
```

Classification below `min_confidence` (default 0.5) downgrades to `UNKNOWN`, whose policy
is the most conservative. The cascade is data (an ordered rule list in `classify.py`),
so new rules are insertions, not redesign.

### 5.4 Recovery Engine (`recovery.py`)

**Responsibility:** execute the recovery ladder for a classified failure *on the same
provider*, within budgets, reporting each step. Recovery actions are small strategy
objects over existing public operations — nothing new touches Playwright directly:

| Action | Realized via (existing surface) |
|---|---|
| `RETRY` | re-invoke `client.complete` (same worker) |
| `SOFT_REFRESH` | enqueue `page.reload()` + `adapter._goto_fresh_chat` on the worker |
| `NEW_CHAT` | `_goto_fresh_chat` only (cheaper than reload; clears stuck composer) |
| `RECOVER_TAB` | close page, reopen from persistent context (profile keeps auth) |
| `RESTART_WORKER` | `BrowserManager.remove_provider` + `start_login_async` + confirm (profile-based re-auth, no credentials needed) |
| `REAUTH` | `RESTART_WORKER` then `is_logged_in` check; if still logged out → `NEEDS_HUMAN` |
| `WAIT_COOLDOWN` | timed backoff (rate limits) with health state QUARANTINED |
| `SHRINK_PROMPT` | re-clip via existing `clip_text` at a lower `max_prompt_chars` (message-too-long) |
| `ESCALATE_HUMAN` | health → NEEDS_HUMAN + SSE/UI banner + toast; run continues elsewhere |
| `DELEGATE` | hand the attempt to the Failover Engine (§5.5) |
| `ABORT_SUBTASK` | surface the original failure to the Executor (its existing repair loop takes over) |

**Budgets:** every action has a cost; a `RecoveryPlan` carries a total budget
(default: 2 in-place actions + 1 delegate per attempt; per-run recovery ceiling to
prevent thrash loops). All steps and outcomes append to the run's checkpoint log and
publish events. §8 defines the per-type ladders.

**Interface:**

```
RecoveryEngine
  plan(event: FailureEvent) -> RecoveryPlan          # ladder for the failure type
  execute(plan) -> RecoveryOutcome                   # RECOVERED | DELEGATED | ESCALATED | ABORTED
  register_action(name, action) -> None              # extension point
```

### 5.5 Failover Engine (`failover.py`)

**Responsibility:** choose the best alternative provider and transfer the attempt.

**Key architectural insight — failover is cheap here.** Workers are *stateless per call*:
the entire input of a worker attempt is `build_worker_messages(task, subtask,
dependency_outputs, repair_feedback)` — instruction, dependency outputs, and (since
Phase 1.4) conversation context, all already in hand. Nothing mid-generation needs
migrating. The unit of transfer is **the attempt**, and "transfer execution + maintain
context" means: re-issue the same message list to a different provider's client, with a
provenance note appended to the FailureEvent trail. (The Planner/Synthesizer calls are the
same shape — also failover-able.)

**Selection scoring** (per candidate provider `p`, for subtask `s`):

```
score(p, s) = base_capability(p, s)          # Router._score — reused, not reimplemented
            × health_multiplier(p)           # HEALTHY 1.0 · DEGRADED 0.7 · SUSPECT 0.35
                                             # QUARANTINED/NEEDS_HUMAN/RECOVERING/OFFLINE 0
            × reliability(p)                 # 0.5 + 0.5 · success_rate_1h  (∈ [0.5, 1.0])
            × latency_fit(p, s)              # 1.0 fast … 0.8 slowest (soft factor)
            × affinity_penalty(p, event)     # 0.5 if p already failed THIS subtask
                                             # this run (retry-different-provider bias)
```

Deterministic, explainable (the decision object records every factor), and cheap.
Tie-break: higher `success_rate_24h`, then lower `latency_ewma`.

**Worked example — "Claude fails mid-subtask (coding task)":**

| candidate | base_cap | health | reliability | latency_fit | affinity | **score** |
|---|---|---|---|---|---|---|
| ChatGPT | 0.92 | 1.0 (HEALTHY) | 0.5+0.5·0.96=0.98 | 0.95 | 1.0 | **0.86** ← chosen |
| DeepSeek | 0.90 | 1.0 (HEALTHY) | 0.5+0.5·0.88=0.94 | 0.85 | 1.0 | 0.72 |
| Gemini | 0.78 | 0.7 (DEGRADED) | 0.5+0.5·0.90=0.95 | 1.00 | 1.0 | 0.52 |

ChatGPT wins on capability × cleanliness; Gemini's DEGRADED state (two recent timeouts)
suppresses it despite the best latency. If ChatGPT also fails this subtask, its affinity
penalty (0.5) drops it to 0.43 and DeepSeek takes attempt 3.

**Interface:**

```
FailoverEngine
  select(subtask_ctx, exclude: set[str]) -> FailoverDecision | None   # None = nobody left
  transfer(call_ctx, decision) -> AttemptHandle                       # re-issues the call
```

`None` (no routable provider) → the failure surfaces to the Executor unchanged (existing
behavior), and if *no* provider is routable for the whole run, the run pauses at a
checkpoint in `WAITING_FOR_PROVIDER` state rather than dying (§5.7).

### 5.6 Provider Registry (`registry.py`)

**Responsibility:** the single directory of providers — static identity + dynamic status
in one queryable place. Composes what exists today (adapter registry in
`webllm/providers.py`, `AgentSpec` strengths in `webllm/pool.py`) with live health.

Registry entry (read model):

```
ProviderEntry
  key, display_name, tier                      # from the adapter/pool (static)
  capabilities: {capability: weight}           # from AgentSpec (static)
  max_prompt_chars, supports_files             # adapter facts (static)
  health: HealthState                          # from HealthManager (live)
  confidence: float                            # composite (live)
  latency_ewma_ms, success_rate_1h             # metrics (live)
  recent_failures: [FailureEvent × N]          # ring buffer (live)
  session: {logged_in, tab_open, worker_alive} # from BrowserManager (live)
```

`confidence = 0.45·success_rate_1h + 0.25·success_rate_24h + 0.15·latency_score +
0.15·probe_freshness` — the single number the UI shows per provider card.

**Interface:** `get(key)`, `all()`, `routable()`, `subscribe(listener)`; plus a read-only
REST projection (`GET /api/reliability/providers`) for the UI. The registry holds **no
logic** — it is a composed view over HealthManager + BrowserManager + pool.

### 5.7 Checkpoint Engine (`checkpoint.py`)

**Responsibility:** persist run progress so crashes resume instead of restart.

**Design: event-sourced run log on the Phase 1.2 storage engine** — the same append-only
JSONL + atomic manifest + per-entity locking already proven in production:

```
.runs/
    <run_id>/
        manifest.json          # RunManifest: task brief ref, conversation id, plan hash,
                               #   status, storage_format_version, updated_at
        checkpoints.jsonl      # append-only CheckpointRecords (below)
```

`CheckpointRecord` kinds (one JSONL line each, ULID-ordered like messages):

| kind | written when | payload |
|---|---|---|
| `RUN_STARTED` | run accepted | task brief, conversation id, provider roster |
| `PLAN_READY` | planner returned | full plan JSON, routing table |
| `SUBTASK_STARTED` | attempt begins | subtask id, provider, attempt #, message-list hash |
| `SUBTASK_COMPLETED` | verifier accepted | subtask id, output artifact (full), provider, timings |
| `FAILURE` | classifier verdict | FailureEvent (compact) |
| `RECOVERY` / `FAILOVER` | engine actions | plan, outcome, decision factors |
| `SYNTHESIS_STARTED` / `RUN_COMPLETED` / `RUN_FAILED` | terminal phases | final answer ref / error |

**Resume protocol** (on server start or operator "resume run"):
replay `checkpoints.jsonl` → rebuild an `ExecutionContext` (the Phase 1.1 model built for
this) → completed subtasks feed the DAG as already-satisfied dependencies (their outputs
come from `SUBTASK_COMPLETED` records) → execution continues with only incomplete
subtasks. In-flight generations at crash time are *lost by design* (browser tabs don't
checkpoint mid-generation) — the subtask simply restarts, which is safe because worker
calls are idempotent from the system's perspective.

**Write cadence:** every record is one append (cheap, atomic at line level). Fsync policy
follows `StorageConfig.fsync_appends`. Checkpointing must never block the hot path: writes
go through a single writer queue per run; if checkpoint IO fails, the run continues and
reliability degrades gracefully (flagged in events) — checkpoints are insurance, not a
dependency.

### 5.8 Health Monitor (`monitor.py`)

**Responsibility:** know provider health *before* a run needs them.

- **Passive first:** every real attempt outcome already feeds HealthManager via the
  ReliabilityClient — zero extra browser traffic. The monitor only fills gaps.
- **Active probes** run **on the provider's existing worker thread** via
  `BrowserManager._enqueue` (never a second Playwright connection — one driver per
  provider is an invariant). Probe = cheap DOM census, *never* a message send:
  `_session_alive` → `page.url` sanity → `is_logged_in` selectors → input-box resolvable
  → captcha/rate-limit banner census. Cost: milliseconds, no provider quota.
- **Cadence:** adaptive. HEALTHY: every 120s. DEGRADED/SUSPECT: 30s. QUARANTINED: at
  cooldown expiry only. NEEDS_HUMAN: on UI "resolved" click + gentle 60s recheck. Idle
  providers (no run in 10min): back off to 300s. All jittered ±20% to avoid synchronized
  probe storms.
- **During runs:** probing yields to real traffic (worker queue priority: jobs first,
  probes only when queue empty).
- **Watchdog duty (the EPIPE lesson):** the monitor also supervises the *workers
  themselves* — a worker whose loop died or whose driver process vanished is detected
  within one cadence tick and scheduled for `RESTART_WORKER`, upgrading the 2026-07-09
  whole-server crash scenario into a single-provider recovery.

**Interface:** `start()`, `stop()`, `probe_now(provider) -> ProbeResult`,
`set_cadence(state, seconds)`.

---

## 6. Diagrams

### 6.1 Class relationships

```
                        ┌────────────────────┐
                        │  ReliabilityClient  │  implements LLMClient
                        │  (one per AgentSpec)│
                        └──────┬──────┬──────┘
              wraps            │      │ consults
   ┌───────────────────────────▼┐   ┌─▼──────────────────┐
   │ WebAutomationLLMClient     │   │ FailureDetection   │──► FailureClassifier
   │ (existing, unchanged)      │   │ Engine             │        │ verdict
   └───────────────┬────────────┘   └────────────────────┘        ▼
                   │ send_prompt              ▲            ┌──────────────┐
   ┌───────────────▼────────────┐   evidence  │            │ Recovery     │
   │ BrowserManager (existing)  │◄────────────┘            │ Engine       │
   │  per-provider _Worker      │◄──── probe jobs ──┐      └──────┬───────┘
   └────────────────────────────┘                   │     in-place│ │delegate
                                                    │             │ ▼
   ┌────────────────────────────┐            ┌──────┴─────┐ ┌────────────┐
   │ ProviderHealthManager      │◄─ outcomes─┤ Health     │ │ Failover   │
   │  (state machine + metrics) │            │ Monitor    │ │ Engine     │
   └──────────────┬─────────────┘            └────────────┘ └──────┬─────┘
                  │ composed into                                  │ scores via
   ┌──────────────▼─────────────┐                          ┌───────▼──────┐
   │ ProviderRegistry (view)    │                          │ Router._score│ (reused)
   └────────────────────────────┘                          └──────────────┘
   ┌────────────────────────────┐   ┌────────────────────────────────────┐
   │ CheckpointEngine           │   │ EventBus (Phase 1.1/1.3, reused)   │
   │  .runs/ on Phase 1.2 store │   │  all components publish here       │
   └────────────────────────────┘   └────────────────────────────────────┘
```

### 6.2 Sequence — failure → in-place recovery

```
Executor          ReliabilityClient    Detection/Classify   Recovery      HealthMgr    Browser tab
   │ complete()          │                     │                │             │             │
   │────────────────────►│  attempt #1         │                │             │             │
   │                     │────────────────────────────────────────────────────────────────►│
   │                     │            ProviderError("Timed out waiting…")                  │
   │                     │◄────────────────────────────────────────────────────────────────│
   │                     │ detect_from_exception + probe_dom    │             │             │
   │                     │────────────────────►│  DOM: logged-in ✓, no banner │             │
   │                     │                     │ verdict: TIMEOUT (0.7)       │             │
   │                     │ plan(TIMEOUT) ──────────────────────►│             │             │
   │                     │                     │   ladder: [RETRY, NEW_CHAT+RETRY, DELEGATE]│
   │                     │                     │                │ record_attempt(fail)      │
   │                     │                     │                │────────────►│ HEALTHY→DEGRADED
   │                     │  execute: RETRY (attempt #2, fresh chat)           │             │
   │                     │────────────────────────────────────────────────────────────────►│
   │                     │                 response text ✓                                  │
   │                     │◄────────────────────────────────────────────────────────────────│
   │                     │                     │                │ record_attempt(success)   │
   │  LLMResponse        │                     │                │────────────►│ streak++    │
   │◄────────────────────│   (Executor never saw any of this)   │             │             │
```

### 6.3 Sequence — unrecoverable → failover (Claude → ChatGPT)

```
Executor      ReliabilityClient(claude)   Recovery     Failover      HealthMgr   ReliabilityClient(gpt)
   │ complete()      │                       │            │              │              │
   │────────────────►│ attempt fails; verdict: LOGIN_REQUIRED (0.9)      │              │
   │                 │ plan: [REAUTH, ESCALATE_HUMAN, DELEGATE]          │              │
   │                 │ REAUTH via worker restart… still logged out       │              │
   │                 │──────────────────────►│ mark NEEDS_HUMAN ────────►│ claude→NEEDS_HUMAN
   │                 │                       │ (UI banner: "Claude needs login")         │
   │                 │ DELEGATE ────────────────────────►│ select(exclude={claude})      │
   │                 │                       │            │ scores: gpt .86 > deepseek .72
   │                 │ CHECKPOINT: FAILOVER record        │              │              │
   │                 │ transfer(call_ctx, gpt) ──────────────────────────────────────►│
   │                 │                       │            │              │  same messages,
   │                 │                       │            │              │  gpt's tab
   │  LLMResponse (from ChatGPT; provenance in metadata)  │              │              │
   │◄────────────────│◄──────────────────────────────────────────────────────────────│
   │   run continues; Claude quarantined for humans; nothing recomputed  │              │
```

### 6.4 State — run execution with checkpoints

```
 RUN_STARTED ──► PLANNING ──► EXECUTING ◄───────────────┐
                    │             │  subtask done: CHECKPOINT
                    │             │─────────────────────┘
                    │             ├── all done ──► SYNTHESIZING ──► COMPLETED
                    │             ├── provider pool empty ──► WAITING_FOR_PROVIDER
                    │             │        (paused at checkpoint; monitor revives pool;
                    │             │         operator notified; resumes → EXECUTING)
                    │             └── fatal/cancelled ──► FAILED / CANCELLED
                    │
              process crash at any point
                    │
                    ▼
              on restart: .runs/<id> replay ──► RESUMED(EXECUTING, only incomplete subtasks)
```

---

## 7. Failure taxonomy

| FailureType | Meaning | Primary evidence | Terminal for provider? |
|---|---|---|---|
| `TIMEOUT` | no response within adaptive deadline; DOM otherwise healthy | Playwright TimeoutError, silent observer | no |
| `GENERATION_STALLED` | generation started, output frozen | stop-button visible + text unchanged ≥ stall_window | no |
| `DOM_CHANGED` | selectors no longer match (provider shipped new UI) | input/response containers unresolvable, adapter census | yes-ish (quarantine; needs selector update) |
| `RATE_LIMIT` | provider throttling this account | banner census, "try again later" text | temporarily (timed) |
| `MODEL_BUSY` | provider-side congestion / queue | busy banner, spinner census | temporarily (short) |
| `LOGIN_REQUIRED` | session expired / logged out | `is_logged_in` false, login wall selectors | until re-auth |
| `CAPTCHA` | human verification wall | captcha frame census | until human |
| `NETWORK_ERROR` | connectivity to provider origin lost | offline, navigation errors, origin ping fail | until network back |
| `BROWSER_CRASH` | driver/browser process died | worker loop dead, EPIPE, `_is_closed_error` on context | until worker restart |
| `TAB_CLOSED` | page/tab gone (user closed it, crash) | `_is_closed_error` on page ops | until tab recover |
| `PROMPT_REJECTED` | provider refused the input mechanically (too long, blocked) | "message too long" bounce, composer error | no |
| `OUTPUT_CORRUPTED` | response scraped but empty/garbled/unparseable | empty scrape, artifact-JSON parse failure | no |
| `UNEXPECTED_UI` | page in an unknown state (interstitial, redirect, A/B dialog) | URL/landmark census mismatch | quarantine + probe |
| `UNKNOWN` | classifier confidence < threshold | — | treated pessimistically |

## 8. Recovery policy matrix

Ladders execute left→right; any success ends the ladder. `Delegate` hands to Failover;
`Abort` surfaces to the Executor's existing repair loop. Budgets bound total time.

| FailureType | Retry? | Refresh/New-chat? | Restart (tab/worker)? | Delegate? | Abort? | Health effect | Notes |
|---|---|---|---|---|---|---|---|
| `TIMEOUT` | ✓ (×1, fresh chat) | ✓ (before retry 2) | — | ✓ | last | → DEGRADED | adaptive deadline from p95 |
| `GENERATION_STALLED` | ✓ (after stop-gen click) | ✓ | — | ✓ | last | → DEGRADED | try provider's own stop button first |
| `DOM_CHANGED` | — | ✓ (×1; may be transient A/B) | — | ✓ immediately | last | → QUARANTINED | raise `selector-drift` diagnostic event with DOM snapshot |
| `RATE_LIMIT` | — | — | — | ✓ immediately | — | → QUARANTINED (timed: parse "try again in X" if present, else ladder 60s→10m) | never hammer a limiter |
| `MODEL_BUSY` | ✓ (×1 after 15s) | — | — | ✓ | — | → DEGRADED | short backoff only |
| `LOGIN_REQUIRED` | — | — | ✓ REAUTH (profile re-open) | ✓ while waiting | — | → NEEDS_HUMAN if reauth fails | profile cookies usually still valid → silent recovery |
| `CAPTCHA` | — | — | — | ✓ immediately | — | → NEEDS_HUMAN + UI banner | never attempt to bypass |
| `NETWORK_ERROR` | ✓ (×1 after 10s) | ✓ | — | ✓ (other providers may share the outage — Failover checks network probe first) | last | → SUSPECT | global-outage detection: if ≥ half of providers hit NETWORK_ERROR in 60s → pause run at checkpoint instead of failing everything |
| `BROWSER_CRASH` | — | — | ✓ RESTART_WORKER | ✓ meanwhile | — | → RECOVERING | delegate first, restart in background |
| `TAB_CLOSED` | — | — | ✓ RECOVER_TAB (reopen from profile) | ✓ if reopen fails | — | → RECOVERING | commonly user-inflicted; cheap recovery |
| `PROMPT_REJECTED` | ✓ with SHRINK_PROMPT | ✓ (clear composer) | — | ✓ (to a bigger-window provider) | last | none (input problem, not provider health) | shrink = lower clip ceiling one notch |
| `OUTPUT_CORRUPTED` | ✓ (×1, re-ask with format reminder) | ✓ | — | ✓ | last | → DEGRADED after repeats | dovetails with Verifier repair |
| `UNEXPECTED_UI` | — | ✓ (×1) | ✓ RECOVER_TAB | ✓ | last | → QUARANTINED + probe | screenshot for diagnosis |
| `UNKNOWN` | ✓ (×1) | ✓ (×1) | — | ✓ | last | → DEGRADED | conservative; full evidence logged |

Global rules:
- The **current prompt attempt is never lost** — it is either answered in place, delegated
  with full context, or surfaced to the Executor exactly as failures surface today.
- **Per-attempt reliability budget:** ≤ 2 in-place actions + 1 delegation before
  surfacing. **Per-run ceiling:** ≤ 12 recovery actions (config) — beyond that the run
  pauses at checkpoint in `WAITING_FOR_PROVIDER` (thrash guard).
- `CancelledError` bypasses everything, always (user intent is sacred).

## 9. Configuration (shape)

```
ReliabilityConfig
  enabled: bool = true                       # master kill-switch → passthrough decorator
  detection:  stall_window_s=45 (adaptive ×p95), dom_probe_timeout_s=5, min_confidence=0.5
  recovery:   per_attempt_budget=2, per_run_budget=12, action_timeouts{…}
  failover:   health_multipliers{…}, affinity_penalty=0.5, allow_suspect=false
  health:     windows{1h,24h}, ewma_alpha=0.2, degraded_after=2, suspect_after=4,
              quarantine_ladder=[30,60,120,300,600]s, offline_after_quarantines=8
  monitor:    cadence{healthy:120, degraded:30, idle:300}s, jitter=0.2
  checkpoint: enabled=true, root=".runs", fsync=inherit(StorageConfig)
```

All defaults land in `dozen/reliability/config.py`, overridable per the same layered
pattern as `ContextConfig` (defaults → file → env), feature-flagged for staged rollout.

## 10. Observability & UI

- Every component publishes typed events (additive `EventType` members:
  `provider.health_changed`, `failure.detected`, `recovery.started/succeeded/failed`,
  `failover.decided`, `checkpoint.written`, `run.resumed`, `provider.needs_human`).
- The run's SSE channel narrates reliability in user language:
  `🩺 Claude timed out — retrying in a fresh chat`,
  `🔀 Delegated to ChatGPT (Claude needs login — click its card to fix)`,
  `💾 Checkpoint: 4/7 subtasks complete`.
- `GET /api/reliability/providers` — registry snapshot for provider cards
  (health chip, confidence, last failure).
- `NEEDS_HUMAN` surfaces as the existing waiting-room pattern: the provider card flips to
  "Needs attention → open window", reusing the login-handoff UX users already know.

## 11. Reliability of the reliability layer

- **Fail-open everywhere:** any exception inside detection/recovery/failover logic is
  caught at the ReliabilityClient boundary; the original provider error then surfaces
  exactly as it does today. The layer can only *add* resilience, never subtract.
- **Watchdog for the layer's blind spot:** worker/driver death (the EPIPE incident) is
  detected by the Health Monitor's supervision tick, not by in-band exceptions.
- **Checkpoint IO failures** degrade to non-checkpointed execution with a warning event —
  never block the run.
- **Kill-switch:** `enabled=false` reduces `ReliabilityClient` to a passthrough; the
  system is bit-for-bit the pre-layer orchestrator.

## 12. Rollout plan (each phase shippable, additive, reversible)

| Phase | Scope | Exit criterion |
|---|---|---|
| **2.1 Foundation** | taxonomy, models, config, registry (static+session view), events | registry API serves live provider snapshot; zero behavior change |
| **2.2 Passive health** | ReliabilityClient records outcomes only (no recovery); HealthManager + metrics; UI health chips | health states track real traffic; still zero behavior change |
| **2.3 Detection & classification** | detectors + classifier wired in-band; FailureEvents logged + narrated | ≥90% of injected failure drills classified correctly |
| **2.4 Recovery** | in-place ladders (retry/refresh/tab/worker/reauth) behind flag | drill: killed tab / stalled generation / login-expiry recover without user action |
| **2.5 Failover** | health-aware selection + attempt transfer | drill: provider hard-down mid-run → run completes on others |
| **2.6 Checkpoints & resume** | run log + resume protocol + WAITING_FOR_PROVIDER | drill: kill server mid-run → resume completes without recomputing done subtasks |
| **2.7 Health Monitor** | background probes, watchdog, adaptive cadence, NEEDS_HUMAN UX | EPIPE-class driver death auto-recovers as single-provider event |

Testing strategy per phase: failure-injection drills (close tab, sever network via
route-blocking, expire session by clearing cookies in the tab, simulate rate-limit banner
via DOM injection) + the existing suite must stay green untouched.

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Providers change DOM faster than selector lists | `DOM_CHANGED` quarantines instead of retry-hammering; selector-drift diagnostics (DOM snapshot events) make updates a data change in one adapter |
| Recovery actions look bot-like to providers (rapid reloads/logins) | budgets + cooldowns + jittered cadence; probes are DOM reads, never sends; rate-limit policy never retries in place |
| Failover mid-conversation loses provider-side chat memory | workers are stateless by design (context injected per attempt — Phase 1.4); provider-side chat history is not relied upon anywhere |
| Checkpoint replay diverges from live semantics | event-sourced records store *outputs*, not intermediate state; replay only marks completion and feeds artifacts — no logic re-execution |
| Thrash loops (recover → fail → recover) | per-run action ceiling + quarantine ladders + WAITING_FOR_PROVIDER pause |
| The layer itself becomes the outage | fail-open decorator, kill-switch, watchdog external to the call path |
| Two orchestrator processes on one profile dir | out of scope v1 (same as today); registry manifest records owner pid as a guard for a future check |

## 14. Open questions (sign-off needed)

1. **Recovery aggressiveness default** — proposed: full auto-recovery ON, auto-failover ON,
   `RESTART_WORKER` ON, with per-provider opt-out in the UI. Confirm?
2. **WAITING_FOR_PROVIDER UX** — when *all* providers are down, pause-and-notify
   (proposed) vs fail-fast?
3. **Checkpoint retention** — proposed: keep last 20 runs or 7 days, whichever is larger;
   completed-run logs compress to the manifest + final answer only.
4. **Probe legal/ToS posture** — probes are passive DOM reads on already-open tabs
   (no synthetic messages). Acceptable posture? (No message-based health pings proposed.)
5. **Confidence weights** (§5.6) — accept proposed weights as calibration defaults?

## 15. Acceptance checklist (architecture)

- [x] All 7 mandated components designed with responsibilities + interfaces
- [x] All 12+ failure types classified with detection evidence per type
- [x] Recovery policy defined for every failure type (retry/refresh/restart/delegate/abort)
- [x] Failover scoring explained with a worked example (Claude → ChatGPT/DeepSeek/Gemini)
- [x] Class, sequence (success/failure/failover), and state diagrams included
- [x] Checkpoint/resume protocol defined on existing storage
- [x] Zero modifications to Planner, Executor, Router, Synthesizer, BrowserManager, adapters
- [x] Browser-tabs-only constraint honored everywhere (no API fallback anywhere)
- [x] Integrates through existing seams (LLMClient decorator, EventBus, Phase 1.2 storage)
- [x] Extension without redesign: detector/action/rule/scoring registries
- [x] Staged, reversible rollout with drills and a kill-switch
