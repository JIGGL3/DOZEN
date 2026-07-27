"""PassiveHealthManager (Phase 2.2.2) — health is COMPUTED, never USED.

Consumes finished ExecutionAttempts (already classified by Phase 2.2.1),
maintains per-provider rolling windows, scores, and a deterministic state
machine. Nothing here retries, probes, routes, or influences execution —
the only outputs are immutable records, query results and events.

State machine (deterministic; every transition carries an explanation):

    UNKNOWN ──success──► HEALTHY ◄────────────── streak+score ──┐
       │                    │ failures/score                    │
       │                    ▼                                   │
       │                DEGRADED ──more──► SUSPECT ──worse──► QUARANTINED
       │                    ▲                │   ▲                │
       └──failure──► (lenient: capped        │   │            success
                      at DEGRADED until      │   └── decay ──┐    │
                      min observations)      │               │    ▼
    NEEDS_HUMAN ◄── login/captcha ×N ────────┘          RECOVERING
       │  (never auto-lifts; human_resolved() or a success → RECOVERING)
       └──────────────── success ───────────────────────────────┘

Rule evaluation order on a FAILURE observation (first match wins):
  1. login/captcha failures in the mid window ≥ needs_human_after → NEEDS_HUMAN
  2. crash-family failures in the mid window ≥ quarantine_crashes → QUARANTINED
  3. consecutive failures ≥ quarantine_after                      → QUARANTINED
  4. score < suspect floor                                        → QUARANTINED
  5. consecutive ≥ suspect_after  OR score < degraded floor       → SUSPECT
  6. consecutive ≥ degraded_after OR score < healthy floor        → DEGRADED
  (providers with fewer than min_scoring_observations decided attempts are
   capped at DEGRADED by rules 4-6 — one early blip never quarantines; the
   explicit counters of rules 1-2 still apply.)

On a SUCCESS observation:
  QUARANTINED/NEEDS_HUMAN/SUSPECT → RECOVERING; RECOVERING/DEGRADED with a
  success streak ≥ recovery_streak → HEALTHY (score permitting); UNKNOWN →
  HEALTHY.

Decay recovery (no observations needed): on READ, if the decayed score has
climbed past a better state's floor and at least one half-life has passed
since the last observation, the state lifts ONE level
(QUARANTINED→RECOVERING, SUSPECT→DEGRADED, DEGRADED→HEALTHY). NEEDS_HUMAN
never lifts by decay — a captcha does not solve itself.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..context.domain.types import ProviderId, Timestamp
from .clock import ReliabilityClock, SystemReliabilityClock
from .config import HealthConfig
from .metrics import WindowMetrics, severity_weight
from .models import ExecutionAttempt, HealthRecord, ProbeResult, ProviderStatistics
from .scoring import HealthScorer, NEUTRAL_SCORE, ScoreBreakdown
from .types import AttemptStatus, FailureType, HealthState
from .windows import Observation, WindowSet, WindowTotals

_CRASH_FAMILY = (FailureType.BROWSER_CRASH, FailureType.TAB_CLOSED)
_HUMAN_FAMILY = (FailureType.LOGIN_REQUIRED, FailureType.CAPTCHA)
_RECENT_FAILURES_KEPT = 10
_HISTORY_PER_PROVIDER = 100
_HISTORY_GLOBAL = 500


@dataclass(frozen=True)
class HealthTransition:
    provider: str
    from_state: HealthState
    to_state: HealthState
    reason: str
    score: float
    at: Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "score": self.score,
            "at": self.at,
        }


@dataclass(frozen=True)
class ProviderHealthRecord:
    """The rich immutable snapshot the queries and debug API publish."""

    provider: str
    current_state: HealthState
    overall_score: float
    confidence: float                      # data confidence (observation volume)
    last_updated: Timestamp
    rolling_windows: dict[str, dict[str, object]] = field(default_factory=dict)
    statistics: ProviderStatistics = field(default_factory=ProviderStatistics)
    recent_failures: tuple[dict[str, object], ...] = ()
    current_latency: Optional[float] = None
    average_latency: Optional[float] = None
    failure_rate: Optional[float] = None
    success_rate: Optional[float] = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    current_quarantine_reason: Optional[str] = None
    score_breakdown: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "current_state": self.current_state.value,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "rolling_windows": {k: dict(v) for k, v in self.rolling_windows.items()},
            "statistics": self.statistics.to_dict(),
            "recent_failures": [dict(f) for f in self.recent_failures],
            "current_latency": self.current_latency,
            "average_latency": self.average_latency,
            "failure_rate": self.failure_rate,
            "success_rate": self.success_rate,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "current_quarantine_reason": self.current_quarantine_reason,
            "score_breakdown": dict(self.score_breakdown),
            "metadata": dict(self.metadata),
        }


class _ProviderTrack:
    """Mutable per-provider bookkeeping (guarded by its own lock)."""

    def __init__(self, config: HealthConfig) -> None:
        self.lock = threading.RLock()
        self.windows = WindowSet(config.window_sizes_s)
        self.state: HealthState = HealthState.UNKNOWN
        self.state_reason: str = "no observations yet"
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_mono: Optional[float] = None
        self.last_observation_mono: Optional[float] = None
        self.last_latency_ms: Optional[float] = None
        self.quarantine_reason: Optional[str] = None
        self.recent_failures: deque = deque(maxlen=_RECENT_FAILURES_KEPT)
        self.last_updated: Timestamp = Timestamp("")
        self.lifetime_successes = 0
        self.lifetime_failures = 0
        self.lifetime_cancellations = 0
        self.last_probe: Optional[dict[str, object]] = None


class PassiveHealthManager:
    """Implements the Phase 2.1.1 ``HealthManager`` port, passively."""

    def __init__(
        self,
        config: Optional[HealthConfig] = None,
        clock: Optional[ReliabilityClock] = None,
        events: Optional["ReliabilityEvents"] = None,  # noqa: F821 (lazy import)
        scorer: Optional[HealthScorer] = None,
    ) -> None:
        self.config = config or HealthConfig()
        self.clock = clock or SystemReliabilityClock()
        self.scorer = scorer or HealthScorer(self.config)
        if events is None:
            from .events import ReliabilityEvents
            events = ReliabilityEvents()
        self.events = events
        self._map_lock = threading.Lock()
        self._providers: dict[str, _ProviderTrack] = {}
        self._history: deque[HealthTransition] = deque(maxlen=_HISTORY_GLOBAL)
        self._history_by_provider: dict[str, deque] = {}

    # ------------------------------------------------------------------ #
    # HealthManager port
    # ------------------------------------------------------------------ #
    def record_attempt(self, attempt: ExecutionAttempt) -> Optional[HealthState]:
        if attempt.status not in (
            AttemptStatus.SUCCEEDED, AttemptStatus.FAILED, AttemptStatus.CANCELLED
        ):
            return None
        track = self._track(str(attempt.provider))
        now = self.clock.monotonic()
        obs = self._observation_of(attempt)
        with track.lock:
            track.windows.add(obs, now)
            track.last_observation_mono = now
            track.last_updated = self.clock.now()
            if obs.latency_ms is not None:
                track.last_latency_ms = obs.latency_ms
            if obs.cancelled:
                track.lifetime_cancellations += 1
                return None  # user intent: streaks and state untouched
            if obs.success:
                track.lifetime_successes += 1
                track.consecutive_successes += 1
                track.consecutive_failures = 0
            else:
                track.lifetime_failures += 1
                track.consecutive_failures += 1
                track.consecutive_successes = 0
                track.last_failure_mono = now
                track.recent_failures.append({
                    "failure_type": (obs.failure_type or FailureType.UNKNOWN).value,
                    "confidence": obs.failure_confidence,
                    "at": track.last_updated,
                    "attempt_id": attempt.attempt_id,
                    "message": str(
                        attempt.result_metadata.get("exception_message", "")
                    )[:160],
                })
            return self._evaluate(track, str(attempt.provider), obs, now)

    def record_probe(self, probe: ProbeResult) -> Optional[HealthState]:
        """Passive phase: probes are stored as context, never acted upon."""
        track = self._track(str(probe.provider))
        with track.lock:
            track.last_probe = probe.to_dict()
        return None

    def state(self, provider: ProviderId) -> HealthState:
        track = self._track(str(provider))
        with track.lock:
            self._maybe_decay_lift(track, str(provider))
            return track.state

    def snapshot(self, provider: ProviderId) -> HealthRecord:
        """Port-shaped compact record (the rich one is health_record())."""
        rich = self.health_record(str(provider))
        return HealthRecord(
            provider=ProviderId(rich.provider),
            state=rich.current_state,
            updated_at=rich.last_updated or self.clock.now(),
            confidence=rich.overall_score,
            statistics=rich.statistics,
            state_reason=self._track(str(provider)).state_reason,
        )

    def routable_providers(self) -> list[str]:
        """Query only — NOTHING routes on this in Phase 2.2.2."""
        routable = (HealthState.HEALTHY, HealthState.DEGRADED, HealthState.UNKNOWN)
        return [p for p in sorted(self._known())
                if self.state(ProviderId(p)) in routable]

    def begin_recovery(self, provider: ProviderId) -> None:
        self._explicit_transition(str(provider), HealthState.RECOVERING,
                                  "recovery started (explicit)")

    def end_recovery(self, provider: ProviderId, success: bool) -> None:
        target = HealthState.HEALTHY if success else HealthState.QUARANTINED
        self._explicit_transition(str(provider), target,
                                  f"recovery finished (success={success})")

    def mark_needs_human(self, provider: ProviderId, reason: str) -> None:
        self._explicit_transition(str(provider), HealthState.NEEDS_HUMAN, reason)

    def human_resolved(self, provider: ProviderId) -> None:
        self._explicit_transition(str(provider), HealthState.RECOVERING,
                                  "operator marked the provider resolved")

    # ------------------------------------------------------------------ #
    # Query API (read-only)
    # ------------------------------------------------------------------ #
    def provider_health(self, provider: str) -> ProviderHealthRecord:
        return self.health_record(provider)

    def all_health(self) -> list[ProviderHealthRecord]:
        return [self.health_record(p) for p in sorted(self._known())]

    def healthy_providers(self) -> list[str]:
        return self._in_states((HealthState.HEALTHY,))

    def degraded_providers(self) -> list[str]:
        return self._in_states((HealthState.DEGRADED, HealthState.SUSPECT))

    def quarantined_providers(self) -> list[str]:
        return self._in_states((HealthState.QUARANTINED, HealthState.NEEDS_HUMAN))

    def health_history(
        self, provider: Optional[str] = None, limit: int = 50
    ) -> list[HealthTransition]:
        """Newest first."""
        with self._map_lock:
            source = (self._history_by_provider.get(provider, deque())
                      if provider else self._history)
            items = list(source)
        return list(reversed(items))[:max(0, limit)]

    def health_statistics(self) -> dict[str, object]:
        records = self.all_health()
        by_state: dict[str, int] = {}
        for r in records:
            by_state[r.current_state.value] = by_state.get(r.current_state.value, 0) + 1
        scores = [r.overall_score for r in records]
        return {
            "providers": len(records),
            "by_state": by_state,
            "average_score": (sum(scores) / len(scores)) if scores else None,
            "transitions_recorded": len(self._history),
        }

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def health_record(self, provider: str) -> ProviderHealthRecord:
        track = self._track(provider)
        now = self.clock.monotonic()
        with track.lock:
            self._maybe_decay_lift(track, provider)
            totals = track.windows.totals(now)
            short, mid, long_, lifetime = self._ordered(totals)
            breakdown = self._score(track, now)
            decided = lifetime.successes + lifetime.failures
            return ProviderHealthRecord(
                provider=provider,
                current_state=track.state,
                overall_score=breakdown.score,
                confidence=min(1.0, decided / 20.0),
                last_updated=track.last_updated or self.clock.now(),
                rolling_windows={
                    label: WindowMetrics.from_totals(label, t).to_dict()
                    for label, t in totals.items()
                },
                statistics=ProviderStatistics(
                    attempts_1h=lifetime.count,          # windows differ by design;
                    successes_1h=lifetime.successes,     # lifetime fills the port model
                    attempts_24h=lifetime.count,
                    successes_24h=lifetime.successes,
                    latency_ewma_ms=mid.average_latency_ms or 0.0,
                    latency_p95_ms=mid.p95_latency_ms or 0.0,
                    consecutive_failures=track.consecutive_failures,
                    failure_histogram=dict(lifetime.failure_types),
                    last_success_at=None, last_failure_at=None,
                ),
                recent_failures=tuple(dict(f) for f in track.recent_failures),
                current_latency=track.last_latency_ms,
                average_latency=mid.average_latency_ms,
                failure_rate=mid.failure_rate,
                success_rate=mid.success_rate,
                consecutive_successes=track.consecutive_successes,
                consecutive_failures=track.consecutive_failures,
                current_quarantine_reason=track.quarantine_reason,
                score_breakdown=breakdown.to_dict(),
                metadata={"last_probe": track.last_probe,
                          "state_reason": track.state_reason,
                          "lifetime_cancellations": track.lifetime_cancellations},
            )

    def emit_snapshot(self, provider: str) -> ProviderHealthRecord:
        record = self.health_record(provider)
        self._emit("health_snapshot", provider, record.current_state,
                   record.current_state, record.overall_score,
                   "on-demand snapshot")
        return record

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _observation_of(self, attempt: ExecutionAttempt) -> Observation:
        if attempt.status is AttemptStatus.CANCELLED:
            return Observation(success=False, cancelled=True,
                               latency_ms=attempt.latency_ms)
        if attempt.status is AttemptStatus.SUCCEEDED:
            return Observation(success=True, latency_ms=attempt.latency_ms)
        failure_type = FailureType.UNKNOWN
        confidence = 0.5
        if attempt.failures:
            failure_type = attempt.failures[-1].failure_type
            confidence = attempt.failures[-1].confidence
        return Observation(
            success=False, latency_ms=attempt.latency_ms,
            failure_type=failure_type, failure_confidence=confidence,
            severity_weight=severity_weight(failure_type, confidence),
        )

    def _ordered(self, totals: dict[str, WindowTotals]):
        labels = [label for label in totals if label != WindowSet.LIFETIME]
        labels.sort(key=lambda l: totals[l].window_s or 0)
        short, mid, long_ = (totals[l] for l in labels[:3])
        return short, mid, long_, totals[WindowSet.LIFETIME]

    def _score(self, track: _ProviderTrack, now: float) -> ScoreBreakdown:
        totals = track.windows.totals(now)
        short, mid, long_, lifetime = self._ordered(totals)
        since_failure = (now - track.last_failure_mono
                         if track.last_failure_mono is not None else None)
        return self.scorer.score(short, mid, long_, lifetime,
                                 track.consecutive_failures, since_failure)

    def _evaluate(
        self, track: _ProviderTrack, provider: str, obs: Observation, now: float
    ) -> Optional[HealthState]:
        cfg = self.config
        breakdown = self._score(track, now)
        score = breakdown.score
        totals = track.windows.totals(now)
        _, mid, _, lifetime = self._ordered(totals)
        decided = lifetime.successes + lifetime.failures
        floors = cfg.state_thresholds
        old = track.state

        if obs.success:
            streak = track.consecutive_successes
            if old is HealthState.UNKNOWN:
                new, reason = HealthState.HEALTHY, "first successful attempt"
            elif old in (HealthState.QUARANTINED, HealthState.NEEDS_HUMAN,
                         HealthState.SUSPECT):
                new, reason = HealthState.RECOVERING, "success after bad state"
            elif old in (HealthState.RECOVERING, HealthState.DEGRADED):
                if streak >= cfg.recovery_streak and score >= floors["healthy"]:
                    new, reason = HealthState.HEALTHY, (
                        f"success streak {streak} with score {score:.2f}"
                    )
                else:
                    new, reason = old, track.state_reason
            else:
                new, reason = HealthState.HEALTHY, "sustained success"
        else:
            human_hits = sum(mid.failure_types.get(t.value, 0) for t in _HUMAN_FAMILY)
            crash_hits = sum(mid.failure_types.get(t.value, 0) for t in _CRASH_FAMILY)
            consec = track.consecutive_failures
            if human_hits >= cfg.needs_human_after:
                new = HealthState.NEEDS_HUMAN
                reason = f"{human_hits} login/captcha failures in the mid window"
            elif crash_hits >= cfg.quarantine_crashes:
                new = HealthState.QUARANTINED
                reason = f"{crash_hits} crash-family failures in the mid window"
            elif consec >= cfg.quarantine_after:
                new = HealthState.QUARANTINED
                reason = f"{consec} consecutive failures"
            elif decided >= cfg.min_scoring_observations and score < floors["suspect"]:
                new = HealthState.QUARANTINED
                reason = f"score {score:.2f} below suspect floor {floors['suspect']}"
            elif consec >= cfg.suspect_after or (
                decided >= cfg.min_scoring_observations and score < floors["degraded"]
            ):
                new = HealthState.SUSPECT
                reason = (f"{consec} consecutive failures" if consec >= cfg.suspect_after
                          else f"score {score:.2f} below degraded floor")
            elif consec >= cfg.degraded_after or score < floors["healthy"]:
                new = HealthState.DEGRADED
                reason = (f"{consec} consecutive failures" if consec >= cfg.degraded_after
                          else f"score {score:.2f} below healthy floor")
            else:
                new, reason = old if old is not HealthState.UNKNOWN else HealthState.DEGRADED, \
                    "isolated failure"
            # NEEDS_HUMAN is sticky against non-human failures:
            if old is HealthState.NEEDS_HUMAN and new not in (
                HealthState.NEEDS_HUMAN, HealthState.QUARANTINED
            ):
                new, reason = old, track.state_reason

        if new is not old:
            self._apply_transition(track, provider, old, new, reason, score)
            return new
        return None

    def _maybe_decay_lift(self, track: _ProviderTrack, provider: str) -> None:
        """Idle recovery: one level up when the decayed score has earned it."""
        if track.state in (HealthState.UNKNOWN, HealthState.HEALTHY,
                           HealthState.RECOVERING, HealthState.NEEDS_HUMAN):
            return
        now = self.clock.monotonic()
        if track.last_observation_mono is None:
            return
        idle = now - track.last_observation_mono
        if idle < self.config.decay_half_life_s:
            return  # give decay at least one half-life before lifting
        score = self._score(track, now).score
        floors = self.config.state_thresholds
        ladder = {
            HealthState.QUARANTINED: (HealthState.RECOVERING, floors["suspect"]),
            HealthState.SUSPECT: (HealthState.DEGRADED, floors["degraded"]),
            HealthState.DEGRADED: (HealthState.HEALTHY, floors["healthy"]),
        }
        target, floor = ladder[track.state]
        if score >= floor:
            self._apply_transition(
                track, provider, track.state, target,
                f"decay recovery: score {score:.2f} ≥ {floor} after "
                f"{idle:.0f}s idle", score,
            )

    def _apply_transition(
        self, track: _ProviderTrack, provider: str,
        old: HealthState, new: HealthState, reason: str, score: float,
    ) -> None:
        track.state = new
        track.state_reason = reason
        track.last_updated = self.clock.now()
        if new is HealthState.QUARANTINED:
            track.quarantine_reason = reason
        elif new in (HealthState.HEALTHY, HealthState.RECOVERING):
            track.quarantine_reason = None
        transition = HealthTransition(
            provider=provider, from_state=old, to_state=new,
            reason=reason, score=score, at=track.last_updated,
        )
        with self._map_lock:
            self._history.append(transition)
            self._history_by_provider.setdefault(
                provider, deque(maxlen=_HISTORY_PER_PROVIDER)
            ).append(transition)
        self._emit_for(transition)

    def _explicit_transition(self, provider: str, new: HealthState, reason: str) -> None:
        track = self._track(provider)
        with track.lock:
            if track.state is new:
                return
            score = self._score(track, self.clock.monotonic()).score
            self._apply_transition(track, provider, track.state, new, reason, score)

    def _emit_for(self, t: HealthTransition) -> None:
        self._emit("provider_health_changed", t.provider, t.from_state,
                   t.to_state, t.score, t.reason)
        if t.to_state in (HealthState.DEGRADED, HealthState.SUSPECT):
            self._emit("provider_degraded", t.provider, t.from_state,
                       t.to_state, t.score, t.reason)
        elif t.to_state in (HealthState.QUARANTINED, HealthState.NEEDS_HUMAN):
            self._emit("provider_quarantined", t.provider, t.from_state,
                       t.to_state, t.score, t.reason)
        elif t.to_state in (HealthState.HEALTHY, HealthState.RECOVERING) and \
                t.from_state not in (HealthState.UNKNOWN, HealthState.HEALTHY):
            self._emit("provider_recovered", t.provider, t.from_state,
                       t.to_state, t.score, t.reason)

    def _emit(self, kind: str, provider: str, old: HealthState,
              new: HealthState, score: float, reason: str) -> None:
        try:
            emit = getattr(self.events, kind)
            emit(provider, old.value, new.value, score, reason)
        except Exception:
            pass  # events are observational; never fatal

    def _track(self, provider: str) -> _ProviderTrack:
        with self._map_lock:
            track = self._providers.get(provider)
            if track is None:
                track = _ProviderTrack(self.config)
                self._providers[provider] = track
            return track

    def _known(self) -> list[str]:
        with self._map_lock:
            return list(self._providers)

    def _in_states(self, states: tuple[HealthState, ...]) -> list[str]:
        return [p for p in sorted(self._known())
                if self.state(ProviderId(p)) in states]


# Process-wide default (mirrors default_recorder()).
_default: Optional[PassiveHealthManager] = None
_default_lock = threading.Lock()


def default_health_manager() -> PassiveHealthManager:
    global _default
    with _default_lock:
        if _default is None:
            _default = PassiveHealthManager()
        return _default
