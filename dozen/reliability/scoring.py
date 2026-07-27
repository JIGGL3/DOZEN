"""HealthScorer (Phase 2.2.2) — deterministic 0..1 provider scoring.

No ML, no randomness. The score is a documented weighted sum of five
components, each itself in [0, 1]:

    score = Ws·S + Wl·L + Wf·F + Wt·T + Wk·K          (weights sum to 1.0)

    S  SUCCESS    success rate in the mid window (5m), falling back to the
                  lifetime rate while the mid window is empty.
    L  LATENCY    1.0 at/below `latency_target_ms`, linearly falling to 0.0
                  at `latency_max_ms` (average latency, mid window).
    F  FAILURE    1 − min(1, weighted_failure_load × decay). The load is the
                  confidence-weighted severity per decided observation
                  (metrics.severity_weight); DECAY (below) makes old failures
                  fade so health recovers with time even while idle.
    T  TREND      0.5 + (success_1m − success_30m) / 2 — above 0.5 means the
                  provider is improving right now, below means worsening.
    K  STREAK     1 / (1 + consecutive_failures): each consecutive failure
                  halves-ish the component; one success fully restores it.

Default weights (HealthConfig.scoring_weights):
    success 0.45 · latency 0.15 · failure 0.25 · trend 0.10 · streak 0.05

DECAY: failure influence is multiplied by 0.5^(elapsed_since_last_failure /
decay_half_life_s). With the default 600 s half-life, a failure loses half
its bite every 10 minutes; after ~5 half-lives it is background noise. This
is why an idle provider recovers naturally with no manual reset.

Neutral prior: with zero decided observations the score is 0.5 (UNKNOWN
territory) — a brand-new provider is neither trusted nor condemned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import HealthConfig
from .windows import WindowTotals

NEUTRAL_SCORE = 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ScoreBreakdown:
    """The score plus every documented component that produced it."""

    score: float
    success_component: float
    latency_component: float
    failure_component: float
    trend_component: float
    streak_component: float
    decay_factor: float
    weights: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "components": {
                "success": self.success_component,
                "latency": self.latency_component,
                "failure": self.failure_component,
                "trend": self.trend_component,
                "streak": self.streak_component,
            },
            "decay_factor": self.decay_factor,
            "weights": dict(self.weights),
            "explanation": self.explanation,
        }


class HealthScorer:
    def __init__(self, config: Optional[HealthConfig] = None) -> None:
        self.config = config or HealthConfig()

    def score(
        self,
        short: WindowTotals,      # 1m
        mid: WindowTotals,        # 5m
        long: WindowTotals,       # 30m
        lifetime: WindowTotals,
        consecutive_failures: int,
        seconds_since_last_failure: Optional[float],
    ) -> ScoreBreakdown:
        cfg = self.config
        weights = dict(cfg.scoring_weights)

        decided = (mid.successes + mid.failures) or (
            lifetime.successes + lifetime.failures
        )
        if decided == 0:
            return ScoreBreakdown(
                score=NEUTRAL_SCORE,
                success_component=NEUTRAL_SCORE, latency_component=1.0,
                failure_component=1.0, trend_component=NEUTRAL_SCORE,
                streak_component=1.0, decay_factor=1.0, weights=weights,
                explanation="no decided observations yet — neutral prior 0.5",
            )

        # S — success
        success = mid.success_rate
        if success is None:
            success = lifetime.success_rate or 0.0

        # L — latency fitness
        avg_latency = mid.average_latency_ms or lifetime.average_latency_ms
        if avg_latency is None or avg_latency <= cfg.latency_target_ms:
            latency = 1.0
        elif avg_latency >= cfg.latency_max_ms:
            latency = 0.0
        else:
            span = cfg.latency_max_ms - cfg.latency_target_ms
            latency = 1.0 - (avg_latency - cfg.latency_target_ms) / span

        # decay — old failures fade with a configurable half-life
        if seconds_since_last_failure is None:
            decay = 0.0  # no failure ever recorded: nothing to decay
        else:
            decay = 0.5 ** (seconds_since_last_failure / cfg.decay_half_life_s)

        # F — confidence-weighted failure load, decayed
        load = max(mid.weighted_failure_load, short.weighted_failure_load)
        failure = _clamp01(1.0 - min(1.0, load) * decay)

        # T — short-term trend vs the long window
        succ_short = short.success_rate
        succ_long = long.success_rate
        if succ_short is None or succ_long is None:
            trend = NEUTRAL_SCORE
        else:
            trend = _clamp01(0.5 + (succ_short - succ_long) / 2.0)

        # K — consecutive-failure penalty
        streak = 1.0 / (1.0 + max(0, consecutive_failures))

        score = _clamp01(
            weights["success"] * success
            + weights["latency"] * latency
            + weights["failure"] * failure
            + weights["trend"] * trend
            + weights["streak"] * streak
        )
        explanation = (
            f"S={success:.2f}·{weights['success']} L={latency:.2f}·{weights['latency']} "
            f"F={failure:.2f}·{weights['failure']} T={trend:.2f}·{weights['trend']} "
            f"K={streak:.2f}·{weights['streak']} (decay={decay:.2f}) -> {score:.3f}"
        )
        return ScoreBreakdown(
            score=score,
            success_component=success, latency_component=latency,
            failure_component=failure, trend_component=trend,
            streak_component=streak, decay_factor=decay,
            weights=weights, explanation=explanation,
        )
