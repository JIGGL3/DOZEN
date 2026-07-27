"""Provider metrics (Phase 2.2.2) — every mandated rate, computed from
window totals. Pure functions and frozen DTOs; no state, no I/O.

Severity table: how much one failure of each type hurts, before confidence
weighting. Grounded in SADD-002 §7/§8 — failures that take the provider out
entirely (crashes, walls needing humans) weigh most; soft/transient ones
least. ``severity × classification-confidence`` is the "confidence-weighted
failure score" every window accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import FailureType
from .windows import WindowTotals

# One failure's weight before confidence scaling (0..1, documented above).
FAILURE_SEVERITY: dict[FailureType, float] = {
    FailureType.BROWSER_CRASH: 1.00,
    FailureType.TAB_CLOSED: 0.90,
    FailureType.CAPTCHA: 0.85,
    FailureType.LOGIN_REQUIRED: 0.85,
    FailureType.DOM_CHANGED: 0.75,
    FailureType.NETWORK_ERROR: 0.70,
    FailureType.RATE_LIMIT: 0.60,
    FailureType.GENERATION_STALLED: 0.60,
    FailureType.UNEXPECTED_UI: 0.60,
    FailureType.TIMEOUT: 0.50,
    FailureType.OUTPUT_CORRUPTED: 0.50,
    FailureType.UNKNOWN: 0.50,
    FailureType.MODEL_BUSY: 0.40,
    FailureType.PROMPT_REJECTED: 0.30,
}


def severity_weight(failure_type: FailureType, confidence: float) -> float:
    """severity × confidence — a low-confidence verdict hurts less."""
    base = FAILURE_SEVERITY.get(failure_type, 0.5)
    return base * max(0.0, min(1.0, confidence))


@dataclass(frozen=True)
class WindowMetrics:
    """All mandated metrics for ONE window, JSON-safe."""

    window: str
    observations: int
    success_rate: Optional[float]
    failure_rate: Optional[float]
    cancellation_rate: Optional[float]
    average_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    timeout_rate: float
    generation_stall_rate: float
    dom_change_rate: float
    rate_limit_frequency: float
    login_failure_frequency: float
    captcha_frequency: float
    network_failure_frequency: float
    browser_crash_frequency: float
    tab_closed_frequency: float
    prompt_rejection_frequency: float
    unknown_failure_frequency: float
    weighted_failure_score: float          # Σ severity×confidence per decided obs
    availability: Optional[float]          # rolling availability == success rate
    failure_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_totals(cls, window: str, totals: WindowTotals) -> "WindowMetrics":
        return cls(
            window=window,
            observations=totals.count,
            success_rate=totals.success_rate,
            failure_rate=totals.failure_rate,
            cancellation_rate=totals.cancellation_rate,
            average_latency_ms=totals.average_latency_ms,
            p95_latency_ms=totals.p95_latency_ms,
            timeout_rate=totals.failure_type_rate(FailureType.TIMEOUT),
            generation_stall_rate=totals.failure_type_rate(FailureType.GENERATION_STALLED),
            dom_change_rate=totals.failure_type_rate(FailureType.DOM_CHANGED),
            rate_limit_frequency=totals.failure_type_rate(FailureType.RATE_LIMIT),
            login_failure_frequency=totals.failure_type_rate(FailureType.LOGIN_REQUIRED),
            captcha_frequency=totals.failure_type_rate(FailureType.CAPTCHA),
            network_failure_frequency=totals.failure_type_rate(FailureType.NETWORK_ERROR),
            browser_crash_frequency=totals.failure_type_rate(FailureType.BROWSER_CRASH),
            tab_closed_frequency=totals.failure_type_rate(FailureType.TAB_CLOSED),
            prompt_rejection_frequency=totals.failure_type_rate(FailureType.PROMPT_REJECTED),
            unknown_failure_frequency=totals.failure_type_rate(FailureType.UNKNOWN),
            weighted_failure_score=totals.weighted_failure_load,
            availability=totals.success_rate,
            failure_counts=dict(totals.failure_types),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "observations": self.observations,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "cancellation_rate": self.cancellation_rate,
            "average_latency_ms": self.average_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "timeout_rate": self.timeout_rate,
            "generation_stall_rate": self.generation_stall_rate,
            "dom_change_rate": self.dom_change_rate,
            "rate_limit_frequency": self.rate_limit_frequency,
            "login_failure_frequency": self.login_failure_frequency,
            "captcha_frequency": self.captcha_frequency,
            "network_failure_frequency": self.network_failure_frequency,
            "browser_crash_frequency": self.browser_crash_frequency,
            "tab_closed_frequency": self.tab_closed_frequency,
            "prompt_rejection_frequency": self.prompt_rejection_frequency,
            "unknown_failure_frequency": self.unknown_failure_frequency,
            "weighted_failure_score": self.weighted_failure_score,
            "availability": self.availability,
            "failure_counts": dict(self.failure_counts),
        }
