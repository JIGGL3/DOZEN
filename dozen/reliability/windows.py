"""Rolling observation windows (Phase 2.2.2).

Incremental time-bucketed aggregation: each window keeps at most
``_BUCKETS_PER_WINDOW`` coarse buckets; an observation lands in the current
bucket in O(1), expired buckets fall off the front, and reading a window sums
the surviving bucket aggregates (≤ a dozen additions). Observations are never
re-scanned and totals are never recomputed from raw history — old data
expires *naturally* by bucket eviction.

The special ``lifetime`` window is a single never-expiring bucket.

P95 latency uses a bounded per-bucket sample reservoir (newest-kept), making
it an approximation with an explicit, documented memory cap.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from .types import FailureType

_BUCKETS_PER_WINDOW = 12
_MIN_BUCKET_S = 5.0
_LATENCY_SAMPLES_PER_BUCKET = 64


@dataclass(frozen=True)
class Observation:
    """One finished attempt, reduced to what health computation needs."""

    success: bool
    cancelled: bool = False
    latency_ms: Optional[float] = None
    failure_type: Optional[FailureType] = None
    failure_confidence: float = 0.0
    severity_weight: float = 0.0          # severity × confidence, precomputed


@dataclass
class _Bucket:
    start: float                          # monotonic seconds
    count: int = 0
    successes: int = 0
    failures: int = 0
    cancellations: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    latency_samples: deque = field(
        default_factory=lambda: deque(maxlen=_LATENCY_SAMPLES_PER_BUCKET)
    )
    failure_types: Counter = field(default_factory=Counter)
    weighted_failure: float = 0.0         # Σ severity×confidence

    def add(self, obs: Observation) -> None:
        self.count += 1
        if obs.cancelled:
            self.cancellations += 1
        elif obs.success:
            self.successes += 1
        else:
            self.failures += 1
            if obs.failure_type is not None:
                self.failure_types[obs.failure_type.value] += 1
            self.weighted_failure += obs.severity_weight
        if obs.latency_ms is not None:
            self.latency_sum += obs.latency_ms
            self.latency_count += 1
            self.latency_samples.append(obs.latency_ms)


@dataclass(frozen=True)
class WindowTotals:
    """Immutable summed view of one window at one instant."""

    window_s: Optional[float]             # None == lifetime
    count: int = 0
    successes: int = 0
    failures: int = 0
    cancellations: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    latency_samples: tuple[float, ...] = ()
    failure_types: dict[str, int] = field(default_factory=dict)
    weighted_failure: float = 0.0

    @property
    def average_latency_ms(self) -> Optional[float]:
        return (self.latency_sum / self.latency_count) if self.latency_count else None

    @property
    def p95_latency_ms(self) -> Optional[float]:
        if not self.latency_samples:
            return None
        ordered = sorted(self.latency_samples)
        index = max(0, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    @property
    def success_rate(self) -> Optional[float]:
        decided = self.successes + self.failures  # cancellations are neutral
        return (self.successes / decided) if decided else None

    @property
    def failure_rate(self) -> Optional[float]:
        decided = self.successes + self.failures
        return (self.failures / decided) if decided else None

    @property
    def cancellation_rate(self) -> Optional[float]:
        return (self.cancellations / self.count) if self.count else None

    def failure_type_rate(self, failure_type: FailureType) -> float:
        if not self.count:
            return 0.0
        return self.failure_types.get(failure_type.value, 0) / self.count

    @property
    def weighted_failure_load(self) -> float:
        """Confidence-weighted failure score per decided observation."""
        decided = self.successes + self.failures
        return (self.weighted_failure / decided) if decided else 0.0


class RollingWindow:
    """One window. Thread-safe. ``window_s=None`` == lifetime (never expires)."""

    def __init__(self, window_s: Optional[float]) -> None:
        self.window_s = window_s
        self._lock = threading.Lock()
        if window_s is None:
            self._bucket_s = None
            self._buckets: deque[_Bucket] = deque([_Bucket(start=0.0)])
        else:
            self._bucket_s = max(_MIN_BUCKET_S, window_s / _BUCKETS_PER_WINDOW)
            self._buckets = deque()

    def add(self, obs: Observation, now: float) -> None:
        with self._lock:
            self._evict(now)
            bucket = self._current_bucket(now)
            bucket.add(obs)

    def totals(self, now: float) -> WindowTotals:
        with self._lock:
            self._evict(now)
            count = successes = failures = cancellations = 0
            latency_sum, latency_count = 0.0, 0
            samples: list[float] = []
            types: Counter = Counter()
            weighted = 0.0
            for b in self._buckets:
                count += b.count
                successes += b.successes
                failures += b.failures
                cancellations += b.cancellations
                latency_sum += b.latency_sum
                latency_count += b.latency_count
                samples.extend(b.latency_samples)
                types.update(b.failure_types)
                weighted += b.weighted_failure
            return WindowTotals(
                window_s=self.window_s,
                count=count, successes=successes, failures=failures,
                cancellations=cancellations,
                latency_sum=latency_sum, latency_count=latency_count,
                latency_samples=tuple(samples),
                failure_types=dict(types),
                weighted_failure=weighted,
            )

    # ------------------------------------------------------------------ #
    def _current_bucket(self, now: float) -> _Bucket:
        if self._bucket_s is None:
            return self._buckets[0]  # lifetime: the one eternal bucket
        aligned = now - (now % self._bucket_s)
        if not self._buckets or self._buckets[-1].start < aligned:
            self._buckets.append(_Bucket(start=aligned))
        return self._buckets[-1]

    def _evict(self, now: float) -> None:
        if self._bucket_s is None:
            return
        horizon = now - self.window_s  # type: ignore[operator]
        while self._buckets and self._buckets[0].start + self._bucket_s <= horizon:
            self._buckets.popleft()


class WindowSet:
    """The per-provider set: the configured short windows + lifetime."""

    LIFETIME = "lifetime"

    def __init__(self, window_sizes_s: list[float]) -> None:
        self.windows: dict[str, RollingWindow] = {
            _label(size): RollingWindow(size) for size in window_sizes_s
        }
        self.windows[self.LIFETIME] = RollingWindow(None)

    def add(self, obs: Observation, now: float) -> None:
        for window in self.windows.values():
            window.add(obs, now)

    def totals(self, now: float) -> dict[str, WindowTotals]:
        return {label: w.totals(now) for label, w in self.windows.items()}


def _label(size_s: float) -> str:
    if size_s % 60 == 0:
        return f"{int(size_s // 60)}m"
    return f"{int(size_s)}s"
