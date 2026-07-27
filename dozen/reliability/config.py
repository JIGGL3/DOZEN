"""Reliability Layer configuration (SADD-002 §9) — same declared-and-
serializable style as ``ContextConfig``. Defaults here, layering (file/env)
in a later phase. No environment loading in Phase 2.1.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping


def _extra_of(data: Mapping[str, object], known: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in known and k != "schema_version"}


@dataclass
class DetectionConfig:
    SCHEMA_VERSION = 1
    _KNOWN = ("stall_window_s", "dom_probe_timeout_s", "min_confidence")

    stall_window_s: float = 45.0           # adaptive ×p95 applied at runtime (later phase)
    dom_probe_timeout_s: float = 5.0
    min_confidence: float = 0.5            # below this a verdict downgrades to UNKNOWN
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "stall_window_s": self.stall_window_s,
            "dom_probe_timeout_s": self.dom_probe_timeout_s,
            "min_confidence": self.min_confidence,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DetectionConfig":
        return cls(
            stall_window_s=float(data.get("stall_window_s", 45.0) or 0.0),
            dom_probe_timeout_s=float(data.get("dom_probe_timeout_s", 5.0) or 0.0),
            min_confidence=float(data.get("min_confidence", 0.5) or 0.0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.stall_window_s <= 0:
            problems.append("stall_window_s must be > 0")
        if self.dom_probe_timeout_s <= 0:
            problems.append("dom_probe_timeout_s must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            problems.append("min_confidence must be within [0, 1]")
        return problems


@dataclass
class RecoveryConfig:
    SCHEMA_VERSION = 1
    _KNOWN = ("per_attempt_budget", "per_run_budget", "action_timeouts_s")

    per_attempt_budget: int = 2            # in-place actions per attempt (SADD §8)
    per_run_budget: int = 12               # total recovery actions per run (thrash guard)
    # action name (RecoveryAction.value) -> timeout override in seconds
    action_timeouts_s: dict[str, float] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "per_attempt_budget": self.per_attempt_budget,
            "per_run_budget": self.per_run_budget,
            "action_timeouts_s": dict(self.action_timeouts_s),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RecoveryConfig":
        timeouts = data.get("action_timeouts_s")
        return cls(
            per_attempt_budget=int(data.get("per_attempt_budget", 2) or 0),
            per_run_budget=int(data.get("per_run_budget", 12) or 0),
            action_timeouts_s={str(k): float(v) for k, v in timeouts.items()}
            if isinstance(timeouts, dict) else {},
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.per_attempt_budget < 0:
            problems.append("per_attempt_budget must be >= 0")
        if self.per_run_budget < 0:
            problems.append("per_run_budget must be >= 0")
        for name, timeout in self.action_timeouts_s.items():
            if timeout <= 0:
                problems.append(f"action_timeouts_s[{name}] must be > 0")
        return problems


@dataclass
class HealthConfig:
    SCHEMA_VERSION = 1
    _KNOWN = (
        "ewma_alpha", "degraded_after", "suspect_after", "quarantine_ladder_s",
        "offline_after_quarantines", "window_1h_s", "window_24h_s",
        "monitor_cadence_s", "cadence_jitter",
        # Phase 2.2.2 passive-health additions (all additive):
        "window_sizes_s", "scoring_weights", "decay_half_life_s",
        "latency_target_ms", "latency_max_ms", "state_thresholds",
        "needs_human_after", "quarantine_crashes", "quarantine_after",
        "recovery_streak", "min_scoring_observations",
    )

    ewma_alpha: float = 0.2
    degraded_after: int = 2                # consecutive failures → DEGRADED
    suspect_after: int = 4                 # consecutive failures → SUSPECT
    quarantine_ladder_s: list[float] = field(
        default_factory=lambda: [30.0, 60.0, 120.0, 300.0, 600.0]
    )
    offline_after_quarantines: int = 8
    window_1h_s: int = 3600
    window_24h_s: int = 86400
    # HealthState.value -> probe cadence in seconds (SADD §5.8)
    monitor_cadence_s: dict[str, float] = field(
        default_factory=lambda: {
            "healthy": 120.0, "degraded": 30.0, "suspect": 30.0,
            "quarantined": 0.0,           # 0 == probe only at cooldown expiry
            "needs_human": 60.0, "idle": 300.0,
        }
    )
    cadence_jitter: float = 0.2
    # ---- Phase 2.2.2: passive health computation ----------------------- #
    # Rolling window sizes (seconds): short / mid / long. Lifetime is implicit.
    window_sizes_s: list[float] = field(default_factory=lambda: [60.0, 300.0, 1800.0])
    # Scoring weights (must sum to 1.0) — documented in scoring.py.
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            "success": 0.45, "latency": 0.15, "failure": 0.25,
            "trend": 0.10, "streak": 0.05,
        }
    )
    decay_half_life_s: float = 600.0       # failure influence halves every 10min
    latency_target_ms: float = 20_000.0    # web UIs are slow; ≤20s is "good"
    latency_max_ms: float = 120_000.0      # ≥2min averages score zero
    # Score floors per state (score >= threshold keeps/earns the state).
    state_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "healthy": 0.70, "degraded": 0.45, "suspect": 0.20,
            # below suspect floor -> quarantined
        }
    )
    needs_human_after: int = 2             # login/captcha failures in mid window
    quarantine_crashes: int = 3            # crash-family failures in mid window
    quarantine_after: int = 8              # consecutive failures hard stop
    recovery_streak: int = 3               # successes to fully recover
    min_scoring_observations: int = 3      # below this, state stays lenient
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "ewma_alpha": self.ewma_alpha,
            "degraded_after": self.degraded_after,
            "suspect_after": self.suspect_after,
            "quarantine_ladder_s": list(self.quarantine_ladder_s),
            "offline_after_quarantines": self.offline_after_quarantines,
            "window_1h_s": self.window_1h_s,
            "window_24h_s": self.window_24h_s,
            "monitor_cadence_s": dict(self.monitor_cadence_s),
            "cadence_jitter": self.cadence_jitter,
            "window_sizes_s": list(self.window_sizes_s),
            "scoring_weights": dict(self.scoring_weights),
            "decay_half_life_s": self.decay_half_life_s,
            "latency_target_ms": self.latency_target_ms,
            "latency_max_ms": self.latency_max_ms,
            "state_thresholds": dict(self.state_thresholds),
            "needs_human_after": self.needs_human_after,
            "quarantine_crashes": self.quarantine_crashes,
            "quarantine_after": self.quarantine_after,
            "recovery_streak": self.recovery_streak,
            "min_scoring_observations": self.min_scoring_observations,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HealthConfig":
        ladder = data.get("quarantine_ladder_s")
        cadence = data.get("monitor_cadence_s")
        sizes = data.get("window_sizes_s")
        weights = data.get("scoring_weights")
        thresholds = data.get("state_thresholds")
        defaults = cls()
        return cls(
            ewma_alpha=float(data.get("ewma_alpha", 0.2) or 0.0),
            degraded_after=int(data.get("degraded_after", 2) or 0),
            suspect_after=int(data.get("suspect_after", 4) or 0),
            quarantine_ladder_s=[float(x) for x in ladder]
            if isinstance(ladder, (list, tuple)) else [30.0, 60.0, 120.0, 300.0, 600.0],
            offline_after_quarantines=int(data.get("offline_after_quarantines", 8) or 0),
            window_1h_s=int(data.get("window_1h_s", 3600) or 0),
            window_24h_s=int(data.get("window_24h_s", 86400) or 0),
            monitor_cadence_s={str(k): float(v) for k, v in cadence.items()}
            if isinstance(cadence, dict) else {},
            cadence_jitter=float(data.get("cadence_jitter", 0.2) or 0.0),
            window_sizes_s=[float(x) for x in sizes]
            if isinstance(sizes, (list, tuple)) else defaults.window_sizes_s,
            scoring_weights={str(k): float(v) for k, v in weights.items()}
            if isinstance(weights, dict) else defaults.scoring_weights,
            decay_half_life_s=float(data.get("decay_half_life_s", 600.0) or 0.0),
            latency_target_ms=float(data.get("latency_target_ms", 20_000.0) or 0.0),
            latency_max_ms=float(data.get("latency_max_ms", 120_000.0) or 0.0),
            state_thresholds={str(k): float(v) for k, v in thresholds.items()}
            if isinstance(thresholds, dict) else defaults.state_thresholds,
            needs_human_after=int(data.get("needs_human_after", 2) or 0),
            quarantine_crashes=int(data.get("quarantine_crashes", 3) or 0),
            quarantine_after=int(data.get("quarantine_after", 8) or 0),
            recovery_streak=int(data.get("recovery_streak", 3) or 0),
            min_scoring_observations=int(data.get("min_scoring_observations", 3) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not 0.0 < self.ewma_alpha <= 1.0:
            problems.append("ewma_alpha must be within (0, 1]")
        if self.degraded_after < 1:
            problems.append("degraded_after must be >= 1")
        if self.suspect_after < self.degraded_after:
            problems.append("suspect_after must be >= degraded_after")
        if not self.quarantine_ladder_s:
            problems.append("quarantine_ladder_s must not be empty")
        if any(x <= 0 for x in self.quarantine_ladder_s):
            problems.append("quarantine_ladder_s entries must be > 0")
        if self.quarantine_ladder_s != sorted(self.quarantine_ladder_s):
            problems.append("quarantine_ladder_s must be non-decreasing")
        if self.offline_after_quarantines < 1:
            problems.append("offline_after_quarantines must be >= 1")
        if self.window_1h_s <= 0 or self.window_24h_s <= 0:
            problems.append("metric windows must be > 0")
        if self.window_24h_s < self.window_1h_s:
            problems.append("window_24h_s must be >= window_1h_s")
        if not 0.0 <= self.cadence_jitter < 1.0:
            problems.append("cadence_jitter must be within [0, 1)")
        for state, seconds in self.monitor_cadence_s.items():
            if seconds < 0:
                problems.append(f"monitor_cadence_s[{state}] must be >= 0")
        # ---- Phase 2.2.2 additions ---------------------------------------- #
        if len(self.window_sizes_s) != 3 or any(s <= 0 for s in self.window_sizes_s):
            problems.append("window_sizes_s must be three positive sizes (short/mid/long)")
        elif self.window_sizes_s != sorted(self.window_sizes_s):
            problems.append("window_sizes_s must be ascending")
        needed = {"success", "latency", "failure", "trend", "streak"}
        if set(self.scoring_weights) != needed:
            problems.append(f"scoring_weights must define exactly {sorted(needed)}")
        elif abs(sum(self.scoring_weights.values()) - 1.0) > 1e-6:
            problems.append("scoring_weights must sum to 1.0")
        if self.decay_half_life_s <= 0:
            problems.append("decay_half_life_s must be > 0")
        if not 0 < self.latency_target_ms < self.latency_max_ms:
            problems.append("latency thresholds must satisfy 0 < target < max")
        for name in ("healthy", "degraded", "suspect"):
            if name not in self.state_thresholds:
                problems.append(f"state_thresholds missing {name!r}")
        floors = self.state_thresholds
        if all(k in floors for k in ("healthy", "degraded", "suspect")) and not (
            floors["healthy"] > floors["degraded"] > floors["suspect"] >= 0
        ):
            problems.append("state_thresholds must be strictly descending and >= 0")
        for name in ("needs_human_after", "quarantine_crashes",
                     "quarantine_after", "recovery_streak"):
            if int(getattr(self, name)) < 1:
                problems.append(f"{name} must be >= 1")
        if self.min_scoring_observations < 0:
            problems.append("min_scoring_observations must be >= 0")
        return problems


@dataclass
class FailoverConfig:
    SCHEMA_VERSION = 1
    _KNOWN = ("health_multipliers", "affinity_penalty", "allow_suspect")

    # HealthState.value -> routing weight multiplier (SADD §5.5)
    health_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "healthy": 1.0, "degraded": 0.7, "suspect": 0.35,
            "recovering": 0.0, "quarantined": 0.0, "needs_human": 0.0,
            "offline": 0.0, "unknown": 0.5,
        }
    )
    affinity_penalty: float = 0.5          # provider already failed THIS subtask
    allow_suspect: bool = False            # route to SUSPECT only if flag set
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "health_multipliers": dict(self.health_multipliers),
            "affinity_penalty": self.affinity_penalty,
            "allow_suspect": self.allow_suspect,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FailoverConfig":
        multipliers = data.get("health_multipliers")
        return cls(
            health_multipliers={str(k): float(v) for k, v in multipliers.items()}
            if isinstance(multipliers, dict) else cls().health_multipliers,
            affinity_penalty=float(data.get("affinity_penalty", 0.5) or 0.0),
            allow_suspect=bool(data.get("allow_suspect", False)),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not 0.0 <= self.affinity_penalty <= 1.0:
            problems.append("affinity_penalty must be within [0, 1]")
        for state, mult in self.health_multipliers.items():
            if not 0.0 <= mult <= 1.0:
                problems.append(f"health_multipliers[{state}] must be within [0, 1]")
        return problems


@dataclass
class CheckpointConfig:
    SCHEMA_VERSION = 1
    _KNOWN = ("enabled", "root_path", "fsync", "retention_runs", "retention_days")

    enabled: bool = True
    root_path: str = ".runs"
    fsync: bool = True
    retention_runs: int = 20
    retention_days: int = 7
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "enabled": self.enabled,
            "root_path": self.root_path,
            "fsync": self.fsync,
            "retention_runs": self.retention_runs,
            "retention_days": self.retention_days,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CheckpointConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            root_path=str(data.get("root_path", ".runs")),
            fsync=bool(data.get("fsync", True)),
            retention_runs=int(data.get("retention_runs", 20) or 0),
            retention_days=int(data.get("retention_days", 7) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.root_path.strip():
            problems.append("root_path must be non-empty")
        if self.retention_runs < 1:
            problems.append("retention_runs must be >= 1")
        if self.retention_days < 1:
            problems.append("retention_days must be >= 1")
        return problems


@dataclass
class ObservabilityConfig:
    """Phase 2.1.3 debug/observability knobs. Everything defaults OFF."""

    SCHEMA_VERSION = 1
    _KNOWN = (
        "enable_debug_api", "enable_attempt_logging",
        "max_attempts_returned", "statistics_cache_seconds",
    )

    enable_debug_api: bool = False
    enable_attempt_logging: bool = False
    max_attempts_returned: int = 100
    statistics_cache_seconds: float = 0.0    # 0 == no caching
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "enable_debug_api": self.enable_debug_api,
            "enable_attempt_logging": self.enable_attempt_logging,
            "max_attempts_returned": self.max_attempts_returned,
            "statistics_cache_seconds": self.statistics_cache_seconds,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ObservabilityConfig":
        return cls(
            enable_debug_api=bool(data.get("enable_debug_api", False)),
            enable_attempt_logging=bool(data.get("enable_attempt_logging", False)),
            max_attempts_returned=int(data.get("max_attempts_returned", 100) or 0),
            statistics_cache_seconds=float(data.get("statistics_cache_seconds", 0.0) or 0.0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.max_attempts_returned < 1:
            problems.append("max_attempts_returned must be >= 1")
        if self.statistics_cache_seconds < 0:
            problems.append("statistics_cache_seconds must be >= 0")
        return problems


@dataclass
class ReliabilityConfig:
    """Aggregate configuration of the whole Reliability Layer."""

    SCHEMA_VERSION = 1
    _KNOWN = ("enabled", "detection", "recovery", "health", "failover",
              "checkpoint", "observability")

    enabled: bool = True                   # master kill-switch (SADD §11)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    failover: FailoverConfig = field(default_factory=FailoverConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def defaults(cls) -> "ReliabilityConfig":
        return cls()

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "enabled": self.enabled,
            "detection": self.detection.to_dict(),
            "recovery": self.recovery.to_dict(),
            "health": self.health.to_dict(),
            "failover": self.failover.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "observability": self.observability.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReliabilityConfig":
        def sub(key: str, model_cls: type, default: object) -> object:
            raw = data.get(key)
            return model_cls.from_dict(raw) if isinstance(raw, dict) else default  # type: ignore[attr-defined]

        return cls(
            enabled=bool(data.get("enabled", True)),
            detection=sub("detection", DetectionConfig, DetectionConfig()),      # type: ignore[arg-type]
            recovery=sub("recovery", RecoveryConfig, RecoveryConfig()),          # type: ignore[arg-type]
            health=sub("health", HealthConfig, HealthConfig()),                  # type: ignore[arg-type]
            failover=sub("failover", FailoverConfig, FailoverConfig()),          # type: ignore[arg-type]
            checkpoint=sub("checkpoint", CheckpointConfig, CheckpointConfig()),  # type: ignore[arg-type]
            observability=sub("observability", ObservabilityConfig, ObservabilityConfig()),  # type: ignore[arg-type]
            extra=_extra_of(data, cls._KNOWN),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "ReliabilityConfig":
        return cls.from_dict(json.loads(payload))

    def validate(self) -> list[str]:
        problems: list[str] = []
        problems.extend(self.detection.validate())
        problems.extend(self.recovery.validate())
        problems.extend(self.health.validate())
        problems.extend(self.failover.validate())
        problems.extend(self.checkpoint.validate())
        problems.extend(self.observability.validate())
        return problems
