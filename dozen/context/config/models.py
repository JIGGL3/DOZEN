"""Strongly typed configuration for the context package.

Every tunable the SADD names (window policy, summary triggers, estimator
ratios, storage paths, pipeline stages, feature flags) lives here as one
declared, serializable policy surface. Layering (defaults -> file -> per-
request override) is applied by later phases; this module defines the shapes
and safe defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def _extra_of(data: dict[str, object], known: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in known and k != "schema_version"}


@dataclass
class WindowPolicy:
    """The structured-window fitting policy (SADD §9.6)."""

    SCHEMA_VERSION = 1
    _KNOWN = (
        "name", "reserved_for_response_tokens", "min_verbatim_tail_messages",
        "min_verbatim_tail_tokens", "pinned_head_max_tokens",
        "summarize_evicted_over_tokens",
    )

    name: str = "structured-v1"
    reserved_for_response_tokens: int = 2048
    min_verbatim_tail_messages: int = 4
    min_verbatim_tail_tokens: int = 1500
    pinned_head_max_tokens: int = 3000
    # Evicted spans larger than this are handed to the SummaryManager.
    summarize_evicted_over_tokens: int = 800
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "name": self.name,
            "reserved_for_response_tokens": self.reserved_for_response_tokens,
            "min_verbatim_tail_messages": self.min_verbatim_tail_messages,
            "min_verbatim_tail_tokens": self.min_verbatim_tail_tokens,
            "pinned_head_max_tokens": self.pinned_head_max_tokens,
            "summarize_evicted_over_tokens": self.summarize_evicted_over_tokens,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WindowPolicy":
        return cls(
            name=str(data.get("name", "structured-v1")),
            reserved_for_response_tokens=int(data.get("reserved_for_response_tokens", 2048) or 0),
            min_verbatim_tail_messages=int(data.get("min_verbatim_tail_messages", 4) or 0),
            min_verbatim_tail_tokens=int(data.get("min_verbatim_tail_tokens", 1500) or 0),
            pinned_head_max_tokens=int(data.get("pinned_head_max_tokens", 3000) or 0),
            summarize_evicted_over_tokens=int(data.get("summarize_evicted_over_tokens", 800) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.name:
            problems.append("window policy name must be non-empty")
        for f_name in (
            "reserved_for_response_tokens", "min_verbatim_tail_messages",
            "min_verbatim_tail_tokens", "pinned_head_max_tokens",
            "summarize_evicted_over_tokens",
        ):
            if int(getattr(self, f_name)) < 0:
                problems.append(f"{f_name} must be >= 0")
        return problems


@dataclass
class SummaryPolicy:
    SCHEMA_VERSION = 1
    _KNOWN = (
        "max_level", "target_compression_ratio", "placeholder_template",
        "summary_max_tokens",
    )

    max_level: int = 3
    target_compression_ratio: float = 0.2   # summary ≈ 20% of source tokens
    placeholder_template: str = "[Summary pending for messages {from_id}–{to_id}]"
    summary_max_tokens: int = 1200
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "max_level": self.max_level,
            "target_compression_ratio": self.target_compression_ratio,
            "placeholder_template": self.placeholder_template,
            "summary_max_tokens": self.summary_max_tokens,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SummaryPolicy":
        return cls(
            max_level=int(data.get("max_level", 3) or 1),
            target_compression_ratio=float(data.get("target_compression_ratio", 0.2) or 0.2),
            placeholder_template=str(
                data.get("placeholder_template", "[Summary pending for messages {from_id}–{to_id}]")
            ),
            summary_max_tokens=int(data.get("summary_max_tokens", 1200) or 0),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.max_level < 1:
            problems.append("max_level must be >= 1")
        if not 0.0 < self.target_compression_ratio <= 1.0:
            problems.append("target_compression_ratio must be within (0, 1]")
        if self.summary_max_tokens < 0:
            problems.append("summary_max_tokens must be >= 0")
        if "{from_id}" not in self.placeholder_template or "{to_id}" not in self.placeholder_template:
            problems.append("placeholder_template must contain {from_id} and {to_id}")
        return problems


@dataclass
class EstimatorConfig:
    """Calibrated heuristic defaults (SADD §9.5); exact tokenizers optional."""

    SCHEMA_VERSION = 1
    _KNOWN = ("chars_per_token", "default_chars_per_token", "safety_margin", "enable_exact_tokenizers")

    # content family -> characters-per-token ratio
    chars_per_token: dict[str, float] = field(
        default_factory=lambda: {"prose": 4.0, "code": 3.3, "json": 2.8}
    )
    default_chars_per_token: float = 4.0
    safety_margin: float = 0.10
    enable_exact_tokenizers: bool = False
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "chars_per_token": dict(self.chars_per_token),
            "default_chars_per_token": self.default_chars_per_token,
            "safety_margin": self.safety_margin,
            "enable_exact_tokenizers": self.enable_exact_tokenizers,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EstimatorConfig":
        ratios = data.get("chars_per_token")
        return cls(
            chars_per_token={str(k): float(v) for k, v in ratios.items()}
            if isinstance(ratios, dict) else {"prose": 4.0, "code": 3.3, "json": 2.8},
            default_chars_per_token=float(data.get("default_chars_per_token", 4.0) or 4.0),
            safety_margin=float(data.get("safety_margin", 0.10) or 0.0),
            enable_exact_tokenizers=bool(data.get("enable_exact_tokenizers", False)),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for family, ratio in self.chars_per_token.items():
            if ratio <= 0:
                problems.append(f"chars_per_token[{family}] must be > 0")
        if self.default_chars_per_token <= 0:
            problems.append("default_chars_per_token must be > 0")
        if not 0.0 <= self.safety_margin < 1.0:
            problems.append("safety_margin must be within [0, 1)")
        return problems


@dataclass
class StorageConfig:
    """Filesystem layout knobs for the v1 persistence adapter."""

    SCHEMA_VERSION = 1
    _KNOWN = ("root_path", "fsync_appends", "shard_directories")

    root_path: str = ".conversations"
    fsync_appends: bool = True
    shard_directories: bool = False
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "root_path": self.root_path,
            "fsync_appends": self.fsync_appends,
            "shard_directories": self.shard_directories,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StorageConfig":
        return cls(
            root_path=str(data.get("root_path", ".conversations")),
            fsync_appends=bool(data.get("fsync_appends", True)),
            shard_directories=bool(data.get("shard_directories", False)),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        return [] if self.root_path.strip() else ["root_path must be non-empty"]


@dataclass
class PersistenceConfig:
    """Which persistence adapter is active + its storage settings."""

    SCHEMA_VERSION = 1
    _KNOWN = ("backend", "storage")

    backend: str = "filesystem"
    storage: StorageConfig = field(default_factory=StorageConfig)
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "backend": self.backend,
            "storage": self.storage.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PersistenceConfig":
        storage = data.get("storage")
        return cls(
            backend=str(data.get("backend", "filesystem")),
            storage=StorageConfig.from_dict(storage) if isinstance(storage, dict) else StorageConfig(),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems = [] if self.backend.strip() else ["backend must be non-empty"]
        problems.extend(self.storage.validate())
        return problems


@dataclass
class PipelineConfig:
    SCHEMA_VERSION = 1
    _KNOWN = ("stage_order", "fail_fast")

    stage_order: list[str] = field(
        default_factory=lambda: ["load", "assemble", "memory", "estimate", "fit", "render"]
    )
    fail_fast: bool = True
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "stage_order": list(self.stage_order),
            "fail_fast": self.fail_fast,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PipelineConfig":
        order = data.get("stage_order")
        return cls(
            stage_order=[str(s) for s in order] if isinstance(order, list) else
            ["load", "assemble", "memory", "estimate", "fit", "render"],
            fail_fast=bool(data.get("fail_fast", True)),
            extra=_extra_of(data, cls._KNOWN),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.stage_order:
            problems.append("stage_order must not be empty")
        if len(set(self.stage_order)) != len(self.stage_order):
            problems.append("stage_order must not contain duplicates")
        return problems


@dataclass
class ContextConfig:
    """The aggregate configuration of the whole context package."""

    SCHEMA_VERSION = 1
    _KNOWN = ("window", "summary", "estimator", "persistence", "pipeline", "feature_flags")

    window: WindowPolicy = field(default_factory=WindowPolicy)
    summary: SummaryPolicy = field(default_factory=SummaryPolicy)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {"memory": False, "exact_tokenizers": False}
    )
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def defaults(cls) -> "ContextConfig":
        return cls()

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "window": self.window.to_dict(),
            "summary": self.summary.to_dict(),
            "estimator": self.estimator.to_dict(),
            "persistence": self.persistence.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "feature_flags": dict(self.feature_flags),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextConfig":
        def sub(key: str, model_cls: type, default: object) -> object:
            raw = data.get(key)
            return model_cls.from_dict(raw) if isinstance(raw, dict) else default  # type: ignore[attr-defined]

        flags = data.get("feature_flags")
        return cls(
            window=sub("window", WindowPolicy, WindowPolicy()),           # type: ignore[arg-type]
            summary=sub("summary", SummaryPolicy, SummaryPolicy()),       # type: ignore[arg-type]
            estimator=sub("estimator", EstimatorConfig, EstimatorConfig()),  # type: ignore[arg-type]
            persistence=sub("persistence", PersistenceConfig, PersistenceConfig()),  # type: ignore[arg-type]
            pipeline=sub("pipeline", PipelineConfig, PipelineConfig()),   # type: ignore[arg-type]
            feature_flags={str(k): bool(v) for k, v in flags.items()} if isinstance(flags, dict)
            else {"memory": False, "exact_tokenizers": False},
            extra=_extra_of(data, cls._KNOWN),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "ContextConfig":
        return cls.from_dict(json.loads(payload))

    def validate(self) -> list[str]:
        problems: list[str] = []
        problems.extend(self.window.validate())
        problems.extend(self.summary.validate())
        problems.extend(self.estimator.validate())
        problems.extend(self.persistence.validate())
        problems.extend(self.pipeline.validate())
        return problems
