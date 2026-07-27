"""FailureEvidence — the classifier's immutable input (Phase 2.2.1).

Everything the classification engine is allowed to look at, gathered into one
frozen value object. Fields that require browser probing (url, page_title,
active_selectors, visible_text_snippet, http_status, connection flags) are
OPTIONAL — in this phase they arrive only if the caller already has them
(e.g. parsed out of an exception message). No browser is ever touched here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .models import ExecutionAttempt
from .types import ExecutionStage, parse_enum


@dataclass(frozen=True)
class FailureEvidence:
    SCHEMA_VERSION = 1
    _KNOWN = (
        "exception_type", "exception_message", "provider", "model", "url",
        "page_title", "active_selectors", "visible_text_snippet",
        "http_status", "browser_connected", "tab_connected", "prompt_length",
        "response_length", "execution_stage", "latency_ms", "metadata",
        "future_screenshot_path", "future_dom_snapshot", "future_network_trace",
    )

    exception_type: str = ""
    exception_message: str = ""
    provider: str = ""
    model: str = ""
    url: Optional[str] = None
    page_title: Optional[str] = None
    active_selectors: tuple[str, ...] = ()
    visible_text_snippet: Optional[str] = None
    http_status: Optional[int] = None
    browser_connected: Optional[bool] = None      # None == not observed
    tab_connected: Optional[bool] = None
    prompt_length: int = 0
    response_length: int = 0
    execution_stage: ExecutionStage = ExecutionStage.UNKNOWN
    latency_ms: Optional[float] = None
    metadata: dict[str, object] = field(default_factory=dict)
    # Reserved placeholders (populated by later capture phases, never here):
    future_screenshot_path: Optional[str] = None
    future_dom_snapshot: Optional[str] = None
    future_network_trace: Optional[str] = None
    extra: dict[str, object] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_attempt(
        cls,
        attempt: ExecutionAttempt,
        extra_metadata: Optional[Mapping[str, object]] = None,
    ) -> "FailureEvidence":
        """Build evidence from a finished attempt. Pure extraction: exception
        details come from the attempt's result_metadata (put there by the
        recording decorator); nothing is probed."""
        meta: dict[str, object] = dict(attempt.result_metadata)
        if extra_metadata:
            meta.update(extra_metadata)

        def opt_str(key: str) -> Optional[str]:
            value = meta.get(key)
            return str(value) if isinstance(value, str) and value else None

        def opt_bool(key: str) -> Optional[bool]:
            value = meta.get(key)
            return bool(value) if isinstance(value, bool) else None

        status = meta.get("http_status")
        selectors = meta.get("active_selectors")
        return cls(
            exception_type=str(meta.get("exception_type", "")),
            exception_message=str(meta.get("exception_message", "")),
            provider=str(attempt.provider),
            model=attempt.model_name or str(meta.get("model", "")),
            url=opt_str("url"),
            page_title=opt_str("page_title"),
            active_selectors=tuple(str(s) for s in selectors)
            if isinstance(selectors, (list, tuple)) else (),
            visible_text_snippet=opt_str("visible_text_snippet"),
            http_status=int(status) if isinstance(status, (int, float)) else None,
            browser_connected=opt_bool("browser_connected"),
            tab_connected=opt_bool("tab_connected"),
            prompt_length=attempt.prompt_character_count,
            response_length=attempt.response_character_count,
            execution_stage=attempt.execution_stage,
            latency_ms=attempt.latency_ms,
            metadata=meta,
        )

    # ------------------------------------------------------------------ #
    def searchable_text(self) -> str:
        """The lowercased haystack pattern rules match against: exception
        text + page signals, never the prompt itself."""
        parts = [
            self.exception_type, self.exception_message,
            self.url or "", self.page_title or "",
            self.visible_text_snippet or "",
        ]
        return "\n".join(p for p in parts if p).lower()

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "provider": self.provider,
            "model": self.model,
            "url": self.url,
            "page_title": self.page_title,
            "active_selectors": list(self.active_selectors),
            "visible_text_snippet": self.visible_text_snippet,
            "http_status": self.http_status,
            "browser_connected": self.browser_connected,
            "tab_connected": self.tab_connected,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "execution_stage": self.execution_stage.value,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
            "future_screenshot_path": self.future_screenshot_path,
            "future_dom_snapshot": self.future_dom_snapshot,
            "future_network_trace": self.future_network_trace,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FailureEvidence":
        extra = {k: v for k, v in data.items()
                 if k not in cls._KNOWN and k != "schema_version"}
        raw_stage = data.get("execution_stage", "unknown")
        stage = parse_enum(ExecutionStage, raw_stage)
        if stage is ExecutionStage.UNKNOWN and str(raw_stage) != "unknown":
            extra["execution_stage__raw"] = raw_stage
        status = data.get("http_status")
        latency = data.get("latency_ms")
        selectors = data.get("active_selectors")
        meta = data.get("metadata")

        def opt_str(key: str) -> Optional[str]:
            value = data.get(key)
            return str(value) if isinstance(value, str) else None

        def opt_bool(key: str) -> Optional[bool]:
            value = data.get(key)
            return bool(value) if isinstance(value, bool) else None

        return cls(
            exception_type=str(data.get("exception_type", "")),
            exception_message=str(data.get("exception_message", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            url=opt_str("url"),
            page_title=opt_str("page_title"),
            active_selectors=tuple(str(s) for s in selectors)
            if isinstance(selectors, (list, tuple)) else (),
            visible_text_snippet=opt_str("visible_text_snippet"),
            http_status=int(status) if isinstance(status, (int, float)) else None,
            browser_connected=opt_bool("browser_connected"),
            tab_connected=opt_bool("tab_connected"),
            prompt_length=int(data.get("prompt_length", 0) or 0),
            response_length=int(data.get("response_length", 0) or 0),
            execution_stage=stage,
            latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
            metadata=dict(meta) if isinstance(meta, dict) else {},
            future_screenshot_path=opt_str("future_screenshot_path"),
            future_dom_snapshot=opt_str("future_dom_snapshot"),
            future_network_trace=opt_str("future_network_trace"),
            extra=extra,
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.prompt_length < 0 or self.response_length < 0:
            problems.append("lengths must be >= 0")
        if self.latency_ms is not None and self.latency_ms < 0:
            problems.append("latency_ms must be >= 0")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            problems.append("http_status must be a valid HTTP status code")
        return problems
