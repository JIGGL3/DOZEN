"""Active interruption of running browser jobs (production-hardening 5B).

WHY THIS MODULE EXISTS
----------------------
Phase 5A gave every physical browser operation an identity, a caller lifecycle
and owner-checked settlement, but a caller that cancels or times out only stopped
*waiting* — the underlying Playwright generation kept running until it naturally
returned, leaving the provider tab occupied and delaying every later job queued
behind it.

Phase 5B adds the *policy and state* needed to actively interrupt the exact
physically-running browser job, safely, through the owning provider worker. This
module owns:

* the small :class:`InterruptionReason` / :class:`InterruptionOutcome` /
  :class:`StopActionStatus` vocabularies,
* one immutable authoritative :class:`InterruptionPolicy` (every numeric value
  lives here — no unbounded waits or retries),
* a provider-neutral :class:`InterruptionActionResult` (what a single adapter
  stop attempt observed),
* an encapsulated, thread-safe :class:`InterruptionRequest` and its immutable
  public :class:`InterruptionSnapshot`,
* a provider-neutral :class:`CancellationObservation` that composes the existing
  external ``should_cancel`` callback with a browser job's internal interruption
  request.

SCOPE — this module owns *interruption request state and policy only*. It never
touches Playwright, never clicks a stop button, never resolves a page. The actual
browser action lives on the provider adapter (``interrupt_generation``) and is
driven exclusively by the owning provider worker thread in ``browser_manager``.

THREADING CONTRACT
------------------
:class:`InterruptionRequest` guards its mutable state with one internal lock. The
caller thread only *records* a request (``request``); the single provider worker
thread performs at most one stop action, guarded by ``begin_action`` (a
compare-and-set that returns ``True`` at most once). No lock is ever held across
Playwright I/O — this module does no I/O at all.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .browser_jobs import SCHEMA_VERSION, BrowserJobId

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class InterruptionReason(str, Enum):
    """Why an active interruption was requested.

    Deliberately tiny. A provider-operation failure or a provider/Playwright
    timeout is NOT a caller interruption request and never appears here.
    """

    CANCELLED = "cancelled"            # explicit caller cancellation
    CALLER_TIMEOUT = "caller_timeout"  # caller-owned browser-job deadline expired

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class InterruptionOutcome(str, Enum):
    """The terminal outcome of an interruption request."""

    NOT_REQUESTED = "not_requested"                  # no interruption was asked for
    NOT_NEEDED = "not_needed"                         # requested, but no action required
    STOPPED = "stopped"                               # stop action succeeded
    SETTLED_BEFORE_ACTION = "settled_before_action"   # ownership lost before any click
    UNSUPPORTED = "unsupported"                       # provider cannot be actively stopped
    FAILED = "failed"                                 # stop action raised / did not register

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class StopActionStatus(str, Enum):
    """What a single provider-adapter stop attempt physically observed."""

    STOPPED = "stopped"            # generation was active and was successfully stopped
    ALREADY_IDLE = "already_idle"  # generation had already settled; nothing to click
    NO_CONTROL = "no_control"      # no usable stop control present on the page
    SETTLED_BEFORE_ACTION = "settled_before_action"  # exact ownership was lost pre-click
    FAILED = "failed"              # the browser interaction failed
    UNSUPPORTED = "unsupported"    # this provider does not support active stopping

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Request sources are free-form short labels; a small vocabulary keeps them
#: content-free and bounded.
REQUEST_SOURCE_MAX = 64


def _bound(value: Optional[str], limit: int) -> Optional[str]:
    """Bound and flatten a public diagnostic/label; never carries content."""
    if value is None:
        return None
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


# --------------------------------------------------------------------------- #
# Policy — ONE authoritative location for every numeric value.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterruptionPolicy:
    """Immutable authoritative interruption policy.

    Every bound lives here so there is exactly one place to reason about how long
    an interruption can take. No value permits an unbounded wait or retry.
    """

    #: Master switch for active browser interruption.
    active_interruption_enabled: bool = True
    #: Explicit cancellation triggers an active stop action.
    interrupt_on_cancel: bool = True
    #: Caller-owned browser-job timeout triggers an active stop action.
    interrupt_on_caller_timeout: bool = True
    #: Hard cap on stop-action attempts per job. Exactly-once by default.
    max_stop_attempts: int = 1
    #: Bound on locating + clicking the stop control (seconds).
    stop_action_timeout_s: float = 5.0
    #: Bound on confirming quiescence after a successful click (seconds).
    post_stop_grace_s: float = 3.0
    #: Cadence for the bounded quiescence poll (seconds).
    quiescence_poll_interval_s: float = 0.2
    #: Hard cap on any sanitized diagnostic copied into snapshots/events.
    max_diagnostic_length: int = 200
    #: Hard cap on an interruption event reason string.
    max_event_reason_length: int = 120
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "active_interruption_enabled",
            "interrupt_on_cancel",
            "interrupt_on_caller_timeout",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if type(self.max_stop_attempts) is not int or self.max_stop_attempts < 1:
            raise ValueError("max_stop_attempts must be an integer >= 1")
        for name in (
            "stop_action_timeout_s",
            "post_stop_grace_s",
            "quiescence_poll_interval_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.quiescence_poll_interval_s > self.post_stop_grace_s:
            raise ValueError("quiescence_poll_interval_s must not exceed post_stop_grace_s")
        for name in ("max_diagnostic_length", "max_event_reason_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")


#: The single default policy instance — active interruption enabled, exactly one
#: stop attempt, no page reload / tab close / worker restart fallback.
DEFAULT_INTERRUPTION_POLICY = InterruptionPolicy()


# --------------------------------------------------------------------------- #
# Provider-neutral stop-attempt result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterruptionActionResult:
    """The provider-neutral result of ONE adapter stop attempt.

    Distinguishes the four physical cases an adapter must report — active and
    successfully stopped, already settled, no usable stop control, and browser
    interaction failure — plus a default ``UNSUPPORTED`` for providers with no
    stop mechanism. Carries only a bounded, content-free diagnostic.
    """

    status: StopActionStatus
    quiescent: bool = False
    diagnostic: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, StopActionStatus):
            raise TypeError("status must be a StopActionStatus")
        if type(self.quiescent) is not bool:
            raise TypeError("quiescent must be a boolean")
        # Bound defensively; adapters should already pass short strings.
        object.__setattr__(self, "diagnostic", _bound(self.diagnostic, 200))

    @classmethod
    def stopped(cls, *, quiescent: bool, diagnostic: Optional[str] = None) -> "InterruptionActionResult":
        return cls(StopActionStatus.STOPPED, quiescent=quiescent, diagnostic=diagnostic)

    @classmethod
    def already_idle(cls, diagnostic: Optional[str] = None) -> "InterruptionActionResult":
        return cls(StopActionStatus.ALREADY_IDLE, quiescent=True, diagnostic=diagnostic)

    @classmethod
    def no_control(cls, diagnostic: Optional[str] = None) -> "InterruptionActionResult":
        return cls(StopActionStatus.NO_CONTROL, quiescent=False, diagnostic=diagnostic)

    @classmethod
    def failed(cls, diagnostic: Optional[str] = None) -> "InterruptionActionResult":
        return cls(StopActionStatus.FAILED, quiescent=False, diagnostic=diagnostic)

    @classmethod
    def unsupported(cls, diagnostic: Optional[str] = None) -> "InterruptionActionResult":
        return cls(StopActionStatus.UNSUPPORTED, quiescent=False, diagnostic=diagnostic)

    @classmethod
    def settled_before_action(
        cls, diagnostic: Optional[str] = None
    ) -> "InterruptionActionResult":
        return cls(
            StopActionStatus.SETTLED_BEFORE_ACTION,
            quiescent=True,
            diagnostic=diagnostic,
        )


#: How an adapter stop-attempt status maps to the request's terminal outcome.
_STATUS_TO_OUTCOME: dict[StopActionStatus, InterruptionOutcome] = {
    StopActionStatus.STOPPED: InterruptionOutcome.STOPPED,
    StopActionStatus.ALREADY_IDLE: InterruptionOutcome.NOT_NEEDED,
    StopActionStatus.NO_CONTROL: InterruptionOutcome.UNSUPPORTED,
    StopActionStatus.SETTLED_BEFORE_ACTION: InterruptionOutcome.SETTLED_BEFORE_ACTION,
    StopActionStatus.UNSUPPORTED: InterruptionOutcome.UNSUPPORTED,
    StopActionStatus.FAILED: InterruptionOutcome.FAILED,
}


def outcome_for_status(status: StopActionStatus) -> InterruptionOutcome:
    """Map a physical stop-attempt status to the request's terminal outcome."""
    return _STATUS_TO_OUTCOME[status]


# --------------------------------------------------------------------------- #
# Immutable public snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterruptionSnapshot:
    """An immutable, content-free view of one job's interruption state.

    Carries NO prompt text, response body, DOM, cookies or traceback — only
    interruption metadata safe to log, serialize and surface to observability.
    """

    job_id: str
    provider: str
    requested: bool = False
    reason: Optional[str] = None
    requested_at: Optional[float] = None
    source: Optional[str] = None
    physical_interruption_required: bool = False
    started: bool = False
    action_attempted: bool = False
    attempt_count: int = 0
    completed_at: Optional[float] = None
    outcome: str = InterruptionOutcome.NOT_REQUESTED.value
    diagnostic: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        BrowserJobId(self.job_id)
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        for name in (
            "requested",
            "physical_interruption_required",
            "started",
            "action_attempted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.reason is not None:
            InterruptionReason(self.reason)
        InterruptionOutcome(self.outcome)
        if type(self.attempt_count) is not int or self.attempt_count < 0:
            raise ValueError("attempt_count must be a non-negative integer")
        if self.started != self.action_attempted:
            raise ValueError("started and action_attempted disagree")
        if self.action_attempted != (self.attempt_count > 0):
            raise ValueError("action_attempted and attempt_count disagree")
        if self.attempt_count > 1:
            raise ValueError("attempt_count exceeds the Phase 5B exactly-once bound")
        for name in ("requested_at", "completed_at"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} must be a number or null")
        if (
            self.requested
            and (self.reason is None or self.requested_at is None)
        ) or (
            not self.requested
            and (self.reason is not None or self.requested_at is not None)
        ):
            raise ValueError("requested, reason, and requested_at disagree")
        if not self.requested and (
            self.source is not None
            or self.physical_interruption_required
            or self.action_attempted
            or self.completed_at is not None
            or self.outcome != InterruptionOutcome.NOT_REQUESTED.value
            or self.diagnostic is not None
        ):
            raise ValueError("unrequested interruption carries lifecycle state")
        if self.completed_at is not None:
            if not self.requested:
                raise ValueError("completed interruption was not requested")
            if self.completed_at < self.requested_at:
                raise ValueError("completed_at precedes requested_at")
            if self.outcome == InterruptionOutcome.NOT_REQUESTED.value:
                raise ValueError("completed interruption has no outcome")
        for name, value, limit in (
            ("source", self.source, REQUEST_SOURCE_MAX),
            ("diagnostic", self.diagnostic, 200),
        ):
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a string or null")
                if len(value) > limit or "\n" in value or "\r" in value:
                    raise ValueError(f"{name} must be flat and at most {limit} characters")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "requested": self.requested,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "source": self.source,
            "physical_interruption_required": self.physical_interruption_required,
            "started": self.started,
            "action_attempted": self.action_attempted,
            "attempt_count": self.attempt_count,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "diagnostic": self.diagnostic,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterruptionSnapshot":
        if not isinstance(data, dict):
            raise TypeError("interruption snapshot must be a mapping")
        bools: dict[str, bool] = {}
        for name in (
            "requested",
            "physical_interruption_required",
            "started",
            "action_attempted",
        ):
            value = data.get(name, False)
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
            bools[name] = value
        attempt_count = data.get("attempt_count", 0)
        if type(attempt_count) is not int:
            raise TypeError("attempt_count must be an integer")
        reason_raw = data.get("reason")
        reason = InterruptionReason(reason_raw).value if reason_raw is not None else None
        outcome = InterruptionOutcome(data.get("outcome", InterruptionOutcome.NOT_REQUESTED.value)).value
        return cls(
            job_id=BrowserJobId(data["job_id"]).value,
            provider=data["provider"],
            requested=bools["requested"],
            reason=reason,
            requested_at=data.get("requested_at"),
            source=data.get("source"),
            physical_interruption_required=bools["physical_interruption_required"],
            started=bools["started"],
            action_attempted=bools["action_attempted"],
            attempt_count=attempt_count,
            completed_at=data.get("completed_at"),
            outcome=outcome,
            diagnostic=data.get("diagnostic"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Encapsulated, thread-safe mutable request state
# --------------------------------------------------------------------------- #
class InterruptionRequest:
    """The authoritative, thread-safe owner of ONE browser job's interruption.

    The caller thread only records a request; the provider worker thread performs
    at most one stop action, guarded by :meth:`begin_action`. All mutation is
    under one internal lock; no lock is held across any Playwright I/O (this class
    performs none).
    """

    def __init__(
        self,
        *,
        job_id: str,
        provider: str,
        policy: InterruptionPolicy = DEFAULT_INTERRUPTION_POLICY,
    ) -> None:
        self._job_id = BrowserJobId(job_id).value
        self._provider = provider
        self._policy = policy
        self._lock = threading.Lock()
        self._requested = False
        self._reason: Optional[InterruptionReason] = None
        self._source: Optional[str] = None
        self._requested_at: Optional[float] = None
        self._required = False
        self._started = False
        self._attempted = False
        self._attempt_count = 0
        self._completed_at: Optional[float] = None
        self._outcome = InterruptionOutcome.NOT_REQUESTED
        self._diagnostic: Optional[str] = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def policy(self) -> InterruptionPolicy:
        return self._policy

    def request(
        self,
        reason: InterruptionReason,
        source: str,
        *,
        require_physical: bool = False,
    ) -> bool:
        """Record cancellation/timeout intent. Idempotent; first reason wins.

        Returns ``True`` only for the first call that actually records the
        request (so a caller can branch on "I was the one who requested it"). A
        later call may only upgrade ``require_physical`` to ``True`` — it never
        overwrites the winning reason or source (Phase 5A caller-outcome
        arbitration remains authoritative).
        """
        if not isinstance(reason, InterruptionReason):
            raise TypeError("reason must be an InterruptionReason")
        with self._lock:
            newly = not self._requested
            if newly:
                self._requested = True
                self._reason = reason
                self._source = _bound(source, REQUEST_SOURCE_MAX)
                self._requested_at = time.time()
                if self._outcome is InterruptionOutcome.NOT_REQUESTED:
                    self._outcome = InterruptionOutcome.NOT_NEEDED
            if require_physical:
                self._required = True
            return newly

    def mark_required(self) -> None:
        """Flag that the job is physically running so a stop action is required."""
        with self._lock:
            if self._requested:
                self._required = True

    def is_requested(self) -> bool:
        with self._lock:
            return self._requested

    @property
    def reason(self) -> Optional[InterruptionReason]:
        with self._lock:
            return self._reason

    @property
    def outcome(self) -> InterruptionOutcome:
        with self._lock:
            return self._outcome

    def begin_action(self) -> bool:
        """Compare-and-set gate: returns ``True`` at most once, ever.

        The single provider worker thread calls this immediately before touching
        the browser. Enforces the exactly-once stop-action guarantee even if two
        cancellation signals race, and respects ``policy.max_stop_attempts``.
        """
        with self._lock:
            if self._attempted or self._attempt_count >= self._policy.max_stop_attempts:
                return False
            self._attempted = True
            self._started = True
            self._attempt_count += 1
            return True

    def complete(
        self, outcome: InterruptionOutcome, diagnostic: Optional[str] = None
    ) -> bool:
        """Record the first terminal interruption outcome, exactly once."""
        if not isinstance(outcome, InterruptionOutcome):
            raise TypeError("outcome must be an InterruptionOutcome")
        with self._lock:
            if self._completed_at is not None:
                return False
            self._outcome = outcome
            self._completed_at = time.time()
            if diagnostic is not None:
                self._diagnostic = _bound(diagnostic, self._policy.max_diagnostic_length)
            return True

    def snapshot(self) -> InterruptionSnapshot:
        with self._lock:
            return InterruptionSnapshot(
                job_id=self._job_id,
                provider=self._provider,
                requested=self._requested,
                reason=self._reason.value if self._reason is not None else None,
                requested_at=self._requested_at,
                source=self._source,
                physical_interruption_required=self._required,
                started=self._started,
                action_attempted=self._attempted,
                attempt_count=self._attempt_count,
                completed_at=self._completed_at,
                outcome=self._outcome.value,
                diagnostic=self._diagnostic,
                schema_version=SCHEMA_VERSION,
            )


# --------------------------------------------------------------------------- #
# Composite cancellation observation (external token + internal request)
# --------------------------------------------------------------------------- #
class CancellationObservation:
    """Provider-neutral view of "should this running operation stop now?".

    Composes the existing external ``should_cancel`` callback (e.g. the
    orchestration cancel token) with a browser job's internal
    :class:`InterruptionRequest`. The running provider operation polls this via
    the unchanged ``should_cancel`` seam; it is cheap, idempotent, content-free,
    and carries job/worker identity for owner checks. The FIRST time it observes
    the external token tripped, it records a ``CANCELLED`` interruption request
    exactly once, so external orchestration cancellation drives one active
    interruption.
    """

    def __init__(
        self,
        *,
        job_id: str,
        provider: str,
        request: InterruptionRequest,
        external: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._job_id = BrowserJobId(job_id).value
        self._provider = provider
        self._request = request
        self._external = external

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def request(self) -> InterruptionRequest:
        return self._request

    def _external_cancelled(self) -> bool:
        if self._external is None:
            return False
        try:
            return bool(self._external())
        except Exception:  # noqa: BLE001 - a bad predicate must fail safe
            # A failing external callback must never corrupt lifecycle state; we
            # treat it as "not cancelled" and let the caller-owned paths decide.
            return False

    def __call__(self) -> bool:
        """True if the operation should stop now. Cheap and idempotent."""
        if self._external_cancelled():
            # Record (idempotently) so the worker actively stops generation.
            self._request.request(
                InterruptionReason.CANCELLED, "external-token", require_physical=True
            )
            return True
        return self._request.is_requested()

    def cancelled(self) -> bool:  # pragma: no cover - alias for readability
        return self.__call__()


__all__ = [
    "InterruptionReason",
    "InterruptionOutcome",
    "StopActionStatus",
    "InterruptionPolicy",
    "DEFAULT_INTERRUPTION_POLICY",
    "InterruptionActionResult",
    "InterruptionSnapshot",
    "InterruptionRequest",
    "CancellationObservation",
    "outcome_for_status",
]
