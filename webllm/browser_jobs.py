"""Browser-job identity and lifecycle state machine (production-hardening 5A).

WHY THIS MODULE EXISTS
----------------------
DOZEN drives provider chat UIs through :class:`~webllm.browser_manager.BrowserManager`,
which runs one Playwright worker thread per provider. A caller can time out or
cancel while the worker is still mid-generation. Before Phase 5A there was no
explicit *identity* for a single physical browser operation and no authoritative
record of its state, so it was hard to reason about — or observe — what happens
when a late browser result arrives after its caller has stopped waiting.

Phase 5A introduces:

* an immutable, process-unique :class:`BrowserJobId`,
* a small :class:`JobState` vocabulary with exact semantics,
* one authoritative, thread-safe transition table (compare-and-transition),
* an immutable public :class:`BrowserJobSnapshot` and an encapsulated mutable
  record,
* a per-``BrowserManager`` :class:`BrowserJobRegistry` that owns lifecycle state,
  provider→current-job ownership, bounded terminal history and an optional
  lifecycle event seam.

SCOPE — this module owns *ownership and observability only*. It never touches
Playwright, never clicks stop buttons, never restarts tabs, and never physically
interrupts a running browser operation. A running operation continues even after
its lifecycle record has moved to ``TIMED_OUT``/``CANCELLED``; the machinery here
merely guarantees that a late result is *discarded* rather than mis-delivered.

THREADING CONTRACT
------------------
Every mutation of registry state happens under one internal lock. External
callbacks (the event sink) are invoked *outside* that lock, so a slow or failing
observer can never stall or corrupt browser execution. Registry locks are never
held while Playwright runs — the registry does no I/O.
"""

from __future__ import annotations

import threading
import time
import uuid
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

# --------------------------------------------------------------------------- #
# Retention / schema policy — ONE authoritative location.
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = 1
#: Upper bound on retained *fully terminal* job records (COMPLETED/FAILED/
#: ABANDONED). Active and settling jobs are never counted toward this bound and
#: are never evicted. Failed/abandoned jobs share this single bound with
#: completed ones — retention is deliberately simple in Phase 5A.
DEFAULT_MAX_TERMINAL_JOBS = 256
#: Optional maximum age (seconds, monotonic) beyond which a terminal record is
#: eligible for eviction even under the count bound. ``None`` disables age-based
#: eviction.
DEFAULT_MAX_TERMINAL_AGE_S: Optional[float] = None
#: Hard cap on the bounded failure message copied into snapshots/events.
FAILURE_MESSAGE_MAX = 300
LABEL_MAX = 128
_UNSAFE_LABEL_CHARS = re.compile(r"[^A-Za-z0-9_.:@/-]+")
_JOB_ID_VALUE = re.compile(r"^[0-9a-f]{32}$")
_PROCESS_JOB_PREFIX = uuid.uuid4().hex[:16]
_JOB_ID_LOCK = threading.Lock()
_JOB_ID_COUNTER = 0


def _require_bounded_string(
    value: Any, field_name: str, *, allow_none: bool = True
) -> Optional[str]:
    """Validate serialized public text without silently stringifying values."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) > FAILURE_MESSAGE_MAX:
        raise ValueError(f"{field_name} exceeds {FAILURE_MESSAGE_MAX} characters")
    return value


def _require_optional_timestamp(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or null")
    return float(value)


def _require_timestamp(value: Any, field_name: str) -> float:
    parsed = _require_optional_timestamp(value, field_name)
    if parsed is None:
        raise TypeError(f"{field_name} must be a number")
    return parsed


# --------------------------------------------------------------------------- #
# Job identity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BrowserJobId:
    """An immutable, process-unique browser-job identifier.

    Backed by a process nonce plus a lock-protected monotonic counter, so it is
    opaque, safe for logs/events, never derived from prompt text, never reused
    after registry eviction, and cannot collide inside this process. It compares
    and serializes as a plain 32-character hex string and is hashable.
    """

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("BrowserJobId value must be a string")
        if _JOB_ID_VALUE.fullmatch(self.value) is None:
            raise ValueError("BrowserJobId must be a lowercase, full 32-character hex string")

    @classmethod
    def new(cls) -> "BrowserJobId":
        global _JOB_ID_COUNTER
        with _JOB_ID_LOCK:
            _JOB_ID_COUNTER += 1
            counter = _JOB_ID_COUNTER
        if counter >= 1 << 64:  # practically unreachable; never wrap/reuse
            raise OverflowError("process-local browser job id space exhausted")
        return cls(f"{_PROCESS_JOB_PREFIX}{counter:016x}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------- #
# Lifecycle states
# --------------------------------------------------------------------------- #
class JobState(str, Enum):
    """The lifecycle of exactly one physical browser operation."""

    QUEUED = "queued"          # accepted by BrowserManager, not yet started
    RUNNING = "running"        # a provider worker has claimed it; browser I/O begun
    COMPLETED = "completed"    # valid result produced AND still owned by caller
    FAILED = "failed"          # execution raised while caller still owned the job
    CANCELLED = "cancelled"    # cancellation observed before a result was accepted
    TIMED_OUT = "timed_out"    # caller deadline expired without an accepted result
    ABANDONED = "abandoned"    # no valid result consumer remains

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class TerminalCause(str, Enum):
    """Trusted caller outcome retained after physical settlement/shutdown."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SHUTDOWN = "shutdown"
    ABANDONED = "abandoned"


#: States from which no transition is ever legal.
_FULLY_TERMINAL: frozenset[JobState] = frozenset(
    {JobState.COMPLETED, JobState.FAILED, JobState.ABANDONED}
)

#: States that count as "finished" for retention/eviction purposes.
_RETAINABLE_TERMINAL: frozenset[JobState] = _FULLY_TERMINAL

#: The single authoritative transition table. Any (old -> new) pair not listed
#: here is rejected by :meth:`BrowserJobRegistry.transition` without mutating
#: state.
_ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset(
        {JobState.RUNNING, JobState.CANCELLED, JobState.TIMED_OUT, JobState.ABANDONED}
    ),
    JobState.RUNNING: frozenset(
        {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMED_OUT,
            JobState.ABANDONED,
        }
    ),
    JobState.TIMED_OUT: frozenset({JobState.ABANDONED}),
    JobState.CANCELLED: frozenset({JobState.ABANDONED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.ABANDONED: frozenset(),
}


def is_terminal(state: JobState) -> bool:
    """True if ``state`` admits no further transitions."""
    return state in _FULLY_TERMINAL


def allowed_transition(old: JobState, new: JobState) -> bool:
    """True if ``old -> new`` is a legal lifecycle transition."""
    return new in _ALLOWED_TRANSITIONS.get(old, frozenset())


def _bounded_message(msg: Optional[str]) -> Optional[str]:
    if msg is None:
        return None
    text = str(msg).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > FAILURE_MESSAGE_MAX:
        text = text[: FAILURE_MESSAGE_MAX - 1].rstrip() + "…"
    return text or None


def _safe_label(value: Optional[str]) -> Optional[str]:
    """Bound an observability label and strip free-form/control characters."""
    text = _bounded_message(value)
    if text is None:
        return None
    text = _UNSAFE_LABEL_CHARS.sub("_", text)[:LABEL_MAX].strip("_")
    return text or None


# --------------------------------------------------------------------------- #
# Public snapshot (immutable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BrowserJobSnapshot:
    """An immutable, content-free view of one browser job.

    Deliberately carries NO prompt text, response body, cookies, DOM, credentials
    or tracebacks — only lifecycle metadata safe to log, serialize and surface to
    future observability. Timestamps are wall-clock seconds (user-facing);
    durations/races are computed by the registry from a monotonic clock and are
    not exposed here.
    """

    job_id: str
    provider: str
    state: str
    created_at: float
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    timeout_at: Optional[float] = None
    cancelled_at: Optional[float] = None
    abandoned_at: Optional[float] = None
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    result_delivered: bool = False
    late_result_discarded: bool = False
    terminal_cause: Optional[str] = None
    physical_settled: bool = False
    physical_settled_at: Optional[float] = None
    queue_position: Optional[int] = None
    correlation: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        BrowserJobId(self.job_id)
        _require_bounded_string(self.provider, "provider", allow_none=False)
        JobState(self.state)
        if self.terminal_cause is not None:
            TerminalCause(self.terminal_cause)
        if type(self.result_delivered) is not bool:
            raise TypeError("result_delivered must be a boolean")
        if type(self.late_result_discarded) is not bool:
            raise TypeError("late_result_discarded must be a boolean")
        if type(self.physical_settled) is not bool:
            raise TypeError("physical_settled must be a boolean")
        if self.physical_settled != (self.physical_settled_at is not None):
            raise ValueError("physical_settled and physical_settled_at disagree")
        created = _require_timestamp(self.created_at, "created_at")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.queue_position is not None and (
            type(self.queue_position) is not int or self.queue_position < 0
        ):
            raise ValueError("queue_position must be a non-negative integer or null")
        for field_name in ("failure_category", "failure_message", "correlation"):
            _require_bounded_string(getattr(self, field_name), field_name)
        for field_name in (
            "queued_at",
            "started_at",
            "finished_at",
            "timeout_at",
            "cancelled_at",
            "abandoned_at",
            "physical_settled_at",
        ):
            value = _require_optional_timestamp(getattr(self, field_name), field_name)
            if value is not None and value < created:
                raise ValueError(f"{field_name} precedes created_at")
        # Reject impossible timestamp combinations so a corrupt snapshot can
        # never be constructed. The state machine already orders transitions;
        # this is defence-in-depth.
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at precedes created_at")
        if (
            self.finished_at is not None
            and self.started_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at precedes started_at")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "state": self.state,
            "created_at": self.created_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "timeout_at": self.timeout_at,
            "cancelled_at": self.cancelled_at,
            "abandoned_at": self.abandoned_at,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "result_delivered": self.result_delivered,
            "late_result_discarded": self.late_result_discarded,
            "terminal_cause": self.terminal_cause,
            "physical_settled": self.physical_settled,
            "physical_settled_at": self.physical_settled_at,
            "queue_position": self.queue_position,
            "correlation": self.correlation,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserJobSnapshot":
        if not isinstance(data, dict):
            raise TypeError("browser job snapshot must be a mapping")
        state = JobState(data["state"]).value
        cause_raw = data.get("terminal_cause")
        cause = TerminalCause(cause_raw).value if cause_raw is not None else None
        result_delivered = data.get("result_delivered", False)
        late_discarded = data.get("late_result_discarded", False)
        physical_settled = data.get("physical_settled", False)
        for field_name, value in (
            ("result_delivered", result_delivered),
            ("late_result_discarded", late_discarded),
            ("physical_settled", physical_settled),
        ):
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be a boolean")
        return cls(
            job_id=BrowserJobId(data["job_id"]).value,
            provider=_require_bounded_string(data["provider"], "provider", allow_none=False),
            state=state,
            created_at=_require_timestamp(data["created_at"], "created_at"),
            queued_at=_require_optional_timestamp(data.get("queued_at"), "queued_at"),
            started_at=_require_optional_timestamp(data.get("started_at"), "started_at"),
            finished_at=_require_optional_timestamp(data.get("finished_at"), "finished_at"),
            timeout_at=_require_optional_timestamp(data.get("timeout_at"), "timeout_at"),
            cancelled_at=_require_optional_timestamp(data.get("cancelled_at"), "cancelled_at"),
            abandoned_at=_require_optional_timestamp(data.get("abandoned_at"), "abandoned_at"),
            failure_category=_require_bounded_string(data.get("failure_category"), "failure_category"),
            failure_message=_require_bounded_string(data.get("failure_message"), "failure_message"),
            result_delivered=result_delivered,
            late_result_discarded=late_discarded,
            terminal_cause=cause,
            physical_settled=physical_settled,
            physical_settled_at=_require_optional_timestamp(
                data.get("physical_settled_at"), "physical_settled_at"
            ),
            queue_position=data.get("queue_position"),
            correlation=data.get("correlation"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Lifecycle event (emitted OUTSIDE the registry lock)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BrowserJobEvent:
    """A concise, content-free lifecycle event for the observability seam.

    Phase 5B adds three optional, content-free interruption fields so active
    cancellation is observable through the same ordered event stream. They default
    to ``None``/``False`` on ordinary lifecycle events, keeping the shape backward
    compatible with Phase 5A observers.
    """

    job_id: str
    provider: str
    old_state: Optional[str]
    new_state: str
    timestamp: float                 # wall-clock seconds
    reason: Optional[str] = None
    late_result_discarded: bool = False
    physical_settled: bool = False
    interruption_reason: Optional[str] = None
    interruption_outcome: Optional[str] = None
    interruption_attempted: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        BrowserJobId(self.job_id)
        _require_bounded_string(self.provider, "provider", allow_none=False)
        if self.old_state is not None:
            JobState(self.old_state)
        JobState(self.new_state)
        _require_timestamp(self.timestamp, "timestamp")
        _require_bounded_string(self.reason, "reason")
        _require_bounded_string(self.interruption_reason, "interruption_reason")
        _require_bounded_string(self.interruption_outcome, "interruption_outcome")
        if type(self.late_result_discarded) is not bool:
            raise TypeError("late_result_discarded must be a boolean")
        if type(self.physical_settled) is not bool:
            raise TypeError("physical_settled must be a boolean")
        if type(self.interruption_attempted) is not bool:
            raise TypeError("interruption_attempted must be a boolean")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "late_result_discarded": self.late_result_discarded,
            "physical_settled": self.physical_settled,
            "interruption_reason": self.interruption_reason,
            "interruption_outcome": self.interruption_outcome,
            "interruption_attempted": self.interruption_attempted,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserJobEvent":
        if not isinstance(data, dict):
            raise TypeError("browser job event must be a mapping")
        late = data.get("late_result_discarded", False)
        settled = data.get("physical_settled", False)
        attempted = data.get("interruption_attempted", False)
        if type(late) is not bool or type(settled) is not bool or type(attempted) is not bool:
            raise TypeError("event boolean fields must be booleans")
        return cls(
            job_id=BrowserJobId(data["job_id"]).value,
            provider=_require_bounded_string(data["provider"], "provider", allow_none=False),
            old_state=(
                JobState(data["old_state"]).value
                if data.get("old_state") is not None
                else None
            ),
            new_state=JobState(data["new_state"]).value,
            timestamp=_require_timestamp(data["timestamp"], "timestamp"),
            reason=_require_bounded_string(data.get("reason"), "reason"),
            late_result_discarded=late,
            physical_settled=settled,
            interruption_reason=_require_bounded_string(
                data.get("interruption_reason"), "interruption_reason"
            ),
            interruption_outcome=_require_bounded_string(
                data.get("interruption_outcome"), "interruption_outcome"
            ),
            interruption_attempted=attempted,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


EventSink = Callable[[BrowserJobEvent], None]


# --------------------------------------------------------------------------- #
# Transition result (structured — no exceptions for ordinary races)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransitionResult:
    """Outcome of a compare-and-transition attempt.

    Ordinary lost races (e.g. completion arriving after timeout) return
    ``transitioned=False`` rather than raising, so callers can branch cleanly.
    ``state`` is the job's state *after* the attempt (unchanged on rejection).
    """

    transitioned: bool
    state: Optional[JobState]
    reason: Optional[str] = None
    snapshot: Optional[BrowserJobSnapshot] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.transitioned


# --------------------------------------------------------------------------- #
# Internal mutable record (encapsulated; only the registry touches it)
# --------------------------------------------------------------------------- #
@dataclass
class _JobRecord:
    job_id: str
    provider: str
    correlation: Optional[str]
    state: JobState = JobState.QUEUED
    # Wall-clock timestamps (user-facing).
    created_at: float = 0.0
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    timeout_at: Optional[float] = None
    cancelled_at: Optional[float] = None
    abandoned_at: Optional[float] = None
    # Monotonic anchor for retention/age calculations only.
    created_mono: float = 0.0
    terminal_mono: Optional[float] = None
    # Bounded failure summary (never a traceback / never prompt/response text).
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    result_delivered: bool = False
    late_result_discarded: bool = False
    terminal_cause: Optional[TerminalCause] = None
    physical_settled: bool = False
    physical_settled_at: Optional[float] = None

    def snapshot(self) -> BrowserJobSnapshot:
        return BrowserJobSnapshot(
            job_id=self.job_id,
            provider=self.provider,
            state=self.state.value,
            created_at=self.created_at,
            queued_at=self.queued_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            timeout_at=self.timeout_at,
            cancelled_at=self.cancelled_at,
            abandoned_at=self.abandoned_at,
            failure_category=self.failure_category,
            failure_message=self.failure_message,
            result_delivered=self.result_delivered,
            late_result_discarded=self.late_result_discarded,
            terminal_cause=(self.terminal_cause.value if self.terminal_cause else None),
            physical_settled=self.physical_settled,
            physical_settled_at=self.physical_settled_at,
            queue_position=None,  # not reliably available under the queue model
            correlation=self.correlation,
            schema_version=SCHEMA_VERSION,
        )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
class BrowserJobRegistry:
    """Thread-safe owner of browser-job lifecycle state for ONE BrowserManager.

    This is intentionally NOT a process-global singleton: each ``BrowserManager``
    instance owns its own registry, so independent managers (and tests) never
    share mutable state.
    """

    def __init__(
        self,
        *,
        event_sink: Optional[EventSink] = None,
        max_terminal_jobs: int = DEFAULT_MAX_TERMINAL_JOBS,
        max_terminal_age_s: Optional[float] = DEFAULT_MAX_TERMINAL_AGE_S,
    ) -> None:
        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be >= 1")
        if max_terminal_age_s is not None and max_terminal_age_s < 0:
            raise ValueError("max_terminal_age_s must be >= 0 or None")
        self._lock = threading.Lock()
        self._records: dict[str, _JobRecord] = {}
        # provider -> job_id currently occupying that provider's tab (RUNNING).
        self._active: dict[str, str] = {}
        # FIFO of fully-terminal job ids, oldest first, for bounded eviction.
        self._terminal_order: "deque[str]" = deque()
        self._event_sink = event_sink
        self._max_terminal_jobs = max_terminal_jobs
        self._max_terminal_age_s = max_terminal_age_s

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self, provider: str, *, correlation: Optional[str] = None
    ) -> BrowserJobId:
        """Create a new ``QUEUED`` job record and return its id."""
        job_id, event = self.register_deferred(provider, correlation=correlation)
        self._emit(event)
        return job_id

    def register_deferred(
        self, provider: str, *, correlation: Optional[str] = None
    ) -> tuple[BrowserJobId, BrowserJobEvent]:
        """Insert a new ``QUEUED`` record WITHOUT emitting its event.

        Returns the id and the prepared ``QUEUED`` event so a caller performing an
        atomic admission decision can create the record under its admission lock
        (a fast dict insert with no callback) and then emit the event *outside*
        that lock. :meth:`register` is the thin emitting wrapper used everywhere
        else. The two paths share one code path so record shape never drifts.
        """
        job_id = BrowserJobId.new()
        now = time.time()
        mono = time.monotonic()
        rec = _JobRecord(
            job_id=job_id.value,
            provider=_safe_label(provider) or "unknown",
            correlation=_safe_label(correlation),
            state=JobState.QUEUED,
            created_at=now,
            queued_at=now,
            created_mono=mono,
        )
        event = BrowserJobEvent(
            job_id=rec.job_id,
            provider=rec.provider,
            old_state=None,
            new_state=JobState.QUEUED.value,
            timestamp=now,
            reason="queued",
        )
        with self._lock:
            self._records[job_id.value] = rec
        return job_id, event

    def rollback_deferred(self, job_id: str | BrowserJobId) -> bool:
        """Remove an unpublished deferred registration after failed admission.

        Only an untouched ``QUEUED`` record can be rolled back.  The manager
        keeps the corresponding queue entry unclaimable until publication, so a
        successful rollback can never erase running or terminal history.
        """
        key = str(job_id)
        with self._lock:
            rec = self._records.get(key)
            if rec is None or rec.state is not JobState.QUEUED:
                return False
            if key in self._active.values():
                return False
            self._records.pop(key, None)
            return True

    def emit(self, event: Optional[BrowserJobEvent]) -> None:
        """Emit a prepared lifecycle event through the sink, outside any lock.

        The public counterpart to the internal ``_emit`` seam, used by an atomic
        admission path that prepared a :meth:`register_deferred` event under its
        own lock and must emit it only after releasing that lock.
        """
        self._emit(event)

    # ------------------------------------------------------------------ #
    # Core compare-and-transition
    # ------------------------------------------------------------------ #
    def transition(
        self,
        job_id: str | BrowserJobId,
        new_state: JobState,
        *,
        reason: Optional[str] = None,
        failure_category: Optional[str] = None,
        failure_message: Optional[str] = None,
        result_delivered: Optional[bool] = None,
        late_result_discarded: Optional[bool] = None,
        terminal_cause: Optional[TerminalCause] = None,
    ) -> TransitionResult:
        """Atomically move ``job_id`` to ``new_state`` if the transition is legal.

        Owner-safe: an illegal transition (including any transition out of a
        fully-terminal state, and any duplicate completion/failure) leaves the
        record untouched and returns ``transitioned=False``. Provider ownership
        is set on entry to ``RUNNING`` and cleared — owner-checked — on exit from
        ``RUNNING``, so a stale job can never clear a successor's ownership.
        """
        key = str(job_id)
        event: Optional[BrowserJobEvent] = None
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return TransitionResult(False, None, reason="unknown-job")
            old = rec.state
            if not allowed_transition(old, new_state):
                return TransitionResult(
                    False, old, reason=f"illegal:{old.value}->{new_state.value}"
                )
            if new_state is JobState.RUNNING:
                owner = self._active.get(rec.provider)
                if owner is not None and owner != rec.job_id:
                    return TransitionResult(False, old, reason="provider-busy")

            now = time.time()
            rec.state = new_state
            self._stamp(rec, old, new_state, now)

            if failure_category is not None:
                rec.failure_category = _safe_label(failure_category)
            if failure_message is not None:
                # Provider/Playwright exception text can contain DOM, prompt,
                # response, cookie, or credential material. Category is enough
                # for lifecycle diagnostics; raw exception text stays only on
                # the private handle for backward-compatible re-raising.
                rec.failure_message = None
            if result_delivered is not None:
                rec.result_delivered = result_delivered
            if late_result_discarded is not None:
                rec.late_result_discarded = late_result_discarded
            if terminal_cause is not None and rec.terminal_cause is None:
                rec.terminal_cause = terminal_cause
            if rec.terminal_cause is None:
                rec.terminal_cause = {
                    JobState.COMPLETED: TerminalCause.COMPLETED,
                    JobState.FAILED: TerminalCause.FAILED,
                    JobState.CANCELLED: TerminalCause.CANCELLED,
                    JobState.TIMED_OUT: TerminalCause.TIMED_OUT,
                    JobState.ABANDONED: TerminalCause.ABANDONED,
                }.get(new_state)

            # Provider ownership is PHYSICAL, not caller-lifecycle ownership.
            # It remains claimed across RUNNING -> TIMED_OUT/CANCELLED/ABANDONED
            # and is released only by mark_physical_settled() from worker cleanup.
            if new_state is JobState.RUNNING:
                self._active[rec.provider] = rec.job_id

            if new_state in _RETAINABLE_TERMINAL and rec.physical_settled:
                rec.terminal_mono = time.monotonic()
                self._terminal_order.append(rec.job_id)

            event = BrowserJobEvent(
                job_id=rec.job_id,
                provider=rec.provider,
                old_state=old.value,
                new_state=new_state.value,
                timestamp=now,
                reason=_safe_label(reason),
                late_result_discarded=rec.late_result_discarded,
                physical_settled=rec.physical_settled,
            )
            snap = rec.snapshot()
            self._evict_locked()

        self._emit(event)
        return TransitionResult(True, new_state, reason=reason, snapshot=snap)

    @staticmethod
    def _stamp(rec: _JobRecord, old: JobState, new: JobState, now: float) -> None:
        """Record the single wall-clock timestamp for this transition."""
        if new is JobState.RUNNING:
            rec.started_at = now
        elif new is JobState.COMPLETED or new is JobState.FAILED:
            rec.finished_at = now
        elif new is JobState.TIMED_OUT:
            rec.timeout_at = now
        elif new is JobState.CANCELLED:
            rec.cancelled_at = now
        elif new is JobState.ABANDONED:
            rec.abandoned_at = now

    # ------------------------------------------------------------------ #
    # Convenience wrappers (thin, so call sites read clearly)
    # ------------------------------------------------------------------ #
    def mark_running(self, job_id: str | BrowserJobId, *, reason: str = "claimed") -> TransitionResult:
        return self.transition(job_id, JobState.RUNNING, reason=reason)

    def mark_completed(
        self, job_id: str | BrowserJobId, *, reason: str = "completed"
    ) -> TransitionResult:
        result = self.transition(
            job_id,
            JobState.COMPLETED,
            reason=reason,
            terminal_cause=TerminalCause.COMPLETED,
        )
        if result.transitioned:
            self.mark_physical_settled(job_id, reason="completed-settled")
        return result

    def mark_failed(
        self,
        job_id: str | BrowserJobId,
        *,
        failure_category: Optional[str] = None,
        failure_message: Optional[str] = None,
        reason: str = "failed",
    ) -> TransitionResult:
        result = self.transition(
            job_id,
            JobState.FAILED,
            reason=reason,
            failure_category=failure_category,
            failure_message=failure_message,
            terminal_cause=TerminalCause.FAILED,
        )
        if result.transitioned:
            self.mark_physical_settled(job_id, reason="failed-settled")
        return result

    def mark_cancelled(
        self, job_id: str | BrowserJobId, *, reason: str = "cancelled"
    ) -> TransitionResult:
        return self.transition(
            job_id,
            JobState.CANCELLED,
            reason=reason,
            terminal_cause=TerminalCause.CANCELLED,
        )

    def mark_timed_out(
        self, job_id: str | BrowserJobId, *, reason: str = "timeout"
    ) -> TransitionResult:
        return self.transition(
            job_id,
            JobState.TIMED_OUT,
            reason=reason,
            terminal_cause=TerminalCause.TIMED_OUT,
        )

    def mark_abandoned(
        self,
        job_id: str | BrowserJobId,
        *,
        reason: str = "abandoned",
        physical_settled: bool = False,
    ) -> TransitionResult:
        cause = (
            TerminalCause.SHUTDOWN
            if "shutdown" in reason
            else TerminalCause.ABANDONED
        )
        result = self.transition(
            job_id, JobState.ABANDONED, reason=reason, terminal_cause=cause
        )
        if result.transitioned and physical_settled:
            self.mark_physical_settled(job_id, reason=f"{reason}-settled")
        return result

    def mark_result_delivered(self, job_id: str | BrowserJobId) -> bool:
        """Record the first actual successful handle read (multi-reader policy)."""
        with self._lock:
            rec = self._records.get(str(job_id))
            if rec is None or rec.state is not JobState.COMPLETED:
                return False
            rec.result_delivered = True
            return True

    def mark_physical_settled(
        self, job_id: str | BrowserJobId, *, reason: str = "physical-settled"
    ) -> bool:
        """Atomically record physical unwind and release provider ownership.

        Caller-terminal CANCELLED/TIMED_OUT records become retainable ABANDONED
        history only here. Their trusted terminal cause is deliberately retained.
        """
        key = str(job_id)
        event: Optional[BrowserJobEvent] = None
        with self._lock:
            rec = self._records.get(key)
            if rec is None or rec.physical_settled:
                return False
            now = time.time()
            old = rec.state
            rec.physical_settled = True
            rec.physical_settled_at = now
            if self._active.get(rec.provider) == rec.job_id:
                del self._active[rec.provider]
            if old in (
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.CANCELLED,
                JobState.TIMED_OUT,
            ):
                rec.state = JobState.ABANDONED
                rec.abandoned_at = now
                if rec.terminal_cause is None:
                    rec.terminal_cause = TerminalCause.ABANDONED
            if rec.state in _RETAINABLE_TERMINAL:
                rec.terminal_mono = time.monotonic()
                self._terminal_order.append(rec.job_id)
            event = BrowserJobEvent(
                job_id=rec.job_id,
                provider=rec.provider,
                old_state=old.value,
                new_state=rec.state.value,
                timestamp=now,
                reason=_safe_label(reason),
                late_result_discarded=rec.late_result_discarded,
                physical_settled=True,
            )
            self._evict_locked()
        self._emit(event)
        return True

    def record_late_result_discarded(
        self, job_id: str | BrowserJobId, *, reason: str = "late-result"
    ) -> None:
        """Flag that a late result/exception was discarded (no state change).

        Emits a dedicated ``late_result_discarded`` event outside the lock. Never
        retains any response content — only the boolean fact.
        """
        key = str(job_id)
        event: Optional[BrowserJobEvent] = None
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return
            if rec.late_result_discarded:
                return
            rec.late_result_discarded = True
            event = BrowserJobEvent(
                job_id=rec.job_id,
                provider=rec.provider,
                old_state=rec.state.value,
                new_state=rec.state.value,
                timestamp=time.time(),
                reason=_safe_label(reason),
                late_result_discarded=True,
                physical_settled=rec.physical_settled,
            )
        self._emit(event)

    def release_provider(self, job_id: str | BrowserJobId) -> bool:
        """Owner-checked release of a provider's active slot.

        A no-op unless ``job_id`` is currently the provider's active job. Safe to
        call from a ``finally`` even when the job never entered ``RUNNING``.
        """
        key = str(job_id)
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return False
            owns = self._active.get(rec.provider) == rec.job_id
        if not owns:
            return False
        return self.mark_physical_settled(job_id, reason="worker-finally")

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #
    def snapshot(self, job_id: str | BrowserJobId) -> Optional[BrowserJobSnapshot]:
        with self._lock:
            self._evict_locked()
            rec = self._records.get(str(job_id))
            return rec.snapshot() if rec is not None else None

    def state(self, job_id: str | BrowserJobId) -> Optional[JobState]:
        with self._lock:
            rec = self._records.get(str(job_id))
            return rec.state if rec is not None else None

    def active_job(self, provider: str) -> Optional[str]:
        with self._lock:
            return self._active.get(provider)

    def snapshots(self) -> list[BrowserJobSnapshot]:
        """All currently-retained snapshots (bounded by retention policy)."""
        with self._lock:
            self._evict_locked()
            return [rec.snapshot() for rec in self._records.values()]

    def active_snapshots(self) -> list[BrowserJobSnapshot]:
        """Snapshots of jobs whose physical operation has not settled."""
        with self._lock:
            return [
                rec.snapshot()
                for rec in self._records.values()
                if not rec.physical_settled
            ]

    def __len__(self) -> int:
        with self._lock:
            self._evict_locked()
            return len(self._records)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _evict_locked(self) -> None:
        """Evict oldest fully-terminal records past the retention bounds.

        Only fully-terminal records live in ``_terminal_order``; active/settling
        jobs are never here and thus never evicted. Called under ``self._lock``.
        """
        max_age = self._max_terminal_age_s
        now_mono = time.monotonic()
        # Age-based eviction first (oldest end of the deque).
        if max_age is not None:
            while self._terminal_order:
                oldest = self._terminal_order[0]
                rec = self._records.get(oldest)
                if rec is None:
                    self._terminal_order.popleft()
                    continue
                if rec.terminal_mono is not None and (now_mono - rec.terminal_mono) > max_age:
                    self._terminal_order.popleft()
                    self._records.pop(oldest, None)
                else:
                    break
        # Count-based eviction.
        while len(self._terminal_order) > self._max_terminal_jobs:
            oldest = self._terminal_order.popleft()
            self._records.pop(oldest, None)

    def _emit(self, event: Optional[BrowserJobEvent]) -> None:
        """Invoke the event sink OUTSIDE the lock; isolate sink failures."""
        if event is None or self._event_sink is None:
            return
        try:
            self._event_sink(event)
        except Exception:  # noqa: BLE001 - an observer must never break the job
            pass


# --------------------------------------------------------------------------- #
# Handle (additive submission result) — see browser_manager for construction.
# --------------------------------------------------------------------------- #
__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MAX_TERMINAL_JOBS",
    "DEFAULT_MAX_TERMINAL_AGE_S",
    "FAILURE_MESSAGE_MAX",
    "BrowserJobId",
    "JobState",
    "TerminalCause",
    "BrowserJobSnapshot",
    "BrowserJobEvent",
    "EventSink",
    "TransitionResult",
    "BrowserJobRegistry",
    "is_terminal",
    "allowed_transition",
]
