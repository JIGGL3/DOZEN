"""Bounded provider queues and admission backpressure (production-hardening 5C).

WHY THIS MODULE EXISTS
----------------------
Phases 5A/5B gave every physical browser operation an identity, a caller
lifecycle, owner-checked settlement and active cancellation. But
:class:`~webllm.browser_manager.BrowserManager` still accepted an effectively
UNBOUNDED number of queued requests: a provider could accumulate one running job
plus arbitrarily many queued jobs, timed-out/cancelled tombstones physically
lingering behind a blocked running job, and new work entering even while the
provider was quarantined (unsafe to reuse after an interruption).

Phase 5C introduces bounded per-provider queues and *deterministic admission
backpressure*:

    request submitted
    → atomically evaluate provider capacity and availability
    → accept into a bounded FIFO queue, OR
    → reject immediately with a typed, content-free reason.

SCOPE — this module owns *queue policy, admission vocabulary, the bounded
provider-job queue, and content-free queue observability only*. It never touches
Playwright, never registers a browser job, never decides quarantine (the manager
observes that from Phase 5B outcomes) and performs no recovery, failover, tab
restart, priority scheduling or run identity.

THREADING CONTRACT
------------------
:class:`ProviderJobQueue` guards its bounded ``deque`` with one internal lock and
a condition variable. The blocking ``get`` used by a provider worker waits on that
condition (no busy polling) and returns a ``None`` sentinel once the queue is
closed and drained. The manager performs the atomic *capacity decision* under its
own admission lock (a strictly outer lock), so admission never over-admits, while
the queue lock alone protects the physical ``deque`` against the worker's dequeue
and against queued cancellation/timeout removal. No callback is ever invoked while
either lock is held — this module emits no events itself; the manager emits them.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .browser_jobs import SCHEMA_VERSION
from .providers import ProviderError

# --------------------------------------------------------------------------- #
# Bounds for public diagnostic text — ONE authoritative location.
# --------------------------------------------------------------------------- #
ADMISSION_DIAGNOSTIC_MAX = 200


def _bound(value: Optional[str], limit: int) -> Optional[str]:
    """Bound and flatten a public diagnostic/reason; never carries content."""
    if value is None:
        return None
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


# --------------------------------------------------------------------------- #
# Part A — Queue policy (ONE immutable authoritative location for every value)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueuePolicy:
    """Immutable authoritative queue/admission policy.

    Every numeric bound and behavioural switch lives here so there is exactly one
    place to reason about queue capacity and backpressure. Conservative defaults;
    no value permits an unbounded queue or a blocking admission.
    """

    #: Maximum QUEUED prompt jobs retained per provider (running excluded).
    max_queued_prompts_per_provider: int = 8
    #: Maximum QUEUED prompt jobs across every provider.
    max_total_queued_prompts: int = 32
    #: Maximum QUEUED control/legacy jobs (login/status/…) per provider.
    max_queued_control_jobs_per_provider: int = 4
    #: Hard cap on how long an accepted prompt may wait in the queue (seconds).
    max_queue_wait_s: float = 120.0
    #: When a queue is full, reject immediately rather than block.
    fail_fast_when_full: bool = True
    #: A queued cancellation releases the logical slot immediately.
    release_capacity_on_queued_cancel: bool = True
    #: A queued timeout/expiry releases the logical slot immediately.
    release_capacity_on_queued_timeout: bool = True
    #: A quarantined provider rejects new prompt admissions.
    reject_quarantined_provider: bool = True
    #: Hard cap on any admission/queue diagnostic copied into snapshots/events.
    max_admission_diagnostic_length: int = ADMISSION_DIAGNOSTIC_MAX
    #: Hard cap on entries returned by a bulk queue snapshot.
    max_queue_snapshot_entries: int = 64
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "max_queued_prompts_per_provider",
            "max_total_queued_prompts",
            "max_queued_control_jobs_per_provider",
            "max_admission_diagnostic_length",
            "max_queue_snapshot_entries",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.max_total_queued_prompts < self.max_queued_prompts_per_provider:
            raise ValueError(
                "max_total_queued_prompts must be >= max_queued_prompts_per_provider"
            )
        if isinstance(self.max_queue_wait_s, bool) or not isinstance(
            self.max_queue_wait_s, (int, float)
        ):
            raise TypeError("max_queue_wait_s must be a number")
        if self.max_queue_wait_s <= 0:
            raise ValueError("max_queue_wait_s must be > 0")
        if not math.isfinite(float(self.max_queue_wait_s)):
            raise ValueError("max_queue_wait_s must be finite")
        for name in (
            "fail_fast_when_full",
            "release_capacity_on_queued_cancel",
            "release_capacity_on_queued_timeout",
            "reject_quarantined_provider",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
            if not getattr(self, name):
                raise ValueError(f"{name} must be true for bounded queue safety")
        if self.max_admission_diagnostic_length > ADMISSION_DIAGNOSTIC_MAX:
            raise ValueError(
                "max_admission_diagnostic_length cannot exceed "
                f"{ADMISSION_DIAGNOSTIC_MAX}"
            )
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_queued_prompts_per_provider": self.max_queued_prompts_per_provider,
            "max_total_queued_prompts": self.max_total_queued_prompts,
            "max_queued_control_jobs_per_provider": self.max_queued_control_jobs_per_provider,
            "max_queue_wait_s": self.max_queue_wait_s,
            "fail_fast_when_full": self.fail_fast_when_full,
            "release_capacity_on_queued_cancel": self.release_capacity_on_queued_cancel,
            "release_capacity_on_queued_timeout": self.release_capacity_on_queued_timeout,
            "reject_quarantined_provider": self.reject_quarantined_provider,
            "max_admission_diagnostic_length": self.max_admission_diagnostic_length,
            "max_queue_snapshot_entries": self.max_queue_snapshot_entries,
            "schema_version": self.schema_version,
        }


#: The single default policy instance — bounded per-provider and global queues,
#: fail-fast when full, immediate logical-capacity release on queued
#: cancel/timeout, and quarantine rejection.
DEFAULT_QUEUE_POLICY = QueuePolicy()


# --------------------------------------------------------------------------- #
# Part B — Admission vocabulary
# --------------------------------------------------------------------------- #
class AdmissionOutcome(str, Enum):
    """The typed outcome of a single admission decision."""

    ACCEPTED = "accepted"
    REJECTED_PROVIDER_FULL = "rejected_provider_full"
    REJECTED_GLOBAL_FULL = "rejected_global_full"
    REJECTED_PROVIDER_QUARANTINED = "rejected_provider_quarantined"
    REJECTED_MANAGER_SHUTTING_DOWN = "rejected_manager_shutting_down"
    REJECTED_PROVIDER_UNAVAILABLE = "rejected_provider_unavailable"
    EXPIRED_BEFORE_START = "expired_before_start"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def accepted(self) -> bool:
        return self is AdmissionOutcome.ACCEPTED


#: Concise, content-free default reasons per outcome (bounded on construction).
_DEFAULT_REASON: dict[AdmissionOutcome, str] = {
    AdmissionOutcome.ACCEPTED: "accepted",
    AdmissionOutcome.REJECTED_PROVIDER_FULL: "provider queue is full",
    AdmissionOutcome.REJECTED_GLOBAL_FULL: "global queue is full",
    AdmissionOutcome.REJECTED_PROVIDER_QUARANTINED: "provider is quarantined",
    AdmissionOutcome.REJECTED_MANAGER_SHUTTING_DOWN: "manager is shutting down",
    AdmissionOutcome.REJECTED_PROVIDER_UNAVAILABLE: "provider is unavailable",
    AdmissionOutcome.EXPIRED_BEFORE_START: "queue wait expired before start",
}


class QueueEventKind(str, Enum):
    """Content-free queue observability event kinds."""

    ADMISSION_ACCEPTED = "admission_accepted"
    ADMISSION_REJECTED = "admission_rejected"
    QUEUE_DEPTH_CHANGED = "queue_depth_changed"
    QUEUED_JOB_REMOVED = "queued_job_removed"
    QUEUE_WAIT_EXPIRED = "queue_wait_expired"
    PROVIDER_QUEUE_CLOSED = "provider_queue_closed"
    PROVIDER_QUARANTINED = "provider_quarantined"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------- #
# Immutable admission snapshot (Part B)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdmissionSnapshot:
    """An immutable, content-free record of ONE admission decision.

    Carries NO prompt text, response body, credentials, cookies, browser state or
    tracebacks — only the outcome, provider identity, capacity accounting and a
    bounded safe reason.
    """

    outcome: str
    provider: str
    timestamp: float
    provider_queued_depth: int
    provider_capacity: int
    global_queued_depth: int
    global_capacity: int
    reason: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        AdmissionOutcome(self.outcome)
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise TypeError("timestamp must be a number")
        for name in (
            "provider_queued_depth",
            "provider_capacity",
            "global_queued_depth",
            "global_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("reason must be a string or null")
            if len(self.reason) > ADMISSION_DIAGNOSTIC_MAX or "\n" in self.reason or "\r" in self.reason:
                raise ValueError(
                    f"reason must be flat and at most {ADMISSION_DIAGNOSTIC_MAX} characters"
                )
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    @property
    def accepted(self) -> bool:
        return self.outcome == AdmissionOutcome.ACCEPTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "provider_queued_depth": self.provider_queued_depth,
            "provider_capacity": self.provider_capacity,
            "global_queued_depth": self.global_queued_depth,
            "global_capacity": self.global_capacity,
            "reason": self.reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdmissionSnapshot":
        if not isinstance(data, dict):
            raise TypeError("admission snapshot must be a mapping")
        return cls(
            outcome=AdmissionOutcome(data["outcome"]).value,
            provider=data["provider"],
            timestamp=data["timestamp"],
            provider_queued_depth=data["provider_queued_depth"],
            provider_capacity=data["provider_capacity"],
            global_queued_depth=data["global_queued_depth"],
            global_capacity=data["global_capacity"],
            reason=data.get("reason"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def make_admission_snapshot(
    outcome: AdmissionOutcome,
    provider: str,
    *,
    provider_queued_depth: int,
    provider_capacity: int,
    global_queued_depth: int,
    global_capacity: int,
    reason: Optional[str] = None,
    policy: QueuePolicy = DEFAULT_QUEUE_POLICY,
) -> AdmissionSnapshot:
    """Build a bounded, validated admission snapshot with a safe default reason."""
    text = _bound(reason, policy.max_admission_diagnostic_length) or _DEFAULT_REASON[outcome]
    return AdmissionSnapshot(
        outcome=outcome.value,
        provider=provider,
        timestamp=time.time(),
        provider_queued_depth=max(0, provider_queued_depth),
        provider_capacity=max(0, provider_capacity),
        global_queued_depth=max(0, global_queued_depth),
        global_capacity=max(0, global_capacity),
        reason=text,
    )


# --------------------------------------------------------------------------- #
# Immutable queue snapshot (Part K)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueueSnapshot:
    """An immutable, content-free view of one provider's queue state."""

    provider: str
    prompt_depth: int
    control_depth: int
    provider_capacity: int
    global_queued_depth: int
    quarantined: bool = False
    running_job_id: Optional[str] = None
    oldest_queued_age_s: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        for name in (
            "prompt_depth",
            "control_depth",
            "provider_capacity",
            "global_queued_depth",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.quarantined) is not bool:
            raise TypeError("quarantined must be a boolean")
        if self.running_job_id is not None and not isinstance(self.running_job_id, str):
            raise TypeError("running_job_id must be a string or null")
        if self.oldest_queued_age_s is not None:
            if isinstance(self.oldest_queued_age_s, bool) or not isinstance(
                self.oldest_queued_age_s, (int, float)
            ):
                raise TypeError("oldest_queued_age_s must be a number or null")
            if self.oldest_queued_age_s < 0:
                raise ValueError("oldest_queued_age_s must be >= 0")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "prompt_depth": self.prompt_depth,
            "control_depth": self.control_depth,
            "provider_capacity": self.provider_capacity,
            "global_queued_depth": self.global_queued_depth,
            "quarantined": self.quarantined,
            "running_job_id": self.running_job_id,
            "oldest_queued_age_s": self.oldest_queued_age_s,
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------------------- #
# Content-free queue event (Part M) — emitted OUTSIDE all locks by the manager
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueueEvent:
    """A concise, content-free queue lifecycle event."""

    kind: str
    provider: str
    timestamp: float
    job_id: Optional[str] = None
    outcome: Optional[str] = None
    prompt_depth: int = 0
    control_depth: int = 0
    global_depth: int = 0
    reason: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        QueueEventKind(self.kind)
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise TypeError("timestamp must be a number")
        if self.job_id is not None and not isinstance(self.job_id, str):
            raise TypeError("job_id must be a string or null")
        if self.outcome is not None:
            AdmissionOutcome(self.outcome)
        for name in ("prompt_depth", "control_depth", "global_depth"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("reason must be a string or null")
            if len(self.reason) > ADMISSION_DIAGNOSTIC_MAX or "\n" in self.reason or "\r" in self.reason:
                raise ValueError("reason must be flat and bounded")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "outcome": self.outcome,
            "prompt_depth": self.prompt_depth,
            "control_depth": self.control_depth,
            "global_depth": self.global_depth,
            "reason": self.reason,
            "schema_version": self.schema_version,
        }


QueueEventSink = Callable[[QueueEvent], None]


# --------------------------------------------------------------------------- #
# Part C — Queue rejection exception (compatible with existing callers)
# --------------------------------------------------------------------------- #
class BrowserQueueRejectedError(ProviderError):
    """Raised when a submission cannot be admitted to a bounded provider queue.

    A subclass of :class:`~webllm.providers.ProviderError`, so every existing
    caller that catches ``ProviderError`` keeps working unchanged. It exposes the
    typed :attr:`outcome` and the immutable :attr:`admission` snapshot while
    carrying a concise, content-free message.
    """

    def __init__(self, admission: AdmissionSnapshot) -> None:
        if not isinstance(admission, AdmissionSnapshot):
            raise TypeError("admission must be an AdmissionSnapshot")
        self.admission = admission
        self.outcome = AdmissionOutcome(admission.outcome)
        reason = admission.reason or _DEFAULT_REASON[self.outcome]
        super().__init__(f"[{admission.provider}] queue admission rejected: {reason}")


# --------------------------------------------------------------------------- #
# Part D — Bounded provider-job queue
# --------------------------------------------------------------------------- #
@dataclass
class QueueEntry:
    """One physically-queued job: a prompt (has a job id) or a control job.

    ``job`` is an opaque payload (the manager's ``_Job``); this module never
    inspects its internals. ``deadline_mono`` bounds a prompt's queue wait; it is
    ``None`` for control jobs, which are not queue-wait-expired.
    """

    job: Any
    is_prompt: bool
    job_id: Optional[str]
    enqueued_mono: float
    deadline_mono: Optional[float]
    # Manager admissions start unpublished so the worker cannot claim the entry
    # before the content-free admission and QUEUED lifecycle events are emitted.
    # Direct queue users retain the ordinary immediately-claimable default.
    claimable: bool = True

    def is_expired(self, now_mono: float) -> bool:
        return self.deadline_mono is not None and now_mono >= self.deadline_mono

    def age_s(self, now_mono: float) -> float:
        return max(0.0, now_mono - self.enqueued_mono)


class ProviderJobQueue:
    """Thread-safe, bounded FIFO job queue for ONE provider worker.

    Physical storage is a bounded ``deque``; a cancelled/timed-out queued job is
    physically *removed* (never left as an unbounded tombstone), so a long-blocked
    running job can never accumulate dead entries behind it. Ordering is strict
    FIFO across both prompt and control jobs. A condition variable wakes the
    blocked worker; ``close`` deterministically releases it with a ``None``
    sentinel. This class enforces the *control* bound itself; the *prompt* bound
    and *global* bound are enforced atomically by the manager's admission lock,
    which is the only writer of prompt entries.
    """

    def __init__(self, key: str, policy: QueuePolicy = DEFAULT_QUEUE_POLICY) -> None:
        self.key = key
        self._policy = policy
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._entries: "deque[QueueEntry]" = deque()
        self._closed = False

    @property
    def policy(self) -> QueuePolicy:
        return self._policy

    # ------------------------------------------------------------------ #
    # Depth / capacity (derived directly from physical contents — no counters)
    # ------------------------------------------------------------------ #
    def _prompt_depth_locked(self) -> int:
        return sum(1 for e in self._entries if e.is_prompt)

    def _control_depth_locked(self) -> int:
        return sum(1 for e in self._entries if not e.is_prompt)

    def prompt_depth(self) -> int:
        with self._lock:
            return self._prompt_depth_locked()

    def control_depth(self) -> int:
        with self._lock:
            return self._control_depth_locked()

    def oldest_prompt_age(self, now_mono: Optional[float] = None) -> Optional[float]:
        now = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            for e in self._entries:
                if e.is_prompt:
                    return e.age_s(now)
        return None

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    # ------------------------------------------------------------------ #
    # Enqueue
    # ------------------------------------------------------------------ #
    def enqueue_prompt(self, entry: QueueEntry) -> None:
        """Append a prompt entry (capacity already reserved by the manager).

        Raises :class:`RuntimeError` if the queue was closed — the manager rolls
        back the just-registered job so no orphan can exist.
        """
        if not entry.is_prompt:
            raise ValueError("enqueue_prompt requires a prompt entry")
        with self._cond:
            if self._closed:
                raise RuntimeError("provider queue is closed")
            if self._prompt_depth_locked() >= self._policy.max_queued_prompts_per_provider:
                raise OverflowError("provider prompt queue is full")
            self._entries.append(entry)
            if entry.claimable:
                self._cond.notify()

    def publish_prompt(self, job_id: str) -> bool:
        """Make one admitted prompt claimable after its events are published."""
        with self._cond:
            for entry in self._entries:
                if entry.is_prompt and entry.job_id == job_id:
                    entry.claimable = True
                    self._cond.notify_all()
                    return True
        return False

    def enqueue_control(self, job: Any) -> bool:
        """Append a control job if under the control bound. Returns admission.

        Control jobs have their own small reserved capacity so login/status calls
        cannot grow without bound and can never starve the prompt bound.
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("provider queue is closed")
            if self._control_depth_locked() >= self._policy.max_queued_control_jobs_per_provider:
                return False
            self._entries.append(
                QueueEntry(
                    job=job,
                    is_prompt=False,
                    job_id=None,
                    enqueued_mono=time.monotonic(),
                    deadline_mono=None,
                )
            )
            self._cond.notify()
            return True

    # ------------------------------------------------------------------ #
    # Dequeue (worker side)
    # ------------------------------------------------------------------ #
    def get(self) -> Optional[QueueEntry]:
        """Block for the next FIFO entry, or ``None`` once closed and drained.

        No busy polling: waits on the condition variable until an entry arrives or
        the queue is closed. A ``None`` return is the deterministic shutdown
        sentinel that tells the worker run loop to stop.
        """
        with self._cond:
            while True:
                while not self._entries and not self._closed:
                    self._cond.wait()
                if not self._entries:
                    return None  # closed and drained
                if self._entries[0].claimable:
                    return self._entries.popleft()
                self._cond.wait()

    # ------------------------------------------------------------------ #
    # Removal (queued cancellation / timeout / expiry)
    # ------------------------------------------------------------------ #
    def remove_prompt(self, job_id: str) -> Optional[QueueEntry]:
        """Physically remove the queued prompt with ``job_id``; return it or None.

        Returns ``None`` if the entry is no longer queued (already claimed by the
        worker or already removed). Physical removal frees the logical slot
        immediately and prevents any tombstone buildup.
        """
        with self._cond:
            for i, e in enumerate(self._entries):
                if e.is_prompt and e.job_id == job_id:
                    del self._entries[i]
                    self._cond.notify_all()
                    return e
        return None

    def collect_expired(self, now_mono: Optional[float] = None) -> list[QueueEntry]:
        """Remove and return every queued prompt whose queue wait has expired.

        FIFO order among the survivors is preserved. Control jobs are never
        expired here.
        """
        now = time.monotonic() if now_mono is None else now_mono
        expired: list[QueueEntry] = []
        with self._cond:
            if not self._entries:
                return expired
            survivors: "deque[QueueEntry]" = deque()
            for e in self._entries:
                if e.is_prompt and e.is_expired(now):
                    expired.append(e)
                else:
                    survivors.append(e)
            if expired:
                self._entries = survivors
                self._cond.notify_all()
        return expired

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def drain(self) -> list[QueueEntry]:
        """Remove and return all pending entries (prompt and control)."""
        with self._lock:
            pending = list(self._entries)
            self._entries.clear()
            return pending

    def close(self) -> None:
        """Mark the queue closed and wake any blocked worker deterministically."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    def snapshot(
        self,
        *,
        global_queued_depth: int,
        quarantined: bool,
        running_job_id: Optional[str],
        now_mono: Optional[float] = None,
    ) -> QueueSnapshot:
        now = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            prompt_depth = self._prompt_depth_locked()
            control_depth = self._control_depth_locked()
            oldest: Optional[float] = None
            for e in self._entries:
                if e.is_prompt:
                    oldest = e.age_s(now)
                    break
        return QueueSnapshot(
            provider=self.key,
            prompt_depth=prompt_depth,
            control_depth=control_depth,
            provider_capacity=self._policy.max_queued_prompts_per_provider,
            global_queued_depth=max(0, global_queued_depth),
            quarantined=bool(quarantined),
            running_job_id=running_job_id,
            oldest_queued_age_s=oldest,
        )


__all__ = [
    "ADMISSION_DIAGNOSTIC_MAX",
    "QueuePolicy",
    "DEFAULT_QUEUE_POLICY",
    "AdmissionOutcome",
    "AdmissionSnapshot",
    "make_admission_snapshot",
    "QueueSnapshot",
    "QueueEventKind",
    "QueueEvent",
    "QueueEventSink",
    "BrowserQueueRejectedError",
    "QueueEntry",
    "ProviderJobQueue",
]
