"""Injectable clock (Phase 2.1.2).

Rule: nothing in the reliability layer reads wall or monotonic time directly —
all time flows through a ``ReliabilityClock`` so tests can be deterministic.

Two time sources on one object, on purpose:
* ``now()``       — wall-clock ``Timestamp`` (canonical ISO-8601 UTC) for records.
* ``monotonic()`` — monotonic seconds for durations (immune to clock jumps).
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from ..context.domain.types import Timestamp
from ..context.utils import utc_now_iso


@runtime_checkable
class ReliabilityClock(Protocol):
    def now(self) -> Timestamp: ...

    def monotonic(self) -> float: ...


class SystemReliabilityClock:
    """Production clock: real UTC wall time + ``time.monotonic``."""

    def now(self) -> Timestamp:
        return Timestamp(utc_now_iso())

    def monotonic(self) -> float:
        return time.monotonic()
