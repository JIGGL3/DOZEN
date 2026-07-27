"""Time utilities — the canonical Timestamp format helpers (pure functions).

The domain's ``Timestamp`` is ISO-8601 UTC with millisecond precision and a
trailing ``Z`` (e.g. ``2026-07-08T09:15:32.123Z``): human-readable in JSONL,
lexicographically sortable, and unambiguous across machines.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Current UTC time in the canonical Timestamp format."""
    return format_iso(datetime.now(timezone.utc))


def format_iso(moment: datetime) -> str:
    """Format an aware datetime as canonical ISO-8601 UTC (…Z, ms precision)."""
    utc = moment.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def parse_iso(value: str) -> datetime:
    """Parse a canonical Timestamp back into an aware UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
