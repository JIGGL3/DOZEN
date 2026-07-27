"""ULID generation — time-sortable identifiers (Crockford base32, spec-compliant).

Pure utility: 48-bit millisecond timestamp + 80 bits of randomness, encoded as
26 characters. Monotonic within one process: two ULIDs minted in the same
millisecond increment the random component, so sort order always matches
creation order — the property the message log and pagination rely on.
"""

from __future__ import annotations

import secrets
import threading
import time

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_TIME_LEN = 10   # 48 bits -> 10 chars
_RAND_LEN = 16   # 80 bits -> 16 chars
_MAX_RANDOM = (1 << 80) - 1

_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(_ENCODING[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def generate_ulid(timestamp_ms: int | None = None) -> str:
    """Return a 26-character ULID; monotonic within this process."""
    global _last_ms, _last_random
    now_ms = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    with _lock:
        if now_ms == _last_ms:
            # Same millisecond: increment randomness to preserve sort order.
            _last_random = (_last_random + 1) & _MAX_RANDOM
        else:
            _last_ms = now_ms
            _last_random = secrets.randbits(80)
        return _encode(now_ms, _TIME_LEN) + _encode(_last_random, _RAND_LEN)


def is_ulid(value: str) -> bool:
    """Structural check: 26 chars, all from the Crockford alphabet."""
    return (
        isinstance(value, str)
        and len(value) == _TIME_LEN + _RAND_LEN
        and all(c in _ENCODING for c in value)
    )
