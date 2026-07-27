"""Content hashing — cache keys for token-estimate memoization (pure function)."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Stable 32-hex-char digest of ``text`` (SHA-256, truncated)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
