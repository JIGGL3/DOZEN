"""Token budgeting for context building.

Phase 1.4 uses the SADD's calibrated character heuristic (~4 chars/token for
prose) — deliberately simple, deliberately replaceable: anything with an
``estimate(text) -> int`` method satisfies the ``TokenEstimator`` protocol,
so a real per-provider tokenizer (Phase 1.5+, ``EstimatorConfig.
enable_exact_tokenizers``) drops in without touching selector or builder.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_TOKEN_BUDGET = 12_000


@runtime_checkable
class TokenEstimator(Protocol):
    """The replaceable estimation seam."""

    def estimate(self, text: str) -> int: ...


class TokenBudgetEstimator:
    """Character-ratio estimator: ``ceil(len(text) / chars_per_token)``.

    Ratio 4.0 matches the SADD prose calibration and the storage layer's
    bookkeeping ratio, so numbers agree across the system.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self.chars_per_token = chars_per_token

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        chars = len(text)
        return int(-(-chars // self.chars_per_token))  # ceil division
