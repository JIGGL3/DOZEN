"""FailureClassifier — evidence in, one explained verdict out (Phase 2.2.1).

    FailureEvidence -> [rules] -> FailureClassification -> FailureType

Deterministic and pure: every registered rule is evaluated, ALL matches are
kept (name + confidence), the highest confidence wins (rule order breaks
ties), and a verdict below ``min_confidence`` downgrades to UNKNOWN while
still reporting what almost matched. No browser actions, no recovery, no
retries — classification only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .config import DetectionConfig
from .evidence import FailureEvidence
from .rules import DEFAULT_RULES, ClassificationRule
from .types import FailureType, parse_enum


@dataclass(frozen=True)
class MatchedRule:
    """One rule that matched, with the confidence it reported."""

    name: str
    failure_type: FailureType
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "failure_type": self.failure_type.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatchedRule":
        return cls(
            name=str(data.get("name", "")),
            failure_type=parse_enum(FailureType, data.get("failure_type", "unknown")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class FailureClassification:
    """The classifier's complete, explainable verdict."""

    SCHEMA_VERSION = 1

    failure_type: FailureType
    confidence: float
    matched_rules: tuple[MatchedRule, ...] = ()
    explanation: str = ""
    raw_evidence: Optional[FailureEvidence] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "failure_type": self.failure_type.value,
            "confidence": self.confidence,
            "matched_rules": [r.to_dict() for r in self.matched_rules],
            "explanation": self.explanation,
            "raw_evidence": self.raw_evidence.to_dict() if self.raw_evidence else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FailureClassification":
        rules = data.get("matched_rules")
        evidence = data.get("raw_evidence")
        return cls(
            failure_type=parse_enum(FailureType, data.get("failure_type", "unknown")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            matched_rules=tuple(
                MatchedRule.from_dict(r) for r in rules if isinstance(r, dict)
            ) if isinstance(rules, (list, tuple)) else (),
            explanation=str(data.get("explanation", "")),
            raw_evidence=FailureEvidence.from_dict(evidence)
            if isinstance(evidence, dict) else None,
        )


_UNMATCHED_CONFIDENCE = 0.3  # UNKNOWN verdicts still carry a floor confidence


class FailureClassifier:
    def __init__(
        self,
        rules: Optional[Sequence[ClassificationRule]] = None,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        self._rules: list[ClassificationRule] = list(
            DEFAULT_RULES if rules is None else rules
        )
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------ #
    def register_rule(self, rule: ClassificationRule) -> None:
        """Extension point: future provider-specific rules append here."""
        self._rules.append(rule)

    @property
    def rules(self) -> tuple[ClassificationRule, ...]:
        return tuple(self._rules)

    # ------------------------------------------------------------------ #
    def classify(self, evidence: FailureEvidence) -> FailureClassification:
        haystack = evidence.searchable_text()
        matched: list[MatchedRule] = []
        for rule in self._rules:
            confidence = rule.evaluate(evidence, haystack)
            if confidence is not None and confidence > 0.0:
                matched.append(MatchedRule(rule.name, rule.failure_type, confidence))

        if not matched:
            return FailureClassification(
                failure_type=FailureType.UNKNOWN,
                confidence=_UNMATCHED_CONFIDENCE,
                matched_rules=(),
                explanation="no classification rule matched the evidence",
                raw_evidence=evidence,
            )

        # Highest confidence wins; list order (== rule order) breaks ties.
        winner = max(matched, key=lambda m: m.confidence)
        if winner.confidence < self.config.min_confidence:
            return FailureClassification(
                failure_type=FailureType.UNKNOWN,
                confidence=winner.confidence,
                matched_rules=tuple(matched),
                explanation=(
                    f"best match {winner.name!r} ({winner.failure_type.value}, "
                    f"{winner.confidence:.2f}) is below the confidence floor "
                    f"{self.config.min_confidence:.2f}"
                ),
                raw_evidence=evidence,
            )

        others = ", ".join(
            f"{m.name}={m.confidence:.2f}" for m in matched if m is not winner
        )
        explanation = (
            f"{winner.failure_type.value} via rule {winner.name!r} "
            f"(confidence {winner.confidence:.2f})"
            + (f"; also matched: {others}" if others else "")
        )
        return FailureClassification(
            failure_type=winner.failure_type,
            confidence=winner.confidence,
            matched_rules=tuple(matched),
            explanation=explanation,
            raw_evidence=evidence,
        )
