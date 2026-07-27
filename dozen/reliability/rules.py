"""Classification rules (Phase 2.2.1) — pure predicates over FailureEvidence.

A rule is data + one pure function: no browser actions, no I/O, no state.
Every rule yields a confidence in [0, 1] when it matches (None when it
doesn't). Multiple rules may match one evidence object; the classifier keeps
them all and the highest confidence wins.

Rule order encodes tie-breaking (earlier wins on equal confidence), mirroring
the SADD-002 §5.3 cascade. New rules — including provider-specific ones in
later phases — are appended via ``FailureClassifier.register_rule`` or by
building a custom rule list; the engine never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from . import patterns as p
from .evidence import FailureEvidence
from .types import FailureType

# A matcher inspects evidence and returns a confidence, or None for no match.
RuleMatcher = Callable[[FailureEvidence, str], Optional[float]]


@dataclass(frozen=True)
class ClassificationRule:
    name: str
    failure_type: FailureType
    matcher: RuleMatcher
    description: str = ""

    def evaluate(self, evidence: FailureEvidence, haystack: str) -> Optional[float]:
        try:
            confidence = self.matcher(evidence, haystack)
        except Exception:
            return None  # a broken rule must never break classification
        if confidence is None:
            return None
        return max(0.0, min(1.0, float(confidence)))


# --------------------------------------------------------------------------- #
# Matchers (evidence, lowercased haystack) -> confidence | None
# --------------------------------------------------------------------------- #
def _browser_crash(e: FailureEvidence, text: str) -> Optional[float]:
    if e.browser_connected is False:
        return 0.95
    # Playwright's combined page-op error ("Target page, context or browser
    # has been closed") is a PAGE-level signal — its "browser has been closed"
    # substring must not trip this rule; the tab-closed rule owns it. Same
    # judgement BrowserManager._is_closed_error applies.
    if p.any_phrase(text, p.TAB_CLOSED_PHRASES):
        return None
    if p.any_phrase(text, p.BROWSER_CRASH_PHRASES):
        return 0.95
    return None


def _tab_closed(e: FailureEvidence, text: str) -> Optional[float]:
    if e.tab_connected is False and e.browser_connected is not False:
        return 0.9
    if p.any_phrase(text, p.TAB_CLOSED_PHRASES):
        return 0.9
    return None


def _captcha(e: FailureEvidence, text: str) -> Optional[float]:
    return 0.95 if p.any_phrase(text, p.CAPTCHA_PHRASES) else None


def _login_required(e: FailureEvidence, text: str) -> Optional[float]:
    if e.url and p.looks_like_login_url(e.url):
        return 0.9
    if p.any_phrase(text, p.LOGIN_PHRASES):
        return 0.85
    return None


def _rate_limit(e: FailureEvidence, text: str) -> Optional[float]:
    if e.http_status in p.RATE_LIMIT_STATUS:
        return 0.9
    if p.any_phrase(text, p.RATE_LIMIT_PHRASES):
        return 0.85
    return None


def _prompt_rejected(e: FailureEvidence, text: str) -> Optional[float]:
    return 0.9 if p.any_phrase(text, p.PROMPT_REJECTED_PHRASES) else None


def _dom_changed(e: FailureEvidence, text: str) -> Optional[float]:
    return 0.8 if p.any_phrase(text, p.DOM_CHANGED_PHRASES) else None


def _generation_stalled(e: FailureEvidence, text: str) -> Optional[float]:
    if p.any_phrase(text, p.EMPTY_RESPONSE_PHRASES):
        return 0.8
    return None


def _network_error(e: FailureEvidence, text: str) -> Optional[float]:
    return 0.85 if p.any_phrase(text, p.NETWORK_PHRASES) else None


def _model_busy(e: FailureEvidence, text: str) -> Optional[float]:
    if e.http_status in p.SERVER_BUSY_STATUS:
        return 0.7
    if p.any_phrase(text, p.MODEL_BUSY_PHRASES):
        return 0.7
    return None


def _output_corrupted(e: FailureEvidence, text: str) -> Optional[float]:
    return 0.75 if p.any_phrase(text, p.OUTPUT_CORRUPTED_PHRASES) else None


def _timeout(e: FailureEvidence, text: str) -> Optional[float]:
    if e.exception_type in p.TIMEOUT_EXCEPTION_TYPES:
        return 0.75
    if p.any_phrase(text, p.TIMEOUT_PHRASES):
        return 0.7
    return None


# --------------------------------------------------------------------------- #
# The default provider-independent ruleset (order == SADD cascade order)
# --------------------------------------------------------------------------- #
DEFAULT_RULES: tuple[ClassificationRule, ...] = (
    ClassificationRule("browser-crash", FailureType.BROWSER_CRASH, _browser_crash,
                       "driver/browser process gone or connection severed"),
    ClassificationRule("tab-closed", FailureType.TAB_CLOSED, _tab_closed,
                       "page/context closed while the browser survives"),
    ClassificationRule("captcha-wall", FailureType.CAPTCHA, _captcha,
                       "captcha / Cloudflare human-verification signals"),
    ClassificationRule("login-required", FailureType.LOGIN_REQUIRED, _login_required,
                       "login wall, expired session, or auth URL"),
    ClassificationRule("rate-limit", FailureType.RATE_LIMIT, _rate_limit,
                       "HTTP 429 or throttling language"),
    ClassificationRule("prompt-rejected", FailureType.PROMPT_REJECTED, _prompt_rejected,
                       "composer refused the input (too long / blocked)"),
    ClassificationRule("dom-changed", FailureType.DOM_CHANGED, _dom_changed,
                       "expected elements unresolvable — UI drift"),
    ClassificationRule("generation-stalled", FailureType.GENERATION_STALLED,
                       _generation_stalled,
                       "generation produced nothing / empty scrape"),
    ClassificationRule("network-error", FailureType.NETWORK_ERROR, _network_error,
                       "connectivity failures below the app layer"),
    ClassificationRule("model-busy", FailureType.MODEL_BUSY, _model_busy,
                       "provider-side congestion (5xx / busy language)"),
    ClassificationRule("output-corrupted", FailureType.OUTPUT_CORRUPTED,
                       _output_corrupted,
                       "response arrived but is garbled/unparseable"),
    ClassificationRule("timeout", FailureType.TIMEOUT, _timeout,
                       "no response within the deadline, DOM otherwise fine"),
)
