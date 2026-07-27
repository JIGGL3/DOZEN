"""Phase 4G — deterministic Gemini prompt-submission state machine.

WHY THIS MODULE EXISTS
----------------------
Live testing showed Gemini submission was intermittent: some prompts were
entered and sent, some never left the composer, and the browser job then waited
for a response Gemini never received. The generic selector-driven ``send`` could
not confirm that (a) the RIGHT composer was chosen, (b) the full prompt actually
landed in it, or (c) Gemini actually accepted the message. It could also press
Enter and click Send, producing duplicate submissions.

This module makes Gemini submission a small, explicit, DETERMINISTIC state
machine that is verifiable WITHOUT a live browser: it operates over a minimal
:class:`GeminiDom` port (backed by Playwright in production and by a fake DOM in
tests). It owns:

* a LAYERED locator strategy (accessible role/name → aria → data → known
  component selectors → conservative structural fallback);
* validated composer selection (visible, enabled, editable, inside the active
  chat composer region — never a search/settings/hidden-template box);
* deterministic text insertion WITH read-back confirmation;
* an acknowledgement state machine (composer cleared / new user bubble whose
  normalized content matches / generation began / message count advanced);
* EXACTLY-ONCE sending — at most one physical send action per attempt, and an
  ambiguous send never triggers a second;
* typed failure categories and bounded, content-free selector-drift diagnostics.

It logs no prompt, response, cookie, DOM text or credential — only lifecycle
metadata (strategy ids, match counts, element state, route category).

SCOPE: Gemini submission only. It does not restart tabs, recover, fail over,
touch login/captcha, or change any other provider's behaviour. The live DOM
audit (Part G) and manual logged-in acceptance are OUT of scope for the code
here; this module is validated against fake DOMs that model the documented
current and previous Gemini DOMs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence


# --------------------------------------------------------------------------- #
# Typed failure vocabulary (Part J / Part L)
# --------------------------------------------------------------------------- #
class SubmissionErrorCode(str, Enum):
    """Why a Gemini submission could not be confirmed. Distinct, content-free."""

    COMPOSER_NOT_FOUND = "composer_not_found"
    COMPOSER_NOT_EDITABLE = "composer_not_editable"
    PROMPT_INSERTION_MISMATCH = "prompt_insertion_mismatch"
    SEND_CONTROL_NOT_FOUND = "send_control_not_found"
    SEND_CONTROL_DISABLED = "send_control_disabled"
    SUBMISSION_NOT_CONFIRMED = "submission_not_confirmed"
    SELECTOR_DRIFT_DETECTED = "selector_drift_detected"
    # Part L: kept distinct from selector drift so callers never conflate them.
    LOGIN_REQUIRED = "login_required"
    CANCELLED = "cancelled"


class SubmissionStatus(str, Enum):
    SUBMITTED = "submitted"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Ordered locator strategies (Part H). Stable signals first, structural last.
class LocatorStrategy(str, Enum):
    ROLE_NAME = "role_name"      # accessible role + accessible name
    ARIA = "aria"                # stable aria-* attributes
    DATA = "data"                # stable data-* attributes
    KNOWN = "known"              # current known Gemini component selectors
    STRUCTURAL = "structural"    # conservative structural fallback


# --------------------------------------------------------------------------- #
# The minimal DOM port
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DomElement:
    """One candidate element, as the state machine sees it.

    ``handle`` is an opaque token the port resolves back to a real element; the
    state machine never inspects it. All boolean/string fields are the trusted,
    already-computed state of the element (visibility, enabled, editable, its
    route/region category, tag and the safe attribute names present).
    """

    handle: object
    tag: str = ""
    role: str = ""
    accessible_name: str = ""
    visible: bool = False
    enabled: bool = False
    editable: bool = False
    is_stop_control: bool = False
    is_attachment_control: bool = False
    region: str = "composer"     # composer|search|settings|feedback|hidden|unknown
    # Content-free identity of the nearest chat/composer region.  Composer and
    # send controls must have the SAME non-empty identity; the coarse ``region``
    # category alone cannot distinguish two simultaneous chat panes.
    chat_region: str = ""
    aria_names: tuple[str, ...] = ()
    data_names: tuple[str, ...] = ()


class GeminiDom(Protocol):
    """The narrow surface the submission state machine drives.

    Backed by Playwright in production (same-thread, on the provider worker) and
    by an in-memory fake DOM in tests. Every method is side-effect-scoped and
    must never raise for ordinary "not found" cases — return empties instead.
    """

    def query(self, selector: str) -> Sequence[DomElement]: ...
    def focus(self, element: DomElement) -> bool: ...
    def insert_text(self, element: DomElement, text: str) -> bool: ...
    def read_composer(self, element: DomElement) -> Optional[str]: ...
    def clear(self, element: DomElement) -> None: ...
    def click(self, element: DomElement) -> bool: ...
    def user_messages(self, element: DomElement) -> Sequence[str]: ...
    def is_generating(self, element: DomElement) -> bool: ...
    def login_required(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Layered selector sets (Part H). Additive over the adapter's own selectors:
# stable signals first; the known Gemini component selectors and a conservative
# structural fallback last.
# --------------------------------------------------------------------------- #
COMPOSER_SELECTORS: tuple[tuple[LocatorStrategy, str], ...] = (
    (LocatorStrategy.ROLE_NAME, "div[role='textbox'][contenteditable='true']"),
    (LocatorStrategy.ARIA, "div[contenteditable='true'][aria-label]"),
    (LocatorStrategy.DATA, "rich-textarea div.ql-editor[contenteditable='true']"),
    (LocatorStrategy.KNOWN, "div.ql-editor[contenteditable='true']"),
    (LocatorStrategy.STRUCTURAL, "div[contenteditable='true']"),
)

SEND_SELECTORS: tuple[tuple[LocatorStrategy, str], ...] = (
    (LocatorStrategy.ROLE_NAME, "button[aria-label='Send message']"),
    (LocatorStrategy.ARIA, "button[aria-label*='Send' i]"),
    (LocatorStrategy.DATA, "button.send-button[mat-icon-button]"),
    (LocatorStrategy.KNOWN, "button.send-button"),
    (LocatorStrategy.STRUCTURAL, "button[type='submit']"),
)

# Composer regions that are NEVER the active chat composer.
_INVALID_REGIONS = frozenset({"search", "settings", "feedback", "hidden"})


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocatorDiagnostics:
    """Bounded, content-free record of a locator pass (Part L)."""

    role: str                       # "composer" | "send"
    strategies_attempted: tuple[str, ...] = ()
    match_counts: dict[str, int] = field(default_factory=dict)
    saw_visible: bool = False
    saw_enabled: bool = False
    saw_editable: bool = False
    rejected_regions: tuple[str, ...] = ()
    chosen_strategy: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "strategies_attempted": list(self.strategies_attempted),
            "match_counts": dict(self.match_counts),
            "saw_visible": self.saw_visible,
            "saw_enabled": self.saw_enabled,
            "saw_editable": self.saw_editable,
            "rejected_regions": list(self.rejected_regions),
            "chosen_strategy": self.chosen_strategy,
        }


@dataclass(frozen=True)
class SubmissionResult:
    status: SubmissionStatus
    error_code: Optional[SubmissionErrorCode] = None
    diagnostic: str = ""
    send_actions: int = 0
    acknowledgement: str = ""            # which ack signal confirmed it
    composer_strategy: Optional[str] = None
    send_strategy: Optional[str] = None
    locator_diagnostics: tuple[LocatorDiagnostics, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is SubmissionStatus.SUBMITTED


# --------------------------------------------------------------------------- #
# Normalization / bounded hashing (content-free surfaces only)
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def bounded_hash(text: str) -> str:
    """A short, content-free digest of normalized text for bubble matching."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def _matches(candidate: str, intended: str) -> bool:
    """Whether ``candidate`` is the intended prompt (exact normalized, or a
    reliable prefix — big pastes may render with a trailing attachment chip)."""
    nc, ni = normalize(candidate), normalize(intended)
    if not ni:
        return False
    if nc == ni:
        return True
    # A rendered attachment chip or trailing UI may add text, but the COMPLETE
    # intended prompt must still be present.  A strong prefix is not sufficient:
    # accepting only the first 200 characters can send a truncated prompt.
    return ni in nc


# --------------------------------------------------------------------------- #
# Locator (Part H)
# --------------------------------------------------------------------------- #
def _validate_composer(el: DomElement) -> bool:
    return bool(
        el.visible and el.enabled and el.editable
        and el.region not in _INVALID_REGIONS
        and bool(el.chat_region)
        and not el.is_stop_control
    )


def _validate_send(el: DomElement) -> bool:
    return bool(
        el.visible and el.enabled
        and not el.is_stop_control
        and not el.is_attachment_control
        and el.region not in _INVALID_REGIONS
        and bool(el.chat_region)
    )


def _locate(
    dom: GeminiDom,
    role: str,
    selectors: Sequence[tuple[LocatorStrategy, str]],
    validate: Callable[[DomElement], bool],
) -> tuple[Optional[DomElement], Optional[LocatorStrategy], LocatorDiagnostics]:
    """One layered locator pass. Returns the first UNAMBIGUOUS valid element.

    A strategy that resolves to exactly one valid element wins. Multiple valid
    matches for one strategy are ambiguous and rejected (never pick the first
    arbitrary one). Diagnostics are bounded and content-free.
    """
    attempted: list[str] = []
    counts: dict[str, int] = {}
    saw_visible = saw_enabled = saw_editable = False
    rejected_regions: set[str] = set()

    for strategy, selector in selectors:
        attempted.append(strategy.value)
        try:
            matches = list(dom.query(selector))
        except Exception:  # noqa: BLE001 - a bad selector is just "no match"
            matches = []
        counts[strategy.value] = len(matches)
        valid: list[DomElement] = []
        for el in matches:
            saw_visible = saw_visible or el.visible
            saw_enabled = saw_enabled or el.enabled
            saw_editable = saw_editable or el.editable
            if el.region in _INVALID_REGIONS:
                rejected_regions.add(el.region)
            if validate(el):
                valid.append(el)
        if len(valid) == 1:
            diag = LocatorDiagnostics(
                role=role, strategies_attempted=tuple(attempted),
                match_counts=dict(counts), saw_visible=saw_visible,
                saw_enabled=saw_enabled, saw_editable=saw_editable,
                rejected_regions=tuple(sorted(rejected_regions)),
                chosen_strategy=strategy.value,
            )
            return valid[0], strategy, diag
        # len(valid) > 1 → ambiguous for this strategy; fall through to the next
        # (more structural) strategy rather than guessing.

    diag = LocatorDiagnostics(
        role=role, strategies_attempted=tuple(attempted),
        match_counts=dict(counts), saw_visible=saw_visible,
        saw_enabled=saw_enabled, saw_editable=saw_editable,
        rejected_regions=tuple(sorted(rejected_regions)),
    )
    return None, None, diag


# --------------------------------------------------------------------------- #
# Bounded polling primitive (no sleeps — the caller/DOM advances state)
# --------------------------------------------------------------------------- #
def _poll(predicate: Callable[[], bool], attempts: int) -> bool:
    for _ in range(max(1, attempts)):
        if predicate():
            return True
    return False


# --------------------------------------------------------------------------- #
# The submission state machine (Parts I + J + K)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubmissionConfig:
    """Bounds for one submission attempt. Deterministic poll counts, no sleeps."""

    send_enable_polls: int = 20
    acknowledgement_polls: int = 40
    clear_stale_composer: bool = True


DEFAULT_SUBMISSION_CONFIG = SubmissionConfig()


def submit_prompt(
    dom: GeminiDom,
    prompt: str,
    *,
    config: SubmissionConfig = DEFAULT_SUBMISSION_CONFIG,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> SubmissionResult:
    """Locate, insert, confirm and send exactly once, then confirm acceptance.

    Returns a typed :class:`SubmissionResult`. Guarantees AT MOST one physical
    send action per call: an ambiguous send never triggers a second, and any
    unconfirmed state returns a typed error rather than silently waiting.
    """
    cancel = should_cancel or (lambda: False)
    diagnostics: list[LocatorDiagnostics] = []

    def cancelled() -> SubmissionResult:
        return SubmissionResult(
            status=SubmissionStatus.CANCELLED,
            error_code=SubmissionErrorCode.CANCELLED,
            diagnostic="cancelled before submission was confirmed",
            locator_diagnostics=tuple(diagnostics),
        )

    if cancel():
        return cancelled()
    if dom.login_required():
        return SubmissionResult(
            status=SubmissionStatus.FAILED,
            error_code=SubmissionErrorCode.LOGIN_REQUIRED,
            diagnostic="an authenticated Gemini chat surface was not present",
        )

    # 1. Locate + validate the active composer.
    composer, composer_strategy, composer_diag = _locate(
        dom, "composer", COMPOSER_SELECTORS, _validate_composer
    )
    diagnostics.append(composer_diag)
    if composer is None:
        # Distinguish "nothing matched at all" (drift) from "matched but not
        # usable" (editable/region problems).
        any_match = any(composer_diag.match_counts.values())
        if not any_match:
            return SubmissionResult(
                status=SubmissionStatus.FAILED,
                error_code=SubmissionErrorCode.SELECTOR_DRIFT_DETECTED,
                diagnostic="no composer matched any locator strategy",
                locator_diagnostics=tuple(diagnostics),
            )
        if composer_diag.saw_visible and not composer_diag.saw_editable:
            return SubmissionResult(
                status=SubmissionStatus.FAILED,
                error_code=SubmissionErrorCode.COMPOSER_NOT_EDITABLE,
                diagnostic="a composer was visible but not editable",
                locator_diagnostics=tuple(diagnostics),
            )
        return SubmissionResult(
            status=SubmissionStatus.FAILED,
            error_code=SubmissionErrorCode.COMPOSER_NOT_FOUND,
            diagnostic="no usable active-composer element was found",
            locator_diagnostics=tuple(diagnostics),
        )

    if cancel():
        return cancelled()

    # 2. Focus, 3. clear stale content only when DOZEN owns it, 4. insert.
    dom.focus(composer)
    if config.clear_stale_composer and normalize(dom.read_composer(composer)):
        dom.clear(composer)
    dom.insert_text(composer, prompt)

    if cancel():
        return cancelled()

    # 5-7. Read back and confirm insertion BEFORE proceeding.
    if not _matches(dom.read_composer(composer), prompt):
        return SubmissionResult(
            status=SubmissionStatus.FAILED,
            error_code=SubmissionErrorCode.PROMPT_INSERTION_MISMATCH,
            diagnostic="the composer content did not match the intended prompt",
            composer_strategy=composer_strategy.value if composer_strategy else None,
            locator_diagnostics=tuple(diagnostics),
        )

    # 8-9. Locate the associated send control, waiting boundedly for it to be
    # usable. ``_locate`` returns only a visible+enabled+associated control, so
    # re-locating each poll picks up a late-enabling button and NEVER yields a
    # disabled one to click.
    ready: Optional[DomElement] = None
    send_strategy = None
    send_diag: Optional[LocatorDiagnostics] = None
    for _ in range(max(1, config.send_enable_polls)):
        if cancel():
            return cancelled()
        candidate, strategy, send_diag = _locate(
            dom,
            "send",
            SEND_SELECTORS,
            lambda element: (
                _validate_send(element)
                and element.chat_region == composer.chat_region
            ),
        )
        if candidate is not None:
            ready, send_strategy = candidate, strategy
            break
    if send_diag is not None:
        diagnostics.append(send_diag)
    if ready is None:
        any_match = bool(send_diag and any(send_diag.match_counts.values()))
        saw_visible = bool(send_diag and send_diag.saw_visible)
        saw_enabled = bool(send_diag and send_diag.saw_enabled)
        code = (
            SubmissionErrorCode.SELECTOR_DRIFT_DETECTED if not any_match
            else SubmissionErrorCode.SEND_CONTROL_DISABLED
            if saw_visible and not saw_enabled
            else SubmissionErrorCode.SEND_CONTROL_NOT_FOUND
        )
        return SubmissionResult(
            status=SubmissionStatus.FAILED, error_code=code,
            diagnostic="no usable, enabled send control associated with the composer",
            composer_strategy=composer_strategy.value if composer_strategy else None,
            locator_diagnostics=tuple(diagnostics),
        )

    if cancel():
        return cancelled()

    # 10. Capture a pre-submission conversation marker.
    before_messages = list(dom.user_messages(composer))
    before_count = len(before_messages)
    intended_hash = bounded_hash(prompt)

    # 11. Perform EXACTLY ONE send action. An exception after dispatch is
    # ambiguous — we never issue a second click; we go straight to confirmation.
    send_actions = 1
    try:
        dom.click(ready)
    except Exception:  # noqa: BLE001 - ambiguous; confirm, never resend
        pass

    # 12. Confirm Gemini accepted the prompt via any trusted signal.
    acknowledgement = ""

    def acknowledged() -> bool:
        nonlocal acknowledgement
        messages = list(dom.user_messages(composer))
        if len(messages) > before_count:
            last = messages[-1]
            if bounded_hash(last) == intended_hash or _matches(last, prompt):
                acknowledgement = "user_bubble_matched"
                return True
        composer_text = dom.read_composer(composer)
        if composer_text is not None and not normalize(composer_text):
            acknowledgement = "composer_cleared"
            return True
        if dom.is_generating(composer):
            acknowledgement = "generation_started"
            return True
        return False

    confirmed = False
    for _ in range(max(1, config.acknowledgement_polls)):
        # Check acknowledgement first: if the send is already confirmed, a
        # simultaneous cancellation must not relabel it as unsubmitted.
        if acknowledged():
            confirmed = True
            break
        if cancel():
            return SubmissionResult(
                status=SubmissionStatus.CANCELLED,
                error_code=SubmissionErrorCode.CANCELLED,
                diagnostic="cancelled during acknowledgement after send action",
                send_actions=send_actions,
                composer_strategy=(
                    composer_strategy.value if composer_strategy else None
                ),
                send_strategy=send_strategy.value if send_strategy else None,
                locator_diagnostics=tuple(diagnostics),
            )
    if not confirmed:
        return SubmissionResult(
            status=SubmissionStatus.FAILED,
            error_code=SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED,
            diagnostic="the send action was not confirmed by any trusted signal",
            send_actions=send_actions,
            composer_strategy=composer_strategy.value if composer_strategy else None,
            send_strategy=send_strategy.value if send_strategy else None,
            locator_diagnostics=tuple(diagnostics),
        )

    return SubmissionResult(
        status=SubmissionStatus.SUBMITTED,
        send_actions=send_actions,
        acknowledgement=acknowledgement,
        composer_strategy=composer_strategy.value if composer_strategy else None,
        send_strategy=send_strategy.value if send_strategy else None,
        locator_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "SubmissionErrorCode",
    "SubmissionStatus",
    "LocatorStrategy",
    "DomElement",
    "GeminiDom",
    "COMPOSER_SELECTORS",
    "SEND_SELECTORS",
    "LocatorDiagnostics",
    "SubmissionResult",
    "SubmissionConfig",
    "DEFAULT_SUBMISSION_CONFIG",
    "submit_prompt",
    "normalize",
    "bounded_hash",
]
