"""Provider adapters: per-LLM web-UI knowledge.

Each adapter encapsulates everything that is specific to one web interface
(ChatGPT, Claude, Gemini): where to log in, where a fresh chat lives, how to
find the input box, how to submit, how to know generation finished, and how to
scrape the final answer.

The actual driving is done by a *generic* ``send`` routine that is selector-
driven and resilient to UI churn:

* Multiple candidate selectors per role (first match wins).
* Text is inserted via ``keyboard.insert_text`` (a single input event) so
  multi-line prompts never accidentally submit on a stray Enter.
* Completion is detected with a hybrid strategy: the provider's "stop
  generating" button must be gone AND the last answer's text must stop changing
  for a short stability window. This survives most selector drift because it
  does not depend on a perfectly-named "done" element.

If a provider tweaks its DOM, you usually only need to add a selector string to
one of the lists below — not rewrite logic.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from dozen.cancellation import CancelledError
from dozen.validation import clip_text

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from playwright.sync_api import Locator, Page

    from .browser_cancellation import (
        CancellationObservation,
        InterruptionActionResult,
        InterruptionPolicy,
    )

# A predicate the browser layer passes down so every long wait/loop can bail out
# the instant the user presses "Stop" (the AbortController analogue).
ShouldCancel = Callable[[], bool]


def _noop_cancel() -> bool:
    return False


class ProviderError(RuntimeError):
    """Raised when an adapter cannot complete an interaction."""


class GeminiSubmissionProviderError(ProviderError):
    """Typed Gemini submission failure preserved through the public client.

    ``send_actions`` is the conservative physical-send count.  In particular,
    a non-zero value means retrying the whole provider call could duplicate a
    prompt whose first send was accepted but whose acknowledgement was delayed.
    """

    def __init__(self, error_code: str, *, send_actions: int = 0) -> None:
        self.error_code = str(error_code or "unknown")
        self.send_actions = max(0, int(send_actions))
        super().__init__(
            f"[Gemini] Gemini submission failed [{self.error_code}] "
            f"(send_actions={self.send_actions})."
        )


@dataclass
class WaitTuning:
    # How long to wait for the very first token / new answer bubble to appear.
    first_token_timeout_s: float = 90.0
    # Hard cap on a single generation.
    generation_timeout_s: float = 300.0
    # Poll cadence while watching for completion.
    poll_interval_s: float = 0.7
    # The answer text must be byte-for-byte stable across this many polls
    # (and the stop button absent) before we call it done.
    stable_polls_required: int = 4
    # --- MutationObserver completion detection (the primary strategy) ----- #
    # The in-page observer declares the response finished once its text has not
    # changed for this long AND the provider's "stop" button is gone.
    stability_ms: int = 1200
    # The observer runs inside the page in short slices so the Python side can
    # re-check the cancellation token between slices and stay responsive.
    observer_slice_ms: int = 250


# In-page DOM -> Markdown serializer. Chat UIs RENDER model markdown into HTML,
# so scraping `innerText` flattens every heading, list marker and bold span
# into same-looking plain lines. Serializing the answer's DOM back to markdown
# preserves that structure end-to-end (worker outputs, synthesis and the final
# answer shown in the console). Injected into both the response observer and
# the last-message reader so old/new message comparisons stay consistent.
_TO_MD_JS = r"""
      const __toMd = (root) => {
        if (!root) return '';
        const ser = (node) => {
          if (node.nodeType === 3) return (node.textContent || '').replace(/\s+/g, ' ');
          if (node.nodeType !== 1) return '';
          const tag = node.tagName.toLowerCase();
          if (['script','style','svg','button','img','video','audio','canvas','input','select'].includes(tag)) return '';
          const kids = () => Array.from(node.childNodes).map(ser).join('');
          if (tag === 'br') return '\n';
          if (tag === 'hr') return '\n\n---\n\n';
          if (/^h[1-6]$/.test(tag)) {
            const k = kids().trim();
            return k ? '\n\n' + '#'.repeat(+tag[1]) + ' ' + k + '\n\n' : '';
          }
          if (tag === 'strong' || tag === 'b') { const k = kids().trim(); return k ? '**' + k + '**' : ''; }
          if (tag === 'em' || tag === 'i') { const k = kids().trim(); return k ? '*' + k + '*' : ''; }
          if (tag === 'pre') return '\n\n```\n' + (node.innerText || '').replace(/\n+$/, '') + '\n```\n\n';
          if (tag === 'code') { const k = kids().trim(); return k ? '`' + k + '`' : ''; }
          if (tag === 'ul' || tag === 'ol') {
            let i = 0;
            const items = Array.from(node.children)
              .filter(c => c.tagName && c.tagName.toLowerCase() === 'li')
              .map(li => {
                i++;
                const inner = ser(li).trim().replace(/\n{2,}/g, '\n');
                return (tag === 'ol' ? i + '. ' : '- ') + inner;
              })
              .filter(x => x.replace(/^(\d+\. |- )/, '').trim());
            return items.length ? '\n\n' + items.join('\n') + '\n\n' : '';
          }
          if (tag === 'li') return kids();
          if (tag === 'blockquote') {
            const k = kids().trim();
            return k ? '\n\n' + k.split('\n').map(l => '> ' + l).join('\n') + '\n\n' : '';
          }
          if (tag === 'table') {
            const rows = Array.from(node.querySelectorAll('tr')).map(tr =>
              '| ' + Array.from(tr.children).map(c => (c.innerText || '').replace(/\s+/g, ' ').trim()).join(' | ') + ' |');
            if (!rows.length) return '';
            const width = (rows[0].match(/\|/g) || []).length - 1;
            const sep = '|' + Array(Math.max(width, 1)).fill(' --- ').join('|') + '|';
            return '\n\n' + rows[0] + '\n' + sep + (rows.length > 1 ? '\n' + rows.slice(1).join('\n') : '') + '\n\n';
          }
          if (['p','div','section','article','header','footer','main'].includes(tag)) {
            const k = kids();
            return k.trim() ? '\n\n' + k.trim() + '\n\n' : '';
          }
          return kids();
        };
        return ser(root).replace(/[ \t]+\n/g, '\n').replace(/\n[ \t]+/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
      };
"""

# Standalone evaluate() wrapper: serialize one element to markdown, falling
# back to plain innerText if the walker produces nothing.
_READ_MD_JS = (
    "(el) => {" + _TO_MD_JS + " return __toMd(el) || el.innerText || el.textContent || ''; }"
)


@dataclass
class ProviderAdapter:
    """Declarative description of one LLM web interface."""

    key: str                      # stable id used everywhere ("openai", ...)
    name: str                     # human label ("ChatGPT")
    login_url: str                # where the user logs in
    new_chat_url: str             # a fresh, empty conversation

    # --- Display metadata (consumed by the frontend grid) -------------- #
    description: str = ""         # one-line pitch shown on the card
    category: str = "Frontier Models"
    color: str = "#6c8cff"        # brand colour for the tile gradient
    short: str = ""               # 1-2 char monogram fallback for the logo

    # Hard cap this provider's composer accepts (0 = no explicit cap). Prompts
    # longer than this are head+tail clipped BEFORE injection, so the send
    # never bounces off the UI's "message too long" wall (Copilot's is tiny).
    max_prompt_chars: int = 0

    # Candidate selectors, tried in order. First one that resolves is used.
    input_selectors: list[str] = field(default_factory=list)
    send_button_selectors: list[str] = field(default_factory=list)
    stop_button_selectors: list[str] = field(default_factory=list)
    # True only for adapters with a provider-specific stop contract. Generic
    # selectors remain useful for generation observation but are not sufficient
    # evidence that clicking is safe for an arbitrary provider.
    supports_active_interruption: bool = False
    response_selectors: list[str] = field(default_factory=list)
    # Presence of any of these implies a logged-in, usable chat surface.
    logged_in_selectors: list[str] = field(default_factory=list)

    tuning: WaitTuning = field(default_factory=WaitTuning)

    # ------------------------------------------------------------------ #
    # Login helpers
    # ------------------------------------------------------------------ #
    def open_login(self, page: "Page") -> None:
        page.goto(self.login_url, wait_until="domcontentloaded")

    def is_logged_in(self, page: "Page") -> bool:
        """Best-effort check that an authenticated chat surface is present."""
        # The most reliable signal is that the chat input exists.
        if self._first_present(page, self.input_selectors, timeout_ms=2500) is not None:
            return True
        for sel in self.logged_in_selectors:
            try:
                if page.locator(sel).first.is_visible(timeout=1500):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------ #
    # The main interaction
    # ------------------------------------------------------------------ #
    def send(
        self,
        page: "Page",
        prompt: str,
        should_cancel: Optional[ShouldCancel] = None,
    ) -> str:
        """Open a fresh chat, send ``prompt``, wait, and scrape the reply.

        ``should_cancel`` is a predicate checked at every wait point so a Stop
        press unwinds the call within a fraction of a second instead of blocking
        until the (slow) generation finishes.
        """
        cancel = should_cancel or _noop_cancel
        self._raise_if_cancelled(cancel)
        self._goto_fresh_chat(page, cancel)

        input_box = self._first_present(
            page, self.input_selectors, timeout_ms=20_000, should_cancel=cancel
        )
        if input_box is None:
            raise ProviderError(
                f"[{self.name}] Could not find the chat input box. "
                "You may be logged out or the UI changed."
            )

        # A previous failed send (e.g. a "message too long" bounce) leaves the
        # composer full; a retry would then stack a SECOND copy on top of it.
        # Always start from an empty composer.
        _clear_editor(page, input_box)

        # Respect the provider's composer limit: clip head+tail up front rather
        # than letting the UI reject the send outright.
        if self.max_prompt_chars and len(prompt) > self.max_prompt_chars:
            prompt = clip_text(prompt, self.max_prompt_chars)

        # Count existing answers so we can detect the *new* one. A fresh chat
        # should have none — but some UIs restore the previous conversation and
        # `_goto_fresh_chat` tolerates soft-nav failures, so if history is still
        # on screen, try once more to land on a truly empty thread.
        resp_selector = self._resolve_response_selector(page)
        before = self._response_count(page, resp_selector)
        if before > 0:
            self._goto_fresh_chat(page, cancel)
            resp_selector = self._resolve_response_selector(page)
            before = self._response_count(page, resp_selector)

        # Remember the last pre-existing message (if any) so the response
        # detector can never mistake a stale answer — e.g. the planner's JSON
        # from a previous call in this window — for the new reply.
        before_text = (
            self._last_response_text(page, resp_selector) if before > 0 else ""
        )

        # Phase 4G: Gemini submits through the deterministic state machine
        # (locate → insert → confirm insertion → send exactly once → confirm
        # acceptance). Every other provider keeps the generic inject+submit
        # path byte-for-byte. On an unconfirmed Gemini submission this raises a
        # TYPED ProviderError instead of silently waiting for a reply that will
        # never arrive.
        if self.key == "google":
            self._submit_gemini(page, prompt, cancel)
        else:
            # Instantly inject the prompt (no human-typing simulation).
            self._raise_if_cancelled(cancel)
            self._inject_text(page, input_box, prompt)

            self._raise_if_cancelled(cancel)
            self._submit(page, input_box, cancel)

        # Watch the chat DOM with a MutationObserver: detect the new answer,
        # wait for it to stop changing, then scrape it. This is what breaks the
        # "keeps re-pasting the prompt" loop — we reliably know when it's done.
        text = self._await_response_via_observer(
            page, resp_selector, before, before_text, cancel
        )
        if not text.strip():
            raise ProviderError(f"[{self.name}] Scraped an empty response.")
        return text.strip()

    @staticmethod
    def _raise_if_cancelled(should_cancel: ShouldCancel) -> None:
        if should_cancel():
            raise CancelledError("Cancelled by user before completion.")

    # ------------------------------------------------------------------ #
    # Internal steps
    # ------------------------------------------------------------------ #
    def _goto_fresh_chat(
        self, page: "Page", should_cancel: Optional[ShouldCancel] = None
    ) -> None:
        cancel = should_cancel or _noop_cancel
        self._raise_if_cancelled(cancel)
        try:
            page.goto(
                self.new_chat_url,
                wait_until="domcontentloaded",
                timeout=1000,
            )
        except Exception:
            # Soft navigation may already have us there; ignore transient errors.
            pass
        self._raise_if_cancelled(cancel)
        page.wait_for_timeout(400)
        self._raise_if_cancelled(cancel)

    def _submit(
        self,
        page: "Page",
        input_box: "Locator",
        should_cancel: Optional[ShouldCancel] = None,
    ) -> None:
        cancel = should_cancel or _noop_cancel
        btn = self._first_enabled(
            page,
            self.send_button_selectors,
            timeout_ms=4000,
            should_cancel=cancel,
        )
        if btn is not None:
            for click in (
                lambda: btn.click(timeout=500),
                lambda: btn.click(timeout=500, force=True),
                lambda: btn.evaluate("el => el.click()"),
            ):
                self._raise_if_cancelled(cancel)
                try:
                    click()
                    return
                except Exception:
                    continue
        # Fallback: most of these UIs submit on Enter.
        self._raise_if_cancelled(cancel)
        try:
            input_box.press("Enter")
        except Exception:
            page.keyboard.press("Enter")

    def _submit_gemini(
        self,
        page: "Page",
        prompt: str,
        should_cancel: Optional[ShouldCancel] = None,
    ) -> None:
        """Submit ``prompt`` to Gemini via the deterministic state machine.

        Raises :class:`CancelledError` on cancellation and a typed
        :class:`ProviderError` (carrying the submission error category) when the
        submission cannot be confirmed — so the browser job fails cleanly with a
        distinguishable reason rather than waiting for a response Gemini never
        received. Never performs more than one physical send action.
        """
        from .gemini_submission import (
            SubmissionErrorCode,
            SubmissionStatus,
            submit_prompt,
        )

        cancel = should_cancel or _noop_cancel
        dom = _PlaywrightGeminiDom(page, self)
        result = submit_prompt(dom, prompt, should_cancel=cancel)
        if result.status is SubmissionStatus.SUBMITTED:
            return
        if result.status is SubmissionStatus.CANCELLED or result.error_code is (
            SubmissionErrorCode.CANCELLED
        ):
            raise CancelledError("Cancelled during Gemini submission.")
        code = result.error_code.value if result.error_code else "unknown"
        # Bounded, content-free: the category plus which locator strategies were
        # attempted — never prompt/response text.
        raise GeminiSubmissionProviderError(
            code, send_actions=result.send_actions
        )

    # ------------------------------------------------------------------ #
    # MutationObserver-based response detection (the loop-breaker)
    # ------------------------------------------------------------------ #
    # A single in-page script installs a MutationObserver over the chat DOM and
    # resolves a Promise once the latest answer bubble (a) exists/grew beyond the
    # count we had before sending, and (b) has stopped mutating for a stability
    # window while the "stop generating" button is gone. It runs in short slices
    # so Python can re-check cancellation between them and never blocks the UI.
    _OBSERVER_JS = r"""
    ({ respSelectors, stopSelectors, beforeCount, beforeText, stabilityMs, sliceMs }) => {
""" + _TO_MD_JS + r"""
      const visible = (el) =>
        !!el && !!(el.offsetParent || el.getClientRects().length) &&
        getComputedStyle(el).visibility !== 'hidden';

      // A successor answer must have a new response identity. Text mutation in
      // the final bubble is not sufficient: a cancelled job can keep mutating
      // its partial bubble after the next job takes its baseline.
      const isNewAnswer = (snap) =>
        snap.count > beforeCount && !!(snap.text || '').trim();

      const stopActive = () =>
        stopSelectors.some((s) => {
          try { return Array.from(document.querySelectorAll(s)).some(visible); }
          catch (e) { return false; }
        });

      // Resolve the response container selector that currently has matches.
      const matches = () => {
        for (const s of respSelectors) {
          let nodes;
          try { nodes = document.querySelectorAll(s); } catch (e) { continue; }
          if (nodes && nodes.length) return Array.from(nodes);
        }
        return [];
      };
      const readLast = () => {
        const nodes = matches();
        if (!nodes.length) return { count: 0, text: '' };
        const last = nodes[nodes.length - 1];
        return {
          count: nodes.length,
          text: (__toMd(last) || last.innerText || last.textContent || ''),
        };
      };

      return new Promise((resolve) => {
        let lastText = readLast().text;
        let lastChange = Date.now();
        const started = Date.now();

        const obs = new MutationObserver(() => {
          const { text } = readLast();
          if (text !== lastText) { lastText = text; lastChange = Date.now(); }
        });
        obs.observe(document.body, {
          childList: true, subtree: true, characterData: true,
        });

        const finish = (status) => {
          obs.disconnect();
          const snap = readLast();
          resolve({ status, text: snap.text, count: snap.count });
        };

        const tick = () => {
          const now = Date.now();
          const snap = readLast();
          // Keep the change clock honest even if the observer missed a batch.
          if (snap.text !== lastText) { lastText = snap.text; lastChange = now; }

          const hasNewAnswer = isNewAnswer(snap);
          const generating = stopActive();
          const stableFor = now - lastChange;

          if (hasNewAnswer && !generating && stableFor >= stabilityMs && snap.text.trim()) {
            finish('done');
            return;
          }
          if (now - started >= sliceMs) {
            finish(generating ? 'generating' : (hasNewAnswer ? 'streaming' : 'waiting'));
            return;
          }
          setTimeout(tick, 120);
        };
        tick();
      });
    }
    """

    def _await_response_via_observer(
        self,
        page: "Page",
        resp_selector: Optional[str],
        before: int,
        before_text: str,
        should_cancel: ShouldCancel,
    ) -> str:
        resp_selectors = (
            [resp_selector] if resp_selector else []
        ) + self.response_selectors
        # De-dupe while preserving order.
        seen: set[str] = set()
        resp_selectors = [s for s in resp_selectors if s and not (s in seen or seen.add(s))]

        payload = {
            "respSelectors": resp_selectors,
            "stopSelectors": self.stop_button_selectors,
            "beforeCount": before,
            "beforeText": before_text,
            "stabilityMs": self.tuning.stability_ms,
            "sliceMs": self.tuning.observer_slice_ms,
        }

        # Same whitespace-insensitive compare the in-page observer uses.
        norm = lambda s: re.sub(r"\s+", " ", s or "").strip()  # noqa: E731
        def is_new(text: str, count: int) -> bool:
            return count > before and bool(norm(text))

        deadline = time.time() + self.tuning.generation_timeout_s
        first_token_deadline = time.time() + self.tuning.first_token_timeout_s
        last_text = ""
        saw_answer = False

        while time.time() < deadline:
            self._raise_if_cancelled(should_cancel)
            try:
                result = page.evaluate(self._OBSERVER_JS, payload)
            except Exception:
                # Page navigated/re-rendered mid-observe; retry on the next slice.
                page.wait_for_timeout(120)
                continue

            status = (result or {}).get("status")
            text = (result or {}).get("text") or ""
            count = (result or {}).get("count")
            if not isinstance(count, int):
                count = 0
            if is_new(text, count):
                last_text, saw_answer = text, True

            if status == "done" and is_new(text, count):
                return text

            # No answer has appeared yet and we've blown the first-token budget.
            if not saw_answer and time.time() > first_token_deadline:
                raise ProviderError(
                    f"[{self.name}] No response appeared within "
                    f"{self.tuning.first_token_timeout_s:.0f}s "
                    "(the model may not have started, or the selector changed)."
                )

        if last_text.strip():
            # Timed out but we captured something new — return it over nothing.
            return last_text
        raise ProviderError(
            f"[{self.name}] Generation did not complete within "
            f"{self.tuning.generation_timeout_s:.0f}s."
        )

    # ------------------------------------------------------------------ #
    # Selector utilities
    # ------------------------------------------------------------------ #
    def _resolve_response_selector(self, page: "Page") -> Optional[str]:
        """Pick whichever response selector currently matches the DOM."""
        for sel in self.response_selectors:
            try:
                if page.locator(sel).count() >= 0:  # selector is at least valid
                    # Prefer the first that already has matches; else keep first valid.
                    if page.locator(sel).count() > 0:
                        return sel
            except Exception:
                continue
        return self.response_selectors[0] if self.response_selectors else None

    def _response_count(self, page: "Page", selector: Optional[str]) -> int:
        if not selector:
            return 0
        try:
            return page.locator(selector).count()
        except Exception:
            return 0

    def _last_response_text(self, page: "Page", selector: Optional[str]) -> str:
        if not selector:
            return ""
        try:
            loc = page.locator(selector)
            n = loc.count()
            if n == 0:
                return ""
            # Same markdown serialization the observer uses, so before/after
            # message comparisons are apples-to-apples.
            return loc.nth(n - 1).evaluate(_READ_MD_JS) or ""
        except Exception:
            return ""

    def _stop_button_visible(self, page: "Page", deadline: float) -> bool:
        for sel in self.stop_button_selectors:
            if time.monotonic() >= deadline:
                return True  # conservative: the idle signal was not observed in-bound
            try:
                base = page.locator(sel)
                for index in range(min(base.count(), 8)):
                    remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                    if base.nth(index).is_visible(timeout=min(remaining_ms, 200)):
                        return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------ #
    # Active interruption seam (production-hardening 5B)
    # ------------------------------------------------------------------ #
    # This is the ONLY provider-specific stop-generation interaction. It runs
    # EXCLUSIVELY on the provider's own Playwright worker thread (Playwright is
    # thread-affine), keyed on THIS adapter's ``stop_button_selectors`` — so
    # BrowserManager never has to know a single selector. The default
    # implementation below serves every current adapter (they all carry stop
    # selectors, provider-specific plus the shared generic fallbacks); a provider
    # with no stop mechanism would report ``UNSUPPORTED``.
    def interrupt_generation(
        self,
        page: "Page",
        observation: Optional["CancellationObservation"] = None,
        policy: Optional["InterruptionPolicy"] = None,
        owner_check: Optional[Callable[[], bool]] = None,
    ) -> "InterruptionActionResult":
        """Attempt to stop the currently-generating response — exactly once.

        Contract (Phase 5B): distinguish generation *active and stopped*,
        generation *already settled*, *no usable stop control*, and *browser
        interaction failure*. It NEVER closes or reloads the page, never starts a
        new chat, never submits another prompt, never touches captcha/login, and
        only ever clicks THIS provider's stop control. It performs at most one
        stop click, then confirms quiescence within a bounded grace period.

        ``observation`` (optional) carries the job identity for owner context; it
        is not required for the DOM work. ``policy`` supplies every bound.
        """
        # Imported lazily to keep providers.py import-light and avoid any cycle.
        from .browser_cancellation import (
            DEFAULT_INTERRUPTION_POLICY,
            InterruptionActionResult,
            StopActionStatus,
        )

        pol = policy or DEFAULT_INTERRUPTION_POLICY
        owns_job = owner_check or (lambda: True)

        if not self.supports_active_interruption:
            return InterruptionActionResult.unsupported("provider-stop-unverified")
        if not self.stop_button_selectors:
            return InterruptionActionResult.unsupported("no-stop-selectors")
        if not owns_job():
            return InterruptionActionResult.settled_before_action("owner-changed-pre-lookup")
        is_closed = getattr(page, "is_closed", None)
        try:
            if callable(is_closed) and is_closed():
                return InterruptionActionResult.failed("page-closed")
        except Exception as exc:  # noqa: BLE001
            return InterruptionActionResult.failed(
                f"page-state-error:{type(exc).__name__}"
            )

        deadline = time.monotonic() + pol.stop_action_timeout_s
        # 1. Find one unambiguous visible, enabled stop control. Absence is not
        # proof that generation is idle.
        try:
            stop, lookup_status, diagnostic = self._visible_stop_button(
                page, deadline, owns_job
            )
        except Exception as exc:  # noqa: BLE001 - locating must never crash the worker
            return InterruptionActionResult.failed(f"locate-error:{type(exc).__name__}")
        if stop is None:
            if lookup_status is StopActionStatus.SETTLED_BEFORE_ACTION:
                return InterruptionActionResult.settled_before_action(diagnostic)
            return InterruptionActionResult.no_control(
                diagnostic or "no-usable-stop-control"
            )

        # 2. Re-check ownership and issue exactly one physical click call. An
        # exception after dispatch is ambiguous, so retrying could click twice.
        if not owns_job():
            return InterruptionActionResult.settled_before_action("owner-changed-pre-click")
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        if remaining_ms <= 1:
            return InterruptionActionResult.failed("stop-action-deadline")
        try:
            stop.click(timeout=min(remaining_ms, 2000))
        except Exception as exc:  # noqa: BLE001 - marshalled as a diagnostic
            return InterruptionActionResult.failed(f"click-error:{type(exc).__name__}")
        # 3. Require stop absence, composer usability, and stable response text.
        quiescent, ownership_lost = self._await_stop_quiescence(
            page, pol, owns_job
        )
        return InterruptionActionResult.stopped(
            quiescent=quiescent,
            diagnostic=(
                None
                if quiescent
                else (
                    "ownership-changed-during-quiescence"
                    if ownership_lost
                    else "quiescence-not-confirmed"
                )
            ),
        )

    def _visible_stop_button(
        self,
        page: "Page",
        deadline: float,
        owner_check: Callable[[], bool],
    ) -> tuple[Optional["Locator"], Optional["StopActionStatus"], Optional[str]]:
        """Return the first currently-visible stop control, or ``None``.

        Bounded by ``timeout_s`` — a single pass over the selectors, never an
        unbounded wait. Only returns an element that is actually visible.
        """
        from .browser_cancellation import StopActionStatus

        saw_match = False
        saw_visible = False
        saw_disabled = False
        while True:
            if not owner_check():
                return None, StopActionStatus.SETTLED_BEFORE_ACTION, "owner-changed-during-lookup"
            for sel in self.stop_button_selectors:
                try:
                    base = page.locator(sel)
                    count = min(base.count(), 8)
                except Exception:
                    continue
                saw_match = saw_match or count > 0
                usable: list["Locator"] = []
                for index in range(count):
                    if not owner_check():
                        return None, StopActionStatus.SETTLED_BEFORE_ACTION, "owner-changed-during-lookup"
                    loc = base.nth(index)
                    remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                    try:
                        if not loc.is_visible(timeout=min(remaining_ms, 200)):
                            continue
                        saw_visible = True
                        if not loc.is_enabled(timeout=min(remaining_ms, 200)):
                            saw_disabled = True
                            continue
                        usable.append(loc)
                    except Exception:
                        continue
                if len(usable) == 1:
                    return usable[0], None, None
                if len(usable) > 1:
                    return None, StopActionStatus.NO_CONTROL, "multiple-visible-stop-controls"
            if time.monotonic() >= deadline:
                if saw_disabled:
                    reason = "stop-control-disabled"
                elif saw_visible:
                    reason = "stop-control-unusable"
                elif saw_match:
                    reason = "stop-control-hidden"
                else:
                    reason = "stop-control-missing"
                return None, StopActionStatus.NO_CONTROL, reason
            if not owner_check():
                return None, StopActionStatus.SETTLED_BEFORE_ACTION, "owner-changed-during-lookup"
            try:
                page.wait_for_timeout(
                    min(120, max(1, int((deadline - time.monotonic()) * 1000)))
                )
            except Exception:
                return None, StopActionStatus.NO_CONTROL, "page-unavailable-during-lookup"

    def _await_stop_quiescence(
        self,
        page: "Page",
        policy: "InterruptionPolicy",
        owner_check: Callable[[], bool],
    ) -> tuple[bool, bool]:
        """Confirm generation has stopped: the stop control is no longer visible.

        Bounded strictly by ``policy.post_stop_grace_s`` and polled at
        ``policy.quiescence_poll_interval_s``. Delivers no response, parses no
        artifact, and makes no assumption that any partial text is valid.
        """
        deadline = time.monotonic() + policy.post_stop_grace_s
        interval_ms = int(policy.quiescence_poll_interval_s * 1000)
        previous_signature: Optional[tuple[str, int, str]] = None
        stable_polls = 0
        while True:
            if not owner_check():
                return False, True
            stop_absent = not self._stop_button_visible(page, deadline)
            if not owner_check():
                return False, True
            composer_usable = self._composer_usable(page, deadline)
            if not owner_check():
                return False, True
            signature = self._response_signature(page, deadline)
            if signature is not None and signature == previous_signature:
                stable_polls += 1
            else:
                stable_polls = 1 if signature is not None else 0
            previous_signature = signature
            if stop_absent and composer_usable and stable_polls >= 2:
                return True, False
            if time.monotonic() >= deadline:
                return False, False
            if not owner_check():
                return False, True
            try:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                page.wait_for_timeout(min(interval_ms, remaining_ms))
            except Exception:
                return False, False

    def _composer_usable(self, page: "Page", deadline: float) -> bool:
        for sel in self.input_selectors:
            if time.monotonic() >= deadline:
                return False
            try:
                base = page.locator(sel)
                count = min(base.count(), 8)
            except Exception:
                continue
            for index in range(count):
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                loc = base.nth(index)
                try:
                    if loc.is_visible(timeout=min(remaining_ms, 200)) and loc.is_enabled(
                        timeout=min(remaining_ms, 200)
                    ):
                        return True
                except Exception:
                    continue
        return False

    def _response_signature(
        self, page: "Page", deadline: float
    ) -> Optional[tuple[str, int, str]]:
        for sel in self.response_selectors:
            if time.monotonic() >= deadline:
                return None
            try:
                loc = page.locator(sel)
                count = loc.count()
                if count:
                    remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                    text = loc.nth(count - 1).text_content(
                        timeout=min(remaining_ms, 200)
                    ) or ""
                    return sel, count, str(text)
            except Exception:
                continue
        return ("", 0, "")

    # ------------------------------------------------------------------ #
    # Instant text injection (no human-typing simulation)
    # ------------------------------------------------------------------ #
    # One JS round-trip drops the whole prompt in and fires the exact events the
    # app's framework (React/ProseMirror/Quill) listens for, so the send button
    # enables immediately. This is O(1) regardless of prompt length — the old
    # per-character path is gone.
    _INJECT_JS = r"""
    ({ value }) => {
      const el = document.activeElement;
      if (!el) return { ok: false, value: '' };
      const tag = (el.tagName || '').toLowerCase();
      const fire = (type, init) => el.dispatchEvent(
        type === 'input'
          ? new InputEvent('input', Object.assign({ bubbles: true }, init))
          : new Event(type, { bubbles: true })
      );

      if (tag === 'textarea' || tag === 'input') {
        // Use the native value setter so React's tracked value updates too.
        const proto = tag === 'textarea'
          ? window.HTMLTextAreaElement.prototype
          : window.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
        setter.call(el, value);
        fire('input', { inputType: 'insertFromPaste', data: value });
        fire('change');
      } else {
        // contenteditable (ProseMirror / Quill / generic rich editors).
        const sel = window.getSelection();
        sel.removeAllRanges();
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.addRange(range);
        let inserted = false;
        try { inserted = document.execCommand('insertText', false, value); }
        catch (e) { inserted = false; }
        if (!inserted) {
          el.textContent = value;
          fire('input', { inputType: 'insertText', data: value });
        }
      }
      // A keyup nudge re-runs send-button enable logic in stubborn UIs.
      el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ' ' }));
      const got = (el.value != null && el.value !== '')
        ? el.value : (el.innerText || el.textContent || '');
      return { ok: got.trim().length > 0, value: got };
    }
    """

    def _inject_text(self, page: "Page", loc: "Locator", text: str) -> None:
        """Focus the editor, then inject ``text`` in a single instant operation.

        For very large prompts we prefer a true clipboard paste (``Ctrl/Cmd+V``):
        it sidesteps any per-keystroke handling, avoids textarea typing limits,
        and lets the site run its big-paste logic (e.g. auto-attaching the text
        as a file on ChatGPT/Gemini). Normal prompts use the instant JS path.
        """
        for focus_attempt in (
            lambda: loc.click(timeout=2500),
            lambda: loc.focus(timeout=1500),
            lambda: loc.evaluate("el => el.focus()"),
        ):
            try:
                focus_attempt()
                break
            except Exception:
                continue

        if len(text) >= _LARGE_PROMPT_THRESHOLD:
            try:
                if _clipboard_paste_into(page, loc, text):
                    return
            except Exception:
                pass  # fall through to the instant JS injection below

        try:
            result = page.evaluate(self._INJECT_JS, {"value": text})
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"[{self.name}] Could not enter text into the chat box: {exc}"
            )

        if result and result.get("ok"):
            return
        # Single fallback: Playwright fill (still instant) for editors that
        # reject programmatic insertText.
        try:
            loc.fill(text, timeout=3000)
            if self._input_has_text(loc, text):
                return
        except Exception:
            pass

        raise ProviderError(
            f"[{self.name}] Text did not register in the chat box "
            "(the input may be covered, disabled, or its UI changed)."
        )

    @staticmethod
    def _input_has_text(loc: "Locator", text: str) -> bool:
        """True if the editor now holds (some of) the text we typed."""
        try:
            val = loc.evaluate(
                "el => (el.value != null && el.value !== '') ? el.value : (el.innerText || '')"
            )
        except Exception:
            return False
        return len((val or "").strip()) >= min(3, len(text.strip()))

    # ------------------------------------------------------------------ #
    # Element resolution (only returns elements that can truly be used)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _actionable(loc: "Locator") -> bool:
        """Real pointer-actionability check beyond Playwright's is_visible().

        Rejects zero/near-zero size, display:none, visibility:hidden,
        opacity≈0, and elements that are fully covered by an overlay (the
        center point resolves to an unrelated element). This is what filters
        out hidden fallback/mirror textareas.
        """
        try:
            return bool(
                loc.evaluate(
                    """el => {
                        const s = getComputedStyle(el);
                        if (s.display === 'none' || s.visibility === 'hidden') return false;
                        if (parseFloat(s.opacity || '1') < 0.05) return false;
                        if (el.getClientRects().length === 0) return false;
                        const r = el.getBoundingClientRect();
                        if (r.width < 4 || r.height < 4) return false;
                        // Covered-by-overlay check: the element (or one of its
                        // descendants/ancestors) should be hit at its center.
                        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
                        const top = document.elementFromPoint(cx, cy);
                        if (top && !(el.contains(top) || top.contains(el))) return false;
                        return true;
                    }"""
                )
            )
        except Exception:
            return False

    @classmethod
    def _first_present(
        cls,
        page: "Page",
        selectors: list[str],
        timeout_ms: int,
        should_cancel: Optional[ShouldCancel] = None,
    ) -> Optional["Locator"]:
        return cls._resolve(
            page, selectors, timeout_ms, require_enabled=False, should_cancel=should_cancel
        )

    @classmethod
    def _first_enabled(
        cls,
        page: "Page",
        selectors: list[str],
        timeout_ms: int,
        should_cancel: Optional[ShouldCancel] = None,
    ) -> Optional["Locator"]:
        return cls._resolve(
            page, selectors, timeout_ms, require_enabled=True, should_cancel=should_cancel
        )

    @classmethod
    def _resolve(
        cls,
        page: "Page",
        selectors: list[str],
        timeout_ms: int,
        require_enabled: bool,
        should_cancel: Optional[ShouldCancel] = None,
    ) -> Optional["Locator"]:
        """Return the first *actionable* element among the selectors.

        Scans each selector's matches (not just ``.first``) so a hidden first
        match (e.g. a fallback textarea) doesn't shadow the real, visible one.
        """
        cancel = should_cancel or _noop_cancel
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            if cancel():
                raise CancelledError("Cancelled while locating the chat input.")
            for sel in selectors:
                try:
                    base = page.locator(sel)
                    count = min(base.count(), 8)
                except Exception:
                    continue
                for i in range(count):
                    loc = base.nth(i)
                    try:
                        if not loc.is_visible(timeout=200):
                            continue
                        if require_enabled and not loc.is_enabled(timeout=200):
                            continue
                        if not cls._actionable(loc):
                            continue
                        try:
                            loc.scroll_into_view_if_needed(timeout=1000)
                        except Exception:
                            pass
                        return loc
                    except Exception:
                        continue
            page.wait_for_timeout(220)
        return None


# --------------------------------------------------------------------------- #
# Large-prompt clipboard paste
# --------------------------------------------------------------------------- #
# Prompts at/above this many characters are injected via a real clipboard paste
# instead of programmatic insertion. Tune to taste.
_LARGE_PROMPT_THRESHOLD = 1000


def _clipboard_modifier() -> str:
    """The OS paste modifier for the host the browser runs on."""
    return "Meta" if sys.platform == "darwin" else "Control"


def _origin_of(page: "Page") -> Optional[str]:
    try:
        from urllib.parse import urlsplit

        u = urlsplit(page.url)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        pass
    return None


def _grant_clipboard(page: "Page") -> None:
    """Grant clipboard read/write so ``navigator.clipboard`` works (Chromium).

    Best-effort: silently ignored on engines that don't know these permission
    names (e.g. Firefox), where we fall back to a synthetic paste event.
    """
    origin = _origin_of(page)
    attempts = ([{"origin": origin}] if origin else []) + [{}]
    for kw in attempts:
        try:
            page.context.grant_permissions(
                ["clipboard-read", "clipboard-write"], **kw
            )
            return
        except Exception:
            continue


# Read the editor's content — from the matched element, falling back to the
# focused element (some UIs, e.g. Copilot, wrap the true editable in a shell
# that matches the selector but never carries the value itself).
_READ_EDITOR_JS = (
    "el => {"
    " const read = (n) => n ? ((n.value != null && n.value !== '') ? n.value"
    " : (n.innerText || n.textContent || '')) : '';"
    " let v = read(el);"
    " if (!v.trim() && document.activeElement && document.activeElement !== el)"
    "   v = read(document.activeElement);"
    " return v; }"
)


def _editor_has_text(loc: "Locator", text: str) -> bool:
    """True if the editor now holds (some of) ``text``."""
    try:
        val = loc.evaluate(_READ_EDITOR_JS)
    except Exception:
        return False
    return len((val or "").strip()) >= min(3, len(text.strip()))


def _editor_full_text(loc: "Locator") -> str:
    """The editor's complete current content ('' on any failure)."""
    try:
        return loc.evaluate(_READ_EDITOR_JS) or ""
    except Exception:
        return ""


def _wait_editor_has_text(
    page: "Page", loc: "Locator", text: str, budget_ms: int
) -> bool:
    """Poll until the editor holds the text. Large pastes render slowly, so a
    single fixed 150ms check races the paste and causes duplicate fallback
    insertions — this waits instead of guessing."""
    deadline = time.time() + budget_ms / 1000.0
    while time.time() < deadline:
        if _editor_has_text(loc, text):
            return True
        page.wait_for_timeout(150)
    return _editor_has_text(loc, text)


def _clear_editor(page: "Page", loc: "Locator") -> None:
    """Empty the composer so injection strategies can never stack copies of
    the prompt on top of each other (the double-paste bug)."""
    try:
        _focus_locator(loc)
        page.keyboard.press(f"{_clipboard_modifier()}+A")
        page.keyboard.press("Delete")
    except Exception:
        pass
    try:
        loc.evaluate(
            """el => {
                const wipe = (n) => {
                    if (!n) return;
                    const tag = (n.tagName || '').toLowerCase();
                    if (tag === 'textarea' || tag === 'input') {
                        const proto = tag === 'textarea'
                            ? window.HTMLTextAreaElement.prototype
                            : window.HTMLInputElement.prototype;
                        Object.getOwnPropertyDescriptor(proto, 'value').set.call(n, '');
                    } else if ((n.innerText || n.textContent || '').trim()) {
                        n.textContent = '';
                    }
                    n.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }));
                };
                wipe(el);
                if (document.activeElement && document.activeElement !== el) wipe(document.activeElement);
            }"""
        )
    except Exception:
        pass


def _ensure_single_copy(page: "Page", loc: "Locator", text: str) -> bool:
    """Guarantee the editor holds the prompt exactly ONCE.

    If a slow paste and a fallback both landed (duplicated text would blow the
    site's message-length limit), wipe the box and set a single copy
    deterministically."""
    probe = re.sub(r"\s+", " ", text.strip())[:80].strip()
    if not probe:
        return True
    content = re.sub(r"\s+", " ", _editor_full_text(loc))
    if content.count(probe) <= 1:
        return True
    _clear_editor(page, loc)
    try:
        if loc.evaluate(_SET_AND_FIRE_JS, text):
            return _editor_has_text(loc, text)
    except Exception:
        pass
    return False


def _fire_input_events(loc: "Locator") -> None:
    """Nudge React/Vue/ProseMirror to register the new content + enable Send."""
    try:
        loc.evaluate(
            """el => {
                el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertFromPaste' }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: ' ' }));
            }"""
        )
    except Exception:
        pass


# Synthetic paste: build a DataTransfer and dispatch a real ClipboardEvent so a
# site's onPaste handler fires (this is what triggers ChatGPT/Gemini's
# big-paste-to-file-attachment behavior even without OS clipboard access).
_SYNTH_PASTE_JS = r"""
(el, text) => {
  el.focus();
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  let ev;
  try {
    ev = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
  } catch (e) {
    ev = new Event('paste', { bubbles: true, cancelable: true });
    try { Object.defineProperty(ev, 'clipboardData', { value: dt }); } catch (_) {}
  }
  return el.dispatchEvent(ev);
}
"""

# Last-resort direct injection: set the value/innerText and fire input/change so
# the framework's tracked state updates.
_SET_AND_FIRE_JS = r"""
(el, value) => {
  el.focus();
  const tag = (el.tagName || '').toLowerCase();
  if (tag === 'textarea' || tag === 'input') {
    const proto = tag === 'textarea'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
  } else {
    const sel = window.getSelection();
    sel.removeAllRanges();
    const range = document.createRange();
    range.selectNodeContents(el);
    sel.addRange(range);
    let ok = false;
    try { ok = document.execCommand('insertText', false, value); } catch (e) { ok = false; }
    if (!ok) el.textContent = value;
  }
  el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertFromPaste', data: value }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  const got = (el.value != null && el.value !== '') ? el.value : (el.innerText || el.textContent || '');
  return got.trim().length > 0;
}
"""


def _focus_locator(loc: "Locator") -> None:
    try:
        loc.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass
    for attempt in (
        lambda: loc.click(timeout=3000),
        lambda: loc.focus(timeout=1500),
        lambda: loc.evaluate("el => el.focus()"),
    ):
        try:
            attempt()
            return
        except Exception:
            continue


def _clipboard_paste_into(page: "Page", loc: "Locator", text: str) -> bool:
    """Core paste routine shared by :func:`paste_large_prompt` and the adapter.

    Tries, in order of fidelity:
      1. real OS clipboard write + a TRUSTED Ctrl/Cmd+V keypress,
      2. a synthetic ``paste`` ClipboardEvent carrying a DataTransfer,
      3. direct value/innerText injection.

    Each strategy is given time to actually register (big pastes render
    slowly), the composer is CLEARED before falling through to the next
    strategy, and the final content is verified to hold exactly ONE copy of
    the prompt — so strategies can never stack duplicates that blow the
    site's message-length limit. Returns True once the text is in place.
    """
    _focus_locator(loc)

    # Composer fingerprint: big-paste UIs (ChatGPT/Gemini) convert a large
    # paste into an attachment chip and leave the editor empty. Counting the
    # composer's elements before/after lets us recognize that success instead
    # of wrongly pasting again.
    def _container_el_count() -> int:
        try:
            return loc.evaluate(
                "el => { const c = el.closest('form') || el.parentElement || el; "
                "return c.querySelectorAll('*').length; }"
            )
        except Exception:
            return -1

    # Wait budget scales with prompt size (rendering 50k chars takes a while).
    budget_ms = min(1500 + len(text) // 4, 6000)

    # --- Strategy 1: real clipboard + trusted paste keystroke --------------- #
    try:
        page.bring_to_front()
    except Exception:
        pass
    _grant_clipboard(page)
    wrote = False
    try:
        wrote = bool(
            page.evaluate(
                "async (t) => { try { await navigator.clipboard.writeText(t); "
                "return true; } catch (e) { return false; } }",
                text,
            )
        )
    except Exception:
        wrote = False
    if wrote:
        before_els = _container_el_count()
        pressed = True
        try:
            page.keyboard.press(f"{_clipboard_modifier()}+V")
        except Exception:
            pressed = False
        if pressed:
            if _wait_editor_has_text(page, loc, text, budget_ms):
                _fire_input_events(loc)
                return _ensure_single_copy(page, loc, text)
            # Editor is empty — the paste may have become an attachment chip.
            # If the composer gained elements, the content IS attached: done,
            # and crucially do NOT inject a second inline copy.
            after_els = _container_el_count()
            if before_els >= 0 and after_els > before_els:
                _fire_input_events(loc)
                return True
            # Nothing registered anywhere: wipe so a late-landing paste can
            # never stack with the next strategy.
            _clear_editor(page, loc)

    # --- Strategy 2: synthetic paste event (fires the site's onPaste) ------- #
    try:
        if loc.evaluate(_SYNTH_PASTE_JS, text):
            if _wait_editor_has_text(page, loc, text, 2000):
                _fire_input_events(loc)
                return _ensure_single_copy(page, loc, text)
    except Exception:
        pass
    _clear_editor(page, loc)

    # --- Strategy 3: direct injection (replaces content — single copy) ------ #
    try:
        if loc.evaluate(_SET_AND_FIRE_JS, text):
            _fire_input_events(loc)
            return _editor_has_text(loc, text)
    except Exception:
        pass

    return False


# --------------------------------------------------------------------------- #
# Phase 4G — Playwright backing for the Gemini submission state machine
# --------------------------------------------------------------------------- #
# Best-effort user-turn and generation selectors used ONLY for acknowledgement.
# Ordered best→fallback; a miss simply means "no bubble yet", never an error.
_GEMINI_USER_TURN_SELECTORS = (
    "user-query .query-text",
    "[data-test-id='user-query'] .query-text",
    "user-query",
    ".user-query-bubble-with-background",
    "[class*='user-query' i]",
)

# One evaluate returns the safe, already-computed state the port needs. It never
# returns prompt/response text — only structural flags and the region category.
_GEMINI_DESCRIBE_JS = r"""
(el) => {
  if (!el) return null;
  const tag = (el.tagName || '').toLowerCase();
  const cs = getComputedStyle(el);
  const editable = el.isContentEditable === true ||
    tag === 'textarea' || tag === 'input';
  const aria = (el.getAttribute && el.getAttribute('aria-label')) || '';
  const role = (el.getAttribute && el.getAttribute('role')) || '';
  const lower = aria.toLowerCase();
  const isStop = /stop/.test(lower);
  const isAttach = /attach|upload|image|file|microphone|voice/.test(lower);
  let region = 'composer';
  if (cs.display === 'none' || cs.visibility === 'hidden') region = 'hidden';
  else if (el.closest('[role="search"], form[role="search"], input[type="search"]')) region = 'search';
  else if (el.closest('[role="dialog"], [aria-label*="Settings" i], mat-dialog-container')) region = 'settings';
  else if (el.closest('[aria-label*="feedback" i]')) region = 'feedback';
  else if (el.closest('mat-bottom-sheet-container, .cdk-visually-hidden, [aria-hidden="true"]')) region = 'hidden';
  const conversationSelector = [
    '[class*="chat-pane" i]', '[class*="chat-window" i]',
    '[class*="conversation-container" i]',
    '[data-test-id*="conversation" i]', 'main', '[role="main"]'
  ].join(',');
  const composerSelector = [
    'form', '[role="form"]', '[data-test-id*="composer" i]',
    '[class*="input-area" i]', '[class*="input-container" i]'
  ].join(',');
  // The form is normally only the input footer.  Prefer the enclosing
  // conversation so sibling user bubbles and generation controls are visible;
  // retain the form/composer root as a bounded fallback for unfamiliar DOMs.
  const regionRoot = el.closest(conversationSelector) || el.closest(composerSelector);
  const roots = regionRoot
    ? Array.from(document.querySelectorAll(`${conversationSelector},${composerSelector}`))
    : [];
  const regionIndex = regionRoot ? roots.indexOf(regionRoot) : -1;
  const chatRegion = regionIndex >= 0
    ? `${(regionRoot.tagName || '').toLowerCase()}:${regionIndex}` : '';
  return { tag, role, editable, aria, isStop, isAttach, region, chatRegion };
}
"""


class _PlaywrightGeminiDom:
    """Adapts a live Playwright ``Page`` to the ``GeminiDom`` port.

    Same-thread only (constructed and used on the provider's own worker thread).
    Every method is defensive: an ordinary DOM miss or transient error yields an
    empty/false result, never an exception, so the state machine's typed outcome
    is always the authority. This class is NOT exercised by the fake-DOM test
    suite (it requires a real browser) and needs the manual logged-in Gemini
    acceptance before it can be trusted.
    """

    def __init__(self, page: "Page", adapter: "ProviderAdapter") -> None:
        self.page = page
        self.adapter = adapter

    def query(self, selector: str):
        from .gemini_submission import DomElement
        out = []
        try:
            base = self.page.locator(selector)
            count = min(base.count(), 8)
        except Exception:
            return out
        for index in range(count):
            loc = base.nth(index)
            try:
                visible = loc.is_visible(timeout=200)
            except Exception:
                visible = False
            try:
                enabled = loc.is_enabled(timeout=200)
            except Exception:
                enabled = True  # non-button editors report no disabled state
            desc = {}
            try:
                desc = loc.evaluate(_GEMINI_DESCRIBE_JS) or {}
            except Exception:
                desc = {}
            out.append(DomElement(
                handle=loc,
                tag=str(desc.get("tag", "")),
                role=str(desc.get("role", "")),
                accessible_name=str(desc.get("aria", "")),
                visible=bool(visible),
                enabled=bool(enabled),
                editable=bool(desc.get("editable", False)),
                is_stop_control=bool(desc.get("isStop", False)),
                is_attachment_control=bool(desc.get("isAttach", False)),
                region=str(desc.get("region", "composer")),
                chat_region=str(desc.get("chatRegion", "")),
            ))
        return out

    def focus(self, element) -> bool:
        try:
            _focus_locator(element.handle)
            return True
        except Exception:
            return False

    def insert_text(self, element, text: str) -> bool:
        try:
            self.adapter._inject_text(self.page, element.handle, text)
            return True
        except Exception:
            return False

    def read_composer(self, element) -> str:
        try:
            return _editor_full_text(element.handle)
        except Exception:
            # Unknown/detached is distinct from a readable empty composer.  The
            # state machine may acknowledge the latter, never the former.
            return None

    def clear(self, element) -> None:
        try:
            _clear_editor(self.page, element.handle)
        except Exception:
            pass

    def click(self, element) -> bool:
        loc = element.handle
        # Exactly one physical action.  A Playwright exception is ambiguous: the
        # browser may have dispatched the click before reporting a detach/timeout.
        # Confirmation decides the outcome; this adapter must never click again.
        try:
            loc.click(timeout=1000)
            return True
        except Exception:
            return False

    def user_messages(self, element):
        # Query only inside the selected composer's chat region.  A user action
        # or restored conversation in another visible pane is not an ack.
        try:
            return element.handle.evaluate(
                r"""
                (el, selectors) => {
                  const conversationSelector = [
                    '[class*="chat-pane" i]', '[class*="chat-window" i]',
                    '[class*="conversation-container" i]',
                    '[data-test-id*="conversation" i]', 'main', '[role="main"]'
                  ].join(',');
                  const composerSelector = [
                    'form', '[role="form"]', '[data-test-id*="composer" i]',
                    '[class*="input-area" i]', '[class*="input-container" i]'
                  ].join(',');
                  const root = el.closest(conversationSelector) ||
                    el.closest(composerSelector);
                  if (!root) return [];
                  for (const selector of selectors) {
                    const nodes = Array.from(root.querySelectorAll(selector));
                    if (nodes.length) {
                      return nodes.slice(0, 64).map(node => node.innerText || '');
                    }
                  }
                  return [];
                }
                """,
                list(_GEMINI_USER_TURN_SELECTORS),
            )
        except Exception:
            return []

    def is_generating(self, element) -> bool:
        # Stop observation is region-scoped for the same reason as user bubbles;
        # selector ordering remains the adapter's Phase-5B-compatible ordering.
        try:
            return bool(element.handle.evaluate(
                r"""
                (el, selectors) => {
                  const conversationSelector = [
                    '[class*="chat-pane" i]', '[class*="chat-window" i]',
                    '[class*="conversation-container" i]',
                    '[data-test-id*="conversation" i]', 'main', '[role="main"]'
                  ].join(',');
                  const composerSelector = [
                    'form', '[role="form"]', '[data-test-id*="composer" i]',
                    '[class*="input-area" i]', '[class*="input-container" i]'
                  ].join(',');
                  const root = el.closest(conversationSelector) ||
                    el.closest(composerSelector);
                  if (!root) return false;
                  const visible = node => !!node &&
                    !!(node.offsetParent || node.getClientRects().length) &&
                    getComputedStyle(node).visibility !== 'hidden';
                  return selectors.some(selector =>
                    Array.from(root.querySelectorAll(selector)).some(visible));
                }
                """,
                list(self.adapter.stop_button_selectors),
            ))
        except Exception:
            return False

    def login_required(self) -> bool:
        try:
            return not self.adapter.is_logged_in(self.page)
        except Exception:
            return False


def paste_large_prompt(page: "Page", selector: str, text: str) -> bool:
    """Inject a (possibly massive) prompt via a true clipboard paste.

    Cross-platform (uses Cmd+V on macOS, Ctrl+V elsewhere) and resilient: it
    writes ``text`` to the clipboard and dispatches a trusted paste keystroke,
    then falls back to a synthetic paste event and finally direct injection. It
    always fires ``input``/``change`` so React/Vue enables the Send button.

    Parameters
    ----------
    page:     the Playwright ``Page``.
    selector: CSS selector for the chat input (textarea or contenteditable).
    text:     the prompt to paste.

    Returns ``True`` if the editor ends up holding the text.
    """
    loc = page.locator(selector).first
    return _clipboard_paste_into(page, loc, text)


# --------------------------------------------------------------------------- #
# Concrete adapters
# --------------------------------------------------------------------------- #
# NOTE: Web UIs change often. These selector lists are ordered best-guess →
# fallback. If an interaction breaks, inspect the live DOM and prepend the
# correct selector to the relevant list.

# Shared, last-resort fallbacks. Most modern chat UIs use a contenteditable
# div or a <textarea> for input and a submit-style button. Appending these to
# every adapter means a brand-new or freshly-redesigned provider usually still
# works without bespoke selectors.
_GENERIC_INPUT = [
    # Prefer the visible rich-text editor; only fall back to a *real* textarea.
    "div[contenteditable='true'][role='textbox']",
    "div[contenteditable='true']",
    "div[role='textbox']",
    # Exclude hidden fallback/mirror textareas that some apps (e.g. ChatGPT)
    # render off-screen beneath the rich editor.
    "textarea:not([class*='fallback' i]):not([aria-hidden='true']):not([readonly])",
]
_GENERIC_SEND = [
    "button[type='submit']",
    "button[aria-label*='Send' i]",
    "button[data-testid*='send' i]",
]
_GENERIC_STOP = [
    "button[aria-label*='Stop' i]",
    "button[data-testid*='stop' i]",
]
_GENERIC_RESP = [
    "[data-message-author-role='assistant']",
    "[class*='assistant' i]",
    "[class*='message' i] .prose",
    ".prose",
    ".markdown",
]
_GENERIC_LOGGEDIN = [
    "div[contenteditable='true']",
    "textarea:not([class*='fallback' i])",
]


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _adapter(
    *,
    key: str,
    name: str,
    login_url: str,
    new_chat_url: str,
    description: str,
    category: str,
    color: str,
    short: str,
    input_selectors: Optional[list[str]] = None,
    send_button_selectors: Optional[list[str]] = None,
    stop_button_selectors: Optional[list[str]] = None,
    response_selectors: Optional[list[str]] = None,
    logged_in_selectors: Optional[list[str]] = None,
    max_prompt_chars: int = 0,
) -> ProviderAdapter:
    """Build an adapter, merging provider-specific selectors with generics."""
    return ProviderAdapter(
        key=key,
        name=name,
        login_url=login_url,
        new_chat_url=new_chat_url,
        description=description,
        category=category,
        color=color,
        short=short,
        max_prompt_chars=max_prompt_chars,
        input_selectors=_dedupe((input_selectors or []) + _GENERIC_INPUT),
        send_button_selectors=_dedupe((send_button_selectors or []) + _GENERIC_SEND),
        stop_button_selectors=_dedupe((stop_button_selectors or []) + _GENERIC_STOP),
        supports_active_interruption=bool(stop_button_selectors),
        response_selectors=_dedupe((response_selectors or []) + _GENERIC_RESP),
        logged_in_selectors=_dedupe((logged_in_selectors or []) + _GENERIC_LOGGEDIN),
    )


# --------------------------------------------------------------------------- #
# The roster. ``provider`` keys here must match the AgentSpec.provider strings
# the orchestrator routes on (see webllm/pool.py).
# --------------------------------------------------------------------------- #
def _build_registry() -> dict[str, ProviderAdapter]:
    adapters = [
        # ---- Frontier Models -------------------------------------------- #
        _adapter(
            key="openai", name="ChatGPT",
            login_url="https://chatgpt.com/auth/login",
            new_chat_url="https://chatgpt.com/",
            description="OpenAI's GPT — strong all-round reasoning, coding & writing.",
            category="Frontier Models", color="#10a37f", short="Gp",
            input_selectors=[
                # Visible rich-text editor only. NEVER the hidden
                # *_fallbackTextarea mirror that sits behind it.
                "div#prompt-textarea[contenteditable='true']",
                "div.ProseMirror#prompt-textarea",
                "div.ProseMirror[contenteditable='true']",
                "div[contenteditable='true'][role='textbox']",
                "[contenteditable='true'][data-virtualkeyboard='true']",
            ],
            send_button_selectors=[
                "button[data-testid='send-button']",
                "button[aria-label='Send prompt']",
                "button#composer-submit-button",
            ],
            stop_button_selectors=[
                "button[data-testid='stop-button']",
                "button[aria-label='Stop generating']",
            ],
            response_selectors=[
                "div[data-message-author-role='assistant']",
                "div.agent-turn",
            ],
            logged_in_selectors=[
                "nav[aria-label='Chat history']",
                "button[data-testid='profile-button']",
            ],
        ),
        _adapter(
            key="anthropic", name="Claude",
            login_url="https://claude.ai/login",
            new_chat_url="https://claude.ai/new",
            description="Anthropic's Claude — top-tier reasoning, long context & prose.",
            category="Frontier Models", color="#d97757", short="Cl",
            input_selectors=[
                "div[contenteditable='true'].ProseMirror",
                "div.ProseMirror[contenteditable='true']",
            ],
            send_button_selectors=[
                "button[aria-label='Send message']",
                "button[aria-label='Send Message']",
            ],
            stop_button_selectors=["button[aria-label='Stop response']"],
            response_selectors=[
                "div.font-claude-message",
                "div[data-testid='assistant-turn']",
            ],
            logged_in_selectors=["a[href='/new']"],
        ),
        _adapter(
            key="google", name="Gemini",
            login_url="https://gemini.google.com/app",
            new_chat_url="https://gemini.google.com/app",
            description="Google's Gemini — huge context windows & multimodal research.",
            category="Frontier Models", color="#1a73e8", short="Ge",
            # Phase 4G: layered, ordered best→fallback. Accessible role/name and
            # aria signals first (survive class churn), then the known Quill
            # editor, then a conservative structural contenteditable. Kept in sync
            # with webllm.gemini_submission.COMPOSER_SELECTORS/SEND_SELECTORS,
            # which the deterministic submission state machine uses.
            input_selectors=[
                "div[role='textbox'][contenteditable='true']",
                "div[contenteditable='true'][aria-label]",
                "rich-textarea div.ql-editor[contenteditable='true']",
                "div.ql-editor[contenteditable='true']",
            ],
            send_button_selectors=[
                "button[aria-label='Send message']",
                "button[aria-label*='Send' i]",
                "button.send-button[mat-icon-button]",
                "button.send-button",
            ],
            stop_button_selectors=[
                "button[aria-label='Stop response']",
                "button[aria-label*='Stop' i]",
                "button.send-button.stop",
                "button.stop",
            ],
            response_selectors=[
                "message-content .model-response-text",
                "message-content",
            ],
            logged_in_selectors=["rich-textarea", "button[aria-label*='Google Account']"],
        ),
        _adapter(
            key="copilot", name="Microsoft Copilot",
            login_url="https://copilot.microsoft.com/",
            new_chat_url="https://copilot.microsoft.com/",
            description="Microsoft Copilot — GPT-backed assistant with web grounding.",
            category="Frontier Models", color="#0e7ee4", short="Co",
            input_selectors=["textarea#userInput", "textarea[placeholder*='Message']"],
            response_selectors=["div[data-content='ai-message']", "cib-message[source='bot']"],
            # Copilot's composer rejects long messages (~10k chars); clip first.
            max_prompt_chars=9500,
        ),
        _adapter(
            key="grok", name="Grok",
            login_url="https://grok.com/",
            new_chat_url="https://grok.com/",
            description="xAI's Grok — witty, real-time answers wired into X.",
            category="Frontier Models", color="#6b7280", short="Gk",
            input_selectors=["textarea[aria-label*='Ask']", "textarea"],
            response_selectors=["div.message-bubble", "[class*='response']"],
        ),
        _adapter(
            key="meta", name="Meta AI",
            login_url="https://www.meta.ai/",
            new_chat_url="https://www.meta.ai/",
            description="Meta's Llama-powered assistant for chat and image gen.",
            category="Frontier Models", color="#0866ff", short="Me",
            input_selectors=["div[contenteditable='true'][role='textbox']", "textarea"],
        ),

        # ---- Search & Research ------------------------------------------ #
        _adapter(
            key="perplexity", name="Perplexity",
            login_url="https://www.perplexity.ai/",
            new_chat_url="https://www.perplexity.ai/",
            description="Answer engine with live citations — great for research.",
            category="Search & Research", color="#20808d", short="Px",
            input_selectors=[
                "textarea[placeholder*='Ask']",
                "div[contenteditable='true']",
            ],
            response_selectors=["div.prose", "[class*='answer']"],
        ),

        # ---- Open Source / Fast ----------------------------------------- #
        _adapter(
            key="mistral", name="Le Chat (Mistral)",
            login_url="https://chat.mistral.ai/",
            new_chat_url="https://chat.mistral.ai/chat",
            description="Mistral's Le Chat — fast European open-weight models.",
            category="Open Source / Fast", color="#fa520f", short="Mi",
            input_selectors=["textarea[placeholder*='Ask']", "textarea"],
        ),
        _adapter(
            key="huggingface", name="HuggingChat",
            login_url="https://huggingface.co/chat/",
            new_chat_url="https://huggingface.co/chat/",
            description="Open-source models (Llama, Qwen, etc.) on Hugging Face.",
            category="Open Source / Fast", color="#ff9d00", short="HF",
            input_selectors=["textarea[placeholder*='Ask']", "textarea"],
            response_selectors=["div.prose", "[class*='message']"],
        ),
        _adapter(
            key="groq", name="Groq",
            login_url="https://chat.groq.com/",
            new_chat_url="https://chat.groq.com/",
            description="Blazing-fast inference on open models via Groq LPUs.",
            category="Open Source / Fast", color="#f55036", short="Gq",
            input_selectors=["textarea[placeholder*='Message']", "textarea"],
        ),
        _adapter(
            key="deepseek", name="DeepSeek",
            login_url="https://chat.deepseek.com/",
            new_chat_url="https://chat.deepseek.com/",
            description="DeepSeek V3/R1 — strong reasoning & coding, low cost.",
            category="Open Source / Fast", color="#4d6bfe", short="DS",
            input_selectors=["textarea#chat-input", "textarea"],
        ),

        # ---- Aggregators & Companions ----------------------------------- #
        _adapter(
            key="poe", name="Poe",
            login_url="https://poe.com/login",
            new_chat_url="https://poe.com/",
            description="Quora's Poe — one login, dozens of bots & models.",
            category="Aggregators & Companions", color="#5d5dff", short="Po",
            input_selectors=["textarea[placeholder*='Talk']", "textarea"],
        ),
        _adapter(
            key="pi", name="Pi",
            login_url="https://pi.ai/talk",
            new_chat_url="https://pi.ai/talk",
            description="Inflection's Pi — a warm, conversational companion.",
            category="Aggregators & Companions", color="#a855f7", short="Pi",
            input_selectors=["textarea[placeholder*='Talk']", "textarea"],
        ),
    ]
    return {a.key: a for a in adapters}


# Registry keyed by the same ``provider`` string used in AgentSpec.
PROVIDERS: dict[str, ProviderAdapter] = _build_registry()

# Display order for the frontend grid.
CATEGORY_ORDER = [
    "Frontier Models",
    "Search & Research",
    "Open Source / Fast",
    "Aggregators & Companions",
]


def get_adapter(provider: str) -> ProviderAdapter:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:  # noqa: TRY003
        raise ProviderError(
            f"No web adapter registered for provider {provider!r}. "
            f"Known: {sorted(PROVIDERS)}"
        ) from exc


def available_providers() -> list[dict[str, str]]:
    """Rich metadata for the frontend grid (logo/colour/category/etc.)."""
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    adapters = sorted(
        PROVIDERS.values(),
        key=lambda a: (order.get(a.category, 99), a.name.lower()),
    )
    return [
        {
            "key": a.key,
            "name": a.name,
            "description": a.description,
            "category": a.category,
            "color": a.color,
            "short": a.short or a.name[:2],
        }
        for a in adapters
    ]
