"use strict";
/*
 * Keyboard policy for the orchestrator prompt box.
 *
 * dozenPromptKeyAction is a PURE function: it receives plain facts about one
 * keydown (no DOM, no globals) and returns what the caller must do. That
 * keeps the decision unit-testable under Node while index.html stays a thin
 * wiring layer that reuses the Send button's existing submission function.
 *
 * Returned actions:
 *   "submit"  — preventDefault and call the shared submit function, once.
 *   "newline" — let the browser insert a newline (Shift+Enter editing).
 *   "block"   — preventDefault, submit nothing (plain Enter reserved for
 *               submit, but the prompt is empty/whitespace, a run is already
 *               submitting, or the key is auto-repeating).
 *   "none"    — not ours to handle: default behavior untouched (non-Enter
 *               keys, IME composition, and undefined modifier combos like
 *               Ctrl/Alt/Meta+Enter).
 */
function dozenPromptKeyAction(k) {
  if (k.key !== "Enter") return "none";
  // An IME composition session commits with Enter; never treat it as submit.
  // keyCode 229 is the classic composition signal older engines emit.
  if (k.isComposing || k.keyCode === 229) return "none";
  if (k.shiftKey) return "newline";
  // Ctrl/Alt/Meta+Enter are not defined by this UI: leave them to the
  // browser rather than surprise-submitting.
  if (k.ctrlKey || k.altKey || k.metaKey) return "none";
  if (k.repeat) return "block";                    // held key must not spam
  if (k.busy) return "block";                      // a run is already active
  if (!k.text || !k.text.trim()) return "block";   // empty/whitespace prompt
  return "submit";
}

/* Node export for the unit tests; browsers just get the global function. */
if (typeof module !== "undefined" && module.exports) {
  module.exports = { dozenPromptKeyAction };
}
