"""Provider-independent failure signal patterns (Phase 2.2.1).

Pure data + tiny match helpers. Nothing here is provider-specific by design:
these are the phrases browsers, Playwright, Cloudflare and LLM web apps use
generically. Future providers plug in by EXTENDING these tuples (or
registering extra rules) — never by editing rule logic.

All matching is case-insensitive substring/regex over the evidence's
searchable text; the prompt content itself is never inspected.
"""

from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# Phrase tables (lowercase; matched with `in` against lowercased haystacks)
# --------------------------------------------------------------------------- #
BROWSER_CRASH_PHRASES: tuple[str, ...] = (
    "browser has been closed",
    "browser closed",
    "browser disconnected",
    "target closed",                 # playwright's classic
    "connection closed",
    "epipe",
    "broken pipe",
    "playwright connection closed",
    "websocket closed",
)

TAB_CLOSED_PHRASES: tuple[str, ...] = (
    "page has been closed",
    "page closed",
    "target page, context or browser has been closed",
    "tab closed",
    "frame was detached",
    "execution context was destroyed",
)

CAPTCHA_PHRASES: tuple[str, ...] = (
    "captcha",
    "cloudflare",
    "cf-turnstile",
    "turnstile",
    "verify you are human",
    "verifying you are human",
    "are you a robot",
    "unusual traffic",
    "security check",
    "checking your browser",
)

LOGIN_PHRASES: tuple[str, ...] = (
    "log in",
    "login",
    "sign in",
    "signin",
    "session expired",
    "session has expired",
    "logged out",
    "not logged in",
    "authentication required",
    "please authenticate",
    "unauthorized",
)

RATE_LIMIT_PHRASES: tuple[str, ...] = (
    "rate limit",
    "rate-limited",
    "too many requests",
    "quota exceeded",
    "usage limit",
    "limit reached",
    "you've reached",
    "you have reached",
    "429",
    "try again later",
)

MODEL_BUSY_PHRASES: tuple[str, ...] = (
    "at capacity",
    "high demand",
    "model is busy",
    "currently overloaded",
    "servers are busy",
    "temporarily unavailable",
    "heavy load",
)

NETWORK_PHRASES: tuple[str, ...] = (
    "net::err_",
    "err_internet_disconnected",
    "err_connection",
    "err_name_not_resolved",
    "econnrefused",
    "econnreset",
    "etimedout",                     # socket-level, distinct from UI timeout
    "dns",
    "network is unreachable",
    "no internet",
    "offline",
)

TIMEOUT_PHRASES: tuple[str, ...] = (
    "timed out",
    "timeout",
    "deadline exceeded",
)

DOM_CHANGED_PHRASES: tuple[str, ...] = (
    "input box not found",
    "could not locate",
    "could not find the chat input",
    "no response container",
    "selector",
    "element not found",
    "unable to resolve",
)

PROMPT_REJECTED_PHRASES: tuple[str, ...] = (
    "message too long",
    "message is too long",
    "prompt too long",
    "maximum length",
    "character limit",
    "input too large",
)

EMPTY_RESPONSE_PHRASES: tuple[str, ...] = (
    "scraped an empty response",
    "empty response",
    "response was empty",
    "no response text",
)

OUTPUT_CORRUPTED_PHRASES: tuple[str, ...] = (
    "invalid json",
    "unparseable",
    "could not parse",
    "malformed artifact",
    "json decode",
)

# Exception type names that are themselves strong signals.
TIMEOUT_EXCEPTION_TYPES: tuple[str, ...] = ("TimeoutError",)

# HTTP status groupings (used when http_status evidence is available).
RATE_LIMIT_STATUS: tuple[int, ...] = (429,)
SERVER_BUSY_STATUS: tuple[int, ...] = (500, 502, 503, 504)


# --------------------------------------------------------------------------- #
# Match helpers
# --------------------------------------------------------------------------- #
def any_phrase(haystack: str, phrases: Iterable[str]) -> bool:
    return any(p in haystack for p in phrases)


def matched_phrases(haystack: str, phrases: Iterable[str]) -> tuple[str, ...]:
    return tuple(p for p in phrases if p in haystack)


_LOGIN_URL_RE = re.compile(r"/(log-?in|sign-?in|auth|sso)([/?#]|$)")


def looks_like_login_url(url: str) -> bool:
    return bool(_LOGIN_URL_RE.search(url.lower()))
