"""A deterministic in-memory Gemini DOM for the submission state machine tests.

Models the documented current and previous Gemini DOMs and lets each test script
element state, insertion behaviour, send-button enabling, click side effects and
acknowledgement — all without Playwright and without sleeps. State advances
across the state machine's bounded polls via a per-query tick counter and an
``on_click`` callback, so races (late-enabling button, ack-after-click) are
exercised with barriers, not timing.
"""

from __future__ import annotations

from typing import Callable, Optional

from webllm.gemini_submission import (
    COMPOSER_SELECTORS,
    SEND_SELECTORS,
    DomElement,
)


class Node:
    """A mutable fake element; ``element()`` snapshots it for the state machine."""

    def __init__(
        self, handle: str, tag: str = "div", *, role: str = "",
        name: str = "", visible: bool = True, enabled: bool = True,
        editable: bool = False, is_stop: bool = False, is_attach: bool = False,
        region: str = "composer", chat_region: str = "active", enable_at: int = 0,
    ) -> None:
        self.handle = handle
        self.tag = tag
        self.role = role
        self.name = name
        self.visible = visible
        self.enabled = enabled
        self.editable = editable
        self.is_stop = is_stop
        self.is_attach = is_attach
        self.region = region
        self.chat_region = chat_region
        # If >0, the node becomes enabled once this many send-locator polls run.
        self.enable_at = enable_at

    def element(self) -> DomElement:
        return DomElement(
            handle=self.handle, tag=self.tag, role=self.role,
            accessible_name=self.name, visible=self.visible,
            enabled=self.enabled, editable=self.editable,
            is_stop_control=self.is_stop, is_attachment_control=self.is_attach,
            region=self.region,
            chat_region=self.chat_region,
        )


class FakeGeminiDom:
    """Implements the ``GeminiDom`` port over scripted nodes and state."""

    def __init__(
        self, *, selector_map: dict[str, list[Node]],
        composer_text: str = "", user_messages: Optional[list[str]] = None,
        generating: bool = False, login: bool = False,
        insert_transform: Optional[Callable[[str], str]] = None,
        on_click: Optional[Callable[["FakeGeminiDom"], None]] = None,
        click_raises: bool = False,
    ) -> None:
        self.selector_map = selector_map
        self.composer_text = composer_text
        self.messages = list(user_messages or [])
        self.generating = generating
        self.login = login
        self.insert_transform = insert_transform
        self.on_click = on_click
        self.click_raises = click_raises
        self.click_count = 0
        self.insert_count = 0
        self._send_polls = 0

    # --- GeminiDom port ------------------------------------------------- #
    def query(self, selector: str):
        # Advance the send-enable schedule whenever a send selector is queried.
        if any(selector == sel for _s, sel in SEND_SELECTORS):
            self._send_polls += 1
            for nodes in self.selector_map.values():
                for node in nodes:
                    if node.enable_at and self._send_polls >= node.enable_at:
                        node.enabled = True
        return [node.element() for node in self.selector_map.get(selector, [])]

    def focus(self, element: DomElement) -> bool:
        return True

    def insert_text(self, element: DomElement, text: str) -> bool:
        self.insert_count += 1
        self.composer_text = (
            self.insert_transform(text) if self.insert_transform else text
        )
        return True

    def read_composer(self, element: DomElement) -> str:
        return self.composer_text

    def clear(self, element: DomElement) -> None:
        self.composer_text = ""

    def click(self, element: DomElement) -> bool:
        self.click_count += 1
        if self.on_click is not None:
            self.on_click(self)
        if self.click_raises:
            raise RuntimeError("playwright click raised after dispatch")
        return True

    def user_messages(self, element: DomElement):
        return list(self.messages)

    def is_generating(self, element: DomElement) -> bool:
        return self.generating

    def login_required(self) -> bool:
        return self.login


# --------------------------------------------------------------------------- #
# Builders for the documented DOMs
# --------------------------------------------------------------------------- #
def _sel(strategy_selectors, strategy_value: str) -> str:
    for strat, selector in strategy_selectors:
        if strat.value == strategy_value:
            return selector
    raise KeyError(strategy_value)


COMPOSER_ROLE_NAME = _sel(COMPOSER_SELECTORS, "role_name")
COMPOSER_KNOWN = _sel(COMPOSER_SELECTORS, "known")
COMPOSER_STRUCTURAL = _sel(COMPOSER_SELECTORS, "structural")
SEND_ROLE_NAME = _sel(SEND_SELECTORS, "role_name")
SEND_KNOWN = _sel(SEND_SELECTORS, "known")
SEND_STRUCTURAL = _sel(SEND_SELECTORS, "structural")


def clears_composer(dom: "FakeGeminiDom") -> None:
    dom.composer_text = ""


def append_matching_bubble(prompt: str):
    def _hook(dom: "FakeGeminiDom") -> None:
        dom.messages.append(prompt)
        dom.composer_text = ""
    return _hook


def starts_generating(dom: "FakeGeminiDom") -> None:
    dom.generating = True


def current_dom(prompt: str, **kw) -> FakeGeminiDom:
    """Current Gemini DOM: role=textbox composer + aria Send button."""
    composer = Node("composer", tag="div", role="textbox", name="Enter a prompt",
                    editable=True, region="composer")
    send = Node("send", tag="button", name="Send message", region="composer")
    selector_map = {
        COMPOSER_ROLE_NAME: [composer],
        COMPOSER_STRUCTURAL: [composer],
        SEND_ROLE_NAME: [send],
        SEND_STRUCTURAL: [send],
    }
    kw.setdefault("on_click", append_matching_bubble(prompt))
    return FakeGeminiDom(selector_map=selector_map, **kw)


def previous_dom(prompt: str, **kw) -> FakeGeminiDom:
    """Previous Gemini DOM: rich-textarea ql-editor + button.send-button."""
    composer = Node("composer", tag="div", editable=True, region="composer")
    send = Node("send", tag="button", name="", region="composer")
    selector_map = {
        COMPOSER_KNOWN: [composer],
        COMPOSER_STRUCTURAL: [composer],
        SEND_KNOWN: [send],
        SEND_STRUCTURAL: [send],
    }
    kw.setdefault("on_click", append_matching_bubble(prompt))
    return FakeGeminiDom(selector_map=selector_map, **kw)
