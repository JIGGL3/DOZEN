"""Phase 4G — the Gemini submission state machine over realistic fake DOMs.

Covers the documented current and previous DOMs, the layered locator strategy,
insertion confirmation, the acknowledgement signals, exactly-once sending, the
cancellation races, and every typed failure category. Barriers/events (a click
hook, a poll-scheduled enable) drive state — no test sleeps.
"""

from __future__ import annotations

import unittest

from dozen.llm_client import LLMMessage
from webllm.client import WebAutomationLLMClient
from webllm.gemini_submission import (
    COMPOSER_SELECTORS,
    SEND_SELECTORS,
    DomElement,
    SubmissionConfig,
    SubmissionErrorCode,
    SubmissionStatus,
    bounded_hash,
    normalize,
    submit_prompt,
)
from webllm.providers import (
    GeminiSubmissionProviderError,
    _PlaywrightGeminiDom,
    get_adapter,
)

from .harness import (
    COMPOSER_ROLE_NAME,
    COMPOSER_STRUCTURAL,
    SEND_ROLE_NAME,
    SEND_STRUCTURAL,
    FakeGeminiDom,
    Node,
    append_matching_bubble,
    clears_composer,
    current_dom,
    previous_dom,
    starts_generating,
)

PROMPT = "Write a Python function that reverses a linked list.\nInclude tests."


class TestHappyPaths(unittest.TestCase):
    def test_current_dom_submits_once(self) -> None:
        dom = current_dom(PROMPT)
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(dom.click_count, 1)
        self.assertEqual(result.send_actions, 1)
        self.assertEqual(result.composer_strategy, "role_name")
        self.assertEqual(result.acknowledgement, "user_bubble_matched")

    def test_previous_dom_still_works(self) -> None:
        dom = previous_dom(PROMPT)
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(result.composer_strategy, "known")
        self.assertEqual(dom.click_count, 1)

    def test_composer_clear_acknowledgement(self) -> None:
        dom = current_dom(PROMPT, on_click=clears_composer)
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(result.acknowledgement, "composer_cleared")

    def test_generation_acknowledgement(self) -> None:
        dom = current_dom(PROMPT, on_click=starts_generating)
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(result.acknowledgement, "generation_started")

    def test_existing_conversation_ack_by_new_bubble(self) -> None:
        dom = current_dom(PROMPT, user_messages=["an earlier question"])
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(len(dom.messages), 2)

    def test_unicode_multiline_and_long_prompts_are_confirmed_in_full(self) -> None:
        prompts = (
            "नमस्ते Gemini — write café tests 🚀",
            "first line\n\n  indented second line\nthird line",
            ("0123456789abcdef" * 256) + " END-OF-PROMPT",
        )
        for prompt in prompts:
            with self.subTest(length=len(prompt)):
                dom = current_dom(prompt)
                result = submit_prompt(dom, prompt)
                self.assertTrue(result.ok)
                self.assertEqual(dom.click_count, 1)


class TestLocatorStrategy(unittest.TestCase):
    def test_structural_contenteditable_only(self) -> None:
        for chat_region in ("desktop-pane", "mobile-narrow-pane"):
            with self.subTest(chat_region=chat_region):
                composer = Node(
                    "c", editable=True, region="composer",
                    chat_region=chat_region,
                )
                send = Node(
                    "s", tag="button", region="composer",
                    chat_region=chat_region,
                )
                dom = FakeGeminiDom(
                    selector_map={COMPOSER_STRUCTURAL: [composer],
                                  SEND_STRUCTURAL: [send]},
                    on_click=append_matching_bubble(PROMPT),
                )
                result = submit_prompt(dom, PROMPT)
                self.assertTrue(result.ok)
                self.assertEqual(result.composer_strategy, "structural")

    def test_hidden_duplicate_composer_is_ignored(self) -> None:
        visible = Node("real", role="textbox", editable=True, region="composer")
        hidden = Node("tmpl", role="textbox", editable=True, region="hidden")
        feedback = Node(
            "feedback", role="textbox", editable=True, region="feedback"
        )
        send = Node("s", tag="button", name="Send message", region="composer")
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [visible, hidden, feedback],
                          SEND_ROLE_NAME: [send]},
            on_click=append_matching_bubble(PROMPT),
        )
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)   # region filter leaves exactly one valid

    def test_ambiguous_composers_fall_through_then_fail(self) -> None:
        # Two equally-valid composers at the SAME strategy is ambiguous; the
        # first arbitrary one is never chosen.
        a = Node("a", role="textbox", editable=True, region="composer")
        b = Node("b", role="textbox", editable=True, region="composer")
        dom = FakeGeminiDom(selector_map={COMPOSER_ROLE_NAME: [a, b]})
        result = submit_prompt(dom, PROMPT)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, SubmissionErrorCode.COMPOSER_NOT_FOUND)

    def test_search_box_is_never_chosen_as_composer(self) -> None:
        search = Node("search", role="textbox", editable=True, region="search")
        dom = FakeGeminiDom(selector_map={COMPOSER_ROLE_NAME: [search]})
        result = submit_prompt(dom, PROMPT)
        self.assertFalse(result.ok)

    def test_composer_and_send_must_share_the_same_chat_region(self) -> None:
        composer = Node("c", role="textbox", editable=True,
                        chat_region="pane-a")
        send = Node("s", tag="button", name="Send message",
                    chat_region="pane-b")
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]},
            on_click=append_matching_bubble(PROMPT),
        )
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SEND_CONTROL_NOT_FOUND)
        self.assertEqual(dom.click_count, 0)

    def test_production_adapter_scopes_ack_to_enclosing_conversation(self) -> None:
        # A Gemini composer form is only the input footer; user bubbles are its
        # siblings.  The production adapter must scope to the enclosing chat
        # pane (and distinguish a second pane), not stop at the form.
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Playwright is not installed")
        html = """
        <main>
          <section class="chat-window pane-a">
            <user-query><div class="query-text">first user</div></user-query>
            <form><div role="textbox" aria-label="Enter a prompt"
              contenteditable="true"></div>
              <button aria-label="Send message">Send</button></form>
          </section>
          <section class="chat-window pane-b">
            <user-query><div class="query-text">other user</div></user-query>
            <form><div role="textbox" aria-label="Enter a prompt"
              contenteditable="true"></div>
              <button aria-label="Send message">Send</button></form>
          </section>
        </main>
        """
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:  # pragma: no cover - environment dependent
                self.skipTest(f"Chromium is unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html)
                dom = _PlaywrightGeminiDom(page, get_adapter("google"))
                composers = dom.query('[role="textbox"]')
                sends = dom.query('button[aria-label="Send message"]')
                self.assertEqual(len(composers), 2)
                self.assertNotEqual(composers[0].chat_region,
                                    composers[1].chat_region)
                self.assertEqual(composers[0].chat_region,
                                 sends[0].chat_region)
                self.assertEqual(dom.user_messages(composers[0]), ["first user"])
                self.assertEqual(dom.user_messages(composers[1]), ["other user"])
            finally:
                browser.close()


class TestTypedFailures(unittest.TestCase):
    def test_composer_not_found_is_drift_when_nothing_matches(self) -> None:
        dom = FakeGeminiDom(selector_map={})
        result = submit_prompt(dom, PROMPT)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SELECTOR_DRIFT_DETECTED)

    def test_composer_not_editable(self) -> None:
        composer = Node("c", role="textbox", editable=False, region="composer")
        dom = FakeGeminiDom(selector_map={COMPOSER_ROLE_NAME: [composer]})
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.COMPOSER_NOT_EDITABLE)

    def test_prompt_insertion_mismatch(self) -> None:
        dom = current_dom(PROMPT, insert_transform=lambda _t: "garbled partial")
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.PROMPT_INSERTION_MISMATCH)
        self.assertEqual(dom.click_count, 0)   # never sent on a bad insert

    def test_long_partial_prefix_is_not_accepted_as_complete_insertion(self) -> None:
        prompt = "A" * 500 + " END"
        dom = current_dom(prompt, insert_transform=lambda text: text[:200])
        result = submit_prompt(dom, prompt)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.PROMPT_INSERTION_MISMATCH)
        self.assertEqual(dom.click_count, 0)

    def test_missing_send_control_is_drift(self) -> None:
        composer = Node("c", role="textbox", editable=True, region="composer")
        dom = FakeGeminiDom(selector_map={COMPOSER_ROLE_NAME: [composer]})
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SELECTOR_DRIFT_DETECTED)
        self.assertEqual(dom.click_count, 0)

    def test_disabled_send_button_never_enables(self) -> None:
        composer = Node("c", role="textbox", editable=True, region="composer")
        send = Node("s", tag="button", name="Send message",
                    enabled=False, region="composer")
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]},
        )
        result = submit_prompt(dom, PROMPT, config=SubmissionConfig(send_enable_polls=5))
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SEND_CONTROL_DISABLED)
        self.assertEqual(dom.click_count, 0)

    def test_stop_control_is_not_mistaken_for_send(self) -> None:
        composer = Node("c", role="textbox", editable=True, region="composer")
        stop = Node("stop", tag="button", name="Stop response",
                    is_stop=True, region="composer")
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [stop]},
        )
        result = submit_prompt(dom, PROMPT)
        self.assertFalse(result.ok)
        self.assertEqual(dom.click_count, 0)

        # During generation Gemini may briefly expose both controls.  The stop
        # control is filtered and the sole actual send control is selected.
        send = Node("send", tag="button", name="Send message", region="composer")
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [composer],
                          SEND_ROLE_NAME: [stop, send]},
            on_click=append_matching_bubble(PROMPT),
        )
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)
        self.assertEqual(dom.click_count, 1)

    def test_submission_not_confirmed(self) -> None:
        # Click does nothing observable: no bubble, composer keeps text, idle.
        dom = current_dom(PROMPT, on_click=lambda _d: None)
        result = submit_prompt(dom, PROMPT,
                               config=SubmissionConfig(acknowledgement_polls=3))
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED)
        self.assertEqual(dom.click_count, 1)   # exactly one send attempt

    def test_unrelated_new_user_bubble_is_not_an_acknowledgement(self) -> None:
        def different_bubble(dom):
            dom.messages.append("a different prompt from another pane")

        dom = current_dom(PROMPT, on_click=different_bubble)
        result = submit_prompt(
            dom, PROMPT, config=SubmissionConfig(acknowledgement_polls=3)
        )
        self.assertEqual(
            result.error_code, SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED
        )
        self.assertEqual(dom.click_count, 1)

    def test_detached_composer_is_not_mistaken_for_a_cleared_composer(self) -> None:
        class DetachedAfterClickDom(FakeGeminiDom):
            def read_composer(self, element):
                if self.click_count:
                    return None
                return super().read_composer(element)

        composer = Node("c", role="textbox", editable=True)
        send = Node("s", tag="button", name="Send message")
        dom = DetachedAfterClickDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]}
        )
        result = submit_prompt(
            dom, PROMPT, config=SubmissionConfig(acknowledgement_polls=3)
        )
        self.assertEqual(
            result.error_code, SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED
        )

    def test_composer_replaced_after_insertion_fails_before_send(self) -> None:
        class ReplacedComposerDom(FakeGeminiDom):
            def read_composer(self, element):
                if self.insert_count:
                    return None
                return super().read_composer(element)

        composer = Node("c", role="textbox", editable=True)
        send = Node("s", tag="button", name="Send message")
        dom = ReplacedComposerDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]}
        )
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(
            result.error_code, SubmissionErrorCode.PROMPT_INSERTION_MISMATCH
        )
        self.assertEqual(dom.click_count, 0)

    def test_login_required_is_distinct_from_drift(self) -> None:
        dom = FakeGeminiDom(selector_map={}, login=True)
        result = submit_prompt(dom, PROMPT)
        self.assertEqual(result.error_code, SubmissionErrorCode.LOGIN_REQUIRED)


class TestSendEnableRace(unittest.TestCase):
    def test_button_enables_just_before_timeout(self) -> None:
        composer = Node("c", role="textbox", editable=True, region="composer")
        send = Node("s", tag="button", name="Send message",
                    enabled=False, region="composer", enable_at=3)
        dom = FakeGeminiDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]},
            on_click=append_matching_bubble(PROMPT),
        )
        result = submit_prompt(dom, PROMPT, config=SubmissionConfig(send_enable_polls=10))
        self.assertTrue(result.ok)
        self.assertEqual(dom.click_count, 1)


class TestExactlyOnce(unittest.TestCase):
    def test_click_raises_but_ack_arrives_no_second_send(self) -> None:
        dom = current_dom(PROMPT, click_raises=True,
                          on_click=append_matching_bubble(PROMPT))
        result = submit_prompt(dom, PROMPT)
        self.assertTrue(result.ok)             # ack confirmed despite the raise
        self.assertEqual(dom.click_count, 1)   # ambiguous send is NEVER retried
        self.assertEqual(result.send_actions, 1)

    def test_click_raises_and_no_ack_is_unconfirmed_not_resent(self) -> None:
        dom = current_dom(PROMPT, click_raises=True, on_click=lambda _d: None)
        result = submit_prompt(dom, PROMPT,
                               config=SubmissionConfig(acknowledgement_polls=3))
        self.assertEqual(result.error_code,
                         SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED)
        self.assertEqual(dom.click_count, 1)   # still exactly one physical send

    def test_production_playwright_port_never_retries_an_ambiguous_click(self) -> None:
        class Locator:
            def __init__(self) -> None:
                self.clicks = 0

            def click(self, **_kwargs) -> None:
                self.clicks += 1
                raise RuntimeError("Playwright raised after dispatch")

        locator = Locator()
        element = type("Element", (), {"handle": locator})()
        dom = _PlaywrightGeminiDom(object(), get_adapter("google"))
        self.assertFalse(dom.click(element))
        self.assertEqual(locator.clicks, 1)

    def test_public_client_preserves_typed_failure_and_never_retries(self) -> None:
        class Browser:
            def __init__(self) -> None:
                self.calls = 0

            def send_prompt(self, provider, prompt, should_cancel):
                self.calls += 1
                raise GeminiSubmissionProviderError(
                    SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED.value,
                    send_actions=1,
                )

        browser = Browser()
        client = WebAutomationLLMClient(
            browser, max_retries=3, retry_backoff_s=0
        )
        with self.assertRaises(GeminiSubmissionProviderError) as ctx:
            client.complete(
                provider="google",
                model="gemini",
                messages=[LLMMessage("user", PROMPT)],
            )
        self.assertEqual(browser.calls, 1)
        self.assertEqual(
            ctx.exception.error_code,
            SubmissionErrorCode.SUBMISSION_NOT_CONFIRMED.value,
        )
        self.assertEqual(ctx.exception.send_actions, 1)

    def test_matching_bubble_delayed_across_polls_still_uses_one_click(self) -> None:
        class DelayedBubbleDom(FakeGeminiDom):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.message_reads = 0

            def user_messages(self, element):
                self.message_reads += 1
                if self.click_count and self.message_reads == 4:
                    self.messages.append(PROMPT)
                return super().user_messages(element)

        composer = Node("c", role="textbox", editable=True)
        send = Node("s", tag="button", name="Send message")
        dom = DelayedBubbleDom(
            selector_map={COMPOSER_ROLE_NAME: [composer], SEND_ROLE_NAME: [send]},
            on_click=lambda _dom: None,
        )
        result = submit_prompt(
            dom, PROMPT, config=SubmissionConfig(acknowledgement_polls=5)
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.acknowledgement, "user_bubble_matched")
        self.assertEqual(dom.click_count, 1)


class TestCancellationRaces(unittest.TestCase):
    def test_cancel_before_insertion(self) -> None:
        dom = current_dom(PROMPT)
        result = submit_prompt(dom, PROMPT, should_cancel=lambda: True)
        self.assertEqual(result.status, SubmissionStatus.CANCELLED)
        self.assertEqual(dom.insert_count, 0)
        self.assertEqual(dom.click_count, 0)

    def test_cancel_after_insertion_before_send(self) -> None:
        state = {"calls": 0}

        def cancel() -> bool:
            state["calls"] += 1
            return state["calls"] > 2   # allow locate+insert, then cancel

        dom = current_dom(PROMPT)
        result = submit_prompt(dom, PROMPT, should_cancel=cancel)
        self.assertEqual(result.status, SubmissionStatus.CANCELLED)
        self.assertEqual(dom.click_count, 0)   # never sent

    def test_cancel_immediately_after_send(self) -> None:
        # Cancel fires only after the click; the send already happened exactly
        # once and acknowledgement is abandoned as cancelled.
        seen = {"clicked": False}

        def on_click(d):
            seen["clicked"] = True

        def cancel() -> bool:
            return seen["clicked"]

        dom = current_dom(PROMPT, on_click=on_click)
        result = submit_prompt(dom, PROMPT, should_cancel=cancel)
        self.assertEqual(result.status, SubmissionStatus.CANCELLED)
        self.assertEqual(dom.click_count, 1)

    def test_confirmed_send_is_not_relabelled_cancelled(self) -> None:
        state = {"clicked": False}

        def on_click(dom):
            state["clicked"] = True
            dom.messages.append(PROMPT)

        dom = current_dom(PROMPT, on_click=on_click)
        result = submit_prompt(
            dom, PROMPT, should_cancel=lambda: state["clicked"]
        )
        self.assertEqual(result.status, SubmissionStatus.SUBMITTED)
        self.assertEqual(result.acknowledgement, "user_bubble_matched")
        self.assertEqual(dom.click_count, 1)


class TestDiagnosticsAreContentFree(unittest.TestCase):
    def test_drift_diagnostics_carry_no_prompt(self) -> None:
        dom = FakeGeminiDom(selector_map={})
        result = submit_prompt(dom, PROMPT)
        blob = repr(result.locator_diagnostics) + result.diagnostic
        self.assertNotIn("linked list", blob)          # no prompt content
        self.assertNotIn(PROMPT[:20], blob)
        # But the safe locator metadata IS present.
        self.assertTrue(result.locator_diagnostics)
        self.assertEqual(result.locator_diagnostics[0].role, "composer")

    def test_success_diagnostics_have_no_response_text(self) -> None:
        dom = current_dom(PROMPT)
        result = submit_prompt(dom, PROMPT)
        for diag in result.locator_diagnostics:
            self.assertNotIn("linked list", str(diag.to_dict()))


class TestNormalization(unittest.TestCase):
    def test_bounded_hash_is_whitespace_insensitive(self) -> None:
        self.assertEqual(bounded_hash("a  b\n c"), bounded_hash("a b c"))

    def test_normalize_collapses_whitespace(self) -> None:
        self.assertEqual(normalize("  a\n\t b  "), "a b")


if __name__ == "__main__":
    unittest.main()
