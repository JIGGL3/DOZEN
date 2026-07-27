"""Backward-compatibility tests: send_prompt semantics, the rejection surfacing
through the existing ProviderError hierarchy, and default-policy wiring."""

from __future__ import annotations

import unittest

from dozen.llm_client import LLMMessage
from webllm.browser_queue import (
    DEFAULT_QUEUE_POLICY,
    AdmissionOutcome,
    BrowserQueueRejectedError,
    QueuePolicy,
    make_admission_snapshot,
)
from webllm.client import WebAutomationLLMClient
from webllm.providers import ProviderError

from ._util import QueueTestManager


class TestSendPromptCompat(unittest.TestCase):
    def test_send_prompt_success_unchanged(self) -> None:
        bm = QueueTestManager()
        try:
            self.assertEqual(bm.send_prompt("openai", "hi"), "reply:hi")
        finally:
            bm.shutdown()

    def test_rejection_surfaces_as_provider_error(self) -> None:
        bm = QueueTestManager(
            queue_policy=QueuePolicy(max_queued_prompts_per_provider=1)
        )
        try:
            bm.install_blocker("openai")
            bm.submit_prompt("openai", "a")  # fills the single slot
            # send_prompt surfaces the queue rejection through ProviderError,
            # exactly what existing callers (WebAutomationLLMClient) already catch.
            with self.assertRaises(ProviderError):
                bm.send_prompt("openai", "rejected")
        finally:
            bm.shutdown()

    def test_default_policy_when_unset(self) -> None:
        bm = QueueTestManager()
        try:
            self.assertEqual(
                bm._queue_policy.max_queued_prompts_per_provider,
                DEFAULT_QUEUE_POLICY.max_queued_prompts_per_provider,
            )
        finally:
            bm.shutdown()

    def test_submit_prompt_handle_semantics_unchanged(self) -> None:
        bm = QueueTestManager()
        try:
            h = bm.submit_prompt("openai", "q")
            self.assertEqual(h.result(timeout=5), "reply:q")
            self.assertEqual(h.provider, "openai")
            self.assertIsNotNone(h.job_id)
        finally:
            bm.shutdown()

    def test_web_client_preserves_typed_queue_rejection_without_retry(self) -> None:
        admission = make_admission_snapshot(
            AdmissionOutcome.REJECTED_GLOBAL_FULL,
            "openai",
            provider_queued_depth=1,
            provider_capacity=8,
            global_queued_depth=32,
            global_capacity=32,
        )
        rejection = BrowserQueueRejectedError(admission)

        class RejectingBrowser:
            def __init__(self) -> None:
                self.calls = 0

            def send_prompt(self, *args, **kwargs):
                self.calls += 1
                raise rejection

        browser = RejectingBrowser()
        client = WebAutomationLLMClient(
            browser, max_retries=3, retry_backoff_s=0
        )
        with self.assertRaises(BrowserQueueRejectedError) as ctx:
            client.complete(
                provider="openai",
                model="web",
                messages=[LLMMessage(role="user", content="content-free-test")],
            )
        self.assertIs(ctx.exception, rejection)
        self.assertEqual(browser.calls, 1)


if __name__ == "__main__":
    unittest.main()
