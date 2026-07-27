"""Backward-compatibility: the pre-5B public surface is unchanged.

``send_prompt`` and the web client keep their exact return/raise contract;
same-provider serialization and cross-provider concurrency still hold; and the
interruption machinery is invisible to callers that never cancel.
"""

from __future__ import annotations

import threading
import time
import unittest

from dozen.cancellation import CancelledError, CancelToken
from dozen.llm_client import LLMMessage
from webllm.client import WebAutomationLLMClient
from webllm.providers import ProviderError

from ._util import Gate, InterruptibleBrowserManager


class TestSendPromptCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_send_prompt_returns_string(self) -> None:
        self.assertEqual(self.bm.send_prompt("openai", "hi"), "reply:hi")
        self.assertEqual(self.bm.stop_calls, 0)

    def test_send_prompt_raises_provider_error(self) -> None:
        self.bm.handlers["boom"] = lambda p, pr, sc: (_ for _ in ()).throw(
            ProviderError("down")
        )
        with self.assertRaises(ProviderError):
            self.bm.send_prompt("openai", "boom")

    def test_send_prompt_raises_cancelled(self) -> None:
        self.bm.handlers["stop"] = lambda p, pr, sc: (_ for _ in ()).throw(
            CancelledError("stop")
        )
        with self.assertRaises(CancelledError):
            self.bm.send_prompt("openai", "stop")

    def test_same_provider_serializes(self) -> None:
        order: list[str] = []
        lock = threading.Lock()

        def make(tag: str):
            def handler(p, pr, sc):
                with lock:
                    order.append(f"start:{tag}")
                time.sleep(0.02)
                with lock:
                    order.append(f"end:{tag}")
                return tag
            return handler

        self.bm.handlers["a"] = make("a")
        self.bm.handlers["b"] = make("b")
        ha = self.bm.submit_prompt("openai", "a")
        hb = self.bm.submit_prompt("openai", "b")
        ha.result(timeout=5)
        hb.result(timeout=5)
        # No interleave: one fully finishes before the next starts.
        self.assertTrue(
            order == ["start:a", "end:a", "start:b", "end:b"]
            or order == ["start:b", "end:b", "start:a", "end:a"]
        )

    def test_cross_provider_runs_in_parallel(self) -> None:
        a = Gate(value="A")
        b = Gate(value="B")
        self.bm.gates["A"] = a
        self.bm.gates["B"] = b
        ha = self.bm.submit_prompt("openai", "A")
        hb = self.bm.submit_prompt("google", "B")
        # Both start before either is released → genuine cross-provider overlap.
        self.assertTrue(a.started.wait(2))
        self.assertTrue(b.started.wait(2))
        a.release.set()
        b.release.set()
        self.assertEqual(ha.result(timeout=5), "A")
        self.assertEqual(hb.result(timeout=5), "B")


class TestWebClientCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = InterruptibleBrowserManager()
        self.client = WebAutomationLLMClient(self.bm, max_retries=2, retry_backoff_s=0.01)

    def tearDown(self) -> None:
        self.bm.shutdown()

    def _msgs(self, text: str):
        return [LLMMessage(role="user", content=text)]

    def test_complete_returns_response(self) -> None:
        resp = self.client.complete(
            provider="openai", model="gpt", messages=self._msgs("hello there")
        )
        self.assertTrue(resp.text.startswith("reply:"))
        self.assertEqual(resp.provider, "openai")

    def test_complete_cancel_token_propagates(self) -> None:
        # A cooperative send that unwinds the instant the composite observation
        # (which wraps the client's cancel token) trips — the real provider loop.
        # The orchestrator installs a real token; do the same here.
        self.client.cancel_token = CancelToken()
        started = threading.Event()

        def send(provider, prompt, should_cancel=None):
            started.set()
            deadline = time.time() + 5
            while time.time() < deadline:
                if should_cancel is not None and should_cancel():
                    raise CancelledError("stopped")
                time.sleep(0.005)
            return "late"

        self.bm._send_prompt = send  # type: ignore[assignment]

        def canceller() -> None:
            started.wait(2)
            self.client.cancel_token.cancel()

        threading.Thread(target=canceller).start()
        with self.assertRaises(CancelledError):
            self.client.complete(
                provider="openai", model="gpt", messages=self._msgs("q")
            )


if __name__ == "__main__":
    unittest.main()
