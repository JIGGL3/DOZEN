"""Backward-compatibility: ``send_prompt`` semantics, the web client, and the
preserved concurrency model (serialize within a provider; parallel across).
"""

from __future__ import annotations

import time
import unittest

from dozen.cancellation import CancelledError
from dozen.llm_client import LLMMessage, LLMResponse
from webllm.client import WebAutomationLLMClient
from webllm.providers import ProviderError

from ._util import ControllableBrowserManager, Gate


class TestSendPromptCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_send_prompt_returns_text(self) -> None:
        self.assertEqual(self.bm.send_prompt("openai", "hi"), "reply:hi")

    def test_send_prompt_raises_provider_error(self) -> None:
        self.bm.handlers["x"] = lambda p, pr, sc: (_ for _ in ()).throw(
            ProviderError("nope")
        )
        with self.assertRaisesRegex(ProviderError, "^nope$"):
            self.bm.send_prompt("openai", "x")

    def test_send_prompt_raises_cancelled(self) -> None:
        self.bm.handlers["c"] = lambda p, pr, sc: (_ for _ in ()).throw(
            CancelledError("stop")
        )
        with self.assertRaisesRegex(CancelledError, "^stop$"):
            self.bm.send_prompt("openai", "c")

    def test_send_prompt_times_out(self) -> None:
        gate = Gate(value="late")
        self.bm.gates["slow"] = gate
        with self.assertRaises(TimeoutError):
            # small explicit timeout keeps the test fast
            self.bm.submit_prompt("openai", "slow", timeout=0.2).result()
        gate.release.set()


class TestWebClientCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()
        self.client = WebAutomationLLMClient(self.bm)

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_complete_returns_llm_response(self) -> None:
        resp = self.client.complete(
            provider="openai",
            model="gpt (web)",
            messages=[LLMMessage("user", "Say hi")],
        )
        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.provider, "openai")
        self.assertTrue(resp.text.startswith("reply:"))


class TestConcurrencyModel(unittest.TestCase):
    def setUp(self) -> None:
        self.bm = ControllableBrowserManager()

    def tearDown(self) -> None:
        self.bm.shutdown()

    def test_same_provider_serializes(self) -> None:
        g1 = Gate()  # returns reply:<prompt>
        g2 = Gate()
        self.bm.gates["first"] = g1
        self.bm.gates["second"] = g2
        h1 = self.bm.submit_prompt("openai", "first")
        h2 = self.bm.submit_prompt("openai", "second")
        self.assertTrue(g1.started.wait(2))
        # Second must NOT start while the first still holds the single worker.
        self.assertFalse(g2.started.wait(0.3))
        g1.release.set()
        self.assertTrue(g2.started.wait(2))       # now it runs
        g2.release.set()
        self.assertEqual(h1.result(timeout=5), "reply:first")
        self.assertEqual(h2.result(timeout=5), "reply:second")

    def test_different_providers_run_in_parallel(self) -> None:
        ga = Gate()  # returns reply:<prompt>
        gb = Gate()
        self.bm.gates["pa"] = ga
        self.bm.gates["pb"] = gb
        ha = self.bm.submit_prompt("openai", "pa")
        hb = self.bm.submit_prompt("google", "pb")
        # Both start concurrently on their own worker threads.
        self.assertTrue(ga.started.wait(2))
        self.assertTrue(gb.started.wait(2))
        ga.release.set()
        gb.release.set()
        self.assertEqual(ha.result(timeout=5), "reply:pa")
        self.assertEqual(hb.result(timeout=5), "reply:pb")


if __name__ == "__main__":
    unittest.main()
