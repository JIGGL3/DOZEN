"""``complete_json`` keeps re-asking until a model actually returns JSON.

A chat UI that answers a structured request conversationally ("It looks like you
uploaded a text file, but there wasn't a question...") has SUCCEEDED at the
transport level, so ``complete``'s retries never fire. Only a re-ask fixes it,
and the re-ask budget is deliberately separate from ``max_retries``.
"""

from __future__ import annotations

import json
import unittest

from dozen.llm_client import (
    JSON_ONLY_DEMAND,
    LLMClient,
    LLMError,
    LLMMessage,
    LLMResponse,
)

# The real reply that motivated this, from a consumer chat UI.
CHATTY = (
    "It looks like you uploaded a text file, but there wasn't a question or "
    "instruction along with it. What would you like me to do with it?"
)

GOOD = {"analysis": "ok", "direct_answer": "hi", "delegations": []}


class ScriptedClient(LLMClient):
    """Returns a scripted sequence of replies, recording each prompt."""

    def __init__(self, replies: list[str], **kw) -> None:
        super().__init__(mock=True, retry_backoff_s=0, **kw)
        self.replies = list(replies)
        self.prompts: list[list[LLMMessage]] = []

    def complete(self, *, provider, model, messages, **kwargs) -> LLMResponse:
        self.prompts.append(list(messages))
        text = self.replies.pop(0) if self.replies else "still not json"
        return LLMResponse(text=text, provider=provider, model=model)

    def ask(self) -> dict:
        return self.complete_json(
            provider="openai",
            model="gpt (web)",
            messages=[
                LLMMessage("system", "You are the MANAGER. Return one JSON object."),
                LLMMessage("user", "Decompose this task: build a dashboard."),
            ],
        )


class TestReAskUntilJson(unittest.TestCase):
    def test_chatty_reply_is_re_asked_and_then_succeeds(self) -> None:
        client = ScriptedClient([CHATTY, json.dumps(GOOD)])
        self.assertEqual(client.ask(), GOOD)
        self.assertEqual(len(client.prompts), 2)

    def test_re_ask_names_the_problem_and_demands_json_only(self) -> None:
        client = ScriptedClient([CHATTY, json.dumps(GOOD)])
        client.ask()
        second = "\n".join(m.content for m in client.prompts[1])
        self.assertIn("could not be read as JSON", second)
        self.assertIn(JSON_ONLY_DEMAND, second)
        # The original request survives the correction.
        self.assertIn("build a dashboard", second)

    def test_keeps_going_well_past_the_transport_retry_budget(self) -> None:
        """The whole point: max_retries=1 must NOT cap the re-asks at one."""
        client = ScriptedClient(
            [CHATTY, CHATTY, CHATTY, CHATTY, json.dumps(GOOD)],
            max_retries=1,
            json_format_retries=5,
        )
        self.assertEqual(client.ask(), GOOD)
        self.assertEqual(len(client.prompts), 5)

    def test_gives_up_after_the_configured_budget(self) -> None:
        client = ScriptedClient([CHATTY] * 10, json_format_retries=3)
        with self.assertRaises(LLMError):
            client.ask()
        self.assertEqual(len(client.prompts), 3)

    def test_succeeds_first_try_without_any_re_ask(self) -> None:
        client = ScriptedClient([json.dumps(GOOD)])
        self.assertEqual(client.ask(), GOOD)
        self.assertEqual(len(client.prompts), 1)

    def test_fenced_json_is_accepted_without_a_re_ask(self) -> None:
        client = ScriptedClient(["```json\n" + json.dumps(GOOD) + "\n```"])
        self.assertEqual(client.ask(), GOOD)
        self.assertEqual(len(client.prompts), 1)

    def test_budget_is_at_least_one_attempt(self) -> None:
        client = ScriptedClient([json.dumps(GOOD)], json_format_retries=0)
        self.assertEqual(client.ask(), GOOD)


class TestEscalation(unittest.TestCase):
    """Corrections escalate; the last attempts re-ask from a clean slate."""

    def test_only_the_first_correction_echoes_the_bad_reply(self) -> None:
        client = ScriptedClient([CHATTY] * 5, json_format_retries=5)
        with self.assertRaises(LLMError):
            client.ask()
        joined = ["\n".join(m.content for m in p) for p in client.prompts]
        # Echo appears in the first correction only — quoting prose back at a
        # model that just wrote prose tends to anchor more of it.
        self.assertIn(CHATTY, joined[1])
        self.assertNotIn(CHATTY, joined[2])
        self.assertNotIn(CHATTY, joined[3])

    def test_final_attempts_strip_back_to_instructions_plus_the_ask(self) -> None:
        client = ScriptedClient([CHATTY] * 5, json_format_retries=5)
        with self.assertRaises(LLMError):
            client.ask()
        final = client.prompts[-1]
        roles = [m.role for m in final]
        self.assertEqual(roles, ["system", "user"])
        text = "\n".join(m.content for m in final)
        self.assertIn("build a dashboard", text)      # the real ask survives
        self.assertIn(JSON_ONLY_DEMAND, text)
        self.assertNotIn(CHATTY, text)                # no accumulated noise

    def test_every_attempt_keeps_the_system_instructions(self) -> None:
        client = ScriptedClient([CHATTY] * 5, json_format_retries=5)
        with self.assertRaises(LLMError):
            client.ask()
        for prompt in client.prompts:
            joined = "\n".join(m.content for m in prompt)
            self.assertIn("You are the MANAGER", joined)


class TestTransportErrorsAreNotReAsked(unittest.TestCase):
    def test_transport_failure_propagates_without_burning_re_asks(self) -> None:
        """``complete`` already retried; a re-ask would just multiply the wait."""

        class Failing(LLMClient):
            def __init__(self) -> None:
                super().__init__(mock=True, retry_backoff_s=0)
                self.calls = 0

            def complete(self, **kwargs):
                self.calls += 1
                raise LLMError("Web automation call failed after 3 attempts")

        client = Failing()
        with self.assertRaises(LLMError):
            client.complete_json(
                provider="openai", model="gpt (web)",
                messages=[LLMMessage("user", "hi")],
            )
        self.assertEqual(client.calls, 1)


class TestRetryReporting(unittest.TestCase):
    def test_observer_is_told_about_each_re_ask_without_the_reply_body(self) -> None:
        client = ScriptedClient([CHATTY, CHATTY, json.dumps(GOOD)])
        seen: list[tuple[int, int, str]] = []
        client.on_json_format_retry = lambda a, t, s: seen.append((a, t, s))
        client.ask()
        self.assertEqual([(a, t) for a, t, _ in seen], [(1, 5), (2, 5)])
        for _, _, shape in seen:
            self.assertNotIn("uploaded a text file", shape)  # content-free

    def test_a_broken_observer_never_breaks_the_run(self) -> None:
        client = ScriptedClient([CHATTY, json.dumps(GOOD)])

        def boom(*_a):
            raise RuntimeError("observer exploded")

        client.on_json_format_retry = boom
        self.assertEqual(client.ask(), GOOD)


if __name__ == "__main__":
    unittest.main()
