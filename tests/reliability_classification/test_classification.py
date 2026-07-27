"""Phase 2.2.1 — failure classification.

Every mandated failure family is injected artificially and classified;
confidence ordering, rule conflicts, serialization and the classify-only
integration (recorder attaches verdicts to FAILED attempts, touches nothing
else) are all verified. The final class re-proves identical orchestration
behavior with classification active.
"""

from __future__ import annotations

import json
import unittest

from dozen.llm_client import LLMClient, LLMMessage
from dozen.reliability import (
    AttemptStatus,
    ClassificationRule,
    ExecutionStage,
    FailureClassification,
    FailureClassifier,
    FailureEvidence,
    FailureType,
    InMemoryExecutionRecorder,
    ReliabilityClientDecorator,
)
from dozen.reliability.config import DetectionConfig


def evidence(message: str = "", exc_type: str = "ProviderError", **kw) -> FailureEvidence:
    return FailureEvidence(exception_type=exc_type, exception_message=message,
                           provider="anthropic", **kw)


class TestFailureTypeRules(unittest.TestCase):
    """Artificial injection of each failure family (acceptance drills)."""

    def setUp(self) -> None:
        self.classifier = FailureClassifier()

    def check(self, ev: FailureEvidence, expected: FailureType, min_conf: float = 0.5):
        verdict = self.classifier.classify(ev)
        self.assertIs(verdict.failure_type, expected,
                      f"{ev.exception_message!r} -> {verdict.explanation}")
        self.assertGreaterEqual(verdict.confidence, min_conf)
        return verdict

    def test_timeout(self) -> None:
        self.check(evidence("Timed out waiting for a fresh response."),
                   FailureType.TIMEOUT)
        self.check(evidence("", exc_type="TimeoutError"), FailureType.TIMEOUT)

    def test_dom_changed(self) -> None:
        self.check(evidence("[anthropic] Input box not found on the page."),
                   FailureType.DOM_CHANGED)
        self.check(evidence("Could not locate the chat input via selector."),
                   FailureType.DOM_CHANGED)

    def test_rate_limit(self) -> None:
        self.check(evidence("You've reached your usage limit. Try again later."),
                   FailureType.RATE_LIMIT)
        self.check(evidence("server said no", http_status=429), FailureType.RATE_LIMIT)

    def test_captcha_including_cloudflare(self) -> None:
        self.check(evidence("Checking your browser — Cloudflare"), FailureType.CAPTCHA)
        self.check(evidence("page shows: Verify you are human"), FailureType.CAPTCHA)

    def test_login_required(self) -> None:
        self.check(evidence("Session expired, please sign in again"),
                   FailureType.LOGIN_REQUIRED)
        self.check(evidence("redirected", url="https://chatgpt.com/auth/login"),
                   FailureType.LOGIN_REQUIRED)

    def test_browser_crash(self) -> None:
        self.check(evidence("Browser has been closed"), FailureType.BROWSER_CRASH)
        self.check(evidence("EPIPE: broken pipe, write"), FailureType.BROWSER_CRASH)
        self.check(evidence("anything", browser_connected=False),
                   FailureType.BROWSER_CRASH)

    def test_tab_closed(self) -> None:
        self.check(
            evidence("Target page, context or browser has been closed"),
            FailureType.TAB_CLOSED, min_conf=0.85,
        )
        self.check(evidence("fine text", tab_connected=False, browser_connected=True),
                   FailureType.TAB_CLOSED)

    def test_empty_response_is_generation_stalled(self) -> None:
        self.check(evidence("[anthropic] Scraped an empty response."),
                   FailureType.GENERATION_STALLED)

    def test_prompt_rejected_network_busy_corrupted(self) -> None:
        self.check(evidence("Copilot: message too long"), FailureType.PROMPT_REJECTED)
        self.check(evidence("net::ERR_INTERNET_DISCONNECTED"), FailureType.NETWORK_ERROR)
        self.check(evidence("The model is at capacity right now"), FailureType.MODEL_BUSY)
        self.check(evidence("worker returned invalid JSON artifact"),
                   FailureType.OUTPUT_CORRUPTED)

    def test_unknown(self) -> None:
        verdict = self.classifier.classify(evidence("zorp gleeble unforeseen"))
        self.assertIs(verdict.failure_type, FailureType.UNKNOWN)
        self.assertEqual(verdict.matched_rules, ())
        self.assertEqual(verdict.confidence, 0.3)
        self.assertIn("no classification rule matched", verdict.explanation)


class TestConfidenceAndConflicts(unittest.TestCase):
    def test_highest_confidence_wins_and_all_matches_kept(self) -> None:
        # 'sign in' (login 0.85) + captcha wording (0.95) in one message:
        verdict = FailureClassifier().classify(
            evidence("Cloudflare check: verify you are human, then sign in")
        )
        self.assertIs(verdict.failure_type, FailureType.CAPTCHA)
        names = {m.name for m in verdict.matched_rules}
        self.assertIn("captcha-wall", names)     # winner stored…
        self.assertIn("login-required", names)   # …and the loser too
        self.assertGreaterEqual(len(verdict.matched_rules), 2)
        self.assertIn("also matched", verdict.explanation)

    def test_confidence_ordering_rate_limit_beats_timeout(self) -> None:
        verdict = FailureClassifier().classify(
            evidence("Request timed out: too many requests")
        )
        # rate-limit (0.85) outranks timeout (0.7)
        self.assertIs(verdict.failure_type, FailureType.RATE_LIMIT)
        types = {m.failure_type for m in verdict.matched_rules}
        self.assertIn(FailureType.TIMEOUT, types)

    def test_below_floor_downgrades_to_unknown_keeping_matches(self) -> None:
        strict = FailureClassifier(config=DetectionConfig(min_confidence=0.99))
        verdict = strict.classify(evidence("Timed out waiting"))
        self.assertIs(verdict.failure_type, FailureType.UNKNOWN)
        self.assertGreater(len(verdict.matched_rules), 0)  # evidence preserved
        self.assertIn("below the confidence floor", verdict.explanation)

    def test_broken_rule_is_contained(self) -> None:
        def bomb(e, text):
            raise RuntimeError("bad rule")

        classifier = FailureClassifier()
        classifier.register_rule(
            ClassificationRule("saboteur", FailureType.CAPTCHA, bomb)
        )
        verdict = classifier.classify(evidence("Timed out waiting"))
        self.assertIs(verdict.failure_type, FailureType.TIMEOUT)  # unharmed

    def test_custom_rule_plugs_in(self) -> None:
        classifier = FailureClassifier()
        classifier.register_rule(ClassificationRule(
            "gemini-quota-banner", FailureType.RATE_LIMIT,
            lambda e, t: 0.97 if "you've hit your gemini limit" in t else None,
        ))
        verdict = classifier.classify(evidence("You've hit your Gemini limit"))
        self.assertIs(verdict.failure_type, FailureType.RATE_LIMIT)
        self.assertEqual(verdict.matched_rules[-1].name
                         if verdict.matched_rules[-1].confidence == 0.97
                         else verdict.matched_rules[0].name, "gemini-quota-banner")


class TestSerialization(unittest.TestCase):
    def test_evidence_round_trip(self) -> None:
        ev = evidence(
            "Timed out", url="https://claude.ai/new", page_title="Claude",
            http_status=504, browser_connected=True, tab_connected=True,
            prompt_length=1200, response_length=0,
            execution_stage=ExecutionStage.WORKER, latency_ms=95000.0,
            active_selectors=("div.input", "button.send"),
        )
        self.assertEqual(ev.validate(), [])
        again = FailureEvidence.from_dict(ev.to_dict())
        self.assertEqual(again.to_dict(), ev.to_dict())
        json.dumps(ev.to_dict())

    def test_classification_round_trip(self) -> None:
        verdict = FailureClassifier().classify(evidence("Timed out waiting"))
        again = FailureClassification.from_dict(verdict.to_dict())
        self.assertIs(again.failure_type, verdict.failure_type)
        self.assertEqual(again.confidence, verdict.confidence)
        self.assertEqual([m.to_dict() for m in again.matched_rules],
                         [m.to_dict() for m in verdict.matched_rules])
        self.assertEqual(again.raw_evidence.exception_message, "Timed out waiting")
        json.dumps(verdict.to_dict())


class TestRecorderIntegration(unittest.TestCase):
    def make(self):
        recorder = InMemoryExecutionRecorder()
        wrapped = LLMClient(mock=True)
        return ReliabilityClientDecorator(wrapped, recorder=recorder), wrapped, recorder

    def ask(self, client):
        return client.complete(provider="anthropic", model="claude",
                               messages=[LLMMessage("user", "hi")])

    def test_failed_attempt_gets_classification(self) -> None:
        decorated, wrapped, recorder = self.make()

        def explode(**kw):
            raise RuntimeError("[anthropic] Timed out waiting for a fresh response.")

        wrapped.complete = explode  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            self.ask(decorated)
        attempt = recorder.peek()
        self.assertIs(attempt.status, AttemptStatus.FAILED)
        self.assertTrue(attempt.failure_present)
        self.assertEqual(len(attempt.failures), 1)
        self.assertIs(attempt.failures[0].failure_type, FailureType.TIMEOUT)
        verdict = attempt.result_metadata["failure_classification"]
        self.assertEqual(verdict["failure_type"], "timeout")
        self.assertGreaterEqual(verdict["confidence"], 0.5)
        self.assertTrue(verdict["matched_rules"])
        # And the attempt still serializes cleanly with the verdict attached:
        json.dumps(attempt.to_dict())

    def test_each_injected_family_classifies_through_the_stack(self) -> None:
        cases = {
            "Browser has been closed": FailureType.BROWSER_CRASH,
            "Target page, context or browser has been closed": FailureType.TAB_CLOSED,
            "Cloudflare: verify you are human": FailureType.CAPTCHA,
            "Session expired — please sign in": FailureType.LOGIN_REQUIRED,
            "429 too many requests": FailureType.RATE_LIMIT,
            "Input box not found": FailureType.DOM_CHANGED,
            "Scraped an empty response.": FailureType.GENERATION_STALLED,
            "message too long": FailureType.PROMPT_REJECTED,
            "net::ERR_CONNECTION_RESET": FailureType.NETWORK_ERROR,
            "totally novel gibberish failure": FailureType.UNKNOWN,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                decorated, wrapped, recorder = self.make()

                def explode(_msg=message, **kw):
                    raise RuntimeError(_msg)

                wrapped.complete = explode  # type: ignore[method-assign]
                with self.assertRaises(RuntimeError):
                    self.ask(decorated)
                self.assertIs(recorder.peek().failures[0].failure_type, expected)

    def test_successful_attempts_remain_unchanged(self) -> None:
        decorated, _, recorder = self.make()
        self.ask(decorated)
        attempt = recorder.peek()
        self.assertIs(attempt.status, AttemptStatus.SUCCEEDED)
        self.assertEqual(attempt.failures, ())
        self.assertFalse(attempt.failure_present)
        self.assertNotIn("failure_classification", attempt.result_metadata)

    def test_cancellation_is_not_classified(self) -> None:
        from dozen.cancellation import CancelledError
        decorated, wrapped, recorder = self.make()

        def cancel(**kw):
            raise CancelledError("stop pressed")

        wrapped.complete = cancel  # type: ignore[method-assign]
        with self.assertRaises(CancelledError):
            self.ask(decorated)
        attempt = recorder.peek()
        self.assertIs(attempt.status, AttemptStatus.CANCELLED)
        self.assertEqual(attempt.failures, ())    # user intent, not a failure

    def test_classifier_crash_never_loses_the_attempt(self) -> None:
        decorated, wrapped, recorder = self.make()
        recorder._classifier = object()  # not a classifier: classify() missing

        def explode(**kw):
            raise RuntimeError("Timed out")

        wrapped.complete = explode  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            self.ask(decorated)
        attempt = recorder.peek()                 # stored despite broken classifier
        self.assertIs(attempt.status, AttemptStatus.FAILED)
        self.assertEqual(attempt.failures, ())


class TestIdenticalBehavior(unittest.TestCase):
    def test_successful_orchestration_output_unchanged(self) -> None:
        from dozen import Task
        from dozen.agent_pool import AgentPool, AgentSpec
        from dozen.config import OrchestratorConfig
        from dozen.orchestrator import Orchestrator

        def run(client) -> str:
            pool = AgentPool([
                AgentSpec(name="alpha", provider="openai", model="gpt",
                          strengths={"reasoning": 0.9, "coding": 0.8}, tier=4),
                AgentSpec(name="beta", provider="anthropic", model="claude",
                          strengths={"writing": 0.9, "reasoning": 0.8}, tier=4),
            ])
            cfg = OrchestratorConfig(max_parallelism=2, max_repair_attempts=1,
                                     verify_outputs=False, use_llm_router=False)
            result = Orchestrator(client=client, pool=pool, config=cfg).run(
                Task(prompt="Define idempotency in one paragraph.")
            )
            return result.final_answer or ""

        recorder = InMemoryExecutionRecorder()
        observed = run(ReliabilityClientDecorator(LLMClient(mock=True), recorder=recorder))
        bare = run(LLMClient(mock=True))
        self.assertEqual(observed, bare)
        self.assertTrue(all(a.status is AttemptStatus.SUCCEEDED
                            for a in recorder.attempts()))
        self.assertTrue(all(not a.failure_present for a in recorder.attempts()))


if __name__ == "__main__":
    unittest.main()
