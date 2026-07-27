"""ContextBuilder: every Phase 1.4 required case — empty conversation,
single exchange, multi-turn, formatting, ordering, token estimation,
truncation, restart persistence, isolation, missing-conversation fallback."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dozen.context.adapters.filesystem import RepositoryFactory
from dozen.context.config.models import PersistenceConfig, StorageConfig
from dozen.context.context_builder import (
    DEFAULT_TOKEN_BUDGET,
    ContextBuilder,
    ContextFormatter,
    TokenBudgetEstimator,
)
from dozen.context.manager import ConversationManager


class BuilderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dozen-ctxbuild-"))
        self.root = str(self.tmp / "conversations")
        self.manager = self.new_manager()
        self.builder = ContextBuilder(self.manager)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def new_manager(self) -> ConversationManager:
        provider = RepositoryFactory.create(
            PersistenceConfig(storage=StorageConfig(root_path=self.root, fsync_appends=False))
        ).unwrap()
        return ConversationManager(provider)

    def conversation_with(self, *turns: tuple[str, str]) -> str:
        cid = self.manager.create_conversation(title="t").unwrap().conversation.id
        for role, content in turns:
            if role == "user":
                self.manager.append_user_message(cid, content)
            else:
                self.manager.append_assistant_message(cid, content)
        return str(cid)


class TestBasicBuilds(BuilderTestCase):
    def test_empty_conversation(self) -> None:
        cid = self.conversation_with()
        built = self.builder.build_context(cid, "first prompt")
        self.assertEqual(built.formatted_history, "")
        self.assertEqual(built.message_count, 0)
        self.assertFalse(built.truncated)
        self.assertFalse(built.fallback)
        self.assertEqual(built.selected_messages, [])
        self.assertEqual(built.estimated_tokens, TokenBudgetEstimator().estimate("first prompt"))
        self.assertEqual(built.budget_remaining, DEFAULT_TOKEN_BUDGET - built.estimated_tokens)

    def test_single_exchange(self) -> None:
        cid = self.conversation_with(("user", "My name is John."), ("assistant", "Nice to meet you, John!"))
        built = self.builder.build_context(cid, "What is my name?")
        self.assertEqual(
            built.formatted_history,
            "User: My name is John.\n\nAssistant: Nice to meet you, John!",
        )
        self.assertEqual(built.message_count, 2)
        self.assertFalse(built.truncated)
        self.assertTrue(built.has_history)

    def test_multi_turn_preserves_chronological_order(self) -> None:
        turns = []
        for i in range(6):
            turns.append(("user", f"question {i}"))
            turns.append(("assistant", f"answer {i}"))
        cid = self.conversation_with(*turns)
        built = self.builder.build_context(cid, "next")
        blocks = built.formatted_history.split("\n\n")
        self.assertEqual(len(blocks), 12)
        self.assertEqual(blocks[0], "User: question 0")
        self.assertEqual(blocks[-1], "Assistant: answer 5")
        # Strictly increasing ids == chronological.
        ids = [m.id for m in built.selected_messages]
        self.assertEqual(ids, sorted(ids))

    def test_formatting_never_modifies_content(self) -> None:
        gnarly = "  line one\n\nUser: fake label inside\n\tweird spacing 🚀  "
        cid = self.conversation_with(("user", gnarly))
        built = self.builder.build_context(cid, "x")
        self.assertEqual(built.formatted_history, f"User: {gnarly}")


class TestTokenEstimation(unittest.TestCase):
    def test_char_ratio(self) -> None:
        est = TokenBudgetEstimator()
        self.assertEqual(est.estimate(""), 0)
        self.assertEqual(est.estimate("abcd"), 1)      # 4 chars -> 1 token
        self.assertEqual(est.estimate("abcde"), 2)     # ceil(5/4)
        self.assertEqual(est.estimate("x" * 4000), 1000)

    def test_custom_ratio_and_validation(self) -> None:
        self.assertEqual(TokenBudgetEstimator(chars_per_token=2.0).estimate("abcd"), 2)
        with self.assertRaises(ValueError):
            TokenBudgetEstimator(chars_per_token=0)

    def test_estimator_is_replaceable(self) -> None:
        """The seam a real tokenizer plugs into: anything with estimate()."""

        class WordCounter:
            def estimate(self, text: str) -> int:
                return len(text.split())

        from dozen.context.context_builder.budget import TokenEstimator
        self.assertIsInstance(WordCounter(), TokenEstimator)


class TestTruncation(BuilderTestCase):
    def setUp(self) -> None:
        super().setUp()
        turns = []
        for i in range(10):
            turns.append(("user", f"question number {i} padded {'x' * 50}"))
            turns.append(("assistant", f"answer number {i} padded {'y' * 50}"))
        self.cid = self.conversation_with(*turns)

    def test_over_budget_keeps_newest_drops_oldest(self) -> None:
        tight = ContextBuilder(self.manager, budget_tokens=120)
        built = tight.build_context(self.cid, "current")
        self.assertTrue(built.truncated)
        self.assertGreater(built.message_count, 0)
        self.assertLess(built.message_count, 20)
        # Newest survived, oldest gone:
        self.assertIn("answer number 9", built.formatted_history)
        self.assertNotIn("question number 0", built.formatted_history)
        # And what survived is still chronological:
        ids = [m.id for m in built.selected_messages]
        self.assertEqual(ids, sorted(ids))
        self.assertLessEqual(built.estimated_tokens, 120)

    def test_current_prompt_is_never_removed(self) -> None:
        huge_prompt = "z" * 4000  # ~1000 tokens on its own
        tiny = ContextBuilder(self.manager, budget_tokens=500)
        built = tiny.build_context(self.cid, huge_prompt)
        # Prompt alone blows the budget: history yields entirely, prompt stays.
        self.assertEqual(built.message_count, 0)
        self.assertTrue(built.truncated)
        self.assertEqual(built.formatted_history, "")
        self.assertEqual(built.estimated_tokens, 1000)
        self.assertEqual(built.budget_remaining, 0)

    def test_within_budget_is_not_truncated(self) -> None:
        roomy = ContextBuilder(self.manager, budget_tokens=100_000)
        built = roomy.build_context(self.cid, "current")
        self.assertFalse(built.truncated)
        self.assertEqual(built.message_count, 20)

    def test_per_build_budget_override(self) -> None:
        built = self.builder.build_context(self.cid, "current", budget_tokens=120)
        self.assertTrue(built.truncated)
        self.assertEqual(built.budget, 120)


class TestPersistenceAndIsolation(BuilderTestCase):
    def test_restart_persistence(self) -> None:
        cid = self.conversation_with(("user", "My name is John."), ("assistant", "Hello John."))
        rebuilt = ContextBuilder(self.new_manager())  # fresh manager == restart
        built = rebuilt.build_context(cid, "What is my name?")
        self.assertEqual(
            built.formatted_history,
            "User: My name is John.\n\nAssistant: Hello John.",
        )

    def test_conversation_isolation(self) -> None:
        cid_a = self.conversation_with(("user", "Secret of A"), ("assistant", "noted A"))
        cid_b = self.conversation_with(("user", "Secret of B"), ("assistant", "noted B"))
        built_a = self.builder.build_context(cid_a, "x")
        built_b = self.builder.build_context(cid_b, "x")
        self.assertIn("Secret of A", built_a.formatted_history)
        self.assertNotIn("Secret of B", built_a.formatted_history)
        self.assertIn("Secret of B", built_b.formatted_history)
        self.assertNotIn("Secret of A", built_b.formatted_history)


class TestFallback(BuilderTestCase):
    def assert_stateless(self, built, reason_fragment: str) -> None:
        self.assertTrue(built.fallback)
        self.assertEqual(built.formatted_history, "")
        self.assertEqual(built.message_count, 0)
        self.assertIn(reason_fragment, built.fallback_reason)

    def test_missing_id(self) -> None:
        self.assert_stateless(self.builder.build_context(None, "x"), "invalid or missing")
        self.assert_stateless(self.builder.build_context("", "x"), "invalid or missing")

    def test_malformed_id(self) -> None:
        self.assert_stateless(self.builder.build_context("../evil", "x"), "invalid or missing")

    def test_unknown_conversation(self) -> None:
        built = self.builder.build_context("01AAAAAAAAAAAAAAAAAAAAAAAA", "x")
        self.assert_stateless(built, "not_found")

    def test_crashing_manager_degrades_not_raises(self) -> None:
        class ExplodingManager:
            def read_messages(self, *a, **k):
                raise RuntimeError("disk on fire")

        exploding = ContextBuilder(ExplodingManager())  # type: ignore[arg-type]
        built = exploding.build_context("01AAAAAAAAAAAAAAAAAAAAAAAA", "x")
        self.assert_stateless(built, "disk on fire")

    def test_fallback_never_blocks_the_prompt(self) -> None:
        built = self.builder.build_context(None, "the run must still happen")
        self.assertGreater(built.estimated_tokens, 0)   # prompt still counted
        self.assertGreater(built.budget_remaining, 0)


class TestFormatterUnit(unittest.TestCase):
    def test_role_labels(self) -> None:
        from dozen.context.domain.enums import MessageRole
        fmt = ContextFormatter()
        self.assertEqual(fmt.label(MessageRole.USER), "User")
        self.assertEqual(fmt.label(MessageRole.ASSISTANT), "Assistant")
        self.assertEqual(fmt.label(MessageRole.SUMMARY), "Summary")
        self.assertEqual(fmt.label(MessageRole.UNKNOWN), "Unknown")


if __name__ == "__main__":
    unittest.main()
