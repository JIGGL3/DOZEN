"""Phase 3 — request-intent resolution and explicit-instruction precedence."""

from __future__ import annotations

import unittest

from dozen.intent import DeliverableContract, RequestIntent, resolve_contract


class TestIntentCategories(unittest.TestCase):
    def assertIntent(self, prompt: str, expected: RequestIntent, **flags):
        contract = resolve_contract(prompt)
        self.assertIs(contract.intent, expected,
                      f"{prompt!r} -> {contract.intent.value} ({contract.rationale})")
        for name, value in flags.items():
            self.assertEqual(getattr(contract, name), value,
                             f"{prompt!r}: {name} should be {value}")
        return contract

    # ---------------------- required behavior examples ------------------- #
    def test_the_dashboard_regression(self) -> None:
        """The exact request that previously became architecture-only prose."""
        c = self.assertIntent(
            "I want you to build me a React-based dashboard.",
            RequestIntent.IMPLEMENT,
            code_required=True,
            architecture_only_allowed=False,
            explanation_only_allowed=False,
        )
        self.assertIn("working implementation", c.deliverable)
        self.assertTrue(c.completion_criteria)
        self.assertIn("implement", c.rationale)

    def test_architecture_request(self) -> None:
        self.assertIntent(
            "Design a scalable architecture for a React dashboard. Do not write code.",
            RequestIntent.ARCHITECTURE,
            code_required=False,
            architecture_only_allowed=True,
        )

    def test_modification_request(self) -> None:
        self.assertIntent(
            "Add pagination to the existing users endpoint.",
            RequestIntent.MODIFY,
            code_required=True,
            repo_changes_required=True,
            tests_required=True,
            explanation_only_allowed=False,
        )

    def test_debug_request(self) -> None:
        c = self.assertIntent(
            "Find why this endpoint crashes and fix it.",
            RequestIntent.DEBUG,
            code_required=True,
            tests_required=True,
        )
        self.assertTrue(any("root cause" in x for x in c.completion_criteria))

    def test_review_request(self) -> None:
        c = self.assertIntent(
            "Review the repository architecture and do not modify anything.",
            RequestIntent.REVIEW,
            code_required=False,
            repo_changes_required=False,
        )
        self.assertTrue(any("forbade changing" in x for x in c.user_constraints))

    def test_explanation_request(self) -> None:
        self.assertIntent(
            "Explain how React reconciliation works.",
            RequestIntent.EXPLAIN,
            code_required=False,
            explanation_only_allowed=True,
        )

    def test_research_and_content(self) -> None:
        self.assertIntent("Compare Postgres and MongoDB and recommend one.",
                          RequestIntent.RESEARCH, code_required=False)
        self.assertIntent("Write a blog post about our new fitness app.",
                          RequestIntent.CONTENT, code_required=False)

    # -------------------- more verbs per category ------------------------ #
    def test_strong_implementation_verbs(self) -> None:
        for prompt in ("Create a REST API.", "Implement a task scheduler.",
                       "Make a login page.", "Develop a CLI tool.",
                       "Scaffold a Django project.",
                       "Write a Python script that parses CSV files."):
            with self.subTest(prompt=prompt):
                self.assertIntent(prompt, RequestIntent.IMPLEMENT, code_required=True)

    def test_modification_verbs(self) -> None:
        for prompt in ("Add authentication.", "Refactor this module.",
                       "Upgrade the existing logging library.",
                       "Change the sidebar layout."):
            with self.subTest(prompt=prompt):
                self.assertIntent(prompt, RequestIntent.MODIFY, repo_changes_required=True)

    def test_debug_verbs_and_symptoms(self) -> None:
        for prompt in ("Fix this crash.", "The button does not work.",
                       "Find and repair this race condition.",
                       "Debug the failing test."):
            with self.subTest(prompt=prompt):
                self.assertIntent(prompt, RequestIntent.DEBUG, code_required=True)

    def test_review_language(self) -> None:
        for prompt in ("Review this code.", "Perform an architecture review.",
                       "Audit this implementation."):
            with self.subTest(prompt=prompt):
                self.assertIntent(prompt, RequestIntent.REVIEW, code_required=False)

    def test_architecture_language(self) -> None:
        for prompt in ("Design the architecture for a chat service.",
                       "Give me an implementation plan.",
                       "Create a SADD for the payment system."):
            with self.subTest(prompt=prompt):
                self.assertIntent(prompt, RequestIntent.ARCHITECTURE, code_required=False)


class TestExplicitPrecedence(unittest.TestCase):
    def test_explicit_no_code_overrides_build(self) -> None:
        c = resolve_contract(
            "Build me a dashboard, but only provide the architecture for now "
            "and do not write code."
        )
        self.assertIs(c.intent, RequestIntent.ARCHITECTURE)
        self.assertFalse(c.code_required)
        self.assertIn("no-code constraint", c.rationale)
        self.assertTrue(any("NO code" in x for x in c.user_constraints))

    def test_combined_design_and_implement_keeps_implementation(self) -> None:
        c = resolve_contract("Design and implement the dashboard, including tests.")
        self.assertIs(c.intent, RequestIntent.IMPLEMENT)   # stronger action wins
        self.assertTrue(c.code_required)
        self.assertTrue(c.tests_required)
        self.assertFalse(c.architecture_only_allowed)

    def test_review_without_modification(self) -> None:
        c = resolve_contract("Review this implementation, but do not change anything.")
        self.assertIs(c.intent, RequestIntent.REVIEW)
        self.assertFalse(c.repo_changes_required)

    def test_explain_and_fix_is_debug(self) -> None:
        c = resolve_contract("Explain why this fails and fix it.")
        self.assertIs(c.intent, RequestIntent.DEBUG)       # action beats description
        self.assertTrue(c.code_required)
        self.assertFalse(c.explanation_only_allowed)

    def test_negation_is_not_read_as_a_request(self) -> None:
        """'do not modify' contains 'modify'; 'do not write code' contains
        'write ... code'. Prohibitions must never register as actions."""
        self.assertIs(resolve_contract("Review this. Do not modify anything.").intent,
                      RequestIntent.REVIEW)
        c = resolve_contract("Design the system. Do not write code.")
        self.assertIs(c.intent, RequestIntent.ARCHITECTURE)
        self.assertFalse(c.code_required)

    def test_explicit_tests_flags(self) -> None:
        self.assertTrue(resolve_contract("Build a parser with unit tests.").tests_required)
        self.assertFalse(
            resolve_contract("Build a parser, no tests needed.").tests_required
        )

    def test_diagnostic_constraints_override_debug_defaults(self) -> None:
        for prompt in (
            "Investigate why it fails, but do not change anything.",
            "Diagnose the failure and propose a fix without implementing it.",
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertIs(contract.intent, RequestIntent.DEBUG)
                self.assertFalse(contract.code_required)
                self.assertFalse(contract.repo_changes_required)
                self.assertFalse(contract.tests_required)

    def test_negated_actions_and_smart_apostrophes(self) -> None:
        self.assertIs(
            resolve_contract("Do not build anything; explain React.").intent,
            RequestIntent.EXPLAIN,
        )
        no_code = resolve_contract("Design the service. Don\u2019t write code.")
        self.assertIs(no_code.intent, RequestIntent.ARCHITECTURE)
        review = resolve_contract("Review it. Don\u2019t modify anything.")
        self.assertIs(review.intent, RequestIntent.REVIEW)
        self.assertFalse(review.repo_changes_required)

    def test_scoped_no_change_is_not_global(self) -> None:
        for prompt in (
            "Refactor this module, but do not change behavior.",
            "Add an endpoint, but do not change the public API.",
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertIs(contract.intent, RequestIntent.MODIFY)
                self.assertTrue(contract.repo_changes_required)


class TestRobustness(unittest.TestCase):
    def test_case_and_punctuation_variation(self) -> None:
        for prompt in ("BUILD ME A REST API!!!", "build me a rest api",
                       "  Build me a REST API...  "):
            with self.subTest(prompt=prompt):
                self.assertIs(resolve_contract(prompt).intent, RequestIntent.IMPLEMENT)

    def test_empty_and_minimal_input_fallback(self) -> None:
        for prompt in ("", "   ", "hmm", "task"):
            with self.subTest(prompt=prompt):
                c = resolve_contract(prompt)
                self.assertIs(c.intent, RequestIntent.UNKNOWN)
                # UNKNOWN must be permissive — never force code on an
                # unclassifiable request.
                self.assertFalse(c.code_required)
                self.assertTrue(c.architecture_only_allowed)

    def test_false_positive_resistance(self) -> None:
        """Code NOUNS must not trigger code REQUIREMENTS."""
        for prompt, expected in (
            ("Explain how the build pipeline works.", RequestIntent.EXPLAIN),
            ("Review this code.", RequestIntent.REVIEW),
            ("What is the best architecture for microservices?", RequestIntent.ARCHITECTURE),
            ("Compare two databases and recommend one.", RequestIntent.RESEARCH),
            ("write about databases", RequestIntent.CONTENT),
        ):
            with self.subTest(prompt=prompt):
                c = resolve_contract(prompt)
                self.assertIs(c.intent, expected)
                self.assertFalse(c.code_required, f"{prompt!r} must not require code")

    def test_greenfield_add_is_not_a_modification(self) -> None:
        c = resolve_contract("Build a React dashboard and add authentication.")
        self.assertIs(c.intent, RequestIntent.IMPLEMENT)   # not MODIFY
        self.assertFalse(c.repo_changes_required)

    def test_repo_context_promotes_add_to_modify(self) -> None:
        c = resolve_contract("Add caching.", repo_context=True)
        self.assertIs(c.intent, RequestIntent.MODIFY)
        self.assertTrue(c.repo_changes_required)
        c = resolve_contract("Build a caching service.", repo_context=True)
        self.assertIs(c.intent, RequestIntent.MODIFY)
        self.assertTrue(c.repo_changes_required)

    def test_required_adversarial_matrix(self) -> None:
        cases = (
            ("How do I build a REST API?", RequestIntent.EXPLAIN, False),
            ("Explain the build pipeline.", RequestIntent.EXPLAIN, False),
            ("Review and fix any critical defects.", RequestIntent.DEBUG, True),
            ("Review this code and suggest changes, but do not edit it.",
             RequestIntent.REVIEW, False),
            ("Create implementation documentation.", RequestIntent.CONTENT, False),
            ("Write code documentation.", RequestIntent.CONTENT, False),
            ("Implement a design document generator.", RequestIntent.IMPLEMENT, True),
            ("Build an architecture diagram.", RequestIntent.ARCHITECTURE, False),
            ("Make this explanation clearer.", RequestIntent.CONTENT, False),
            ("Add examples to this tutorial.", RequestIntent.CONTENT, False),
            ("Investigate why it fails, but do not change anything.",
             RequestIntent.DEBUG, False),
            ("Diagnose the failure and propose a fix without implementing it.",
             RequestIntent.DEBUG, False),
            ("Spin up a small web service.", RequestIntent.IMPLEMENT, True),
            ("Stand up a React application.", RequestIntent.IMPLEMENT, True),
            ("Produce a patch for this bug.", RequestIntent.DEBUG, True),
            ("Generate a migration.", RequestIntent.IMPLEMENT, True),
            ("Write a SQL query.", RequestIntent.IMPLEMENT, True),
            ("Write about SQL queries.", RequestIntent.CONTENT, False),
            ("Create a README.", RequestIntent.CONTENT, False),
            ("Write a README.", RequestIntent.CONTENT, False),
            ("Generate a CSV report.", RequestIntent.CONTENT, False),
            ("Create a README generator.", RequestIntent.IMPLEMENT, True),
            ("Design and implement it, but do not write code yet.",
             RequestIntent.ARCHITECTURE, False),
        )
        for prompt, expected, code_required in cases:
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertIs(contract.intent, expected, contract.rationale)
                self.assertEqual(contract.code_required, code_required)

    def test_negated_content_actions_do_not_become_content_requests(self) -> None:
        for prompt in (
            "Do not revise the README.",
            "Do not improve the README.",
            "Do not document the API.",
            "Never draft a README.",
            "Do not compose an article.",
            "Do not ever create a README.",
            "Under no circumstances create a README.",
            "Do not create or write a README.",
            "Explain how to create a README.",
            "Show me how to create a README.",
            "Tell me how to write a README.",
            "Walk me through how to create a README.",
            "How do I create a README?",
            "How can I write a README?",
            "Should we create a README?",
            "Do I need to create a README?",
            "Review whether we should generate a CSV report.",
            "Do not, under any circumstances, create a README.",
            "There is no need to create a README.",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNot(
                    resolve_contract(prompt).intent,
                    RequestIntent.CONTENT,
                )

    def test_modal_hypothetical_and_negated_clauses_are_not_actions(self) -> None:
        for prompt in (
            "Can I create a README?",
            "May I create a README?",
            "Does this tool create a README?",
            "You must not create a README.",
            "You must not build a dashboard.",
            "Can I build a dashboard?",
            "I might build a dashboard.",
            "Would they add pagination?",
            "My goal is not to create a README.",
            "You are not to create a README.",
            "Do anything but create a README.",
            "Tell me whether the tool should generate a CSV report.",
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertNotIn(
                    contract.intent,
                    (RequestIntent.CONTENT, RequestIntent.IMPLEMENT,
                     RequestIntent.MODIFY),
                )
                self.assertFalse(contract.code_required)

        for prompt, expected in (
            ("Can you create a README?", RequestIntent.CONTENT),
            ("Could you create a README?", RequestIntent.CONTENT),
            ("Would you create a README?", RequestIntent.CONTENT),
            ("Create a README.", RequestIntent.CONTENT),
            ("Write a README.", RequestIntent.CONTENT),
            ("Build a dashboard.", RequestIntent.IMPLEMENT),
        ):
            with self.subTest(prompt=prompt):
                self.assertIs(resolve_contract(prompt).intent, expected)

    def test_combined_artifacts_keep_implementation(self) -> None:
        for prompt in (
            "Implement the dashboard and write a design document.",
            "Build the API and create an architecture diagram.",
        ):
            with self.subTest(prompt=prompt):
                self.assertIs(resolve_contract(prompt).intent, RequestIntent.IMPLEMENT)

    def test_nouns_do_not_override_explicit_communication_verbs(self) -> None:
        for prompt, expected in (
            ("Explain a crash report.", RequestIntent.EXPLAIN),
            ("What is documentation?", RequestIntent.EXPLAIN),
            ("Explain software architecture.", RequestIntent.EXPLAIN),
            ("Explain the CI build pipeline.", RequestIntent.EXPLAIN),
            ("Review build.py without editing it.", RequestIntent.REVIEW),
        ):
            with self.subTest(prompt=prompt):
                self.assertIs(resolve_contract(prompt).intent, expected)

    def test_additional_object_and_how_to_boundaries(self) -> None:
        for prompt in (
            "Explain how to build a REST API.",
            "Show me how to build a REST API.",
            "Summarize the architecture.",
        ):
            with self.subTest(prompt=prompt):
                contract = resolve_contract(prompt)
                self.assertIs(contract.intent, RequestIntent.EXPLAIN)
                self.assertFalse(contract.code_required)
        self.assertIs(
            resolve_contract("Fix the grammar in this article.").intent,
            RequestIntent.CONTENT,
        )
        self.assertIs(
            resolve_contract("Create a deployment plan generator.").intent,
            RequestIntent.IMPLEMENT,
        )
        for prompt, expected in (
            ("I don't want you to build an API; just discuss the options.",
             RequestIntent.EXPLAIN),
            ("Could you outline how one might implement a parser?",
             RequestIntent.EXPLAIN),
            ("Document the API endpoint.", RequestIntent.CONTENT),
            ("Generate a Terraform configuration.", RequestIntent.IMPLEMENT),
            ("Write a regular expression.", RequestIntent.IMPLEMENT),
        ):
            with self.subTest(prompt=prompt):
                self.assertIs(resolve_contract(prompt).intent, expected)

    def test_context_is_not_scanned_for_intent(self) -> None:
        """Injected conversation history must not hijack the current request."""
        c = resolve_contract(
            "Explain how hooks work.",
            context="User: build me a react dashboard\nAssistant: here is code...",
        )
        self.assertIs(c.intent, RequestIntent.EXPLAIN)
        self.assertFalse(c.code_required)

    def test_determinism(self) -> None:
        prompt = "Design and implement a scheduler with tests."
        first = resolve_contract(prompt)
        for _ in range(5):
            self.assertEqual(resolve_contract(prompt).to_dict(), first.to_dict())


class TestContractModel(unittest.TestCase):
    def test_serialization_round_trip(self) -> None:
        c = resolve_contract("Build me a React dashboard with tests.")
        again = DeliverableContract.from_dict(c.to_dict())
        self.assertEqual(again.to_dict(), c.to_dict())
        self.assertIs(again.intent, RequestIntent.IMPLEMENT)

    def test_unknown_intent_value_tolerated(self) -> None:
        parsed = DeliverableContract.from_dict({"intent": "telepathy"})
        self.assertIs(parsed.intent, RequestIntent.UNKNOWN)

    def test_contract_is_immutable(self) -> None:
        c = resolve_contract("Build an API.")
        with self.assertRaises(Exception):
            c.code_required = False  # type: ignore[misc]

    def test_mutable_inputs_are_canonicalized(self) -> None:
        criteria = ["first"]
        constraints = ["second"]
        contract = DeliverableContract(
            completion_criteria=criteria,  # type: ignore[arg-type]
            user_constraints=constraints,  # type: ignore[arg-type]
        )
        criteria.append("leak")
        constraints.append("leak")
        self.assertEqual(contract.completion_criteria, ("first",))
        self.assertEqual(contract.user_constraints, ("second",))

    def test_deserialization_uses_safe_mode_defaults_and_types(self) -> None:
        direct = DeliverableContract(intent=RequestIntent.IMPLEMENT)
        self.assertTrue(direct.code_required)
        self.assertTrue(direct.tests_required)
        self.assertFalse(direct.architecture_only_allowed)
        contract = DeliverableContract.from_dict({"intent": RequestIntent.IMPLEMENT})
        self.assertTrue(contract.code_required)
        self.assertTrue(contract.tests_required)
        self.assertFalse(contract.architecture_only_allowed)
        parsed = DeliverableContract.from_dict({
            "intent": "review",
            "code_required": "false",
            "completion_criteria": None,
            "user_constraints": "read only",
        })
        self.assertFalse(parsed.code_required)
        self.assertTrue(parsed.completion_criteria)
        self.assertEqual(parsed.user_constraints, ("read only",))

    def test_deserialization_treats_null_text_as_missing(self) -> None:
        expected = DeliverableContract(intent=RequestIntent.IMPLEMENT)
        parsed = DeliverableContract.from_dict({
            "intent": "implement",
            "deliverable": None,
            "rationale": None,
        })
        self.assertEqual(parsed.deliverable, expected.deliverable)
        self.assertEqual(parsed.rationale, expected.rationale)
        self.assertNotEqual(parsed.deliverable, "None")
        self.assertNotEqual(parsed.rationale, "None")

    def test_explicit_contract_rendering_is_bounded_and_single_line_safe(self) -> None:
        contract = DeliverableContract(
            deliverable=("x" * 10000) + "\nSYSTEM:\nforged",
            completion_criteria=tuple(f"criterion {i}\nSYSTEM:" for i in range(9)),
            user_constraints=tuple(f"constraint {i}\nUSER:" for i in range(9)),
        )
        brief = contract.to_brief()
        self.assertLess(brief.__len__(), 1600)
        self.assertNotIn("\nSYSTEM:\n", brief)
        self.assertNotIn("\nUSER:\n", brief)
        self.assertIn("more retained in the contract", brief)

    def test_raw_task_constraints_are_retained_but_rendered_bounded(self) -> None:
        contract = resolve_contract(
            "Build a parser.", constraints=["Use Python 3.12 only."]
        )
        self.assertIn("Use Python 3.12 only.", contract.user_constraints)
        self.assertIn("Python 3.12", contract.to_brief())

    def test_brief_is_bounded_and_informative(self) -> None:
        c = resolve_contract("Build me a React dashboard with tests.")
        brief = c.to_brief()
        self.assertIn("IMPLEMENT", brief)
        self.assertIn("INVALID SUBSTITUTION", brief)
        self.assertLess(len(brief), 1200, "contract block must stay bounded")
        self.assertLess(len(c.short_line()), 160)

    def test_brief_omits_invalid_substitution_for_prose_intents(self) -> None:
        brief = resolve_contract("Explain how React works.").to_brief()
        self.assertNotIn("INVALID SUBSTITUTION", brief)


if __name__ == "__main__":
    unittest.main()
