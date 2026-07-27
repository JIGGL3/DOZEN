"""Phase 4F Part H — refusal / non-delivery classification."""

from __future__ import annotations

import json
import unittest

from dozen.finalization import (
    FATAL_DELIVERY_CODES,
    WorkerDeliveryCode,
    classify_scoped_delivery,
)
from dozen.models import TaskStatus

from ..artifact_decomposition.harness import react_work_plan
from ..artifact_results.harness import CONTENT, collect_for, entry, scope_for
from .harness import SIZE_REFUSAL_TEXT


def scope() -> object:
    return scope_for("shell")


class TestPrimaryEvidenceIsScope(unittest.TestCase):
    def test_full_delivery_is_never_a_refusal_even_if_text_says_so(self) -> None:
        # False positive guard: valid source code containing a refusal-like
        # phrase (e.g. a docstring) must not be classified as non-delivery.
        payload = {
            "summary": "Done", "key_decisions": [],
            "artifacts": [
                entry(a, content=CONTENT[a] + "// too large? never mind, done.\n")
                if a == "src/app/App.tsx" else entry(a)
                for a in ("src/app/App.tsx", "src/app/router.tsx",
                          "src/components/layout/DashboardLayout.tsx")
            ],
            "confidence": 0.9,
        }
        collection = collect_for("shell", payload)
        classification = classify_scoped_delivery(
            scope(), collection, json.dumps(payload)
        )
        self.assertEqual(classification.code, WorkerDeliveryCode.DELIVERED)

    def test_partial_delivery_is_missing_required_not_refusal(self) -> None:
        payload = {
            "summary": "Partial", "key_decisions": [],
            "artifacts": [entry("src/app/App.tsx")],
            "confidence": 0.9,
        }
        collection = collect_for("shell", payload)
        classification = classify_scoped_delivery(
            scope(), collection, json.dumps(payload)
        )
        self.assertEqual(
            classification.code, WorkerDeliveryCode.MISSING_REQUIRED_ARTIFACTS
        )
        self.assertNotIn(classification.code, FATAL_DELIVERY_CODES)


class TestZeroDeliveryClassification(unittest.TestCase):
    def test_explicit_size_refusal(self) -> None:
        collection = collect_for("shell", {"summary": SIZE_REFUSAL_TEXT,
                                           "key_decisions": [], "artifacts": [],
                                           "confidence": 0.1})
        classification = classify_scoped_delivery(
            scope(), collection, SIZE_REFUSAL_TEXT
        )
        self.assertEqual(classification.code, WorkerDeliveryCode.RESPONSE_SIZE_REFUSAL)
        self.assertIn(classification.code, FATAL_DELIVERY_CODES)

    def test_indirect_size_refusal_phrasing(self) -> None:
        text = "This project is far too large to return in a single reply."
        collection = collect_for("shell", {"summary": text, "key_decisions": [],
                                           "artifacts": [], "confidence": 0.1})
        classification = classify_scoped_delivery(scope(), collection, text)
        self.assertEqual(classification.code, WorkerDeliveryCode.RESPONSE_SIZE_REFUSAL)

    def test_split_message_offer(self) -> None:
        text = "I can split this into multiple messages if that works for you."
        collection = collect_for("shell", {"summary": text, "key_decisions": [],
                                           "artifacts": [], "confidence": 0.1})
        classification = classify_scoped_delivery(scope(), collection, text)
        self.assertEqual(classification.code, WorkerDeliveryCode.RESPONSE_SIZE_REFUSAL)

    def test_missing_source_refusal(self) -> None:
        text = "The source material is missing, so I cannot produce the requested project."
        collection = collect_for("shell", {"summary": text, "key_decisions": [],
                                           "artifacts": [], "confidence": 0.1})
        classification = classify_scoped_delivery(scope(), collection, text)
        self.assertEqual(classification.code, WorkerDeliveryCode.ARTIFACT_NON_DELIVERY)
        self.assertIn(classification.code, FATAL_DELIVERY_CODES)

    def test_empty_artifact_envelope_with_no_refusal_text(self) -> None:
        payload = {"summary": "Done.", "key_decisions": [], "artifacts": [],
                  "confidence": 0.9}
        collection = collect_for("shell", payload)
        classification = classify_scoped_delivery(
            scope(), collection, json.dumps(payload)
        )
        self.assertEqual(classification.code, WorkerDeliveryCode.ARTIFACT_NON_DELIVERY)

    def test_explanation_only_response(self) -> None:
        text = "The application shell should use React Router with a layout wrapper."
        collection = collect_for("shell", text)  # not a typed envelope at all
        classification = classify_scoped_delivery(scope(), collection, text)
        self.assertEqual(classification.code,
                         WorkerDeliveryCode.EXPLANATION_ONLY_RESPONSE)
        self.assertIn(classification.code, FATAL_DELIVERY_CODES)

    def test_legacy_concrete_code_is_repairable_not_explanation_only(self) -> None:
        text = "```tsx\nexport default function App() { return <main />; }\n```"
        collection = collect_for("shell", text)
        classification = classify_scoped_delivery(scope(), collection, text)
        self.assertEqual(
            classification.code, WorkerDeliveryCode.MISSING_REQUIRED_ARTIFACTS
        )
        self.assertNotIn(classification.code, FATAL_DELIVERY_CODES)

    def test_policy_refusal_keeps_a_typed_delivery_code_through_retries(self) -> None:
        from .harness import FinalizationClient, run_failed_prompt

        refusal = "I cannot assist with this because it is against my policy."
        result = run_failed_prompt(FinalizationClient(
            worker_replies={"s2": [refusal, refusal]}
        ))
        shell = next(item for item in result.subtask_results
                     if item.delivery_code)
        self.assertEqual(shell.status, TaskStatus.FAILED)
        self.assertEqual(
            shell.delivery_code, WorkerDeliveryCode.ARTIFACT_NON_DELIVERY.value
        )
        self.assertNotIn(refusal, result.final_answer)

    def test_false_positive_valid_code_with_refusal_phrase_survives(self) -> None:
        # A worker returning ZERO owned artifacts but whose prose happens to
        # discuss size is still correctly classified — the point is that a
        # worker who DID deliver is never caught by phrase matching (covered
        # above); here we confirm the phrase-based refinement only applies to
        # the zero-delivery case, never demotes a satisfied delivery.
        payload = {
            "summary": "This is not too large; done in full.",
            "key_decisions": [],
            "artifacts": [
                entry(a) for a in ("src/app/App.tsx", "src/app/router.tsx",
                                   "src/components/layout/DashboardLayout.tsx")
            ],
            "confidence": 0.9,
        }
        collection = collect_for("shell", payload)
        classification = classify_scoped_delivery(
            scope(), collection, json.dumps(payload)
        )
        self.assertEqual(classification.code, WorkerDeliveryCode.DELIVERED)


class TestExistingRepairIntegration(unittest.TestCase):
    def test_missing_required_reuses_existing_repair_feedback_channel(self) -> None:
        # MISSING_REQUIRED_ARTIFACTS is explicitly non-fatal here: the existing
        # targeted-repair loop (Phase 4E) is the correction path, not a NEW
        # retry loop this phase introduces.
        payload = {"summary": "Partial", "key_decisions": [],
                  "artifacts": [entry("src/app/App.tsx")], "confidence": 0.9}
        collection = collect_for("shell", payload)
        self.assertFalse(collection.satisfied)
        classification = classify_scoped_delivery(
            scope(), collection, json.dumps(payload)
        )
        self.assertNotIn(classification.code, FATAL_DELIVERY_CODES)
        # The unmet obligation still carries an actionable feedback message
        # for the existing repair loop (never a new one).
        from dozen.artifact_results import DEFAULT_RESULT_POLICY
        self.assertTrue(collection.feedback(DEFAULT_RESULT_POLICY).strip())


if __name__ == "__main__":
    unittest.main()
