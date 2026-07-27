"""Phase 4C — the ONE authoritative produced-artifact parser."""

from __future__ import annotations

import unittest

from dozen.artifact_results import (
    CollectionErrorCode,
    parse_artifact_candidates,
)
from dozen.validation import parse_worker_artifact, render_artifact

from .harness import (
    CONTENT,
    entry,
    envelope,
    legacy_envelope,
    scope_for,
    worker_envelope_for,
)


class TestParser(unittest.TestCase):
    def parse(self, subtask_id: str, payload):
        return parse_artifact_candidates(payload, scope_for(subtask_id))

    def test_valid_worker_envelope(self) -> None:
        accepted, rejected, engaged = self.parse(
            "shared-ui", worker_envelope_for("shared-ui")
        )
        self.assertTrue(engaged)
        self.assertEqual(rejected, ())
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].path, "src/components/ui/Card.tsx")
        self.assertEqual(accepted[0].content, CONTENT["src/components/ui/Card.tsx"])

    def test_multiple_valid_artifacts_keep_submission_order(self) -> None:
        accepted, rejected, _ = self.parse("shell", worker_envelope_for("shell"))
        self.assertEqual(rejected, ())
        self.assertEqual(
            [c.artifact_id for c in accepted],
            ["src/app/App.tsx", "src/app/router.tsx",
             "src/components/layout/DashboardLayout.tsx"],
        )

    def test_parsing_is_deterministic(self) -> None:
        first = self.parse("shell", worker_envelope_for("shell"))
        second = self.parse("shell", worker_envelope_for("shell"))
        self.assertEqual(first, second)

    def test_duplicate_candidate_in_one_response(self) -> None:
        accepted, rejected, _ = self.parse(
            "shell",
            envelope(
                entry("src/app/App.tsx"),
                entry("src/app/App.tsx", content="a different App"),
                entry("src/app/router.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            ),
        )
        self.assertEqual(len(accepted), 3)
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.DUPLICATE_IN_RESPONSE]
        )
        # The FIRST submission is the one kept; content is never merged.
        kept = next(c for c in accepted if c.artifact_id == "src/app/App.tsx")
        self.assertEqual(kept.content, CONTENT["src/app/App.tsx"])

    def test_missing_artifact_id(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", envelope({"path": "src/app/App.tsx", "content": "x"})
        )
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.MISSING_ARTIFACT_ID]
        )

    def test_missing_path(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", envelope({"artifact_id": "src/app/App.tsx", "content": "x"})
        )
        self.assertEqual([r.code for r in rejected], [CollectionErrorCode.MISSING_PATH])

    def test_missing_content(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell", envelope({"artifact_id": "src/app/App.tsx",
                               "path": "src/app/App.tsx"})
        )
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.MISSING_CONTENT]
        )

    def test_mixed_valid_and_invalid_candidates(self) -> None:
        accepted, rejected, _ = self.parse(
            "shell",
            envelope(
                entry("src/app/App.tsx"),
                entry("package.json"),                       # another package's
                entry("src/app/router.tsx"),
                entry("src/nope.tsx", path="src/nope.tsx"),  # not in the manifest
                entry("src/components/layout/DashboardLayout.tsx"),
            ),
        )
        self.assertEqual(
            [c.artifact_id for c in accepted],
            ["src/app/App.tsx", "src/app/router.tsx",
             "src/components/layout/DashboardLayout.tsx"],
        )
        self.assertEqual(
            [r.code for r in rejected],
            [CollectionErrorCode.OUT_OF_SCOPE, CollectionErrorCode.UNKNOWN_ARTIFACT_ID],
        )

    def test_nested_malformed_structures(self) -> None:
        accepted, rejected, _ = self.parse(
            "shell",
            envelope(
                "just a string",
                ["a", "list"],
                None,
                entry("src/app/App.tsx"),
            ),
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            [r.code for r in rejected],
            [CollectionErrorCode.MALFORMED_ENTRY] * 3,
        )
        self.assertEqual([r.index for r in rejected], [0, 1, 2])

    def test_no_base64_or_binary_payload_transport(self) -> None:
        _ok, rejected, _ = self.parse(
            "shell",
            envelope(entry("src/app/App.tsx",
                           content={"encoding": "base64", "data": "QUJD"})),
        )
        self.assertEqual(
            [r.code for r in rejected], [CollectionErrorCode.INVALID_CONTENT_TYPE]
        )

    def test_empty_artifact_list_is_parsed_but_produces_nothing(self) -> None:
        accepted, rejected, engaged = self.parse("shell", envelope())
        self.assertTrue(engaged)
        self.assertEqual(accepted, ())
        self.assertEqual(rejected, ())

    def test_rejections_are_never_silent(self) -> None:
        _ok, rejected, _ = self.parse("dashboard", envelope(entry("package.json")))
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0].message)
        self.assertEqual(rejected[0].artifact_id, "package.json")
        self.assertEqual(rejected[0].subtask_id, "dashboard")


class TestLegacyEnvelopeCompatibility(unittest.TestCase):
    def test_legacy_mapping_envelope_is_not_engaged(self) -> None:
        accepted, rejected, engaged = parse_artifact_candidates(
            legacy_envelope(**{"App.tsx": "export default function App() {}"}),
            scope_for("shell"),
        )
        self.assertFalse(engaged)
        self.assertEqual(accepted, ())
        self.assertEqual(rejected, ())

    def test_absent_envelope_is_not_engaged(self) -> None:
        for payload in ({}, None, {"summary": "no artifacts key"}):
            _a, _r, engaged = parse_artifact_candidates(payload, scope_for("shell"))
            self.assertFalse(engaged)

    def test_legacy_envelope_still_renders_for_synthesis(self) -> None:
        rendered = render_artifact(
            legacy_envelope(**{"app.py": "print('hi')"})
        )
        self.assertEqual(rendered, "print('hi')")

    def test_typed_envelope_renders_by_path(self) -> None:
        rendered = render_artifact(worker_envelope_for("shell"))
        self.assertIn("### src/app/App.tsx", rendered)
        self.assertIn("### src/app/router.tsx", rendered)
        self.assertIn(CONTENT["src/app/App.tsx"], rendered)

    def test_typed_envelope_survives_json_extraction(self) -> None:
        import json

        text = "```json\n" + json.dumps(worker_envelope_for("shared-ui")) + "\n```"
        payload = parse_worker_artifact(text)
        self.assertIsNotNone(payload)
        accepted, rejected, engaged = parse_artifact_candidates(
            payload, scope_for("shared-ui")
        )
        self.assertTrue(engaged)
        self.assertEqual(rejected, ())
        self.assertEqual(accepted[0].content, CONTENT["src/components/ui/Card.tsx"])


if __name__ == "__main__":
    unittest.main()
