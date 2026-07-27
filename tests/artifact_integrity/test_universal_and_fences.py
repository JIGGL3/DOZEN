"""Phase 4D Parts C/D — universal truncation indicators and fenced content."""

from __future__ import annotations

import unittest

from dozen.artifact_integrity import IntegrityIssueCode, IntegrityStatus
from dozen.artifacts import ArtifactKind

from .harness import assert_invalid, assert_suspicious, assert_valid, check

TSX = "src/components/ui/Card.tsx"
DOC = "docs/design.md"


class TestUniversalIndicators(unittest.TestCase):
    def test_explicit_truncation_phrase_is_fatal(self) -> None:
        assert_invalid(
            self,
            "export const a = 1;\n// [response truncated]\n",
            TSX, IntegrityIssueCode.TRUNCATION_MARKER,
        )

    def test_output_cut_off_marker_is_fatal(self) -> None:
        assert_invalid(
            self, "vite build\ndone\n... output cut off", "build-result",
            IntegrityIssueCode.TRUNCATION_MARKER, kind=ArtifactKind.BUILD_RESULT,
        )

    def test_placeholder_ending_is_fatal(self) -> None:
        assert_invalid(
            self,
            "export function Card() {\n  return null;\n}\n// ... rest of the component\n",
            TSX, IntegrityIssueCode.TRUNCATION_MARKER,
        )

    def test_bare_ellipsis_ending_is_fatal(self) -> None:
        assert_invalid(
            self, "export const a = 1;\n...\n", TSX,
            IntegrityIssueCode.PLACEHOLDER_ENDING,
        )

    def test_todo_continue_ending_is_fatal(self) -> None:
        assert_invalid(
            self, "export const a = 1;\n// TODO: continue\n", TSX,
            IntegrityIssueCode.TRUNCATION_MARKER,
        )

    def test_an_ordinary_ellipsis_inside_a_string_is_accepted(self) -> None:
        assert_valid(
            self,
            'const msg = "Loading...";\nexport default msg;\n',
            "src/msg.ts",
        )

    def test_a_legitimate_final_comment_ending_in_ellipsis_is_accepted(self) -> None:
        assert_valid(
            self,
            "export const retries = 3;\n// Retry up to three times...\n",
            "src/retry.ts",
        )

    def test_truncation_words_inside_a_fixture_string_are_accepted(self) -> None:
        assert_valid(
            self,
            'export const fixture = "response truncated";\n',
            "tests/fixture.ts",
        )

    def test_an_ellipsis_in_ordinary_prose_is_accepted(self) -> None:
        assert_valid(
            self,
            "# Design\n\nThe cache warms slowly... then it holds steady.\n",
            DOC, kind=ArtifactKind.DOCUMENT,
        )

    def test_a_spread_operator_ending_is_not_a_placeholder(self) -> None:
        assert_valid(
            self,
            "export const merged = { ...base, id: 1 };\n",
            "src/merge.ts",
        )

    def test_unicode_content_is_accepted(self) -> None:
        assert_valid(
            self,
            'export const greeting = "こんにちは — naïve café 🚀";\n',
            "src/i18n.ts",
        )

    def test_a_trailing_replacement_character_is_suspicious(self) -> None:
        result = check('export const flag = "ok";\n�', "src/flag.ts")
        self.assertIs(result.status, IntegrityStatus.SUSPICIOUS)
        self.assertIn(
            IntegrityIssueCode.REPLACEMENT_CHARACTER,
            [issue.code for issue in result.issues],
        )

    def test_a_dangling_continuation_character_is_fatal(self) -> None:
        assert_invalid(
            self, "set -e\necho hello \\", "scripts/run.sh",
            IntegrityIssueCode.CONTINUATION_ENDING,
        )

    def test_an_unfinished_escape_sequence_is_fatal(self) -> None:
        result = assert_invalid(
            self, "body {\n  content: 'x';\n}\n/* \\u00", "src/app.css",
        )
        self.assertTrue(result.truncation_suspected)

    def test_refusal_text_instead_of_content_is_fatal(self) -> None:
        assert_invalid(
            self,
            "I'm sorry, but I can't create that file for you.",
            TSX, IntegrityIssueCode.REFUSAL_CONTENT,
        )

    def test_a_document_ending_with_a_placeholder_is_flagged_not_condemned(self) -> None:
        # Prose is allowed to trail off; code is not.
        assert_suspicious(
            self, "# Notes\n\nSome design thoughts.\n\n...\n", DOC,
            IntegrityIssueCode.PLACEHOLDER_ENDING, kind=ArtifactKind.DOCUMENT,
        )


class TestFences(unittest.TestCase):
    def test_a_raw_unfenced_file_is_the_normal_case(self) -> None:
        result = check("export const a = 1;\n", "src/a.ts")
        self.assertIs(result.status, IntegrityStatus.VALID)
        self.assertEqual(result.advisories, ())

    def test_a_fully_fenced_raw_file_is_rejected(self) -> None:
        assert_invalid(
            self, "```tsx\nexport const a = 1;\n```\n", "src/a.tsx",
            IntegrityIssueCode.FENCED_CONTENT,
        )

    def test_a_missing_closing_fence_is_fatal(self) -> None:
        assert_invalid(
            self, "```tsx\nexport const a = 1;\n", "src/a.tsx",
            IntegrityIssueCode.UNCLOSED_FENCE,
        )

    def test_a_mismatched_fence_length_never_closes_the_block(self) -> None:
        assert_invalid(
            self, "````tsx\nexport const a = 1;\n```\n", "src/a.tsx",
            IntegrityIssueCode.UNCLOSED_FENCE,
        )

    def test_a_mismatched_fence_character_never_closes_the_block(self) -> None:
        assert_invalid(
            self, "```tsx\nexport const a = 1;\n~~~\n", "src/a.tsx",
            IntegrityIssueCode.UNCLOSED_FENCE,
        )

    def test_an_empty_fenced_body_is_fatal(self) -> None:
        assert_invalid(
            self, "```tsx\n```\n", "src/a.tsx", IntegrityIssueCode.EMPTY_FENCE,
        )

    def test_multiple_fenced_blocks_for_one_raw_file_are_invalid(self) -> None:
        assert_invalid(
            self,
            "```tsx\nexport const a = 1;\n```\n\n```tsx\nexport const b = 2;\n```\n",
            "src/a.tsx", IntegrityIssueCode.MULTIPLE_FENCES,
        )

    def test_prose_around_a_closed_raw_file_fence_is_invalid(self) -> None:
        assert_invalid(
            self, "Here is the file:\n\n```tsx\nexport const a = 1;\n```\n",
            "src/a.tsx", IntegrityIssueCode.TRAILING_PROSE,
        )

    def test_a_fenced_json_file_is_rejected_as_a_wrapper_first(self) -> None:
        assert_invalid(
            self, '```json\n{\n  "name": "x"\n```\n', "package.json",
            IntegrityIssueCode.FENCED_CONTENT, kind=ArtifactKind.CONFIG_FILE,
        )

    def test_a_document_with_several_code_blocks_is_perfectly_normal(self) -> None:
        assert_valid(
            self,
            "# Guide\n\n```ts\nconst a = 1;\n```\n\nThen:\n\n```ts\nconst b = 2;\n```\n",
            DOC, kind=ArtifactKind.DOCUMENT,
        )

    def test_a_document_with_an_unclosed_fence_is_fatal(self) -> None:
        assert_invalid(
            self, "# Guide\n\n```ts\nconst a = 1;\n", DOC,
            IntegrityIssueCode.UNCLOSED_FENCE, kind=ArtifactKind.DOCUMENT,
        )

    def test_backticks_inside_a_template_literal_are_not_a_fence(self) -> None:
        # Fence analysis outside prose only engages when the file STARTS fenced.
        assert_valid(
            self,
            'export const md = "```";\nexport const more = 1;\n',
            "src/md.ts",
        )

    def test_integrity_never_rewrites_the_submitted_content(self) -> None:
        body = "```tsx\nexport const a = 1;\n```\n"
        result = check(body, "src/a.tsx")
        # The result carries a verdict and bounded evidence, never a rewritten body.
        self.assertEqual(result.content_chars, len(body))
        self.assertNotIn("content", result.to_dict())


if __name__ == "__main__":
    unittest.main()
