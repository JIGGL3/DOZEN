"""Phase 4D Parts F/G — source-code structural checks and the lexical scanner.

Python, JavaScript/TypeScript/JSX/TSX, HTML, CSS, shell, SQL and unified diffs.
Each check is deliberately structural: none of them claims full language,
type or semantic validity, and none of them executes the submitted content.
"""

from __future__ import annotations

import unittest

from dozen.artifact_integrity import (
    DEFAULT_INTEGRITY_POLICY,
    IntegrityIssueCode,
    IntegrityStatus,
    scan_lexical,
)
from dozen.artifact_integrity import _JS_DIALECT  # structural scanner under test
from dozen.artifacts import ArtifactKind

from .harness import (
    CARD_BRACES_IN_STRINGS,
    LAYOUT_UNCLOSED_TAG,
    ROUTER_CUT_AFTER_IMPORT,
    TEST_UNCLOSED_TEMPLATE,
    assert_invalid,
    assert_suspicious,
    assert_valid,
    check,
)

TEST_FILE = ArtifactKind.TEST_FILE


class TestLexicalScanner(unittest.TestCase):
    def test_delimiters_inside_strings_are_ignored(self) -> None:
        scan = scan_lexical('const a = "{ ( [";\n', _JS_DIALECT)
        self.assertTrue(scan.clean)
        self.assertTrue(scan.balanced)

    def test_delimiters_inside_comments_are_ignored(self) -> None:
        scan = scan_lexical("// { ( [\n/* } ) ] */\nconst a = 1;\n", _JS_DIALECT)
        self.assertTrue(scan.clean)

    def test_the_masked_view_preserves_offsets_and_lines(self) -> None:
        body = 'const a = "xyz";\nconst b = 2;\n'
        scan = scan_lexical(body, _JS_DIALECT)
        self.assertEqual(len(scan.masked), len(body))
        self.assertEqual(scan.masked.count("\n"), body.count("\n"))

    def test_the_scan_result_is_immutable_tuples(self) -> None:
        scan = scan_lexical("function f( {\n", _JS_DIALECT)
        self.assertIsInstance(scan.unbalanced_open, tuple)
        self.assertIsInstance(scan.unbalanced_close, tuple)

    def test_nesting_is_bounded_by_policy(self) -> None:
        from dozen.artifact_integrity import IntegrityPolicy

        scan = scan_lexical(
            "(" * 50, _JS_DIALECT, policy=IntegrityPolicy(max_nesting_depth=4)
        )
        self.assertTrue(scan.depth_exceeded)
        self.assertLessEqual(len(scan.unbalanced_open), 4)

    def test_one_level_beyond_nesting_limit_is_not_applicable(self) -> None:
        from dozen.artifact_integrity import IntegrityPolicy

        result = check(
            "(" * 5 + "x" + ")" * 5,
            "src/deep.ts",
            policy=IntegrityPolicy(max_nesting_depth=4),
        )
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertFalse(result.conclusive)

    def test_a_regex_literal_containing_a_quote_is_not_a_string(self) -> None:
        scan = scan_lexical(
            "const clean = raw.replace(/'/g, \"\");\nconst n = 1;\n", _JS_DIALECT
        )
        self.assertTrue(scan.clean)


class TestPython(unittest.TestCase):
    def test_valid_python_is_accepted(self) -> None:
        assert_valid(
            self,
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            "src/calc.py",
        )

    def test_an_unterminated_string_is_incomplete(self) -> None:
        result = assert_invalid(
            self, 'MESSAGE = "hello\n', "src/calc.py",
            IntegrityIssueCode.PYTHON_INCOMPLETE,
        )
        self.assertTrue(result.truncation_suspected)

    def test_a_missing_delimiter_is_incomplete(self) -> None:
        assert_invalid(
            self, "def add(a, b):\n    return sum([a, b\n", "src/calc.py",
            IntegrityIssueCode.PYTHON_INCOMPLETE,
        )

    def test_an_incomplete_function_body_is_incomplete(self) -> None:
        assert_invalid(
            self, "def add(a, b):\n", "src/calc.py",
            IntegrityIssueCode.PYTHON_INCOMPLETE,
        )

    def test_a_general_syntax_error_is_recorded_as_a_syntax_error(self) -> None:
        result = assert_invalid(
            self, "def add(a, b):\n    return a +* b\n\nx = 1\n", "src/calc.py",
        )
        self.assertIn(
            IntegrityIssueCode.PYTHON_SYNTAX_ERROR,
            [issue.code for issue in result.issues],
        )
        self.assertFalse(result.truncation_suspected)

    def test_a_last_line_general_syntax_error_is_not_called_truncation(self) -> None:
        result = assert_invalid(self, "x = )\n", "src/calc.py")
        self.assertIn(IntegrityIssueCode.PYTHON_SYNTAX_ERROR,
                      [issue.code for issue in result.issues])
        self.assertFalse(result.truncation_suspected)

    def test_braces_inside_python_strings_are_accepted(self) -> None:
        assert_valid(
            self,
            'TEMPLATE = "{ unbalanced ( brace"\nOTHER = \'\'\'} ) ]\'\'\'\n',
            "src/calc.py",
        )

    def test_comments_containing_delimiters_are_accepted(self) -> None:
        assert_valid(
            self, "# a ( b { c [\nVALUE = 1\n", "src/calc.py",
        )

    def test_ast_parse_never_executes_the_submitted_code(self) -> None:
        # If this were executed the test process would die; parsing is inert.
        assert_valid(
            self,
            "import os\n\ndef wipe():\n    os._exit(1)\n",
            "src/danger.py",
        )


class TestJavaScriptFamily(unittest.TestCase):
    def test_valid_structural_balance_is_accepted(self) -> None:
        assert_valid(self, CARD_BRACES_IN_STRINGS, "src/components/ui/Card.tsx")

    def test_an_unterminated_string_is_fatal(self) -> None:
        assert_invalid(
            self, 'export const title = "Revenue', "src/title.ts",
            IntegrityIssueCode.UNTERMINATED_STRING,
        )

    def test_an_unterminated_template_string_is_fatal(self) -> None:
        result = assert_invalid(
            self, TEST_UNCLOSED_TEMPLATE, "tests/dashboard.test.tsx",
            IntegrityIssueCode.UNTERMINATED_TEMPLATE, kind=TEST_FILE,
        )
        self.assertTrue(result.truncation_suspected)

    def test_an_unclosed_brace_is_fatal(self) -> None:
        assert_invalid(
            self, "export function Card() {\n  const a = 1;\n", "src/Card.tsx",
            IntegrityIssueCode.UNBALANCED_DELIMITER,
        )

    def test_an_unclosed_jsx_tag_is_fatal(self) -> None:
        result = assert_invalid(
            self, LAYOUT_UNCLOSED_TAG,
            "src/components/layout/DashboardLayout.tsx",
            IntegrityIssueCode.UNCLOSED_TAG,
        )
        self.assertIn("section", result.problem_summary())

    def test_a_cut_off_opening_tag_is_fatal(self) -> None:
        assert_invalid(
            self,
            "export const A = () => {\n  return <div className={cls}\n};\n",
            "src/A.tsx", IntegrityIssueCode.INCOMPLETE_TAG,
        )

    def test_self_closing_jsx_is_accepted(self) -> None:
        assert_valid(
            self,
            'import App from "./App";\n\n'
            "export const Root = () => <App title=\"x\" />;\n",
            "src/Root.tsx",
        )

    def test_jsx_fragments_are_accepted(self) -> None:
        assert_valid(
            self,
            "export const Pair = () => (\n  <>\n    <A />\n    <B />\n  </>\n);\n",
            "src/Pair.tsx",
        )

    def test_an_arrow_function_in_a_jsx_attribute_is_accepted(self) -> None:
        # The '>' of '=>' must never be read as the end of the opening tag.
        assert_valid(
            self,
            "export const Btn = ({ go }) => (\n"
            "  <button onClick={() => go(1)}>Go</button>\n"
            ");\n",
            "src/Btn.tsx",
        )

    def test_an_apostrophe_in_jsx_text_is_accepted(self) -> None:
        assert_valid(
            self,
            "export const Hi = () => <p>It's ready, don't worry</p>;\n",
            "src/Hi.tsx",
        )

    def test_typescript_generics_are_not_jsx_tags(self) -> None:
        assert_valid(
            self,
            'import { useState } from "react";\n\n'
            "export const useRows = () => useState<Record<string, number>>({});\n",
            "src/useRows.ts",
        )

    def test_tsx_generic_arrow_is_not_an_unclosed_jsx_tag(self) -> None:
        assert_valid(
            self, "const identity = <T extends {}>(value: T) => value;\n",
            "src/identity.tsx",
        )

    def test_fragment_inside_a_complex_jsx_attribute_is_accepted(self) -> None:
        assert_valid(
            self,
            "const A = () => <Foo render={(v) => <><B />{v}</>} />;\n",
            "src/A.tsx",
        )

    def test_template_interpolation_is_accepted(self) -> None:
        assert_valid(
            self,
            "export const cls = (n) => `w-${n} ${n > 2 ? `deep-${n}` : \"\"}`;\n",
            "src/cls.ts",
        )

    def test_comments_containing_delimiters_are_accepted(self) -> None:
        assert_valid(
            self,
            "// { ( [ unbalanced\n/* } ) ] also unbalanced */\nexport const a = 1;\n",
            "src/a.ts",
        )

    def test_braces_inside_strings_are_accepted(self) -> None:
        assert_valid(
            self, 'export const cls = "rounded { } p-4";\n', "src/cls.ts",
        )

    def test_an_unfinished_import_is_suspicious(self) -> None:
        result = assert_suspicious(
            self, ROUTER_CUT_AFTER_IMPORT, "src/app/router.tsx",
            IntegrityIssueCode.UNFINISHED_STATEMENT,
        )
        self.assertTrue(result.truncation_suspected)


class TestHtml(unittest.TestCase):
    def test_valid_html_is_accepted(self) -> None:
        assert_valid(
            self,
            "<!DOCTYPE html>\n<html>\n<head><title>App</title></head>\n"
            '<body><div id="root"></div><script src="/main.js"></script></body>\n'
            "</html>\n",
            "index.html",
        )

    def test_optional_closing_tags_are_accepted(self) -> None:
        assert_valid(
            self,
            "<ul>\n  <li>one\n  <li>two\n</ul>\n<p>a paragraph\n<p>another\n",
            "index.html",
        )

    def test_void_elements_are_accepted(self) -> None:
        assert_valid(
            self,
            '<div><img src="a.png"><br><input type="text"></div>\n',
            "index.html",
        )

    def test_a_truncated_opening_tag_is_fatal(self) -> None:
        assert_invalid(
            self, '<html>\n<body>\n<div class="wrap"\n', "index.html",
            IntegrityIssueCode.INCOMPLETE_TAG,
        )

    def test_an_unterminated_attribute_is_fatal(self) -> None:
        assert_invalid(
            self, '<html>\n<body>\n<div class="wrap\n', "index.html",
            IntegrityIssueCode.UNTERMINATED_ATTRIBUTE,
        )

    def test_an_unterminated_comment_is_fatal(self) -> None:
        assert_invalid(
            self, "<html>\n<!-- todo\n", "index.html",
            IntegrityIssueCode.UNTERMINATED_COMMENT,
        )

    def test_an_unclosed_container_is_fatal(self) -> None:
        assert_invalid(
            self, "<html>\n<body>\n<div>content</div>\n</body>\n", "index.html",
            IntegrityIssueCode.UNCLOSED_TAG,
        )

    def test_an_unclosed_script_is_fatal(self) -> None:
        assert_invalid(
            self, "<html><body><script>\nconst a = 1;\n", "index.html",
            IntegrityIssueCode.UNCLOSED_TAG,
        )


class TestCss(unittest.TestCase):
    def test_valid_css_is_accepted(self) -> None:
        assert_valid(
            self,
            '/* base */\n.card {\n  content: "{ }";\n  padding: 1rem;\n}\n',
            "src/app.css",
        )

    def test_an_unclosed_block_is_fatal(self) -> None:
        assert_invalid(
            self, ".card {\n  padding: 1rem;\n", "src/app.css",
            IntegrityIssueCode.UNBALANCED_DELIMITER,
        )

    def test_an_unterminated_string_is_fatal(self) -> None:
        assert_invalid(
            self, '.card::after {\n  content: "hello', "src/app.css",
            IntegrityIssueCode.UNTERMINATED_STRING,
        )

    def test_an_unterminated_comment_is_fatal(self) -> None:
        assert_invalid(
            self, ".card { padding: 1rem; }\n/* more to come\n", "src/app.css",
            IntegrityIssueCode.UNTERMINATED_COMMENT,
        )

    def test_an_ending_cut_after_a_selector_is_suspicious(self) -> None:
        # Balanced braces, closed strings — but the file stops on a bare selector.
        assert_suspicious(
            self, ".card {\n  padding: 1rem;\n}\n\n.title\n", "src/app.css",
            IntegrityIssueCode.UNFINISHED_STATEMENT,
        )


class TestShellAndSql(unittest.TestCase):
    def test_valid_shell_is_accepted(self) -> None:
        assert_valid(
            self,
            '#!/usr/bin/env bash\nset -euo pipefail\necho "building..."\nnpm run build\n',
            "scripts/build.sh",
        )

    def test_an_unterminated_shell_quote_is_fatal(self) -> None:
        assert_invalid(
            self, '#!/bin/sh\necho "starting the build', "scripts/build.sh",
            IntegrityIssueCode.UNTERMINATED_STRING,
        )

    def test_a_heredoc_is_honestly_not_inspected(self) -> None:
        result = check(
            "cat <<EOF > out.txt\nit's fine to have one quote here\nEOF\n",
            "scripts/write.sh",
        )
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)

    def test_valid_sql_is_accepted(self) -> None:
        assert_valid(
            self,
            "-- seed\nINSERT INTO users (name) VALUES ('O''Brien');\n"
            "SELECT * FROM users WHERE name LIKE '%a%';\n",
            "db/seed.sql",
        )

    def test_postgres_dollar_quoted_sql_is_accepted(self) -> None:
        assert_valid(
            self, "SELECT $body$ (not structure) 'still text' $body$;\n",
            "db/query.sql",
        )

    def test_an_unterminated_sql_string_is_fatal(self) -> None:
        assert_invalid(
            self, "INSERT INTO users (name) VALUES ('Alice", "db/seed.sql",
            IntegrityIssueCode.UNTERMINATED_STRING,
        )

    def test_an_unbalanced_sql_paren_is_fatal(self) -> None:
        assert_invalid(
            self, "SELECT count(* FROM users;\n", "db/seed.sql",
            IntegrityIssueCode.UNBALANCED_DELIMITER,
        )


VALID_DIFF = (
    "diff --git a/src/app.ts b/src/app.ts\n"
    "--- a/src/app.ts\n"
    "+++ b/src/app.ts\n"
    "@@ -1,3 +1,4 @@\n"
    " const a = 1;\n"
    "-const b = 2;\n"
    "+const b = 3;\n"
    "+const c = 4;\n"
    " export { a };\n"
)


class TestUnifiedDiff(unittest.TestCase):
    def test_a_valid_unified_diff_is_accepted(self) -> None:
        assert_valid(self, VALID_DIFF, "fix.patch", kind=ArtifactKind.PATCH)

    def test_binary_diff_declaration_is_accepted(self) -> None:
        assert_valid(
            self,
            "diff --git a/a.png b/a.png\nBinary files a/a.png and b/a.png differ\n",
            "binary.patch", kind=ArtifactKind.PATCH,
        )

    def test_rename_only_diff_is_accepted(self) -> None:
        assert_valid(
            self,
            "diff --git a/a.txt b/b.txt\nsimilarity index 100%\n"
            "rename from a.txt\nrename to b.txt\n",
            "rename.patch", kind=ArtifactKind.PATCH,
        )

    def test_a_header_only_diff_is_fatal(self) -> None:
        assert_invalid(
            self,
            "diff --git a/src/app.ts b/src/app.ts\n--- a/src/app.ts\n+++ b/src/app.ts\n",
            "fix.patch", IntegrityIssueCode.MALFORMED_DIFF, kind=ArtifactKind.PATCH,
        )

    def test_an_incomplete_final_hunk_is_fatal(self) -> None:
        truncated = VALID_DIFF[: VALID_DIFF.index("+const c = 4;")]
        result = assert_invalid(
            self, truncated, "fix.patch", IntegrityIssueCode.INCOMPLETE_HUNK,
            kind=ArtifactKind.PATCH,
        )
        self.assertTrue(result.truncation_suspected)

    def test_a_truncated_final_line_is_fatal(self) -> None:
        assert_invalid(
            self, VALID_DIFF[: -len(" export { a };\n")], "fix.patch",
            IntegrityIssueCode.INCOMPLETE_HUNK, kind=ArtifactKind.PATCH,
        )

    def test_a_bad_hunk_prefix_is_fatal(self) -> None:
        broken = VALID_DIFF.replace("-const b = 2;", "const b = 2;")
        assert_invalid(
            self, broken, "fix.patch", IntegrityIssueCode.MALFORMED_DIFF,
            kind=ArtifactKind.PATCH,
        )

    def test_a_patch_without_a_file_header_is_fatal(self) -> None:
        assert_invalid(
            self, "@@ -1,1 +1,1 @@\n-a\n+b\n", "fix.patch",
            IntegrityIssueCode.MALFORMED_DIFF, kind=ArtifactKind.PATCH,
        )

    def test_a_no_newline_marker_is_accepted(self) -> None:
        assert_valid(
            self,
            "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-a\n+b\n"
            "\\ No newline at end of file\n",
            "fix.patch", kind=ArtifactKind.PATCH,
        )


if __name__ == "__main__":
    unittest.main()
