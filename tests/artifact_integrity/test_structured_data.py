"""Phase 4D Part E — structured data: JSON, JSON Lines, TOML, XML, YAML."""

from __future__ import annotations

import unittest

from dozen.artifact_integrity import IntegrityIssueCode, IntegrityStatus
from dozen.artifacts import ArtifactKind

from .harness import PACKAGE_JSON_MISSING_BRACE, assert_invalid, assert_valid, check

CONFIG = ArtifactKind.CONFIG_FILE
DATA = ArtifactKind.DATA_FILE


class TestJson(unittest.TestCase):
    def test_valid_json_is_accepted(self) -> None:
        assert_valid(
            self, '{\n  "name": "dashboard",\n  "version": "1.0.0"\n}\n',
            "package.json", kind=CONFIG,
        )

    def test_a_missing_final_brace_is_fatal(self) -> None:
        result = assert_invalid(
            self, PACKAGE_JSON_MISSING_BRACE, "package.json",
            IntegrityIssueCode.MALFORMED_JSON, kind=CONFIG,
        )
        self.assertTrue(result.truncation_suspected)

    def test_an_unterminated_json_string_is_fatal(self) -> None:
        result = assert_invalid(
            self, '{\n  "name": "dash', "package.json",
            IntegrityIssueCode.MALFORMED_JSON, kind=CONFIG,
        )
        self.assertTrue(result.truncation_suspected)

    def test_a_valid_tsconfig_is_accepted(self) -> None:
        assert_valid(
            self, '{\n  "compilerOptions": {"jsx": "react-jsx", "strict": true}\n}\n',
            "tsconfig.json", kind=CONFIG,
        )

    def test_plain_json_never_silently_becomes_jsonc(self) -> None:
        result = assert_invalid(
            self,
            '{\n  // the React 17+ transform\n  "compilerOptions": {"jsx": "react-jsx"}\n}\n',
            "tsconfig.json", IntegrityIssueCode.MALFORMED_JSON, kind=CONFIG,
        )
        self.assertFalse(result.truncation_suspected)

    def test_explicit_jsonc_accepts_comments(self) -> None:
        result = check(
            '{\n  // comment\n  "compilerOptions": {"strict": true}\n}\n',
            "tsconfig.jsonc", kind=CONFIG,
        )
        self.assertIs(result.status, IntegrityStatus.VALID)
        self.assertIn(IntegrityIssueCode.JSONC_COMMENTS,
                      [issue.code for issue in result.advisories])

    def test_jsonc_language_and_media_type_are_explicit(self) -> None:
        body = '{"x": 1 // comment\n}\n'
        by_language = check(body, "config", kind=CONFIG, language="jsonc")
        by_media = check(body, "config", kind=CONFIG,
                         media_type="application/jsonc")
        self.assertIs(by_language.status, IntegrityStatus.VALID)
        self.assertIs(by_media.status, IntegrityStatus.VALID)

    def test_a_double_slash_inside_a_string_is_not_a_comment(self) -> None:
        assert_valid(
            self, '{"homepage": "https://example.com/app"}\n', "package.json",
            kind=CONFIG,
        )

    def test_a_schema_violation_is_not_a_structural_violation(self) -> None:
        # Structurally perfect JSON that no package.json schema would accept.
        assert_valid(self, '{"totally": ["unexpected", 1, null]}\n',
                     "package.json", kind=CONFIG)


class TestJsonLines(unittest.TestCase):
    def test_valid_jsonl_is_accepted(self) -> None:
        assert_valid(
            self, '{"a": 1}\n{"a": 2}\n{"a": 3}\n', "data/events.jsonl", kind=DATA,
        )

    def test_a_truncated_final_record_is_fatal(self) -> None:
        result = assert_invalid(
            self, '{"a": 1}\n{"a": 2}\n{"a": ', "data/events.jsonl",
            IntegrityIssueCode.MALFORMED_JSONL, kind=DATA,
        )
        self.assertTrue(result.truncation_suspected)
        self.assertIn("final", result.issues[0].message)

    def test_a_malformed_middle_record_is_fatal_but_not_truncation(self) -> None:
        result = assert_invalid(
            self, '{"a": 1}\n{oops}\n{"a": 3}\n', "data/events.jsonl",
            IntegrityIssueCode.MALFORMED_JSONL, kind=DATA,
        )
        self.assertFalse(result.truncation_suspected)


class TestToml(unittest.TestCase):
    def test_valid_toml_is_accepted(self) -> None:
        assert_valid(
            self, '[project]\nname = "dozen"\nversion = "1.0"\n',
            "pyproject.toml", kind=CONFIG,
        )

    def test_a_truncated_toml_string_is_fatal(self) -> None:
        assert_invalid(
            self, '[project]\nname = "doz', "pyproject.toml",
            IntegrityIssueCode.MALFORMED_TOML, kind=CONFIG,
        )

    def test_an_invalid_toml_table_is_fatal(self) -> None:
        assert_invalid(
            self, "[project\nname = 1\n", "pyproject.toml",
            IntegrityIssueCode.MALFORMED_TOML, kind=CONFIG,
        )


class TestXml(unittest.TestCase):
    def test_valid_xml_is_accepted(self) -> None:
        assert_valid(
            self, '<?xml version="1.0"?>\n<root><item id="1">x</item></root>\n',
            "config/app.xml", kind=CONFIG,
        )

    def test_truncated_xml_is_fatal(self) -> None:
        assert_invalid(
            self, "<root>\n  <item>x</item>\n", "config/app.xml",
            IntegrityIssueCode.MALFORMED_XML, kind=CONFIG,
        )

    def test_a_cut_off_xml_tag_is_fatal(self) -> None:
        assert_invalid(
            self, "<root>\n  <item id=\"1", "config/app.xml",
            IntegrityIssueCode.MALFORMED_XML, kind=CONFIG,
        )

    def test_entity_declarations_are_never_expanded(self) -> None:
        # The one XML feature with an expansion hazard: reported, never parsed.
        result = check(
            '<?xml version="1.0"?>\n<!DOCTYPE r [<!ENTITY a "b">]>\n<r>&a;</r>\n',
            "config/app.xml", kind=CONFIG,
        )
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertIn(
            IntegrityIssueCode.UNSUPPORTED_FORMAT,
            [issue.code for issue in result.issues],
        )


class TestYaml(unittest.TestCase):
    def test_yaml_is_not_inspected_and_never_falsely_invalid(self) -> None:
        result = check("services:\n  web:\n    image: nginx\n",
                       "docker-compose.yml", kind=CONFIG)
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertIn(
            IntegrityIssueCode.UNSUPPORTED_FORMAT,
            [issue.code for issue in result.issues],
        )

    def test_yaml_still_gets_the_universal_truncation_indicators(self) -> None:
        assert_invalid(
            self,
            "services:\n  web:\n    image: nginx\n# ... rest of file omitted\n",
            "docker-compose.yml", IntegrityIssueCode.TRUNCATION_MARKER, kind=CONFIG,
        )


class TestUnsupportedFormats(unittest.TestCase):
    def test_an_unknown_language_is_not_condemned_for_being_unknown(self) -> None:
        result = check(
            "const std = @import(\"std\");\npub fn main() void {}\n",
            "src/main.zig",
        )
        self.assertIs(result.status, IntegrityStatus.NOT_APPLICABLE)
        self.assertTrue(result.structurally_complete)

    def test_an_unknown_language_still_fails_on_strong_evidence(self) -> None:
        assert_invalid(
            self,
            "pub fn main() void {}\n// [... output truncated ...]\n",
            "src/main.zig", IntegrityIssueCode.TRUNCATION_MARKER,
        )


if __name__ == "__main__":
    unittest.main()
