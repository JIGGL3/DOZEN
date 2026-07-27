"""Phase 6 — frontend rendering contract (source-level, always runs).

The chat renderer in webllm/static/index.html must display only the approved
``final_answer`` string, never stringify unknown objects into the chat, and
keep protocol fields hidden. These are wiring-contract checks over the shipped
source, mirroring the established pattern in tests/usability.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_INDEX = Path("webllm/static/index.html")


class TestFrontendRenderingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _INDEX.read_text(encoding="utf-8")

    def test_final_answer_must_be_a_string_to_render(self) -> None:
        self.assertIn('typeof r.final_answer === "string"', self.src)

    def test_typed_json_bypasses_protocol_unwrap_and_preserves_formatting(self) -> None:
        self.assertIn('r.answer_kind === "json_data"', self.src)
        self.assertIn(
            '((isJsonData || isLiteral) ? r.final_answer : '
            'unwrapAnswer(r.final_answer))',
            self.src,
        )
        self.assertIn('r.answer_kind === "literal"', self.src)
        self.assertIn('`<pre><code>${esc(answerText)}</code></pre>`', self.src)
        # A non-string result degrades to a clean notice, never JSON text.
        self.assertIn("The run returned a non-text result", self.src)

    def test_no_object_is_stringified_into_the_chat(self) -> None:
        # Every JSON.stringify in the page is a REQUEST body, not display text.
        for match in re.finditer(r"JSON\.stringify\([^)]*\)", self.src):
            line_start = self.src.rfind("\n", 0, match.start())
            line = self.src[line_start: match.end()]
            self.assertIn("body:", line,
                          f"JSON.stringify used outside a request body: {line!r}")

    def test_envelope_summary_fallback_is_gated_on_protocol_fields(self) -> None:
        self.assertIn("Array.isArray(obj.key_decisions)", self.src)
        self.assertIn('"confidence" in obj', self.src)

    def test_internal_planning_data_never_renders_raw(self) -> None:
        self.assertIn("Array.isArray(obj.delegations)", self.src)
        self.assertIn("internal planning data instead of an answer", self.src)

    def test_markdown_renderer_escapes_before_rebuilding(self) -> None:
        # mdToHtml escapes the whole source before rebuilding markup, and code
        # blocks render inside <pre><code>.
        self.assertIn('esc(src || "")', self.src)
        self.assertIn("<pre><code>", self.src)

    def test_error_text_is_escaped(self) -> None:
        self.assertIn("esc(String(r.error))", self.src)

    def test_answer_uses_unwrap_only_for_strings(self) -> None:
        self.assertIn("unwrapAnswer(r.final_answer)", self.src)


if __name__ == "__main__":
    unittest.main()
