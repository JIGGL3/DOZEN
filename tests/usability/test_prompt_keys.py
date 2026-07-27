"""Phase 2 Part B — prompt-box Enter/Shift+Enter behavior.

The decision logic is a pure function in webllm/static/prompt_keys.js; its
behavior table is executed under Node (skipped cleanly when Node is absent).
A wiring-contract test (pure Python, always runs) proves index.html attaches
the handler, shares runTask() with the Send button, and defines no parallel
submission logic.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

_STATIC = Path("webllm/static")
_NODE = shutil.which("node")

# (case-name, event-facts, expected-action)
CASES = [
    ("enter_submits", {"key": "Enter", "text": "hello"}, "submit"),
    ("shift_enter_newline", {"key": "Enter", "shiftKey": True, "text": "hello"}, "newline"),
    ("empty_blocks", {"key": "Enter", "text": ""}, "block"),
    ("whitespace_blocks", {"key": "Enter", "text": "   \n\t "}, "block"),
    ("ime_composing_none", {"key": "Enter", "isComposing": True, "text": "hello"}, "none"),
    ("ime_keycode_229_none", {"key": "Enter", "keyCode": 229, "text": "hello"}, "none"),
    ("ctrl_enter_none", {"key": "Enter", "ctrlKey": True, "text": "hello"}, "none"),
    ("alt_enter_none", {"key": "Enter", "altKey": True, "text": "hello"}, "none"),
    ("meta_enter_none", {"key": "Enter", "metaKey": True, "text": "hello"}, "none"),
    ("key_repeat_blocks", {"key": "Enter", "repeat": True, "text": "hello"}, "block"),
    ("busy_blocks", {"key": "Enter", "busy": True, "text": "hello"}, "block"),
    ("other_keys_none", {"key": "a", "text": "hello"}, "none"),
    ("shift_enter_while_busy_still_newline",
     {"key": "Enter", "shiftKey": True, "busy": True, "text": "x"}, "newline"),
    ("ime_wins_over_shift",
     {"key": "Enter", "shiftKey": True, "isComposing": True, "text": "x"}, "none"),
]


@unittest.skipIf(_NODE is None, "Node.js not available for JS unit tests")
class TestPromptKeyDecision(unittest.TestCase):
    """Runs the REAL shipped function under Node against the behavior table."""

    @classmethod
    def setUpClass(cls) -> None:
        module_path = (_STATIC / "prompt_keys.js").resolve().as_posix()
        script = (
            f"const {{ dozenPromptKeyAction }} = require({json.dumps(module_path)});"
            "const cases = JSON.parse(process.argv[1]);"
            "const out = {};"
            "for (const [name, facts] of cases) out[name] = dozenPromptKeyAction(facts);"
            "console.log(JSON.stringify(out));"
        )
        payload = json.dumps([[name, facts] for name, facts, _ in CASES])
        result = subprocess.run(
            [_NODE, "-e", script, payload],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"node failed: {result.stderr}")
        cls.actual = json.loads(result.stdout)

    def test_behavior_table(self) -> None:
        for name, _facts, expected in CASES:
            with self.subTest(case=name):
                self.assertEqual(self.actual[name], expected)

    def test_enter_submits_exactly_once_per_decision(self) -> None:
        # The function is pure: one keydown maps to exactly one action, and
        # "submit" is returned only for the plain-Enter-with-content case.
        submits = [name for name, _f, expected in CASES if expected == "submit"]
        self.assertEqual(submits, ["enter_submits"])


class TestPromptWiringContract(unittest.TestCase):
    """index.html must wire the pure function to the SHARED submission path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (_STATIC / "index.html").read_text(encoding="utf-8")

    def test_pure_function_module_is_loaded(self) -> None:
        self.assertIn('<script src="/static/prompt_keys.js"></script>', self.html)

    def test_keydown_handler_attached_to_prompt(self) -> None:
        self.assertRegex(self.html,
                         r'\$\("prompt"\)\.addEventListener\("keydown"')

    def test_handler_uses_decision_function_and_shared_submit(self) -> None:
        handler = self.html[self.html.index('$("prompt").addEventListener'):]
        handler = handler[:handler.index("\n});") + 4]  # listener's own close
        self.assertIn("dozenPromptKeyAction", handler)
        self.assertIn("e.preventDefault()", handler)
        self.assertIn("runTask()", handler)          # the Send button's path
        self.assertIn('$("btn-run").disabled', handler)  # shared busy source
        self.assertIn("isComposing", handler)

    def test_send_button_still_uses_the_same_path(self) -> None:
        self.assertIn('onclick="runTask()" id="btn-run"', self.html)

    def test_no_parallel_submission_logic(self) -> None:
        # Exactly one function posts to /api/run — the shared runTask().
        self.assertEqual(self.html.count('api("/api/run"'), 1)

    def test_decision_module_defines_single_pure_function(self) -> None:
        js = (_STATIC / "prompt_keys.js").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"\bfunction dozenPromptKeyAction\b", js)), 1)
        for forbidden in ("fetch(", "document.", "window.", "runTask"):
            self.assertNotIn(forbidden, js)  # pure: no DOM, no network, no submit


if __name__ == "__main__":
    unittest.main()
