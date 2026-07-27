"""Phase 4E Part F — the targeted repair prompt."""

from __future__ import annotations

import unittest

from dozen.artifact_repair import (
    DEFAULT_REPAIR_POLICY,
    ArtifactRepairRequest,
    ArtifactRepairPolicy,
    PreservedArtifact,
    RepairTarget,
    derive_repair_scope,
)
from dozen.artifact_results import content_hash
from dozen.intent import resolve_contract
from dozen.models import SubTask, Task
from dozen.prompts import (
    SCOPED_WORKER_ARTIFACT_CONTRACT,
    WORKER_ARTIFACT_CONTRACT,
    build_worker_messages,
)

from .harness import (
    APP,
    CARD,
    CONTENT,
    LAYOUT,
    PKG,
    ROUTER,
    decide,
    entry,
    envelope,
    envelope_with,
    first_attempt,
    scope_for,
)

IMPLEMENT = resolve_contract("Build me a React dashboard with tests.")
TRUNCATED_LAYOUT = "export function DashboardLayout({ children }) {\n  return <main>"


def repair_request(payload=None):
    state = first_attempt(
        payload or envelope_with("shell", **{LAYOUT: TRUNCATED_LAYOUT})
    )
    return decide(state).request


def worker_prompt(request=None, *, scope=None) -> str:
    task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
    subtask = SubTask(title="Application shell", instruction="Write the shell.",
                      id="shell")
    if request is not None and scope is None:
        scope = derive_repair_scope(scope_for("shell"), request)
    messages = build_worker_messages(task, subtask, {}, "", scope=scope,
                                     repair=request)
    return messages[1].content


class TestRepairBlock(unittest.TestCase):
    def test_preserved_artifacts_are_listed_without_their_content(self) -> None:
        block = repair_request().to_worker_block()
        self.assertIn(APP, block)
        self.assertIn(ROUTER, block)
        self.assertNotIn(CONTENT[APP].strip(), block)
        self.assertNotIn("createRoot", block)
        self.assertIn("Do NOT return them", block)

    def test_preserved_hashes_are_bounded(self) -> None:
        request = repair_request()
        block = request.to_worker_block()
        for item in request.preserved:
            self.assertNotIn(item.content_hash, block)  # the FULL hash never appears
            self.assertIn(item.content_hash[: DEFAULT_REPAIR_POLICY.max_hash_chars - 1],
                          block)

    def test_only_the_repair_targets_are_requested(self) -> None:
        block = repair_request().to_worker_block()
        self.assertIn("Return ONLY these repair artifacts", block)
        self.assertIn(LAYOUT, block.split("Return ONLY these repair artifacts")[1])
        self.assertNotIn(PKG, block)
        self.assertNotIn(CARD, block)

    def test_reasons_and_evidence_are_included(self) -> None:
        block = repair_request().to_worker_block()
        self.assertIn("Reason:", block)
        self.assertIn("Evidence:", block)
        self.assertIn("incomplete", block)

    def test_no_fragment_or_continuation_is_requested(self) -> None:
        block = repair_request().to_worker_block().lower()
        self.assertIn("complete raw file contents", block)
        self.assertNotIn("continue from", block)
        self.assertNotIn("resume where", block)

    def test_the_block_prevents_newline_injection(self) -> None:
        payload = envelope_with("shell", **{
            LAYOUT: "export function X() {\n  return <main>",
        })
        request = repair_request(payload)
        injected = request.to_worker_block()
        # Every rendered line is one of ours: no diagnostic can open a new
        # instruction line of its own.
        for line in injected.splitlines():
            self.assertFalse(line.startswith("Reply with"), line)
            self.assertFalse(line.startswith("Ignore"), line)

    def test_the_block_is_bounded(self) -> None:
        request = repair_request(envelope())  # three targets
        policy = ArtifactRepairPolicy(max_prompt_chars=120)
        self.assertLessEqual(len(request.to_worker_block(policy)), 120)
        self.assertLessEqual(
            len(request.to_worker_block()), DEFAULT_REPAIR_POLICY.max_prompt_chars
        )

    def test_maximum_block_keeps_every_target_and_the_final_rule(self) -> None:
        targets = tuple(
            RepairTarget(
                artifact_id=f"artifact-{i}",
                path=f"src/{'x' * 180}/{i}.tsx",
                evidence="e" * DEFAULT_REPAIR_POLICY.max_diagnostic_chars,
            )
            for i in range(DEFAULT_REPAIR_POLICY.max_targets_per_attempt)
        )
        preserved = tuple(
            PreservedArtifact(
                artifact_id=f"preserved-{i}",
                path=f"src/{'y' * 180}/{i}.tsx",
                content_hash=content_hash(f"body-{i}"),
            )
            for i in range(DEFAULT_REPAIR_POLICY.max_preserved_per_package)
        )
        request = ArtifactRepairRequest(
            manifest_id="m", subtask_id="s", targets=targets,
            preserved=preserved,
        )
        block = request.to_worker_block()
        self.assertLessEqual(len(block), DEFAULT_REPAIR_POLICY.max_prompt_chars)
        for target in targets:
            self.assertIn(target.path, block)
        self.assertIn("Return one typed artifact envelope", block)

    def test_worker_prompt_honors_the_supplied_repair_policy(self) -> None:
        request = repair_request()
        scope = derive_repair_scope(scope_for("shell"), request)
        task = Task(prompt="Build me a React dashboard.", contract=IMPLEMENT)
        subtask = SubTask(
            title="Application shell", instruction="Write the shell.", id="shell"
        )
        messages = build_worker_messages(
            task, subtask, {}, "", scope=scope, repair=request,
            repair_policy=ArtifactRepairPolicy(max_prompt_chars=120),
        )
        repair_block = messages[1].content.split("TARGETED ARTIFACT REPAIR", 1)[1]
        repair_block = "TARGETED ARTIFACT REPAIR" + repair_block.split(
            SCOPED_WORKER_ARTIFACT_CONTRACT, 1
        )[0]
        self.assertLessEqual(len(repair_block.strip()), 120)

    def test_rendering_is_deterministic(self) -> None:
        self.assertEqual(
            repair_request().to_worker_block(), repair_request().to_worker_block()
        )


class TestRepairWorkerPrompt(unittest.TestCase):
    def test_the_repair_prompt_asks_only_for_the_targets(self) -> None:
        prompt = worker_prompt(repair_request())
        self.assertIn("TARGETED ARTIFACT REPAIR", prompt)
        self.assertIn(LAYOUT, prompt)
        self.assertNotIn(CARD, prompt)
        self.assertNotIn(PKG, prompt)

    def test_the_repair_prompt_keeps_the_typed_envelope_contract(self) -> None:
        prompt = worker_prompt(repair_request())
        self.assertIn(SCOPED_WORKER_ARTIFACT_CONTRACT, prompt)
        self.assertIn('"artifact_id"', prompt)

    def test_the_repair_prompt_never_re_requests_the_whole_package(self) -> None:
        prompt = worker_prompt(repair_request())
        # The narrow scope block owns only the target...
        scope_block = prompt.split("ASSIGNED ARTIFACT PACKAGE")[1].split(
            "TARGETED ARTIFACT REPAIR"
        )[0]
        self.assertIn(LAYOUT, scope_block)
        self.assertNotIn("You own (complete contents required for each):\n- "
                         "src/app/App.tsx", scope_block)

    def test_the_repair_prompt_carries_no_preserved_file_contents(self) -> None:
        prompt = worker_prompt(repair_request())
        self.assertNotIn(CONTENT[APP], prompt)
        self.assertNotIn(CONTENT[ROUTER], prompt)

    def test_the_first_attempt_prompt_is_unchanged(self) -> None:
        full = worker_prompt(None, scope=scope_for("shell"))
        self.assertNotIn("TARGETED ARTIFACT REPAIR", full)
        self.assertIn("ASSIGNED ARTIFACT PACKAGE", full)
        self.assertIn(APP, full)
        self.assertIn(ROUTER, full)
        self.assertIn(LAYOUT, full)

    def test_the_legacy_unscoped_prompt_is_unchanged(self) -> None:
        prompt = worker_prompt(None, scope=None)
        self.assertNotIn("TARGETED ARTIFACT REPAIR", prompt)
        self.assertNotIn("ASSIGNED ARTIFACT PACKAGE", prompt)
        self.assertIn(WORKER_ARTIFACT_CONTRACT, prompt)


if __name__ == "__main__":
    unittest.main()
