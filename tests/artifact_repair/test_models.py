"""Phase 4E Parts A/B/C/O — the repair vocabulary, policy, request and bounds."""

from __future__ import annotations

import dataclasses
import json
import unittest

from dozen.artifact_repair import (
    DEFAULT_REPAIR_POLICY,
    ArtifactRepairError,
    ArtifactRepairPolicy,
    ArtifactRepairReport,
    ArtifactRepairRequest,
    PreservedArtifact,
    RepairDecision,
    RepairReason,
    RepairStatus,
    RepairTarget,
    short_hash,
)
from dozen.artifact_results import content_hash

from .harness import APP, LAYOUT, ROUTER

HASH = content_hash("export default function App() {}\n")


def target(artifact_id: str = LAYOUT, **kwargs) -> RepairTarget:
    kwargs.setdefault("path", artifact_id)
    kwargs.setdefault("reason", RepairReason.TRUNCATION_SUSPECTED)
    return RepairTarget(artifact_id=artifact_id, **kwargs)


def preserved(artifact_id: str = APP, digest: str = HASH) -> PreservedArtifact:
    return PreservedArtifact(
        artifact_id=artifact_id, path=artifact_id, content_hash=digest, attempt=1
    )


def request(**kwargs) -> ArtifactRepairRequest:
    kwargs.setdefault("manifest_id", "m1")
    kwargs.setdefault("subtask_id", "shell")
    kwargs.setdefault("package_ids", ("application-shell",))
    kwargs.setdefault("attempt", 2)
    kwargs.setdefault("targets", (target(),))
    kwargs.setdefault("preserved", (preserved(),))
    return ArtifactRepairRequest(**kwargs)


class TestRepairReasons(unittest.TestCase):
    def test_every_reason_has_worker_readable_semantics(self) -> None:
        for reason in RepairReason:
            described = RepairTarget(
                artifact_id="a", path="a.tsx", reason=reason
            ).describe()
            self.assertTrue(described)
            self.assertNotIn("\n", described)

    def test_unknown_reason_falls_back_without_raising(self) -> None:
        item = RepairTarget(artifact_id="a", path="a.tsx", reason="wat")
        self.assertIs(item.reason, RepairReason.MISSING_REQUIRED)

    def test_unknown_status_falls_back_to_the_safe_verdict(self) -> None:
        decision = RepairDecision(status="something-new")
        self.assertIs(decision.status, RepairStatus.NON_REPAIRABLE)

    def test_reasons_and_statuses_are_string_enums(self) -> None:
        self.assertEqual(RepairReason.PATH_MISMATCH.value, "path_mismatch")
        self.assertEqual(RepairStatus.NO_PROGRESS.value, "no_progress")
        self.assertEqual(json.dumps({"r": RepairReason.MISSING_CONTENT}), '{"r": "missing_content"}')


class TestRepairPolicy(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        policy = DEFAULT_REPAIR_POLICY
        self.assertFalse(policy.repair_missing_optional)
        self.assertTrue(policy.repair_rejected_optional)
        self.assertFalse(policy.allow_preserved_in_response)
        self.assertTrue(policy.tolerate_identical_preserved_echo)
        self.assertTrue(policy.stop_on_no_progress)
        self.assertTrue(policy.stop_on_repeated_content_hash)
        self.assertEqual(policy.max_no_progress_attempts, 1)
        self.assertEqual(policy.max_attempts_per_artifact, 3)

    def test_the_policy_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_REPAIR_POLICY.max_targets_per_attempt = 99  # type: ignore[misc]

    def test_the_policy_declares_no_independent_round_counter(self) -> None:
        # The number of worker CALLS is the executor's existing budget. A second
        # counter here could exceed max_repair_attempts, so none may exist.
        fields = {f.name for f in dataclasses.fields(ArtifactRepairPolicy)}
        self.assertNotIn("max_repair_rounds", fields)
        self.assertNotIn("max_repair_attempts", fields)

    def test_numeric_limits_live_in_one_place(self) -> None:
        custom = ArtifactRepairPolicy(max_targets_per_attempt=1)
        self.assertEqual(custom.max_targets_per_attempt, 1)
        self.assertEqual(DEFAULT_REPAIR_POLICY.max_targets_per_attempt, 12)

    def test_invalid_numeric_limits_fail_at_configuration_time(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            ArtifactRepairPolicy(max_targets_per_attempt=0)
        with self.assertRaises(ArtifactRepairError):
            ArtifactRepairPolicy(max_prompt_chars=0)
        with self.assertRaises(ArtifactRepairError):
            ArtifactRepairPolicy(max_preserved_per_package=-1)


class TestRepairRequest(unittest.TestCase):
    def test_a_request_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request().attempt = 9  # type: ignore[misc]

    def test_a_request_must_target_something(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            request(targets=())

    def test_a_preserved_artifact_may_not_also_be_a_target(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            request(targets=(target(APP),), preserved=(preserved(APP),))

    def test_duplicate_targets_are_refused(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            request(targets=(target(LAYOUT), target(LAYOUT)))

    def test_a_preserved_artifact_requires_a_hash(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            PreservedArtifact(artifact_id=APP, path=APP, content_hash="")

    def test_a_preserved_artifact_rejects_a_noncanonical_hash(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            PreservedArtifact(
                artifact_id=APP, path=APP, content_hash="worker-supplied-value"
            )

    def test_the_request_never_carries_preserved_content(self) -> None:
        fields = {f.name for f in dataclasses.fields(PreservedArtifact)}
        self.assertNotIn("content", fields)
        self.assertNotIn("body", fields)
        serialized = json.dumps(request().to_dict())
        self.assertNotIn("export default", serialized)

    def test_targets_are_bounded(self) -> None:
        many = tuple(
            target(f"src/f{i}.tsx", path=f"src/f{i}.tsx") for i in range(40)
        )
        self.assertEqual(
            len(request(targets=many).targets),
            DEFAULT_REPAIR_POLICY.max_targets_per_attempt,
        )

    def test_preserved_entries_are_bounded(self) -> None:
        many = tuple(
            PreservedArtifact(
                artifact_id=f"src/p{i}.tsx", path=f"src/p{i}.tsx", content_hash=HASH
            )
            for i in range(60)
        )
        self.assertEqual(
            len(request(preserved=many).preserved),
            DEFAULT_REPAIR_POLICY.max_preserved_per_package,
        )

    def test_diagnostics_are_bounded_and_newline_free(self) -> None:
        req = request(diagnostics=("x\ny" * 400, "b", "c", "d", "e", "f", "g", "h", "i"))
        self.assertLessEqual(len(req.diagnostics), DEFAULT_REPAIR_POLICY.max_diagnostics)
        for item in req.diagnostics:
            self.assertNotIn("\n", item)
            self.assertLessEqual(
                len(item), DEFAULT_REPAIR_POLICY.max_diagnostic_chars
            )

    def test_evidence_is_bounded_and_newline_free(self) -> None:
        item = target(evidence="line one\nline two\n" * 90)
        self.assertNotIn("\n", item.evidence)
        self.assertLessEqual(
            len(item.evidence), DEFAULT_REPAIR_POLICY.max_diagnostic_chars
        )

    def test_hashes_are_bounded_for_display(self) -> None:
        self.assertLessEqual(
            len(short_hash(HASH)), DEFAULT_REPAIR_POLICY.max_hash_chars
        )
        self.assertTrue(short_hash(HASH).startswith("sha256:"))

    def test_tuple_canonicalization(self) -> None:
        req = request(package_ids=["application-shell"], diagnostics=["a"])
        self.assertIsInstance(req.package_ids, tuple)
        self.assertIsInstance(req.diagnostics, tuple)
        self.assertIsInstance(req.targets, tuple)

    def test_queries_are_deterministic(self) -> None:
        req = request(targets=(target(LAYOUT), target(ROUTER)))
        self.assertEqual(req.target_artifact_ids, (LAYOUT, ROUTER))
        self.assertEqual(req.target_paths, (LAYOUT, ROUTER))
        self.assertEqual(req.preserved_artifact_ids, (APP,))
        self.assertEqual(req.preserved_hashes(), {APP: HASH})
        self.assertIsNone(req.target_for("nope"))
        self.assertIsNotNone(req.target_for(LAYOUT))


class TestSerialization(unittest.TestCase):
    def test_request_round_trips(self) -> None:
        req = request(
            completion_criteria=("the app builds",),
            validation_ids=("v-build",),
            origin_output_artifact_ids=(APP, ROUTER, LAYOUT),
            diagnostics=("layout is truncated",),
        )
        self.assertEqual(ArtifactRepairRequest.from_dict(req.to_dict()), req)

    def test_request_survives_json(self) -> None:
        req = request()
        self.assertEqual(
            ArtifactRepairRequest.from_dict(json.loads(json.dumps(req.to_dict()))),
            req,
        )

    def test_missing_and_null_fields_are_tolerated(self) -> None:
        data = {
            "manifest_id": "m1",
            "subtask_id": "shell",
            "targets": [{"artifact_id": LAYOUT, "path": LAYOUT}],
            "preserved": None,
            "diagnostics": None,
            "package_ids": None,
        }
        req = ArtifactRepairRequest.from_dict(data)
        self.assertEqual(req.preserved, ())
        self.assertEqual(req.diagnostics, ())
        self.assertIs(req.targets[0].reason, RepairReason.MISSING_REQUIRED)

    def test_unknown_enum_falls_back_on_deserialization(self) -> None:
        data = request().to_dict()
        data["targets"][0]["reason"] = "invented_reason"
        self.assertIs(
            ArtifactRepairRequest.from_dict(data).targets[0].reason,
            RepairReason.MISSING_REQUIRED,
        )

    def test_explicit_false_is_preserved(self) -> None:
        req = request(targets=(target(LAYOUT, required=False),))
        self.assertFalse(req.targets[0].required)
        self.assertFalse(
            ArtifactRepairRequest.from_dict(req.to_dict()).targets[0].required
        )

    def test_no_none_string_leaks(self) -> None:
        req = ArtifactRepairRequest.from_dict({
            "manifest_id": "m1",
            "subtask_id": "shell",
            "targets": [{"artifact_id": LAYOUT, "path": None, "evidence": None}],
        })
        self.assertEqual(req.targets[0].path, "")
        self.assertEqual(req.targets[0].evidence, "")
        self.assertNotIn("None", json.dumps(req.to_dict()))

    def test_report_round_trips_and_summarizes(self) -> None:
        report = ArtifactRepairReport(
            subtask_id="shell", attempted=True, attempts_used=1,
            preserved_artifact_ids=(APP, ROUTER), repaired_artifact_ids=(LAYOUT,),
            remaining_target_ids=(), no_progress_attempts=0,
            termination=RepairStatus.SATISFIED,
        )
        self.assertEqual(ArtifactRepairReport.from_dict(report.to_dict()), report)
        summary = report.summary()
        self.assertEqual(summary["artifacts_preserved"], 2)
        self.assertEqual(summary["artifacts_repaired"], 1)
        self.assertEqual(summary["remaining_repair_targets"], 0)
        self.assertNotIn("content", json.dumps(summary))

    def test_asdict_compatibility(self) -> None:
        # The report is a plain frozen dataclass, so it drops into
        # dataclasses.asdict() of a SubTaskResult without special handling.
        data = dataclasses.asdict(
            ArtifactRepairReport(subtask_id="shell", attempted=True)
        )
        self.assertEqual(data["subtask_id"], "shell")
        self.assertTrue(data["attempted"])

    def test_decision_serializes(self) -> None:
        decision = RepairDecision(
            status=RepairStatus.REPAIRABLE, request=request(), reason="one target"
        )
        data = decision.to_dict()
        self.assertEqual(data["status"], "repairable")
        self.assertEqual(data["request"]["subtask_id"], "shell")

    def test_a_repairable_decision_requires_a_request(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            RepairDecision(status=RepairStatus.REPAIRABLE)

    def test_a_terminal_decision_may_not_carry_a_request(self) -> None:
        with self.assertRaises(ArtifactRepairError):
            RepairDecision(status=RepairStatus.SATISFIED, request=request())


class TestMutationResistance(unittest.TestCase):
    def test_target_tuples_cannot_be_mutated_through_the_request(self) -> None:
        req = request()
        with self.assertRaises(AttributeError):
            req.targets.append(target(ROUTER))  # type: ignore[attr-defined]

    def test_a_supplied_list_is_copied_not_aliased(self) -> None:
        targets = [target(LAYOUT)]
        req = request(targets=targets)
        targets.append(target(ROUTER))
        self.assertEqual(len(req.targets), 1)


if __name__ == "__main__":
    unittest.main()
