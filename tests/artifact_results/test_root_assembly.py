"""Phase 4C — root collection, duplicate/conflict policy and assembly status.

Covers the required six-package React-dashboard example and every prescribed
variant. Nothing here writes a file: assembly is entirely in memory.
"""

from __future__ import annotations

import unittest

from dozen.artifact_results import (
    ArtifactCandidate,
    AssembledDeliverable,
    AssemblyStatus,
    CollectionErrorCode,
    ConflictKind,
    SubtaskCollectionResult,
    assemble_deliverable,
    merge_subtask_collections,
)
from dozen.artifacts import ArtifactKind, ArtifactOperation

from .harness import (
    CONTENT,
    assemble_dashboard,
    collect_dashboard,
    collect_for,
    entry,
    envelope,
    react_work_plan,
    scope_for,
    worker_envelope_for,
)

CARD = "src/components/ui/Card.tsx"


def branch(subtask_id: str, payload, **kwargs) -> SubtaskCollectionResult:
    """One recursive producing branch's collection for a scope."""
    return collect_for(subtask_id, payload, **kwargs)


def two_branch_collection(payload_a, payload_b) -> SubtaskCollectionResult:
    """Two producing delegations inside the shared-ui package's recursion."""
    scope = scope_for("shared-ui")
    return merge_subtask_collections(
        [
            branch("shared-ui", payload_a, producer_subtask_id="inner-a",
                   recursion_depth=1),
            branch("shared-ui", payload_b, producer_subtask_id="inner-b",
                   recursion_depth=1),
        ],
        scope,
    )


class TestCompleteDashboard(unittest.TestCase):
    def test_the_six_package_dashboard_assembles_completely(self) -> None:
        assembly = assemble_dashboard()
        self.assertEqual(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(len(assembly.artifacts), 11)
        self.assertEqual(len(assembly.expected_required_artifact_ids), 11)
        self.assertEqual(assembly.missing_required_artifact_ids, ())
        self.assertEqual(assembly.conflicts, ())
        self.assertEqual(assembly.duplicates, ())
        self.assertEqual(assembly.rejected, ())
        self.assertTrue(assembly.complete)

    def test_summary_counts(self) -> None:
        summary = assemble_dashboard().summary()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["expected_required"], 11)
        self.assertEqual(summary["assembled"], 11)
        self.assertEqual(summary["missing_required"], 0)
        self.assertEqual(summary["conflicts"], 0)
        self.assertEqual(summary["validation_results"], 2)

    def test_validation_result_artifacts_are_assembled(self) -> None:
        assembly = assemble_dashboard()
        self.assertEqual(
            assembly.validation_result_artifact_ids, ("build-result", "test-result")
        )
        build = assembly.by_artifact()["build-result"]
        self.assertEqual(build.kind, ArtifactKind.BUILD_RESULT)
        self.assertEqual(build.content, CONTENT["build-result"])

    def test_content_is_preserved_exactly(self) -> None:
        by_id = assemble_dashboard().by_artifact()
        for artifact_id, content in CONTENT.items():
            self.assertEqual(by_id[artifact_id].content, content)

    def test_assembly_is_in_manifest_order_and_deterministic(self) -> None:
        first = assemble_dashboard()
        second = assemble_dashboard()
        self.assertEqual(first, second)
        self.assertEqual(
            [c.artifact_id for c in first.artifacts],
            [spec.id for spec in react_work_plan().manifest.artifacts],
        )

    def test_provenance_is_preserved(self) -> None:
        assembly = assemble_dashboard()
        index = dict(assembly.provenance_index())
        self.assertIn("application-shell", index["src/app/App.tsx"])
        self.assertIn("shared-ui", index[CARD])
        by_id = assembly.by_artifact()
        self.assertEqual(by_id["build-result"].package_id, "integration-validation")
        self.assertEqual(by_id["build-result"].subtask_id, "validation")

    def test_every_package_reports_complete(self) -> None:
        status = dict(assemble_dashboard().package_status)
        self.assertEqual(set(status.values()), {"complete"})
        self.assertEqual(len(status), 6)

    def test_serialization_round_trip(self) -> None:
        assembly = assemble_dashboard()
        self.assertEqual(
            AssembledDeliverable.from_dict(assembly.to_dict()), assembly
        )


class TestVariants(unittest.TestCase):
    def test_1_missing_router_is_partial(self) -> None:
        assembly = assemble_dashboard({
            "shell": envelope(
                entry("src/app/App.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            )
        })
        self.assertEqual(assembly.status, AssemblyStatus.PARTIAL)
        self.assertEqual(
            assembly.missing_required_artifact_ids, ("src/app/router.tsx",)
        )
        self.assertEqual(len(assembly.artifacts), 10)
        self.assertEqual(assembly.conflicts, ())
        self.assertEqual(
            dict(assembly.package_status)["application-shell"], "incomplete"
        )
        self.assertIn("src/app/router.tsx", assembly.problem_summary())

    def test_2_two_different_cards_conflict(self) -> None:
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [
                two_branch_collection(
                    envelope(entry(CARD, content="export function Card() { return 1; }")),
                    envelope(entry(CARD, content="export function Card() { return 2; }")),
                )
            ],
        )
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        self.assertEqual(len(assembly.conflicts), 1)
        conflict = assembly.conflicts[0]
        self.assertEqual(conflict.kind, ConflictKind.CONTENT)
        self.assertEqual(conflict.artifact_id, CARD)
        self.assertEqual(conflict.path, CARD)
        self.assertEqual(len(conflict.entries), 2)
        # Both hashes, both provenances, deterministic order — nothing discarded.
        self.assertEqual(len({e.content_hash for e in conflict.entries}), 2)
        self.assertEqual(
            [e.producer_subtask_id for e in conflict.entries], ["inner-a", "inner-b"]
        )
        self.assertTrue(all(e.recursion_depth == 1 for e in conflict.entries))
        # The conflicted artifact is NOT in the successful set.
        self.assertNotIn(CARD, assembly.by_artifact())
        self.assertIn(CARD, assembly.missing_required_artifact_ids)
        self.assertEqual(dict(assembly.package_status)["shared-ui"], "conflicted")

    def test_3_identical_duplicate_cards_are_recorded_not_discarded(self) -> None:
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [
                two_branch_collection(
                    worker_envelope_for("shared-ui"),
                    worker_envelope_for("shared-ui"),
                )
            ],
        )
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        self.assertEqual(len(assembly.duplicates), 1)
        duplicate = assembly.duplicates[0]
        self.assertEqual(duplicate.artifact_id, CARD)
        self.assertEqual(len(duplicate.provenance), 2)
        self.assertIn("inner-a", " ".join(duplicate.provenance))
        self.assertIn("inner-b", " ".join(duplicate.provenance))
        # It is still assembled exactly once, with the identical content.
        self.assertEqual(assembly.by_artifact()[CARD].content, CONTENT[CARD])
        self.assertEqual(len(assembly.artifacts), 11)
        self.assertTrue(any("identical content" in w for w in assembly.warnings))

    def test_4_dashboard_worker_returning_package_json_is_out_of_scope(self) -> None:
        assembly = assemble_dashboard({
            "dashboard": envelope(
                entry("src/features/dashboard/DashboardPage.tsx"),
                entry("package.json"),
            )
        })
        # package.json still assembles (its real owner produced it), but the
        # dashboard worker's claim on it is rejected, never silently accepted.
        self.assertEqual(len(assembly.rejected), 1)
        self.assertEqual(assembly.rejected[0].code, CollectionErrorCode.OUT_OF_SCOPE)
        self.assertEqual(assembly.rejected[0].subtask_id, "dashboard")
        self.assertEqual(
            assembly.by_artifact()["package.json"].package_id, "project-foundation"
        )
        self.assertEqual(len(assembly.artifacts), 11)
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)

    def test_5_shell_worker_with_the_wrong_app_path_is_rejected(self) -> None:
        assembly = assemble_dashboard({
            "shell": envelope(
                entry("src/app/App.tsx", path="src/App.tsx"),
                entry("src/app/router.tsx"),
                entry("src/components/layout/DashboardLayout.tsx"),
            )
        })
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        self.assertEqual(
            [r.code for r in assembly.rejected], [CollectionErrorCode.PATH_MISMATCH]
        )
        self.assertEqual(
            assembly.missing_required_artifact_ids, ("src/app/App.tsx",)
        )

    def test_6_validation_worker_returning_a_source_file_is_rejected(self) -> None:
        assembly = assemble_dashboard({
            "validation": envelope(
                entry("build-result"),
                entry("test-result"),
                entry("src/app/App.tsx", content="export default function App() {}"),
            )
        })
        self.assertEqual(
            [r.code for r in assembly.rejected], [CollectionErrorCode.OUT_OF_SCOPE]
        )
        self.assertEqual(assembly.rejected[0].subtask_id, "validation")
        # The shell's own App.tsx is unaffected.
        self.assertEqual(
            assembly.by_artifact()["src/app/App.tsx"].package_id, "application-shell"
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)

    def test_7_recursive_shell_producer_reaches_the_root(self) -> None:
        recursive_shell = merge_subtask_collections(
            [collect_for("shell", worker_envelope_for("shell"),
                         producer_subtask_id="shell-impl", recursion_depth=1)],
            scope_for("shell"),
        )
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shell": None}) + [recursive_shell],
        )
        self.assertEqual(assembly.status, AssemblyStatus.COMPLETE)
        self.assertEqual(len(assembly.artifacts), 11)
        shell_artifacts = [
            c for c in assembly.artifacts if c.package_id == "application-shell"
        ]
        self.assertEqual(len(shell_artifacts), 3)
        for candidate in shell_artifacts:
            self.assertEqual(candidate.subtask_id, "shell")
            self.assertEqual(candidate.producer_subtask_id, "shell-impl")
            self.assertEqual(candidate.recursion_depth, 1)


class TestConflictPolicy(unittest.TestCase):
    def test_worker_operation_override_is_rejected_at_the_trust_boundary(self) -> None:
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [
                two_branch_collection(
                    worker_envelope_for("shared-ui"),
                    envelope(entry(CARD, content="", operation="delete")),
                )
            ],
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        self.assertEqual(assembly.conflicts, ())
        self.assertEqual(
            [rejection.code for rejection in assembly.rejected],
            [CollectionErrorCode.INVALID_OPERATION],
        )
        self.assertEqual(assembly.by_artifact()[CARD].content, CONTENT[CARD])

    def test_defensive_operation_conflict_is_recorded_not_resolved(self) -> None:
        valid = collect_for("shared-ui", worker_envelope_for("shared-ui")).accepted[0]
        rogue = ArtifactCandidate(
            artifact_id=valid.artifact_id,
            path=valid.path,
            subtask_id=valid.subtask_id,
            package_id=valid.package_id,
            content="",
            kind=valid.kind,
            operation=ArtifactOperation.DELETE,
            producer_subtask_id="inner-rogue",
            recursion_depth=1,
        )
        malformed = SubtaskCollectionResult(
            subtask_id="shared-ui",
            package_ids=("shared-ui",),
            accepted=(valid, rogue),
            engaged=True,
        )
        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard({"shared-ui": None}) + [malformed]
        )
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        self.assertEqual(assembly.conflicts[0].kind, ConflictKind.OPERATION)
        self.assertNotIn(CARD, assembly.by_artifact())

    def test_wrong_mapped_subtask_is_a_fatal_ownership_conflict(self) -> None:
        valid = collect_for("shared-ui", worker_envelope_for("shared-ui")).accepted[0]
        rogue = ArtifactCandidate(
            artifact_id=valid.artifact_id,
            path=valid.path,
            subtask_id="dashboard",
            package_id=valid.package_id,
            content=valid.content,
            kind=valid.kind,
            operation=valid.operation,
        )
        malformed = SubtaskCollectionResult(
            subtask_id="dashboard",
            package_ids=(valid.package_id,),
            accepted=(rogue,),
            engaged=True,
        )
        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard({"shared-ui": None}) + [malformed]
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        self.assertEqual(assembly.conflicts[0].kind, ConflictKind.OWNERSHIP)

    def test_arrival_order_does_not_change_bounded_assembly(self) -> None:
        from dozen.artifact_results import ResultPolicy

        collections = collect_dashboard()
        policy = ResultPolicy(max_total_content_chars=200)
        forward = assemble_deliverable(react_work_plan(), collections, policy=policy)
        reverse = assemble_deliverable(
            react_work_plan(), list(reversed(collections)), policy=policy
        )
        self.assertEqual(forward, reverse)

    def test_two_same_subtask_collections_are_not_last_writer_wins(self) -> None:
        first = collect_for(
            "shared-ui",
            envelope(entry(CARD, content="first")),
            producer_subtask_id="branch-a",
            recursion_depth=1,
        )
        second = collect_for(
            "shared-ui",
            envelope(entry(CARD, content="second")),
            producer_subtask_id="branch-b",
            recursion_depth=1,
        )
        base = collect_dashboard({"shared-ui": None})
        forward = assemble_deliverable(react_work_plan(), base + [first, second])
        reverse = assemble_deliverable(react_work_plan(), base + [second, first])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.status, AssemblyStatus.CONFLICTED)
        self.assertEqual(len(forward.conflicts[0].entries), 2)

    def test_path_collision_from_malformed_data_fails_safely(self) -> None:
        plan = react_work_plan()
        # Malformed: a hand-built candidate for router.tsx aimed at App.tsx's
        # path. A valid Phase 4A manifest prevents this; the collector must not
        # silently overwrite when it happens anyway.
        rogue = ArtifactCandidate(
            artifact_id="src/app/router.tsx",
            path="src/app/App.tsx",
            subtask_id="shell",
            package_id="application-shell",
            content="export function Router() {}\n",
            kind=ArtifactKind.SOURCE_FILE,
        )
        shell = SubtaskCollectionResult(
            subtask_id="shell",
            package_ids=("application-shell",),
            accepted=(
                ArtifactCandidate(
                    artifact_id="src/app/App.tsx", path="src/app/App.tsx",
                    subtask_id="shell", package_id="application-shell",
                    content=CONTENT["src/app/App.tsx"], kind=ArtifactKind.SOURCE_FILE,
                ),
                rogue,
                ArtifactCandidate(
                    artifact_id="src/components/layout/DashboardLayout.tsx",
                    path="src/components/layout/DashboardLayout.tsx",
                    subtask_id="shell", package_id="application-shell",
                    content=CONTENT["src/components/layout/DashboardLayout.tsx"],
                    kind=ArtifactKind.SOURCE_FILE,
                ),
            ),
            engaged=True,
        )
        assembly = assemble_deliverable(
            plan, collect_dashboard({"shell": None}) + [shell]
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        collision = next(
            c for c in assembly.conflicts if c.kind is ConflictKind.PATH_COLLISION
        )
        self.assertEqual(collision.path, "src/app/App.tsx")
        self.assertEqual(len(collision.entries), 2)
        self.assertNotIn("src/app/App.tsx", assembly.by_artifact())
        self.assertNotIn("src/app/router.tsx", assembly.by_artifact())

    def test_ownership_violation_from_malformed_data_fails_safely(self) -> None:
        stolen = SubtaskCollectionResult(
            subtask_id="dashboard",
            package_ids=("dashboard-feature",),
            accepted=(
                ArtifactCandidate(
                    artifact_id=CARD, path=CARD, subtask_id="dashboard",
                    package_id="dashboard-feature",  # NOT the declared owner
                    content=CONTENT[CARD], kind=ArtifactKind.SOURCE_FILE,
                ),
                ArtifactCandidate(
                    artifact_id="src/features/dashboard/DashboardPage.tsx",
                    path="src/features/dashboard/DashboardPage.tsx",
                    subtask_id="dashboard", package_id="dashboard-feature",
                    content=CONTENT["src/features/dashboard/DashboardPage.tsx"],
                    kind=ArtifactKind.SOURCE_FILE,
                ),
            ),
            engaged=True,
        )
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"dashboard": None, "shared-ui": None}) + [stolen],
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        conflict = assembly.conflicts[0]
        self.assertEqual(conflict.kind, ConflictKind.OWNERSHIP)
        self.assertEqual(conflict.artifact_id, CARD)
        self.assertIn("shared-ui", conflict.message)

    def test_multiple_conflicts_are_all_retained_in_stable_order(self) -> None:
        shell_conflict = merge_subtask_collections(
            [
                collect_for("shell", worker_envelope_for("shell"),
                            producer_subtask_id="a", recursion_depth=1),
                collect_for("shell",
                            envelope(entry("src/app/App.tsx", content="other App"),
                                     entry("src/app/router.tsx", content="other router")),
                            producer_subtask_id="b", recursion_depth=1),
            ],
            scope_for("shell"),
        )
        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard({"shell": None}) + [shell_conflict]
        )
        self.assertEqual(assembly.status, AssemblyStatus.CONFLICTED)
        self.assertEqual(
            [c.artifact_id for c in assembly.conflicts],
            ["src/app/App.tsx", "src/app/router.tsx"],
        )
        self.assertEqual(assemble_deliverable(
            react_work_plan(), collect_dashboard({"shell": None}) + [shell_conflict]
        ), assembly)

    def test_no_latest_wins_and_no_merging(self) -> None:
        assembly = assemble_deliverable(
            react_work_plan(),
            collect_dashboard({"shared-ui": None}) + [
                two_branch_collection(
                    envelope(entry(CARD, content="first")),
                    envelope(entry(CARD, content="second")),
                )
            ],
        )
        contents = [c.content for c in assembly.artifacts]
        self.assertNotIn("first", contents)
        self.assertNotIn("second", contents)
        self.assertNotIn("firstsecond", contents)

    def test_nothing_assembled_at_all_is_failed(self) -> None:
        assembly = assemble_deliverable(
            react_work_plan(),
            [collect_for("shell", envelope())],
        )
        self.assertEqual(assembly.status, AssemblyStatus.FAILED)
        self.assertEqual(assembly.artifacts, ())
        self.assertEqual(len(assembly.missing_required_artifact_ids), 11)

    def test_optional_only_legacy_collection_is_not_complete(self) -> None:
        from dozen.artifacts import ArtifactSpec, WorkPackage
        from ..artifact_decomposition.harness import manifest_of, work_plan_of

        plan = work_plan_of(
            manifest_of([ArtifactSpec(id="b", path="b.py", required=False)]),
            [WorkPackage(id="p1", title="impl", owns=("b",))],
            (("p1", "s1"),),
        )
        collection = collect_for(
            "s1", {"artifacts": {"b.py": "print(1)\n"}}, work_plan=plan
        )
        assembly = assemble_deliverable(plan, [collection])
        self.assertEqual(assembly.status, AssemblyStatus.PARTIAL)
        self.assertEqual(assembly.missing_optional_artifact_ids, ("b",))


class TestAggregateBounds(unittest.TestCase):
    def test_total_content_is_bounded(self) -> None:
        from dozen.artifact_results import ResultPolicy

        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard(),
            policy=ResultPolicy(max_total_content_chars=200),
        )
        self.assertLess(len(assembly.artifacts), 11)
        self.assertTrue(any("aggregate limit" in w for w in assembly.warnings))

    def test_total_candidate_count_is_bounded(self) -> None:
        from dozen.artifact_results import ResultPolicy

        assembly = assemble_deliverable(
            react_work_plan(), collect_dashboard(),
            policy=ResultPolicy(max_total_candidates=3),
        )
        self.assertEqual(len(assembly.artifacts), 3)
        self.assertEqual(
            [candidate.artifact_id for candidate in assembly.artifacts],
            ["package.json", "tsconfig.json", "src/main.tsx"],
        )
        self.assertTrue(any("candidates" in w for w in assembly.warnings))


if __name__ == "__main__":
    unittest.main()
