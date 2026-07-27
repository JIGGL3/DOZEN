"""Phase 4F Part I — output-aware package sizing."""

from __future__ import annotations

import unittest

from dozen.artifacts import ArtifactKind
from dozen.decomposition import (
    PackageSizingPolicy,
    estimate_artifact_output_chars,
    provider_output_budget_chars,
    validate_package_sizing,
)

from ..artifact_decomposition.harness import (
    artifact,
    manifest_of,
    plan_of,
    work_plan_of,
)
from ..artifact_decomposition.harness import react_plan


def package(pid, owns, subtask, **kwargs):
    from dozen.artifacts import WorkPackage
    return WorkPackage(id=pid, title=pid, objective=f"produce {pid}",
                       owns=tuple(owns), **kwargs), subtask


def plan_with_packages(specs, packages_and_subtasks):
    manifest = manifest_of(specs)
    packages = [p for p, _s in packages_and_subtasks]
    subtask_map = tuple((p.id, s) for p, s in packages_and_subtasks)
    subtask_ids = sorted({s for _p, s in packages_and_subtasks})
    work_plan = work_plan_of(manifest, packages, subtask_map)
    return plan_of([(sid, []) for sid in subtask_ids], work_plan)


TIGHT = PackageSizingPolicy(
    max_package_output_chars=6000,
    max_substantial_files_per_package=2,
    max_small_files_per_package=3,
    substantial_file_chars=1000,
    large_artifact_chars=3000,
    # Explicit per-kind bases so "substantial" (source) and "small" (config)
    # in these fixtures land unambiguously on either side of the threshold
    # above, independent of the module's production defaults.
    source_file_chars=1500,
    config_file_chars=300,
)


class TestSizingEstimates(unittest.TestCase):
    def test_source_file_estimate_uses_the_kind_base(self) -> None:
        spec = artifact("a.py", kind=ArtifactKind.SOURCE_FILE)
        estimate = estimate_artifact_output_chars(spec)
        self.assertGreater(estimate, 0)

    def test_config_extension_estimates_smaller_than_source(self) -> None:
        source = artifact("a.py", kind=ArtifactKind.SOURCE_FILE)
        config = artifact("a.json", kind=ArtifactKind.SOURCE_FILE)
        self.assertLess(
            estimate_artifact_output_chars(config),
            estimate_artifact_output_chars(source),
        )

    def test_external_and_delete_artifacts_estimate_zero(self) -> None:
        from dozen.artifacts import ArtifactOperation
        external = artifact("a.py", external=True)
        deleted = artifact("b.py", operation=ArtifactOperation.DELETE)
        self.assertEqual(estimate_artifact_output_chars(external), 0)
        self.assertEqual(estimate_artifact_output_chars(deleted), 0)

    def test_completion_criteria_add_a_bounded_bonus(self) -> None:
        bare = artifact("a.py", kind=ArtifactKind.SOURCE_FILE)
        detailed = artifact(
            "b.py", kind=ArtifactKind.SOURCE_FILE,
            completion_criteria=("must handle edge case one",
                                 "must handle edge case two"),
        )
        self.assertGreater(
            estimate_artifact_output_chars(detailed),
            estimate_artifact_output_chars(bare),
        )


class TestOneLargeFile(unittest.TestCase):
    def test_oversized_single_file_requires_logical_redesign(self) -> None:
        specs = [artifact("big.py", kind=ArtifactKind.SOURCE_FILE,
                          completion_criteria=tuple(f"c{i}" for i in range(20)))]
        pkg, sid = package("p1", ["big.py"], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertFalse(check.ok)
        self.assertIn("smaller logical modules", check.feedback())
        self.assertIn("never fragment", check.feedback())


class TestSeveralModerateFiles(unittest.TestCase):
    def test_two_moderate_files_fit_one_package(self) -> None:
        specs = [artifact(f"m{i}.py", kind=ArtifactKind.SOURCE_FILE)
                for i in range(2)]
        pkg, sid = package("p1", [s.id for s in specs], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertTrue(check.ok)


class TestManySmallConfigFiles(unittest.TestCase):
    def test_bounded_shared_config_package_passes(self) -> None:
        specs = [artifact(f"c{i}.json", kind=ArtifactKind.CONFIG_FILE)
                for i in range(3)]
        pkg, sid = package("p1", [s.id for s in specs], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertTrue(check.ok)

    def test_too_many_small_files_fails(self) -> None:
        specs = [artifact(f"c{i}.json", kind=ArtifactKind.CONFIG_FILE)
                for i in range(6)]
        pkg, sid = package("p1", [s.id for s in specs], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertFalse(check.ok)
        self.assertIn("small files", check.feedback())


class TestLargeSubsystemSplit(unittest.TestCase):
    def test_too_many_substantial_files_fails(self) -> None:
        specs = [artifact(f"s{i}.py", kind=ArtifactKind.SOURCE_FILE)
                for i in range(4)]
        pkg, sid = package("p1", [s.id for s in specs], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertFalse(check.ok)

    def test_splitting_into_dependency_aware_packages_passes(self) -> None:
        specs = [artifact(f"s{i}.py", kind=ArtifactKind.SOURCE_FILE)
                for i in range(4)]
        pkg1, sid1 = package("p1", ["s0.py", "s1.py"], "s1")
        pkg2, sid2 = package("p2", ["s2.py", "s3.py"], "s2",
                             depends_on=("p1",))
        plan = plan_with_packages(specs, [(pkg1, sid1), (pkg2, sid2)])
        plan.subtasks[1].depends_on = [plan.subtasks[0].id]
        check = validate_package_sizing(plan, policy=TIGHT)
        self.assertTrue(check.ok)


class TestStableGroupingAndLimits(unittest.TestCase):
    def test_estimate_is_deterministic(self) -> None:
        specs = [artifact("a.py", kind=ArtifactKind.SOURCE_FILE)]
        pkg, sid = package("p1", ["a.py"], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        first = validate_package_sizing(plan, policy=TIGHT)
        second = validate_package_sizing(plan, policy=TIGHT)
        self.assertEqual(first.fatal, second.fatal)
        self.assertEqual(first.advisory, second.advisory)

    def test_one_character_over_the_boundary_fails(self) -> None:
        # Two config files (kept "small" so only the aggregate-budget rule is
        # in play) at a known margin-adjusted total: a budget exactly at that
        # total fits; one character less fails — the boundary is exact.
        specs = [artifact("a.json", kind=ArtifactKind.CONFIG_FILE),
                artifact("b.json", kind=ArtifactKind.CONFIG_FILE)]
        total = sum(estimate_artifact_output_chars(spec, TIGHT) for spec in specs)
        margin = TIGHT.with_margin(total)

        def policy_with_budget(budget: int) -> PackageSizingPolicy:
            return PackageSizingPolicy(
                max_package_output_chars=budget,
                max_substantial_files_per_package=TIGHT.max_substantial_files_per_package,
                max_small_files_per_package=TIGHT.max_small_files_per_package,
                substantial_file_chars=TIGHT.substantial_file_chars,
                large_artifact_chars=TIGHT.large_artifact_chars,
                source_file_chars=TIGHT.source_file_chars,
                config_file_chars=TIGHT.config_file_chars,
            )

        pkg, sid = package("p1", ["a.json", "b.json"], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        self.assertTrue(
            validate_package_sizing(plan, policy=policy_with_budget(margin)).ok
        )
        self.assertFalse(
            validate_package_sizing(plan, policy=policy_with_budget(margin - 1)).ok
        )

    def test_package_count_limit_is_not_exceeded_by_sizing_alone(self) -> None:
        # Sizing never invents new packages; it only judges the ones given.
        check = validate_package_sizing(react_plan())
        self.assertTrue(check.ok or check.advisory)

    def test_selected_provider_budget_is_applied_to_combined_scope(self) -> None:
        specs = [artifact(f"m{i}.py", kind=ArtifactKind.SOURCE_FILE)
                 for i in range(2)]
        pkg, sid = package("p1", [s.id for s in specs], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        self.assertTrue(validate_package_sizing(plan, policy=TIGHT).ok)
        check = validate_package_sizing(
            plan, policy=TIGHT, subtask_output_budgets={"s1": 3000}
        )
        self.assertFalse(check.ok)
        self.assertIn("selected provider", check.feedback())

    def test_provider_token_budget_conversion_is_bounded(self) -> None:
        self.assertEqual(provider_output_budget_chars(1000, TIGHT), 4500)
        self.assertEqual(provider_output_budget_chars(100000, TIGHT), 6000)


class TestProviderBudgetIntegration(unittest.TestCase):
    def test_planner_replans_once_for_selected_provider_budget(self) -> None:
        from dozen import AgentPool, AgentSpec
        from dozen.intent import resolve_contract
        from dozen.models import Task
        from dozen.planner import PlanError, Planner
        from ..artifact_decomposition.harness import (
            QueuedPlannerClient,
            react_plan_json,
        )

        payload = react_plan_json()
        client = QueuedPlannerClient(payload, payload)
        agent = AgentSpec(
            name="alpha", provider="openai", model="small",
            strengths={"coding": 1.0}, max_tokens=1000,
        )
        planner = Planner(client, agent, AgentPool([agent]))
        task = Task(
            prompt="Build a React dashboard with tests.",
            contract=resolve_contract("Build a React dashboard with tests."),
        )
        with self.assertRaises(PlanError) as ctx:
            planner.plan(task, 0, 2)
        self.assertEqual(client.calls, 2)
        self.assertIn("selected provider", str(ctx.exception).lower())

    def test_executor_never_calls_an_undersized_provider(self) -> None:
        from dozen.intent import resolve_contract
        from dozen.models import Task
        from ..artifact_decomposition.harness import react_plan
        from ..artifact_decomposition.test_execution import (
            RecordingClient,
            make_executor,
        )

        client = RecordingClient()
        executor = make_executor(client)
        executor.pool.get("alpha").max_tokens = 100
        results = executor.run(
            Task(
                prompt="Build a React dashboard.",
                contract=resolve_contract("Build a React dashboard."),
            ),
            react_plan(),
            depth=0,
        )
        self.assertEqual(client.worker_prompts, [])
        self.assertTrue(any("safe output budget" in result.error for result in results))


class TestNoSourceFragmentation(unittest.TestCase):
    def test_sizing_never_recommends_splitting_a_single_artifact(self) -> None:
        # There is only ONE way to satisfy an oversized single-artifact
        # package: give it its own package. Sizing never asks for less than
        # the complete artifact, and the fatal/advisory messages never suggest
        # partial file delivery.
        specs = [artifact("huge.py", kind=ArtifactKind.SOURCE_FILE,
                          completion_criteria=tuple(f"c{i}" for i in range(30)))]
        pkg, sid = package("p1", ["huge.py"], "s1")
        plan = plan_with_packages(specs, [(pkg, sid)])
        check = validate_package_sizing(plan, policy=TIGHT)
        combined = " ".join(check.fatal) + " ".join(check.advisory)
        self.assertIn("never fragment", combined.lower())
        self.assertNotIn("partial", combined.lower())


if __name__ == "__main__":
    unittest.main()
