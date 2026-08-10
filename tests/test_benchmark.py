import unittest
from types import SimpleNamespace

from benchmark.runner import BenchmarkRunner
from benchmark.quality import calibrate_quality_point
from benchmark.session_adapter import SessionStatementExecutor
from benchmark.workloads import ExplanationAnalyticsWorkloads
from result_schema import ExplanationResult, QueryExecutionResult


class BenchmarkTest(unittest.TestCase):
    def test_suite_covers_required_workloads(self):
        cases = ExplanationAnalyticsWorkloads.build((1, 2), class_label=0)
        categories = {case.category for case in cases}
        self.assertTrue(
            {
                "single_node",
                "class_level",
                "full_test_set",
                "multi_layer",
                "result_filter",
                "common_pattern",
                "session",
                "quality_constrained_approximation",
                "materialized_reuse",
            }.issubset(categories)
        )

    def test_runner_collects_latency_and_plan_metadata(self):
        def execute(_statement):
            return SimpleNamespace(
                filtered_results=2,
                algorithm="MS",
                plan=SimpleNamespace(physical={"estimated_cost": 1.2}),
                cache_stats={"generation_entries": 1},
            )

        summary = BenchmarkRunner(execute).run(
            ExplanationAnalyticsWorkloads.build((1, 2))[:1],
            repetitions=2,
            warmups=0,
        )
        self.assertEqual(len(summary.records), 2)
        self.assertTrue(all(record.success for record in summary.records))
        self.assertEqual(summary.records[0].estimated_cost, 1.2)

    def test_quality_calibration_compares_exact_and_approximate_results(self):
        exact = QueryExecutionResult(
            query={},
            algorithm="SS",
            results=[ExplanationResult(node_id=1, nodes=[1, 2], fidelity_plus=0.8)],
        )
        approximate = QueryExecutionResult(
            query={},
            algorithm="SS",
            results=[ExplanationResult(node_id=1, nodes=[1, 3], fidelity_plus=0.7)],
        )
        point = calibrate_quality_point(exact, [approximate, approximate], 0.5, confidence=0.95)
        self.assertEqual(point.sample_ratio, 0.5)
        self.assertEqual(point.observations, 2)
        self.assertGreater(point.observed_max_error, 0.0)

    def test_session_adapter_executes_named_result_chain(self):
        class FakeExecutor:
            def execute(self, query):
                return SimpleNamespace(query=query, filtered_results=1)

            def filter_saved_result(self, saved, query, name):
                return SimpleNamespace(source=saved, query=query, name=name, filtered_results=1)

            def rank_saved_result(self, saved, metric, name):
                return SimpleNamespace(source=saved, metric=metric, name=name, filtered_results=1)

            def compare_saved_result(self, saved, metric, name):
                return SimpleNamespace(source=saved, metric=metric, name=name, filtered_results=1)

        adapter = SessionStatementExecutor(FakeExecutor())
        adapter("LET Q1 = EXPLAIN NODE 1")
        adapter("LET Q2 = FILTER Q1 WHERE FACTUAL = TRUE")
        ranked = adapter("RANK Q2 BY FIDELITY_PLUS")
        self.assertEqual(ranked.metric, "fidelity_plus")


if __name__ == "__main__":
    unittest.main()
