import unittest

from planner.optimizer import QueryOptimizer
from planner.quality import ApproximationQualityProfile, QualityCalibrationPoint
from query_parser import ExplainQuery


class QueryOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = QueryOptimizer()
        self.cache_stats = {
            "subfunction_entries": 0,
            "explanatory_entries": 0,
            "modelslice_entries": 0,
        }

    def test_single_node_maps_to_ss(self):
        query = ExplainQuery(target="node", node_ids=[519], K=6, gamma=0.5)
        plan = self.optimizer.plan(query, [519], self.cache_stats)
        self.assertEqual(plan.physical.algorithm, "SS")
        self.assertEqual(plan.physical.executor_op, "SingleNodeExactPlan")

    def test_multi_node_maps_to_ms(self):
        query = ExplainQuery(target="node", node_ids=[519, 537], K=6, gamma=0.5)
        plan = self.optimizer.plan(query, [519, 537], self.cache_stats)
        self.assertEqual(plan.physical.algorithm, "MS")

    def test_all_layers_maps_to_mm(self):
        query = ExplainQuery(target="node", node_ids=[519], layer=-1, K=6, gamma=0.5)
        plan = self.optimizer.plan(query, [519], self.cache_stats)
        self.assertEqual(plan.physical.algorithm, "MM")
        self.assertEqual(plan.logical.layer_scope, "all_layers")

    def test_all_layers_with_constraints_uses_semantics_preserving_plan(self):
        query = ExplainQuery(
            target="node",
            node_ids=[519],
            layer=-1,
            K=6,
            gamma=0.5,
            include_nodes=[517],
        )
        plan = self.optimizer.plan(query, [519], self.cache_stats)
        self.assertEqual(plan.physical.algorithm, "SS_LAYERED")

    def test_multi_node_enumerates_and_selects_shared_plan(self):
        query = ExplainQuery(target="node", node_ids=[1, 2], K=6, gamma=0.5)
        plan = self.optimizer.plan(query, [1, 2], self.cache_stats)
        algorithms = {candidate["algorithm"] for candidate in plan.physical.candidates}
        self.assertEqual(algorithms, {"SS", "MS"})
        self.assertEqual(plan.physical.shared_scope, "cross_node")

    def test_materialized_generation_rewrite_and_cache_plan(self):
        query = ExplainQuery(target="node", node_ids=[1], K=6, gamma=0.5)
        stats = dict(self.cache_stats, exact_generation_hit=True, resumable_nodes=0)
        plan = self.optimizer.plan(query, [1], stats)
        self.assertEqual(plan.physical.algorithm, "CACHE")
        self.assertIn("ReuseMaterializedGeneration", plan.logical.rewrite_rules)

    def test_quality_constrained_approximation(self):
        query = ExplainQuery(
            target="all",
            K=6,
            gamma=0.5,
            approximate=True,
            max_error=0.1,
            min_confidence=0.9,
            time_budget_seconds=10,
        )
        plan = self.optimizer.plan(query, list(range(10)), self.cache_stats)
        self.assertAlmostEqual(plan.physical.sample_ratio, 0.9)
        self.assertTrue(plan.physical.quality["quality_constrained"])
        self.assertFalse(plan.physical.quality["calibrated"])

    def test_calibrated_quality_profile_selects_lowest_feasible_ratio(self):
        optimizer = QueryOptimizer(
            quality_profile=ApproximationQualityProfile(
                [
                    QualityCalibrationPoint(0.4, 0.2, 0.8, 20),
                    QualityCalibrationPoint(0.7, 0.08, 0.93, 20),
                    QualityCalibrationPoint(1.0, 0.0, 1.0, 20),
                ]
            )
        )
        query = ExplainQuery(
            target="all",
            K=6,
            gamma=0.5,
            approximate=True,
            sample_ratio=0.3,
            max_error=0.1,
            min_confidence=0.9,
        )
        plan = optimizer.plan(query, list(range(10)), self.cache_stats)
        self.assertEqual(plan.physical.sample_ratio, 0.7)
        self.assertTrue(plan.physical.quality["calibrated"])


if __name__ == "__main__":
    unittest.main()
