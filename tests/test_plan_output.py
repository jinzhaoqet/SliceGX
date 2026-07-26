import unittest

from query_formatter import format_result
from result_schema import QueryExecutionResult


class PlanOutputTest(unittest.TestCase):
    def test_format_plan_only_result(self):
        result = QueryExecutionResult(
            query={"target": "node", "node_ids": [519], "plan_only": True},
            algorithm="SS",
            plan={
                "logical": {
                    "target_op": "TargetNodeLookup",
                    "filter_ops": [],
                    "compare_op": "none",
                    "layer_scope": "single_layer",
                    "approximate": False,
                },
                "physical": {
                    "algorithm": "SS",
                    "executor_op": "SingleNodeExactPlan",
                    "cache_mode": "reuse_enabled",
                    "sample_ratio": 1.0,
                    "estimated_cost": 0.6,
                    "reasons": ["Single-node query maps to the single-start execution path."],
                },
            },
            total_results=0,
            filtered_results=0,
            results=[],
            time_seconds=0.001,
            cache_stats={},
            plan_only=True,
        )
        output = format_result(result)
        self.assertIn("=== SliceGX Query Plan ===", output)
        self.assertIn("Logical: target=TargetNodeLookup", output)
        self.assertIn("Plan: SingleNodeExactPlan", output)
        self.assertIn("Planning Time:", output)


if __name__ == "__main__":
    unittest.main()
