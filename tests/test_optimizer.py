import unittest

from planner.optimizer import QueryOptimizer
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


if __name__ == "__main__":
    unittest.main()
