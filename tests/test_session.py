import unittest

from query_session import QuerySessionStore
from result_operations import ResultOperations
from result_schema import ExplanationResult, QueryExecutionResult


class SessionStoreTest(unittest.TestCase):
    def test_save_and_get_named_result(self):
        store = QuerySessionStore()
        result = QueryExecutionResult(
            query={"target": "node"},
            algorithm="SS",
            total_results=1,
            filtered_results=1,
            results=[ExplanationResult(node_id=519, nodes=[519], factual=True)],
        )
        store.save("Q1", result)
        loaded = store.get("q1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.filtered_results, 1)


class CompareSavedResultTest(unittest.TestCase):
    def test_compare_saved_result_common_nodes(self):
        saved = QueryExecutionResult(
            query={"target": "saved"},
            algorithm="SS",
            total_results=2,
            filtered_results=2,
            results=[
                ExplanationResult(node_id=1, nodes=[1, 2, 3], factual=True),
                ExplanationResult(node_id=2, nodes=[2, 3, 4], factual=True),
            ],
        )
        compared = ResultOperations.compare(saved, "common_nodes", "Q1")
        self.assertIsNotNone(compared.comparison)
        payload = compared.comparison.to_dict()
        self.assertEqual(payload["type"], "common_nodes")
        self.assertIn(2, payload["common_nodes"])
        self.assertIn(3, payload["common_nodes"])

    def test_filter_saved_result_factual(self):
        saved = QueryExecutionResult(
            query={"target": "saved"},
            algorithm="SS",
            total_results=2,
            filtered_results=2,
            results=[
                ExplanationResult(node_id=1, nodes=[1, 2], factual=True, raw={"node_id": 1, "nodes": [1, 2], "factual": True}),
                ExplanationResult(node_id=2, nodes=[2, 3], factual=False, raw={"node_id": 2, "nodes": [2, 3], "factual": False}),
            ],
        )

        class FilterQuery:
            require_factual = True
            require_counterfactual = None
            fid_plus_threshold = None
            fid_minus_threshold = None

        filtered = ResultOperations.filter(saved, FilterQuery(), "Q1")
        self.assertEqual(filtered.filtered_results, 1)
        self.assertEqual(filtered.results[0].node_id, 1)

    def test_rank_saved_result_fidelity_plus(self):
        saved = QueryExecutionResult(
            query={"target": "saved"},
            algorithm="SS",
            total_results=3,
            filtered_results=3,
            results=[
                ExplanationResult(node_id=1, nodes=[1], fidelity_plus=0.2),
                ExplanationResult(node_id=2, nodes=[2], fidelity_plus=0.8),
                ExplanationResult(node_id=3, nodes=[3], fidelity_plus=0.5),
            ],
        )
        ranked = ResultOperations.rank(saved, "fidelity_plus", "Q1")
        self.assertEqual([item.node_id for item in ranked.results], [2, 3, 1])


if __name__ == "__main__":
    unittest.main()
