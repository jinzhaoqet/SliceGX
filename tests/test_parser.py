import unittest

from query_parser import QueryParser


class QueryParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser()

    def test_parse_basic_query(self):
        query = self.parser.parse("EXPLAIN NODE 519 WITH K 6")
        self.assertEqual(query.target, "node")
        self.assertEqual(query.node_ids, [519])
        self.assertEqual(query.K, 6)
        self.assertFalse(query.plan_only)

    def test_parse_explain_plan_prefix(self):
        query = self.parser.parse("EXPLAIN PLAN FOR EXPLAIN NODE 519 WITH K 6")
        self.assertTrue(query.plan_only)
        self.assertEqual(query.target, "node")
        self.assertEqual(query.node_ids, [519])
        self.assertEqual(query.K, 6)

    def test_parse_approximate_query(self):
        query = self.parser.parse("EXPLAIN NODE 556 WITH APPROXIMATE 0.3")
        self.assertTrue(query.approximate)
        self.assertAlmostEqual(query.sample_ratio, 0.3)

    def test_parse_inline_rank_query(self):
        query = self.parser.parse("EXPLAIN ALL WHERE FACTUAL = TRUE RANK BY FIDELITY_PLUS")
        self.assertEqual(query.target, "all")
        self.assertTrue(query.require_factual)
        self.assertEqual(query.rank_by, "fidelity_plus")

    def test_parse_result_algebra_and_quality_constraints(self):
        query = self.parser.parse(
            "EXPLAIN ALL WHERE FACTUAL = TRUE AND FIDELITY_PLUS > 0.6 "
            "GROUP BY LAYER PATTERN MIN_SUPPORT 0.5 "
            "PROJECT NODE_ID,LAYER,FIDELITY_PLUS MATERIALIZE AS Q1 "
            "WITH MAX_ERROR 0.1 WITH MIN_CONFIDENCE 0.9 WITH TIME_BUDGET 10"
        )
        self.assertTrue(query.require_factual)
        self.assertEqual(query.fid_plus_threshold, 0.6)
        self.assertEqual(query.group_by, "layer")
        self.assertEqual(query.pattern_min_support, 0.5)
        self.assertEqual(query.project_fields, ["node_id", "layer", "fidelity_plus"])
        self.assertEqual(query.materialize_as, "Q1")
        self.assertEqual(query.max_error, 0.1)
        self.assertEqual(query.min_confidence, 0.9)
        self.assertEqual(query.time_budget_seconds, 10.0)

    def test_rejects_unknown_tokens_and_wrong_comparators(self):
        with self.assertRaises(ValueError):
            self.parser.parse("EXPLAIN NODE 519 UNKNOWN")
        with self.assertRaises(ValueError):
            self.parser.parse("EXPLAIN NODE 519 WHERE FIDELITY_PLUS < 0.5")


if __name__ == "__main__":
    unittest.main()
