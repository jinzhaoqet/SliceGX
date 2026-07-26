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


if __name__ == "__main__":
    unittest.main()
