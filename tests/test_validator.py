import unittest

from query_parser import ExplainQuery
from query_validator import QueryValidationError, QueryValidator


class QueryValidatorTest(unittest.TestCase):
    def setUp(self):
        self.validator = QueryValidator()

    def test_accepts_valid_query(self):
        query = ExplainQuery(target="node", node_ids=[519], gamma=0.5)
        self.validator.validate(query)

    def test_rejects_conflicting_include_exclude(self):
        query = ExplainQuery(
            target="node",
            node_ids=[519],
            include_nodes=[1],
            exclude_nodes=[1],
        )
        with self.assertRaises(QueryValidationError):
            self.validator.validate(query)

    def test_rejects_invalid_gamma(self):
        query = ExplainQuery(target="node", node_ids=[519], gamma=1.5)
        with self.assertRaises(QueryValidationError):
            self.validator.validate(query)


if __name__ == "__main__":
    unittest.main()
