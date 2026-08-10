import json
import unittest

from llm.result2nl import Result2NLService
from result_schema import ExplanationResult, QueryExecutionResult


class CapturingProvider:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def complete(self, messages, temperature=0.0):
        self.messages = messages
        return self.response


class Result2NLServiceTest(unittest.TestCase):
    def test_narrates_only_structured_evidence(self):
        provider = CapturingProvider(
            json.dumps({"summary": "节点 17 出现在公共结构中；该结果不证明因果关系。"}, ensure_ascii=False)
        )
        result = QueryExecutionResult(
            query={"target": "all"},
            algorithm="MS",
            total_results=2,
            filtered_results=2,
            results=[
                ExplanationResult(node_id=1, nodes=[1, 17], fidelity_plus=0.8),
                ExplanationResult(node_id=2, nodes=[2, 17], fidelity_plus=0.7),
            ],
            comparison={"type": "common_nodes", "common_nodes": [17], "support": {"17": 1.0}},
        )
        summary = Result2NLService(provider).narrate(result)
        evidence = json.loads(provider.messages[-1]["content"])
        self.assertIn("节点 17", summary)
        self.assertEqual(evidence["comparison"]["common_nodes"], [17])
        self.assertFalse(evidence["evidence_policy"]["causal_claims_supported"])

    def test_marks_truncated_evidence(self):
        provider = CapturingProvider(json.dumps({"summary": "摘要"}, ensure_ascii=False))
        result = QueryExecutionResult(
            query={"target": "all"},
            algorithm="MS",
            filtered_results=2,
            results=[
                ExplanationResult(node_id=1),
                ExplanationResult(node_id=2),
            ],
        )
        evidence = Result2NLService(provider, max_results=1).build_evidence(result)
        self.assertEqual(len(evidence["results"]), 1)
        self.assertTrue(evidence["evidence_policy"]["results_truncated"])


if __name__ == "__main__":
    unittest.main()
