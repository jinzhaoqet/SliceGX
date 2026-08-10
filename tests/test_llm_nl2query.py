import json
import unittest

from llm.nl2query import NL2QueryService


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.responses.pop(0)


def intent_response(**overrides):
    intent = {
        "target": {"type": "class", "node_ids": [], "class_label": 1},
        "filters": {
            "factual": True,
            "counterfactual": None,
            "fidelity_plus_gt": 0.6,
            "fidelity_minus_lt": None,
            "subgraph_size_lte": None,
        },
        "layer": {"mode": "default", "index": None},
        "structure": {"include_nodes": [], "exclude_nodes": []},
        "compare_by": "common_nodes",
        "rank_by": None,
        "parameters": {
            "K": None,
            "h": None,
            "theta": None,
            "gamma": None,
            "approximate_ratio": None,
        },
    }
    intent.update(overrides)
    return json.dumps(
        {
            "needs_clarification": False,
            "clarification_question": None,
            "intent": intent,
        }
    )


class NL2QueryServiceTest(unittest.TestCase):
    def test_translates_structured_intent_and_multiple_filters(self):
        provider = FakeProvider([intent_response()])
        result = NL2QueryService(provider).translate("解释类别1中高保真的事实解释并找公共节点")
        self.assertFalse(result.needs_clarification)
        self.assertIn("WHERE FACTUAL = TRUE AND FIDELITY_PLUS > 0.6", result.query_text)
        self.assertTrue(result.query.require_factual)
        self.assertEqual(result.query.fid_plus_threshold, 0.6)
        self.assertEqual(result.query.compare_by, "common_nodes")

    def test_repairs_invalid_first_response(self):
        invalid = intent_response(
            parameters={
                "K": None,
                "h": None,
                "theta": None,
                "gamma": 2.0,
                "approximate_ratio": None,
            }
        )
        provider = FakeProvider([invalid, intent_response()])
        result = NL2QueryService(provider).translate("解释类别1")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("failed deterministic validation", provider.calls[1][-1]["content"])

    def test_returns_clarification_without_execution_query(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "needs_clarification": True,
                        "clarification_question": "你说的效果不好是预测错误还是低保真度？",
                        "intent": None,
                    },
                    ensure_ascii=False,
                )
            ]
        )
        result = NL2QueryService(provider).translate("解释效果不好的节点")
        self.assertTrue(result.needs_clarification)
        self.assertIsNone(result.query)
        self.assertIn("预测错误", result.clarification_question)

    def test_translates_pattern_projection_materialization_and_quality(self):
        response = intent_response(
            compare_by=None,
            group_by="layer",
            pattern_min_support=0.5,
            project_fields=["node_id", "layer", "fidelity_plus"],
            materialize_as="fraud_patterns",
            parameters={
                "K": None,
                "h": None,
                "theta": None,
                "gamma": None,
                "approximate_ratio": None,
                "max_error": 0.1,
                "min_confidence": 0.9,
                "time_budget_seconds": 10,
            },
        )
        result = NL2QueryService(FakeProvider([response])).translate("按层找公共模式并保存")
        self.assertIn("GROUP BY LAYER", result.query_text)
        self.assertIn("PATTERN MIN_SUPPORT 0.5", result.query_text)
        self.assertIn("MATERIALIZE AS FRAUD_PATTERNS", result.query_text)
        self.assertEqual(result.query.project_fields, ["node_id", "layer", "fidelity_plus"])
        self.assertEqual(result.query.max_error, 0.1)


if __name__ == "__main__":
    unittest.main()
