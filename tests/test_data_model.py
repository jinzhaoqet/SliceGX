import unittest

from analytics.data_model import Graph, LayerExplanation, Model, Prediction


class DataModelTest(unittest.TestCase):
    def test_graph_model_prediction_and_layer_explanation(self):
        graph = Graph(graph_id="g1", node_ids=(1, 2), edges=((1, 2),), feature_names=("x",))
        model = Model(model_id="m1", architecture="GCN", task="node_classification", layer_count=3)
        prediction = Prediction(graph_id="g1", model_id="m1", node_id=1, label=0, score=0.8, layer=1)
        explanation = LayerExplanation(
            explanation_id="e1",
            graph_id=graph.graph_id,
            model_id=model.model_id,
            node_id=prediction.node_id,
            nodes=(1, 2),
            layer=1,
            factual=True,
            fidelity_plus=0.7,
        )
        self.assertEqual(explanation.subgraph_size, 2)
        self.assertEqual(explanation.get("layer"), 1)

    def test_graph_rejects_unknown_edge_nodes(self):
        with self.assertRaises(ValueError):
            Graph(graph_id="g1", node_ids=(1,), edges=((1, 2),))


if __name__ == "__main__":
    unittest.main()
