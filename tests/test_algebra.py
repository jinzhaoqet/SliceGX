import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analytics.algebra import (
    AlgebraExecutor,
    Compare,
    Filter,
    GroupPattern,
    MaterializationCatalog,
    Materialize,
    Project,
    Rank,
)
from analytics.data_model import ExplanationSet, LayerExplanation


def explanation(explanation_id, node_id, nodes, layer, fidelity, factual=True):
    return LayerExplanation(
        explanation_id=explanation_id,
        graph_id="g",
        model_id="m",
        node_id=node_id,
        nodes=tuple(nodes),
        layer=layer,
        fidelity_plus=fidelity,
        factual=factual,
    )


class AlgebraExecutorTest(unittest.TestCase):
    def setUp(self):
        self.value = ExplanationSet(
            set_id="Q1",
            explanations=[
                explanation("e1", 1, [1, 17], 0, 0.8),
                explanation("e2", 2, [2, 17], 0, 0.6),
                explanation("e3", 3, [3, 23], 1, 0.4, factual=False),
            ],
        )
        self.catalog = MaterializationCatalog()
        self.executor = AlgebraExecutor(self.catalog)

    def test_filter_rank_project(self):
        filtered = self.executor.execute(self.value, Filter("factual", "=", True))
        ranked = self.executor.execute(filtered, Rank("fidelity_plus"))
        projected = self.executor.execute(ranked, Project(("node_id", "layer", "fidelity_plus")))
        self.assertEqual([row["node_id"] for row in projected.rows], [1, 2])

    def test_compare_and_group_pattern(self):
        compared = self.executor.execute(self.value, Compare("fidelity_plus"))
        patterns = self.executor.execute(self.value, GroupPattern(group_by="layer", min_support=0.5))
        self.assertEqual(compared.payload["best_node"], 1)
        layer_zero_patterns = {pattern.nodes for pattern in patterns.patterns if pattern.group_key == 0}
        self.assertIn((17,), layer_zero_patterns)

    def test_materialize(self):
        returned = self.executor.execute(self.value, Materialize("base"))
        self.assertIs(returned, self.value)
        self.assertIs(self.catalog.get("BASE"), self.value)

    def test_persistent_materialization_round_trip(self):
        with TemporaryDirectory() as directory:
            catalog = MaterializationCatalog(Path(directory))
            AlgebraExecutor(catalog).execute(self.value, Materialize("base"))
            restored = MaterializationCatalog(Path(directory)).get("BASE")
            self.assertEqual(restored.set_id, "Q1")
            self.assertEqual([item.node_id for item in restored], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
