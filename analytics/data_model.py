from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Graph:
    graph_id: str
    node_ids: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    feature_names: Tuple[str, ...] = ()
    label_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        node_set = set(self.node_ids)
        if len(node_set) != len(self.node_ids):
            raise ValueError("Graph node_ids must be unique.")
        missing = sorted({node for edge in self.edges for node in edge if node not in node_set})
        if missing:
            raise ValueError(f"Graph edges reference unknown nodes: {missing}")


@dataclass(frozen=True)
class Model:
    model_id: str
    architecture: str
    task: str
    layer_count: int
    checkpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.layer_count <= 0:
            raise ValueError("Model layer_count must be positive.")


@dataclass(frozen=True)
class Prediction:
    graph_id: str
    model_id: str
    node_id: int
    label: int
    score: float
    layer: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Prediction score must be within [0, 1].")


@dataclass(frozen=True)
class Explanation:
    explanation_id: str
    graph_id: str
    model_id: str
    node_id: int
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...] = ()
    factual: bool = False
    counterfactual: bool = False
    fidelity_plus: float = 0.0
    fidelity_minus: float = 0.0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def subgraph_size(self) -> int:
        return len(self.nodes)

    def get(self, field_name: str) -> Any:
        aliases = {
            "id": "explanation_id",
            "layer": "layer",
            "subgraph_size": "subgraph_size",
        }
        attribute = aliases.get(field_name.lower(), field_name.lower())
        if attribute == "layer":
            return getattr(self, "layer", None)
        if not hasattr(self, attribute):
            raise KeyError(f"Unknown explanation field: {field_name}")
        return getattr(self, attribute)

    def to_record(self) -> Dict[str, Any]:
        record = {
            "explanation_id": self.explanation_id,
            "graph_id": self.graph_id,
            "model_id": self.model_id,
            "node_id": self.node_id,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "fidelity_plus": self.fidelity_plus,
            "fidelity_minus": self.fidelity_minus,
            "score": self.score,
            "subgraph_size": self.subgraph_size,
        }
        if hasattr(self, "layer"):
            record["layer"] = getattr(self, "layer")
        return record

    @classmethod
    def from_result(
        cls,
        result: Any,
        graph_id: str = "graph",
        model_id: str = "model",
        ordinal: int = 0,
    ) -> "Explanation":
        layer = getattr(result, "layer", None)
        target_cls = LayerExplanation if layer is not None else cls
        fields = {
            "explanation_id": f"{graph_id}:{model_id}:{result.node_id}:{layer}:{ordinal}",
            "graph_id": graph_id,
            "model_id": model_id,
            "node_id": int(result.node_id),
            "nodes": tuple(int(node) for node in result.nodes),
            "factual": bool(result.factual),
            "counterfactual": bool(result.counterfactual),
            "fidelity_plus": float(result.fidelity_plus),
            "fidelity_minus": float(result.fidelity_minus),
            "score": float(result.score),
        }
        if layer is not None:
            fields["layer"] = int(layer)
        return target_cls(**fields)


@dataclass(frozen=True)
class LayerExplanation(Explanation):
    layer: int = 0

    def __post_init__(self):
        if self.layer < 0:
            raise ValueError("LayerExplanation layer must be non-negative.")


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    nodes: Tuple[int, ...]
    support: float
    source_explanation_ids: Tuple[str, ...]
    group_key: Optional[Any] = None

    def __post_init__(self):
        if not 0.0 <= self.support <= 1.0:
            raise ValueError("Pattern support must be within [0, 1].")


@dataclass
class ExplanationSet:
    set_id: str
    explanations: List[Explanation] = field(default_factory=list)
    source_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.explanations)

    def __len__(self) -> int:
        return len(self.explanations)

    @classmethod
    def from_results(
        cls,
        results: Sequence[Any],
        set_id: str = "result",
        graph_id: str = "graph",
        model_id: str = "model",
        source_query: Optional[str] = None,
    ) -> "ExplanationSet":
        explanations = [
            Explanation.from_result(result, graph_id, model_id, ordinal)
            for ordinal, result in enumerate(results)
        ]
        return cls(set_id=set_id, explanations=explanations, source_query=source_query)

    def copy_with(self, explanations: Iterable[Explanation], suffix: str) -> "ExplanationSet":
        return ExplanationSet(
            set_id=f"{self.set_id}:{suffix}",
            explanations=list(explanations),
            source_query=self.source_query,
            metadata=dict(self.metadata),
        )
