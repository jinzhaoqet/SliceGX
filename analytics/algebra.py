import json
import operator
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from analytics.data_model import Explanation, ExplanationSet, LayerExplanation, Pattern


@dataclass(frozen=True)
class Explain:
    target_type: str
    target_values: Tuple[int, ...] = ()
    layer_scope: Union[int, str] = 0

    input_type: str = field(default="Graph×Model", init=False)
    output_type: str = field(default="ExplanationSet", init=False)


@dataclass(frozen=True)
class Filter:
    field_name: str
    comparator: str
    value: Any

    input_type: str = field(default="ExplanationSet", init=False)
    output_type: str = field(default="ExplanationSet", init=False)


@dataclass(frozen=True)
class Project:
    fields: Tuple[str, ...]

    input_type: str = field(default="ExplanationSet", init=False)
    output_type: str = field(default="Table", init=False)


@dataclass(frozen=True)
class Rank:
    field_name: str
    descending: bool = True
    limit: Optional[int] = None

    input_type: str = field(default="ExplanationSet", init=False)
    output_type: str = field(default="ExplanationSet", init=False)


@dataclass(frozen=True)
class Compare:
    metric: str

    input_type: str = field(default="ExplanationSet", init=False)
    output_type: str = field(default="Comparison", init=False)


@dataclass(frozen=True)
class GroupPattern:
    group_by: Optional[str] = None
    min_support: float = 0.5
    max_pattern_size: int = 2

    input_type: str = field(default="ExplanationSet", init=False)
    output_type: str = field(default="PatternSet", init=False)


@dataclass(frozen=True)
class Materialize:
    name: str

    input_type: str = field(default="AnyResult", init=False)
    output_type: str = field(default="MaterializedResult", init=False)


@dataclass
class Table:
    fields: Tuple[str, ...]
    rows: List[Dict[str, Any]]


@dataclass
class Comparison:
    metric: str
    payload: Dict[str, Any]


@dataclass
class PatternSet:
    patterns: List[Pattern]
    group_by: Optional[str] = None


AlgebraValue = Union[ExplanationSet, Table, Comparison, PatternSet]


class MaterializationCatalog:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory is not None else None
        self._values: Dict[str, AlgebraValue] = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, value: AlgebraValue) -> None:
        key = name.upper()
        self._values[key] = value
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            path.write_text(json.dumps(self._serialize(value), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, name: str) -> Optional[AlgebraValue]:
        key = name.upper()
        if key in self._values:
            return self._values[key]
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        value = self._deserialize(json.loads(path.read_text(encoding="utf-8")))
        self._values[key] = value
        return value

    def names(self) -> List[str]:
        return sorted(self._values)

    @staticmethod
    def _serialize(value: AlgebraValue) -> Dict[str, Any]:
        if isinstance(value, ExplanationSet):
            return {
                "type": "ExplanationSet",
                "set_id": value.set_id,
                "source_query": value.source_query,
                "rows": [item.to_record() for item in value],
            }
        if isinstance(value, Table):
            return {"type": "Table", "fields": list(value.fields), "rows": value.rows}
        if isinstance(value, Comparison):
            return {"type": "Comparison", "metric": value.metric, "payload": value.payload}
        return {
            "type": "PatternSet",
            "group_by": value.group_by,
            "patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "nodes": list(pattern.nodes),
                    "support": pattern.support,
                    "source_explanation_ids": list(pattern.source_explanation_ids),
                    "group_key": pattern.group_key,
                }
                for pattern in value.patterns
            ],
        }

    @staticmethod
    def _deserialize(payload: Dict[str, Any]) -> AlgebraValue:
        value_type = payload.get("type")
        if value_type == "ExplanationSet":
            explanations = []
            for row in payload.get("rows", []):
                explanation_type = LayerExplanation if row.get("layer") is not None else Explanation
                kwargs = {
                    "explanation_id": row["explanation_id"],
                    "graph_id": row["graph_id"],
                    "model_id": row["model_id"],
                    "node_id": row["node_id"],
                    "nodes": tuple(row.get("nodes", [])),
                    "edges": tuple(tuple(edge) for edge in row.get("edges", [])),
                    "factual": row.get("factual", False),
                    "counterfactual": row.get("counterfactual", False),
                    "fidelity_plus": row.get("fidelity_plus", 0.0),
                    "fidelity_minus": row.get("fidelity_minus", 0.0),
                    "score": row.get("score", 0.0),
                }
                if explanation_type is LayerExplanation:
                    kwargs["layer"] = row["layer"]
                explanations.append(explanation_type(**kwargs))
            return ExplanationSet(
                set_id=payload["set_id"],
                explanations=explanations,
                source_query=payload.get("source_query"),
            )
        if value_type == "Table":
            return Table(fields=tuple(payload.get("fields", [])), rows=list(payload.get("rows", [])))
        if value_type == "Comparison":
            return Comparison(metric=payload["metric"], payload=dict(payload.get("payload", {})))
        if value_type == "PatternSet":
            return PatternSet(
                group_by=payload.get("group_by"),
                patterns=[
                    Pattern(
                        pattern_id=item["pattern_id"],
                        nodes=tuple(item.get("nodes", [])),
                        support=item["support"],
                        source_explanation_ids=tuple(item.get("source_explanation_ids", [])),
                        group_key=item.get("group_key"),
                    )
                    for item in payload.get("patterns", [])
                ],
            )
        raise ValueError(f"Unsupported materialized value type: {value_type}")


class AlgebraExecutor:
    COMPARATORS = {
        "=": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }

    def __init__(self, catalog: Optional[MaterializationCatalog] = None):
        self.catalog = catalog or MaterializationCatalog()

    def execute(self, value: AlgebraValue, operation: Any) -> AlgebraValue:
        if isinstance(operation, Filter):
            return self._filter(self._require_explanation_set(value), operation)
        if isinstance(operation, Project):
            return self._project(self._require_explanation_set(value), operation)
        if isinstance(operation, Rank):
            return self._rank(self._require_explanation_set(value), operation)
        if isinstance(operation, Compare):
            return self._compare(self._require_explanation_set(value), operation)
        if isinstance(operation, GroupPattern):
            return self._group_pattern(self._require_explanation_set(value), operation)
        if isinstance(operation, Materialize):
            self.catalog.put(operation.name, value)
            return value
        raise TypeError(f"Unsupported algebra operation: {type(operation).__name__}")

    def pipeline(self, value: AlgebraValue, operations: Sequence[Any]) -> AlgebraValue:
        current = value
        for operation_item in operations:
            current = self.execute(current, operation_item)
        return current

    def _filter(self, value: ExplanationSet, operation_item: Filter) -> ExplanationSet:
        comparator = self.COMPARATORS.get(operation_item.comparator)
        if comparator is None:
            raise ValueError(f"Unsupported comparator: {operation_item.comparator}")
        selected = [
            explanation
            for explanation in value
            if comparator(explanation.get(operation_item.field_name), operation_item.value)
        ]
        return value.copy_with(selected, f"filter-{operation_item.field_name}")

    @staticmethod
    def _project(value: ExplanationSet, operation_item: Project) -> Table:
        rows = []
        for explanation in value:
            record = explanation.to_record()
            unknown = [field_name for field_name in operation_item.fields if field_name not in record]
            if unknown:
                raise KeyError(f"Unknown projection fields: {unknown}")
            rows.append({field_name: record[field_name] for field_name in operation_item.fields})
        return Table(fields=operation_item.fields, rows=rows)

    @staticmethod
    def _rank(value: ExplanationSet, operation_item: Rank) -> ExplanationSet:
        ranked = sorted(
            value.explanations,
            key=lambda explanation: explanation.get(operation_item.field_name),
            reverse=operation_item.descending,
        )
        if operation_item.limit is not None:
            ranked = ranked[: operation_item.limit]
        return value.copy_with(ranked, f"rank-{operation_item.field_name}")

    @staticmethod
    def _compare(value: ExplanationSet, operation_item: Compare) -> Comparison:
        if not value.explanations:
            return Comparison(metric=operation_item.metric, payload={"count": 0})
        if operation_item.metric == "fidelity_plus":
            best = max(value, key=lambda explanation: explanation.fidelity_plus)
            return Comparison(
                metric=operation_item.metric,
                payload={
                    "best_explanation_id": best.explanation_id,
                    "best_node": best.node_id,
                    "best_fidelity_plus": best.fidelity_plus,
                    "best_nodes": list(best.nodes),
                },
            )
        if operation_item.metric == "common_nodes":
            counts = Counter(node for explanation in value for node in set(explanation.nodes))
            total = len(value)
            return Comparison(
                metric=operation_item.metric,
                payload={
                    "total_explanations": total,
                    "support": {
                        str(node): round(count / total, 6)
                        for node, count in counts.most_common()
                    },
                },
            )
        raise ValueError(f"Unsupported comparison metric: {operation_item.metric}")

    @staticmethod
    def _group_pattern(value: ExplanationSet, operation_item: GroupPattern) -> PatternSet:
        if not 0.0 < operation_item.min_support <= 1.0:
            raise ValueError("min_support must be within (0, 1].")
        if operation_item.max_pattern_size <= 0:
            raise ValueError("max_pattern_size must be positive.")
        groups: Dict[Any, List[Explanation]] = defaultdict(list)
        if operation_item.group_by is None:
            groups[None] = list(value.explanations)
        else:
            for explanation in value:
                groups[explanation.get(operation_item.group_by)].append(explanation)

        patterns = []
        for group_key, explanations in groups.items():
            if not explanations:
                continue
            candidate_counts = Counter()
            for explanation in explanations:
                unique_nodes = sorted(set(explanation.nodes))
                for size in range(1, min(operation_item.max_pattern_size, len(unique_nodes)) + 1):
                    candidate_counts.update(combinations(unique_nodes, size))
            for node_pattern, count in candidate_counts.most_common():
                support = count / len(explanations)
                if support >= operation_item.min_support:
                    pattern_nodes = tuple(node_pattern)
                    sources = tuple(
                        explanation.explanation_id
                        for explanation in explanations
                        if set(pattern_nodes).issubset(explanation.nodes)
                    )
                    rendered = "-".join(str(node) for node in pattern_nodes)
                    patterns.append(
                        Pattern(
                            pattern_id=f"{value.set_id}:{group_key}:{rendered}",
                            nodes=pattern_nodes,
                            support=support,
                            source_explanation_ids=sources,
                            group_key=group_key,
                        )
                    )
        return PatternSet(patterns=patterns, group_by=operation_item.group_by)

    @staticmethod
    def _require_explanation_set(value: AlgebraValue) -> ExplanationSet:
        if not isinstance(value, ExplanationSet):
            raise TypeError(f"Operation requires ExplanationSet, received {type(value).__name__}.")
        return value
