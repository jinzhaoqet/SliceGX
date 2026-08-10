from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class IntentValidationError(ValueError):
    pass


@dataclass
class QueryIntent:
    target_type: str = "node"
    node_ids: List[int] = field(default_factory=list)
    class_label: Optional[int] = None
    layer: Optional[int] = None
    all_layers: bool = False
    require_factual: Optional[bool] = None
    require_counterfactual: Optional[bool] = None
    fidelity_plus_gt: Optional[float] = None
    fidelity_minus_lt: Optional[float] = None
    subgraph_size_lte: Optional[int] = None
    include_nodes: List[int] = field(default_factory=list)
    exclude_nodes: List[int] = field(default_factory=list)
    compare_by: Optional[str] = None
    rank_by: Optional[str] = None
    project_fields: List[str] = field(default_factory=list)
    group_by: Optional[str] = None
    pattern_min_support: Optional[float] = None
    materialize_as: Optional[str] = None
    K: Optional[int] = None
    h: Optional[float] = None
    theta: Optional[float] = None
    gamma: Optional[float] = None
    approximate_ratio: Optional[float] = None
    max_error: Optional[float] = None
    min_confidence: Optional[float] = None
    time_budget_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QueryIntent":
        if not isinstance(payload, dict):
            raise IntentValidationError("QueryIntent must be a JSON object.")
        target = payload.get("target", {})
        filters = payload.get("filters", {})
        layer = payload.get("layer", {})
        structure = payload.get("structure", {})
        parameters = payload.get("parameters", {})
        for name, value in (
            ("target", target),
            ("filters", filters),
            ("layer", layer),
            ("structure", structure),
            ("parameters", parameters),
        ):
            if not isinstance(value, dict):
                raise IntentValidationError(f"{name} must be a JSON object.")

        target_type = str(target.get("type", "node")).lower()
        node_ids = cls._int_list(target.get("node_ids", []), "target.node_ids")
        class_label = cls._optional_int(target.get("class_label"), "target.class_label")
        layer_mode = str(layer.get("mode", "default")).lower()
        layer_index = cls._optional_int(layer.get("index"), "layer.index")
        if layer_mode not in {"default", "single", "all"}:
            raise IntentValidationError("layer.mode must be default, single, or all.")
        if layer_mode == "single" and layer_index is None:
            raise IntentValidationError("layer.index is required when layer.mode is single.")

        return cls(
            target_type=target_type,
            node_ids=node_ids,
            class_label=class_label,
            layer=layer_index if layer_mode == "single" else None,
            all_layers=layer_mode == "all",
            require_factual=cls._optional_bool(filters.get("factual"), "filters.factual"),
            require_counterfactual=cls._optional_bool(
                filters.get("counterfactual"), "filters.counterfactual"
            ),
            fidelity_plus_gt=cls._optional_float(
                filters.get("fidelity_plus_gt"), "filters.fidelity_plus_gt"
            ),
            fidelity_minus_lt=cls._optional_float(
                filters.get("fidelity_minus_lt"), "filters.fidelity_minus_lt"
            ),
            subgraph_size_lte=cls._optional_int(
                filters.get("subgraph_size_lte"), "filters.subgraph_size_lte"
            ),
            include_nodes=cls._int_list(structure.get("include_nodes", []), "structure.include_nodes"),
            exclude_nodes=cls._int_list(structure.get("exclude_nodes", []), "structure.exclude_nodes"),
            compare_by=cls._optional_string(payload.get("compare_by")),
            rank_by=cls._optional_string(payload.get("rank_by")),
            project_fields=cls._string_list(payload.get("project_fields", []), "project_fields"),
            group_by=cls._optional_string(payload.get("group_by")),
            pattern_min_support=cls._optional_float(
                payload.get("pattern_min_support"), "pattern_min_support"
            ),
            materialize_as=cls._optional_string(payload.get("materialize_as")),
            K=cls._optional_int(parameters.get("K"), "parameters.K"),
            h=cls._optional_float(parameters.get("h"), "parameters.h"),
            theta=cls._optional_float(parameters.get("theta"), "parameters.theta"),
            gamma=cls._optional_float(parameters.get("gamma"), "parameters.gamma"),
            approximate_ratio=cls._optional_float(
                parameters.get("approximate_ratio"), "parameters.approximate_ratio"
            ),
            max_error=cls._optional_float(parameters.get("max_error"), "parameters.max_error"),
            min_confidence=cls._optional_float(
                parameters.get("min_confidence"), "parameters.min_confidence"
            ),
            time_budget_seconds=cls._optional_float(
                parameters.get("time_budget_seconds"), "parameters.time_budget_seconds"
            ),
        )

    @staticmethod
    def _int_list(value: Any, name: str) -> List[int]:
        if value is None:
            return []
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise IntentValidationError(f"{name} must be a list of integers.")
        return list(value)

    @staticmethod
    def _optional_bool(value: Any, name: str) -> Optional[bool]:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise IntentValidationError(f"{name} must be true, false, or null.")
        return value

    @staticmethod
    def _optional_int(value: Any, name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise IntentValidationError(f"{name} must be an integer or null.")
        return value

    @staticmethod
    def _optional_float(value: Any, name: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IntentValidationError(f"{name} must be numeric or null.")
        return float(value)

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value).lower()

    @staticmethod
    def _string_list(value: Any, name: str) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise IntentValidationError(f"{name} must be a list of strings.")
        return [item.lower() for item in value]


class QueryIntentCompiler:
    def compile(self, intent: QueryIntent) -> str:
        clauses = [self._target_clause(intent)]
        filters = []
        if intent.require_factual is not None:
            filters.append(f"FACTUAL = {str(intent.require_factual).upper()}")
        if intent.require_counterfactual is not None:
            filters.append(f"COUNTERFACTUAL = {str(intent.require_counterfactual).upper()}")
        if intent.fidelity_plus_gt is not None:
            filters.append(f"FIDELITY_PLUS > {intent.fidelity_plus_gt:g}")
        if intent.fidelity_minus_lt is not None:
            filters.append(f"FIDELITY_MINUS < {intent.fidelity_minus_lt:g}")
        if intent.subgraph_size_lte is not None:
            filters.append(f"SUBGRAPH_SIZE <= {intent.subgraph_size_lte}")
        if filters:
            clauses.append("WHERE " + " AND ".join(filters))

        if intent.all_layers:
            clauses.append("AT ALL LAYERS")
        elif intent.layer is not None:
            clauses.append(f"AT LAYER {intent.layer}")
        if intent.include_nodes:
            clauses.append("INCLUDE " + ",".join(str(node) for node in intent.include_nodes))
        if intent.exclude_nodes:
            clauses.append("EXCLUDE " + ",".join(str(node) for node in intent.exclude_nodes))
        if intent.compare_by:
            clauses.append(f"COMPARE BY {intent.compare_by.upper()}")
        if intent.rank_by:
            clauses.append(f"RANK BY {intent.rank_by.upper()}")
        if intent.group_by:
            clauses.append(f"GROUP BY {intent.group_by.upper()}")
        if intent.pattern_min_support is not None:
            clauses.append(f"PATTERN MIN_SUPPORT {intent.pattern_min_support:g}")
        if intent.project_fields:
            clauses.append("PROJECT " + ",".join(field_name.upper() for field_name in intent.project_fields))
        if intent.materialize_as:
            clauses.append(f"MATERIALIZE AS {intent.materialize_as.upper()}")
        for name, value in (
            ("K", intent.K),
            ("H", intent.h),
            ("THETA", intent.theta),
            ("GAMMA", intent.gamma),
        ):
            if value is not None:
                rendered = f"{value:g}" if isinstance(value, float) else str(value)
                clauses.append(f"WITH {name} {rendered}")
        if intent.approximate_ratio is not None:
            clauses.append(f"WITH APPROXIMATE {intent.approximate_ratio:g}")
        if intent.max_error is not None:
            clauses.append(f"WITH MAX_ERROR {intent.max_error:g}")
        if intent.min_confidence is not None:
            clauses.append(f"WITH MIN_CONFIDENCE {intent.min_confidence:g}")
        if intent.time_budget_seconds is not None:
            clauses.append(f"WITH TIME_BUDGET {intent.time_budget_seconds:g}")
        return " ".join(clauses)

    @staticmethod
    def _target_clause(intent: QueryIntent) -> str:
        if intent.target_type == "all":
            return "EXPLAIN ALL"
        if intent.target_type == "class":
            if intent.class_label is None:
                raise IntentValidationError("target.class_label is required for a class target.")
            return f"EXPLAIN CLASS {intent.class_label}"
        if intent.target_type != "node":
            raise IntentValidationError(f"Unsupported target type: {intent.target_type}")
        if not intent.node_ids:
            raise IntentValidationError("target.node_ids is required for a node target.")
        if len(intent.node_ids) == 1:
            return f"EXPLAIN NODE {intent.node_ids[0]}"
        return "EXPLAIN NODES " + ",".join(str(node) for node in intent.node_ids)
