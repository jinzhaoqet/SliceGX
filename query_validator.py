from dataclasses import dataclass
from typing import List


@dataclass
class QueryValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class QueryValidator:
    """Validate parsed SliceGX declarative queries before execution."""

    VALID_TARGETS = {"node", "all", "class"}
    VALID_COMPARE_BY = {None, "fidelity_plus", "common_nodes"}
    VALID_RANK_BY = {None, "fidelity_plus"}
    VALID_PROJECT_FIELDS = {
        "explanation_id",
        "node_id",
        "nodes",
        "factual",
        "counterfactual",
        "fidelity_plus",
        "fidelity_minus",
        "score",
        "subgraph_size",
        "layer",
    }
    VALID_GROUP_FIELDS = {None, "layer", "factual", "counterfactual"}

    def validate(self, query) -> None:
        errors: List[str] = []

        if query.target not in self.VALID_TARGETS:
            errors.append(f"Unsupported target type: {query.target}")

        if query.target == "node" and not query.node_ids:
            errors.append("EXPLAIN NODE/NODES must provide at least one node id.")

        if query.target == "all" and query.node_ids:
            errors.append("EXPLAIN ALL cannot be combined with explicit node ids.")

        if query.target == "class" and query.class_label < 0:
            errors.append("EXPLAIN CLASS requires a non-negative class label.")

        if query.node_ids and any(node < 0 for node in query.node_ids):
            errors.append("Node ids must be non-negative integers.")

        if query.include_nodes and any(node < 0 for node in query.include_nodes):
            errors.append("Included node ids must be non-negative integers.")

        if query.exclude_nodes and any(node < 0 for node in query.exclude_nodes):
            errors.append("Excluded node ids must be non-negative integers.")

        overlap = set(query.include_nodes).intersection(query.exclude_nodes)
        if overlap:
            errors.append(f"INCLUDE and EXCLUDE overlap on nodes: {sorted(overlap)}")

        if query.layer < -1:
            errors.append("Layer must be >= 0, or use AT ALL LAYERS.")

        if query.K is not None and query.K <= 0:
            errors.append("K must be a positive integer.")

        if query.h is not None and query.h < 0:
            errors.append("H must be >= 0.")

        if query.theta is not None and query.theta < 0:
            errors.append("THETA must be >= 0.")

        if query.gamma is not None and not (0.0 <= query.gamma <= 1.0):
            errors.append("GAMMA must be within [0, 1].")

        if query.approximate and not (0.0 < query.sample_ratio <= 1.0):
            errors.append("APPROXIMATE ratio must be within (0, 1].")

        if query.compare_by not in self.VALID_COMPARE_BY:
            errors.append(f"Unsupported COMPARE BY metric: {query.compare_by}")

        if query.rank_by not in self.VALID_RANK_BY:
            errors.append(f"Unsupported RANK BY metric: {query.rank_by}")

        if query.max_subgraph_size is not None and query.max_subgraph_size <= 0:
            errors.append("SUBGRAPH_SIZE upper bound must be positive.")

        unknown_project_fields = set(query.project_fields).difference(self.VALID_PROJECT_FIELDS)
        if unknown_project_fields:
            errors.append(f"Unsupported PROJECT fields: {sorted(unknown_project_fields)}")

        if query.group_by not in self.VALID_GROUP_FIELDS:
            errors.append(f"Unsupported GROUP BY field: {query.group_by}")

        if query.pattern_min_support is not None and not (0.0 < query.pattern_min_support <= 1.0):
            errors.append("PATTERN MIN_SUPPORT must be within (0, 1].")

        if query.pattern_min_support is not None and query.compare_by is not None:
            errors.append("PATTERN and COMPARE cannot be requested in the same query.")

        if query.materialize_as is not None:
            if not query.materialize_as.replace("_", "a").isalnum() or query.materialize_as[0].isdigit():
                errors.append("MATERIALIZE name must be an identifier.")

        if query.max_error is not None and not (0.0 < query.max_error < 1.0):
            errors.append("MAX_ERROR must be within (0, 1).")

        if query.min_confidence is not None and not (0.0 < query.min_confidence <= 1.0):
            errors.append("MIN_CONFIDENCE must be within (0, 1].")

        if query.time_budget_seconds is not None and query.time_budget_seconds <= 0:
            errors.append("TIME_BUDGET must be positive.")

        if errors:
            raise QueryValidationError(" | ".join(errors))
