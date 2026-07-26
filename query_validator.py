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

        if errors:
            raise QueryValidationError(" | ".join(errors))
