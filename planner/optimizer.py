from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LogicalPlan:
    target_op: str
    filter_ops: List[str] = field(default_factory=list)
    compare_op: str = "none"
    layer_scope: str = "single_layer"
    approximate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_op": self.target_op,
            "filter_ops": list(self.filter_ops),
            "compare_op": self.compare_op,
            "layer_scope": self.layer_scope,
            "approximate": self.approximate,
        }


@dataclass
class PhysicalPlan:
    algorithm: str
    executor_op: str
    cache_mode: str
    sample_ratio: float
    estimated_cost: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "executor_op": self.executor_op,
            "cache_mode": self.cache_mode,
            "sample_ratio": self.sample_ratio,
            "estimated_cost": self.estimated_cost,
            "reasons": list(self.reasons),
        }


@dataclass
class QueryPlan:
    logical: LogicalPlan
    physical: PhysicalPlan
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical": self.logical.to_dict(),
            "physical": self.physical.to_dict(),
            "stats": dict(self.stats),
        }


class QueryOptimizer:
    """A lightweight rule-based optimizer for SliceGX explanation queries."""

    def plan(self, query: Any, test_nodes: List[int], cache_stats: Dict[str, int]) -> QueryPlan:
        logical = self._build_logical_plan(query)
        physical = self._build_physical_plan(query, test_nodes, cache_stats, logical)
        stats = {
            "target_count": len(test_nodes),
            "has_constraints": bool(query.include_nodes or query.exclude_nodes),
            "cache_entries": dict(cache_stats),
        }
        return QueryPlan(logical=logical, physical=physical, stats=stats)

    def _build_logical_plan(self, query: Any) -> LogicalPlan:
        target_op = {
            "node": "TargetNodeLookup",
            "all": "TargetTestSetScan",
            "class": "TargetClassScan",
        }.get(query.target, "TargetUnknown")

        filter_ops = []
        if query.require_factual is not None:
            filter_ops.append("FilterFactual")
        if query.require_counterfactual is not None:
            filter_ops.append("FilterCounterfactual")
        if query.fid_plus_threshold is not None:
            filter_ops.append("FilterFidelityPlus")
        if query.fid_minus_threshold is not None:
            filter_ops.append("FilterFidelityMinus")
        if query.max_subgraph_size is not None:
            filter_ops.append("FilterSubgraphSize")
        if query.include_nodes:
            filter_ops.append("ForceIncludeNodes")
        if query.exclude_nodes:
            filter_ops.append("ForceExcludeNodes")

        compare_op = "none"
        if query.compare_by == "fidelity_plus":
            compare_op = "CompareBestFidelityPlus"
        elif query.compare_by == "common_nodes":
            compare_op = "CompareCommonNodes"

        layer_scope = "all_layers" if query.layer == -1 else "single_layer"
        return LogicalPlan(
            target_op=target_op,
            filter_ops=filter_ops,
            compare_op=compare_op,
            layer_scope=layer_scope,
            approximate=query.approximate,
        )

    def _build_physical_plan(
        self,
        query: Any,
        test_nodes: List[int],
        cache_stats: Dict[str, int],
        logical: LogicalPlan,
    ) -> PhysicalPlan:
        target_count = len(test_nodes)
        sample_ratio = query.sample_ratio if query.approximate else 1.0
        estimated_cost = self._estimate_cost(query, target_count, sample_ratio)
        reasons: List[str] = []

        if logical.layer_scope == "all_layers":
            algorithm = "MM"
            executor_op = "MultiLayerHopJumpPlan"
            reasons.append("AT ALL LAYERS requires multi-layer execution.")
        elif target_count > 1 or query.target in ("all", "class"):
            algorithm = "MS"
            executor_op = "MultiNodeSharedCandidatePlan"
            reasons.append("Multiple target nodes benefit from shared-candidate execution.")
        else:
            algorithm = "SS"
            executor_op = "SingleNodeExactPlan"
            reasons.append("Single-node query maps to the single-start execution path.")

        if query.approximate:
            reasons.append(f"Approximate execution enabled with sample_ratio={sample_ratio}.")

        if query.include_nodes or query.exclude_nodes:
            reasons.append("Structural constraints disable greedy-state reuse for safety.")

        if algorithm == "SS" and cache_stats.get("subfunction_entries", 0) > 0:
            reasons.append("Existing cache entries may reduce repeated single-node query cost.")

        cache_mode = "reuse_enabled"
        if query.include_nodes or query.exclude_nodes:
            cache_mode = "partial_reuse_only"
        if algorithm in ("MS", "MM"):
            cache_mode = "backend_managed"

        return PhysicalPlan(
            algorithm=algorithm,
            executor_op=executor_op,
            cache_mode=cache_mode,
            sample_ratio=sample_ratio,
            estimated_cost=estimated_cost,
            reasons=reasons,
        )

    @staticmethod
    def _estimate_cost(query: Any, target_count: int, sample_ratio: float) -> float:
        layer_factor = 3.0 if query.layer == -1 else 1.0 + max(query.layer, 0) * 0.5
        k_factor = float(query.K or 10)
        constraint_factor = 1.25 if (query.include_nodes or query.exclude_nodes) else 1.0
        compare_factor = 1.15 if query.compare_by else 1.0
        approx_factor = max(sample_ratio, 0.1)
        base = max(target_count, 1) * layer_factor * max(k_factor / 10.0, 0.5)
        return round(base * constraint_factor * compare_factor * approx_factor, 3)
