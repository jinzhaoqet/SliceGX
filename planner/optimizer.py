from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from planner.cost_model import CandidatePlan, ExplanationCostModel
from planner.quality import ApproximationQualityProfile
from planner.rewrites import LogicalRewriter


@dataclass
class LogicalPlan:
    target_op: str
    filter_ops: List[str] = field(default_factory=list)
    compare_op: str = "none"
    layer_scope: str = "single_layer"
    approximate: bool = False
    operators: List[str] = field(default_factory=list)
    rewrite_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_op": self.target_op,
            "filter_ops": list(self.filter_ops),
            "compare_op": self.compare_op,
            "layer_scope": self.layer_scope,
            "approximate": self.approximate,
            "operators": list(self.operators),
            "rewrite_rules": list(self.rewrite_rules),
        }


@dataclass
class PhysicalPlan:
    algorithm: str
    executor_op: str
    cache_mode: str
    sample_ratio: float
    estimated_cost: float
    reasons: List[str] = field(default_factory=list)
    shared_scope: str = "none"
    incremental: bool = False
    quality: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "executor_op": self.executor_op,
            "cache_mode": self.cache_mode,
            "sample_ratio": self.sample_ratio,
            "estimated_cost": self.estimated_cost,
            "reasons": list(self.reasons),
            "shared_scope": self.shared_scope,
            "incremental": self.incremental,
            "quality": dict(self.quality),
            "candidates": list(self.candidates),
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
    """Rule rewriting and candidate-based physical planning for SliceGX queries."""

    def __init__(
        self,
        layer_count: int = 3,
        quality_profile: Optional[ApproximationQualityProfile] = None,
    ):
        self.layer_count = layer_count
        self.rewriter = LogicalRewriter()
        self.cost_model = ExplanationCostModel()
        self.quality_profile = quality_profile or ApproximationQualityProfile()
        self.execution_profiles: Dict[str, Dict[str, float]] = {}

    def plan(self, query: Any, test_nodes: List[int], cache_stats: Dict[str, Any]) -> QueryPlan:
        sample_ratio, quality = self.resolve_sample_ratio(query)
        logical = self._build_logical_plan(query, cache_stats)
        candidates = self.cost_model.enumerate(
            query,
            len(test_nodes),
            self.layer_count,
            sample_ratio,
            cache_stats,
        )
        selected = self.cost_model.choose(candidates)
        profile = self.execution_profiles.get(selected.algorithm, {})
        seconds_per_cost_unit = profile.get("seconds_per_cost_unit", 1.0)
        estimated_seconds = selected.estimated_cost * seconds_per_cost_unit
        if query.time_budget_seconds is not None and estimated_seconds > query.time_budget_seconds:
            if query.max_error is None and query.min_confidence is None and query.approximate:
                budget_ratio = max(
                    0.05,
                    sample_ratio * query.time_budget_seconds / max(estimated_seconds, 1e-9),
                )
                sample_ratio = budget_ratio
                candidates = self.cost_model.enumerate(
                    query,
                    len(test_nodes),
                    self.layer_count,
                    sample_ratio,
                    cache_stats,
                )
                selected = self.cost_model.choose(candidates)
                profile = self.execution_profiles.get(selected.algorithm, {})
                estimated_seconds = selected.estimated_cost * profile.get("seconds_per_cost_unit", 1.0)
            else:
                quality["budget_feasible"] = False
        quality["estimated_seconds"] = round(estimated_seconds, 6)
        quality["cost_calibration_observations"] = int(profile.get("observations", 0))
        physical = self._physical_plan(selected, candidates, sample_ratio, quality)
        stats = {
            "target_count": len(test_nodes),
            "has_constraints": bool(query.include_nodes or query.exclude_nodes),
            "cache_entries": dict(cache_stats),
            "candidate_count": len(candidates),
        }
        return QueryPlan(logical=logical, physical=physical, stats=stats)

    def record_execution(self, algorithm: str, estimated_cost: float, elapsed_seconds: float) -> None:
        if estimated_cost <= 0 or elapsed_seconds < 0:
            return
        observed_ratio = elapsed_seconds / estimated_cost
        profile = self.execution_profiles.setdefault(
            algorithm,
            {"seconds_per_cost_unit": observed_ratio, "observations": 0},
        )
        if profile["observations"] == 0:
            profile["seconds_per_cost_unit"] = observed_ratio
        else:
            profile["seconds_per_cost_unit"] = (
                0.8 * profile["seconds_per_cost_unit"] + 0.2 * observed_ratio
            )
        profile["observations"] += 1

    def _build_logical_plan(self, query: Any, cache_context: Dict[str, Any]) -> LogicalPlan:
        target_op = {
            "node": "TargetNodeLookup",
            "all": "TargetTestSetScan",
            "class": "TargetClassScan",
        }.get(query.target, "TargetUnknown")
        filters = []
        if query.require_factual is not None:
            filters.append("FilterFactual")
        if query.require_counterfactual is not None:
            filters.append("FilterCounterfactual")
        if query.fid_plus_threshold is not None:
            filters.append("FilterFidelityPlus")
        if query.fid_minus_threshold is not None:
            filters.append("FilterFidelityMinus")
        if query.max_subgraph_size is not None:
            filters.append("FilterSubgraphSize")

        operators = [target_op]
        if query.include_nodes:
            operators.append("ForceIncludeNodes")
        if query.exclude_nodes:
            operators.append("ForceExcludeNodes")
        operators.append("Explain")
        operators.extend(filters)
        if query.rank_by:
            operators.append(f"Rank[{query.rank_by}]")
        if query.compare_by:
            operators.append(f"Compare[{query.compare_by}]")
        if query.pattern_min_support is not None:
            operators.append(f"GroupPattern[{query.group_by},{query.pattern_min_support}]")
        if query.project_fields:
            operators.append("Project[" + ",".join(query.project_fields) + "]")
        if query.materialize_as:
            operators.append(f"Materialize[{query.materialize_as}]")
        rewrite = self.rewriter.rewrite(operators, cache_context)

        compare_op = "none"
        if query.compare_by == "fidelity_plus":
            compare_op = "CompareBestFidelityPlus"
        elif query.compare_by == "common_nodes":
            compare_op = "CompareCommonNodes"
        return LogicalPlan(
            target_op=target_op,
            filter_ops=filters,
            compare_op=compare_op,
            layer_scope="all_layers" if query.layer == -1 else "single_layer",
            approximate=query.approximate,
            operators=rewrite.operators,
            rewrite_rules=rewrite.applied_rules,
        )

    @staticmethod
    def _physical_plan(
        selected: CandidatePlan,
        candidates: List[CandidatePlan],
        sample_ratio: float,
        quality: Dict[str, Any],
    ) -> PhysicalPlan:
        reasons = [selected.reason]
        if quality.get("quality_constrained"):
            reasons.append(
                "Sampling ratio was derived from MAX_ERROR/MIN_CONFIDENCE constraints; "
                "the estimate must be calibrated empirically for each dataset."
            )
        if not quality.get("budget_feasible", True):
            reasons.append("The requested time budget conflicts with the quality lower bound.")
        return PhysicalPlan(
            algorithm=selected.algorithm,
            executor_op=selected.executor_op,
            cache_mode=selected.cache_mode,
            sample_ratio=sample_ratio,
            estimated_cost=selected.estimated_cost,
            reasons=reasons,
            shared_scope=selected.shared_scope,
            incremental=selected.incremental,
            quality=quality,
            candidates=[candidate.to_dict() for candidate in candidates],
        )

    def resolve_sample_ratio(self, query: Any):
        if not query.approximate:
            return 1.0, {
                "quality_constrained": False,
                "estimated_max_error": 0.0,
                "estimated_confidence": 1.0,
                "budget_feasible": True,
            }
        lower_bound = max(float(query.sample_ratio), 0.05)
        calibrated_point = self.quality_profile.select(
            query.max_error,
            query.min_confidence,
            minimum_ratio=lower_bound,
        )
        if calibrated_point is not None:
            return calibrated_point.sample_ratio, {
                "quality_constrained": query.max_error is not None or query.min_confidence is not None,
                "requested_max_error": query.max_error,
                "requested_min_confidence": query.min_confidence,
                "estimated_max_error": calibrated_point.observed_max_error,
                "estimated_confidence": calibrated_point.observed_confidence,
                "time_budget_seconds": query.time_budget_seconds,
                "budget_feasible": True,
                "calibrated": True,
                "quality_observations": calibrated_point.observations,
            }
        if query.max_error is not None:
            lower_bound = max(lower_bound, 1.0 - float(query.max_error))
        if query.min_confidence is not None:
            lower_bound = max(lower_bound, float(query.min_confidence))
        ratio = min(lower_bound, 1.0)
        return ratio, {
            "quality_constrained": query.max_error is not None or query.min_confidence is not None,
            "requested_max_error": query.max_error,
            "requested_min_confidence": query.min_confidence,
            "estimated_max_error": round(1.0 - ratio, 6),
            "estimated_confidence": round(ratio, 6),
            "time_budget_seconds": query.time_budget_seconds,
            "budget_feasible": True,
            "calibrated": False,
        }
