from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CandidatePlan:
    algorithm: str
    executor_op: str
    estimated_cost: float
    shared_scope: str
    cache_mode: str
    incremental: bool = False
    feasible: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "executor_op": self.executor_op,
            "estimated_cost": self.estimated_cost,
            "shared_scope": self.shared_scope,
            "cache_mode": self.cache_mode,
            "incremental": self.incremental,
            "feasible": self.feasible,
            "reason": self.reason,
        }


class ExplanationCostModel:
    def enumerate(
        self,
        query: Any,
        target_count: int,
        layer_count: int,
        sample_ratio: float,
        cache_context: Dict[str, Any],
    ) -> List[CandidatePlan]:
        if cache_context.get("exact_generation_hit"):
            return [
                CandidatePlan(
                    algorithm="CACHE",
                    executor_op="MaterializedGenerationScan",
                    estimated_cost=0.01,
                    shared_scope="cross_query",
                    cache_mode="exact_generation_reuse",
                    reason="An exact generation result is materialized in the executor cache.",
                )
            ]

        constraints = bool(query.include_nodes or query.exclude_nodes)
        k_factor = max(float(query.K or 10) / 10.0, 0.5)
        approximation_factor = max(sample_ratio, 0.05)
        candidates: List[CandidatePlan] = []

        if query.layer == -1:
            cost = max(target_count, 1) * max(layer_count, 1) * 0.62 * k_factor * approximation_factor
            if constraints:
                candidates.append(
                    CandidatePlan(
                        algorithm="SS_LAYERED",
                        executor_op="ConstraintAwareLayeredSingleNodePlan",
                        estimated_cost=round(cost / 0.62, 4),
                        shared_scope="cross_query+cross_layer-slice-cache",
                        cache_mode="incremental_resume",
                        reason=(
                            "Layered SS preserves INCLUDE/EXCLUDE semantics while reusing "
                            "per-layer model slices and incremental states."
                        ),
                    )
                )
                return candidates
            candidates.append(
                CandidatePlan(
                    algorithm="MM",
                    executor_op="MultiLayerSharedSlicePlan",
                    estimated_cost=round(cost, 4),
                    shared_scope="cross_node+cross_layer",
                    cache_mode="backend_managed",
                    reason="MM shares model-slice and layer-transition work across requested layers.",
                )
            )
            return candidates

        ss_cost = max(target_count, 1) * k_factor * approximation_factor
        incremental = cache_context.get("resumable_nodes", 0) > 0
        if incremental:
            resumed_fraction = min(cache_context["resumable_nodes"] / max(target_count, 1), 1.0)
            ss_cost *= 1.0 - 0.55 * resumed_fraction
        candidates.append(
            CandidatePlan(
                algorithm="SS",
                executor_op="IncrementalSingleNodePlan" if incremental else "SingleNodeExactPlan",
                estimated_cost=round(ss_cost, 4),
                shared_scope="cross_query" if incremental else "none",
                cache_mode="incremental_resume" if incremental else "reuse_enabled",
                incremental=incremental,
                reason=(
                    "SS resumes cached greedy states for a larger K."
                    if incremental
                    else "SS independently explains each target node."
                ),
            )
        )

        if target_count > 1 and not constraints:
            ms_cost = (0.45 + 0.42 * target_count) * k_factor * approximation_factor
            candidates.append(
                CandidatePlan(
                    algorithm="MS",
                    executor_op="MultiNodeSharedCandidatePlan",
                    estimated_cost=round(ms_cost, 4),
                    shared_scope="cross_node",
                    cache_mode="backend_managed",
                    reason="MS shares candidate-quality computation across target nodes.",
                )
            )
        return candidates

    @staticmethod
    def choose(candidates: List[CandidatePlan]) -> CandidatePlan:
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            raise ValueError("No feasible physical plan is available.")
        return min(feasible, key=lambda candidate: candidate.estimated_cost)
