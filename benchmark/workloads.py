from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    statements: Tuple[str, ...]
    description: str
    scale: Dict[str, Any] = field(default_factory=dict)


class ExplanationAnalyticsWorkloads:
    """Canonical TKDE workloads spanning generation, analytics, and sessions."""

    @staticmethod
    def build(
        node_ids: Sequence[int] = (519, 537),
        class_label: int = 1,
        fidelity_threshold: float = 0.5,
        pattern_support: float = 0.5,
    ) -> List[BenchmarkCase]:
        if len(node_ids) < 2:
            raise ValueError("At least two node ids are required for the workload suite.")
        first_node, second_node = int(node_ids[0]), int(node_ids[1])
        return [
            BenchmarkCase(
                case_id="W1-single-node",
                category="single_node",
                statements=(f"EXPLAIN NODE {first_node} AT LAYER 1 WITH K 6",),
                description="Exact explanation for one target node and one layer.",
                scale={"target_count": 1, "layer_count": 1},
            ),
            BenchmarkCase(
                case_id="W2-class-level",
                category="class_level",
                statements=(f"EXPLAIN CLASS {class_label} RANK BY FIDELITY_PLUS",),
                description="Generate and rank explanations for one target class.",
                scale={"class_label": class_label},
            ),
            BenchmarkCase(
                case_id="W3-full-test-set",
                category="full_test_set",
                statements=("EXPLAIN ALL WITH APPROXIMATE 0.3",),
                description="Approximate explanation generation over the full test set.",
            ),
            BenchmarkCase(
                case_id="W4-multi-layer-diagnosis",
                category="multi_layer",
                statements=(f"EXPLAIN NODES {first_node},{second_node} AT ALL LAYERS",),
                description="Cross-node and cross-layer diagnostic workload.",
                scale={"target_count": 2, "layer_scope": "all"},
            ),
            BenchmarkCase(
                case_id="W5-result-filter",
                category="result_filter",
                statements=(
                    "EXPLAIN ALL WHERE FACTUAL = TRUE "
                    f"AND FIDELITY_PLUS > {fidelity_threshold:g} "
                    "PROJECT NODE_ID,LAYER,FIDELITY_PLUS",
                ),
                description="Filter and project an explanation result set.",
            ),
            BenchmarkCase(
                case_id="W6-common-pattern",
                category="common_pattern",
                statements=(
                    f"EXPLAIN CLASS {class_label} AT ALL LAYERS "
                    f"GROUP BY LAYER PATTERN MIN_SUPPORT {pattern_support:g}",
                ),
                description="Discover common explanation nodes within each layer.",
            ),
            BenchmarkCase(
                case_id="W7-session-analysis",
                category="session",
                statements=(
                    "LET Q1 = EXPLAIN ALL WHERE FACTUAL = TRUE",
                    f"LET Q2 = FILTER Q1 WHERE FIDELITY_PLUS > {fidelity_threshold:g}",
                    "LET Q3 = RANK Q2 BY FIDELITY_PLUS",
                    "COMPARE Q3 BY COMMON_NODES",
                ),
                description="Materialize, refine, rank, and compare a named explanation set.",
            ),
            BenchmarkCase(
                case_id="W8-quality-constrained-approximation",
                category="quality_constrained_approximation",
                statements=(
                    "EXPLAIN ALL WITH MAX_ERROR 0.1 "
                    "WITH MIN_CONFIDENCE 0.9 WITH TIME_BUDGET 10",
                ),
                description="Approximate execution constrained by declared quality and time goals.",
            ),
            BenchmarkCase(
                case_id="W9-materialized-reuse",
                category="materialized_reuse",
                statements=(
                    f"EXPLAIN NODES {first_node},{second_node} WITH K 6 MATERIALIZE AS BASE",
                    f"EXPLAIN NODES {first_node},{second_node} WITH K 6 "
                    "WHERE FIDELITY_PLUS > 0.4",
                ),
                description="Reuse a materialized generation result for a new result predicate.",
            ),
        ]
