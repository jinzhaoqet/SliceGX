from collections import Counter

from result_schema import ComparisonSummary, QueryExecutionResult


class ResultOperations:
    @staticmethod
    def compare(saved_result: QueryExecutionResult, compare_by: str, source_name: str) -> QueryExecutionResult:
        metric = compare_by.lower()
        comparison = None
        if saved_result.results and metric == "fidelity_plus":
            best = max(saved_result.results, key=lambda item: item.fidelity_plus)
            comparison = {
                "type": "best_fidelity_plus",
                "best_node": best.node_id,
                "best_fid_plus": best.fidelity_plus,
                "best_nodes": list(best.nodes),
            }
        elif saved_result.results and metric == "common_nodes":
            counts = Counter(node for item in saved_result.results for node in set(item.nodes))
            total = len(saved_result.results)
            comparison = {
                "type": "common_nodes",
                "common_nodes": [node for node, count in counts.most_common() if count >= 0.5 * total],
                "total_explanations": total,
                "support": {
                    str(node): round(count / total, 3)
                    for node, count in counts.most_common(20)
                },
            }
        elif metric not in {"fidelity_plus", "common_nodes"}:
            raise ValueError(f"Unsupported comparison metric: {metric}")
        return QueryExecutionResult(
            query={"target": "saved_result", "source_name": source_name, "compare_by": metric},
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.total_results,
            filtered_results=saved_result.filtered_results,
            results=list(saved_result.results),
            comparison=ComparisonSummary.from_raw(comparison) if comparison else None,
            cache_stats=dict(saved_result.cache_stats),
        )

    @staticmethod
    def filter(saved_result: QueryExecutionResult, query, source_name: str) -> QueryExecutionResult:
        selected = []
        for item in saved_result.results:
            if getattr(query, "require_factual", None) is not None and item.factual != query.require_factual:
                continue
            if getattr(query, "require_counterfactual", None) is not None and item.counterfactual != query.require_counterfactual:
                continue
            if getattr(query, "fid_plus_threshold", None) is not None and item.fidelity_plus <= query.fid_plus_threshold:
                continue
            if getattr(query, "fid_minus_threshold", None) is not None and item.fidelity_minus >= query.fid_minus_threshold:
                continue
            if getattr(query, "max_subgraph_size", None) is not None and len(item.nodes) > query.max_subgraph_size:
                continue
            selected.append(item)
        return QueryExecutionResult(
            query={
                "target": "saved_result",
                "source_name": source_name,
                "filter_applied": {
                    "require_factual": getattr(query, "require_factual", None),
                    "require_counterfactual": getattr(query, "require_counterfactual", None),
                    "fid_plus_threshold": getattr(query, "fid_plus_threshold", None),
                    "fid_minus_threshold": getattr(query, "fid_minus_threshold", None),
                    "max_subgraph_size": getattr(query, "max_subgraph_size", None),
                },
            },
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.filtered_results,
            filtered_results=len(selected),
            results=selected,
            cache_stats=dict(saved_result.cache_stats),
        )

    @staticmethod
    def rank(saved_result: QueryExecutionResult, rank_by: str, source_name: str) -> QueryExecutionResult:
        metric = rank_by.lower()
        if metric != "fidelity_plus":
            raise ValueError(f"Unsupported rank metric: {metric}")
        ranked = sorted(saved_result.results, key=lambda item: item.fidelity_plus, reverse=True)
        return QueryExecutionResult(
            query={"target": "saved_result", "source_name": source_name, "rank_by": metric},
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.filtered_results,
            filtered_results=len(ranked),
            results=ranked,
            cache_stats=dict(saved_result.cache_stats),
        )
