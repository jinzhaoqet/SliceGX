import json

from result_schema import QueryExecutionResult


def format_result(result: QueryExecutionResult) -> str:
    """将执行结果格式化为易读文本"""
    lines = []
    if result.error:
        return f"Error: {result.error.message}"

    lines.append("=== SliceGX Query Result ===")
    lines.append(f"Algorithm: {result.algorithm}")
    logical = result.plan.logical
    physical = result.plan.physical
    if result.plan_only:
        lines[0] = "=== SliceGX Query Plan ==="
        if logical:
            lines.append(
                f"Logical: target={logical.get('target_op', '?')} "
                f"filters={logical.get('filter_ops', [])} "
                f"compare={logical.get('compare_op', 'none')} "
                f"layer_scope={logical.get('layer_scope', '?')} "
                f"approximate={logical.get('approximate', False)}"
            )
    if physical:
        lines.append(
            f"Plan: {physical.get('executor_op', '?')} "
            f"(cost={physical.get('estimated_cost', '?')}, "
            f"cache={physical.get('cache_mode', '?')})"
        )
        reasons = physical.get('reasons', [])
        if reasons:
            lines.append("Reasons:")
            for reason in reasons:
                lines.append(f"  - {reason}")
    if result.plan_only:
        lines.append(f"Planning Time: {result.time_seconds}s")
        return '\n'.join(lines)

    lines.append(f"Results: {result.filtered_results}/{result.total_results} (passed filters/total)")
    lines.append(f"Time: {result.time_seconds}s")

    cache = result.cache_stats
    if cache:
        lines.append(
            f"Cache: sf={cache.get('subfunction_entries', 0)}, "
            f"exp={cache.get('explanatory_entries', 0)}, "
            f"slice={cache.get('modelslice_entries', 0)}"
        )

    comparison = result.comparison.to_dict() if result.comparison else None
    if comparison:
        lines.append(f"\n--- Comparison ({comparison.get('type', '')}) ---")
        if comparison.get('type') == 'best_fidelity_plus':
            fid_value = comparison.get('best_fid_plus')
            lines.append(f"  Best node: {comparison.get('best_node')}")
            lines.append(f"  Fid+: {(0.0 if fid_value is None else fid_value):.4f}")
            lines.append(f"  Subgraph: {comparison.get('best_nodes')}")
        elif comparison.get('type') == 'common_nodes':
            lines.append(f"  Common nodes (>=50% support): {comparison.get('common_nodes')}")
            lines.append(f"  Top support: {comparison.get('support')}")

    for i, r in enumerate(result.results[:10]):
        layer_info = f" layer={r.layer}" if r.layer is not None else ""
        lines.append(
            f"\n  [{i}] node={r.node_id}{layer_info} | "
            f"factual={r.factual} counter={r.counterfactual} | "
            f"Fid+={r.fidelity_plus:.4f} Fid-={r.fidelity_minus:.4f} | "
            f"subgraph({len(r.nodes)})={r.nodes}"
        )

    if result.filtered_results > 10:
        lines.append(f"\n  ... and {result.filtered_results - 10} more results")

    return '\n'.join(lines)


def result_to_json(result: QueryExecutionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
