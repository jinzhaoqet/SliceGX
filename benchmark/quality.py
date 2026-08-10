from typing import Any, Iterable

from planner.quality import QualityCalibrationPoint


def explanation_result_error(exact_result: Any, approximate_result: Any) -> float:
    exact_by_node = {
        (item.node_id, item.layer): item
        for item in getattr(exact_result, "results", [])
    }
    approximate_by_node = {
        (item.node_id, item.layer): item
        for item in getattr(approximate_result, "results", [])
    }
    keys = set(exact_by_node).union(approximate_by_node)
    if not keys:
        return 0.0
    errors = []
    for key in keys:
        exact = exact_by_node.get(key)
        approximate = approximate_by_node.get(key)
        if exact is None or approximate is None:
            errors.append(1.0)
            continue
        exact_nodes = set(exact.nodes)
        approximate_nodes = set(approximate.nodes)
        union = exact_nodes.union(approximate_nodes)
        jaccard_error = 0.0 if not union else 1.0 - len(exact_nodes.intersection(approximate_nodes)) / len(union)
        fidelity_error = min(abs(exact.fidelity_plus - approximate.fidelity_plus), 1.0)
        errors.append(max(jaccard_error, fidelity_error))
    return max(errors)


def calibrate_quality_point(
    exact_result: Any,
    approximate_results: Iterable[Any],
    sample_ratio: float,
    confidence: float = 0.95,
) -> QualityCalibrationPoint:
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be within (0, 1].")
    errors = sorted(explanation_result_error(exact_result, result) for result in approximate_results)
    if not errors:
        raise ValueError("At least one approximate result is required for calibration.")
    quantile_index = min(int((len(errors) - 1) * confidence), len(errors) - 1)
    return QualityCalibrationPoint(
        sample_ratio=sample_ratio,
        observed_max_error=errors[quantile_index],
        observed_confidence=confidence,
        observations=len(errors),
    )
