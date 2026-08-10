from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class RewriteResult:
    operators: List[str]
    applied_rules: List[str]


class LogicalRewriter:
    PRIORITY = {
        "Target": 0,
        "ForceIncludeNodes": 1,
        "ForceExcludeNodes": 1,
        "Explain": 2,
        "ReuseMaterializedExplain": 2,
        "Filter": 3,
        "Rank": 4,
        "Compare": 5,
        "GroupPattern": 5,
        "Project": 6,
        "Materialize": 7,
    }

    def rewrite(self, operators: List[str], context: Dict[str, Any]) -> RewriteResult:
        rewritten = list(operators)
        rules = []

        filter_ops = [item for item in rewritten if item.startswith("Filter")]
        if len(filter_ops) > 1:
            rewritten = [item for item in rewritten if not item.startswith("Filter")]
            rewritten.append("Filter[" + ",".join(filter_ops) + "]")
            rules.append("MergeResultFilters")

        if any(item.startswith("Force") for item in rewritten):
            rules.append("PushStructuralConstraintsBeforeExplain")

        if context.get("exact_generation_hit") and "Explain" in rewritten:
            rewritten[rewritten.index("Explain")] = "ReuseMaterializedExplain"
            rules.append("ReuseMaterializedGeneration")

        if any(item.startswith("Project") for item in rewritten):
            rules.append("DeferProjectionUntilAfterAnalytics")

        ordered = sorted(rewritten, key=self._priority)
        if ordered != operators:
            rules.append("CanonicalOperatorOrdering")
        return RewriteResult(operators=ordered, applied_rules=rules)

    def _priority(self, operator_name: str) -> Tuple[int, str]:
        prefix = operator_name.split("[")[0]
        if prefix.startswith("Target"):
            prefix = "Target"
        return self.PRIORITY.get(prefix, 99), operator_name
