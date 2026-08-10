from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanSummary:
    logical: Dict[str, Any] = field(default_factory=dict)
    physical: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> "PlanSummary":
        if raw is None:
            return cls()
        return cls(
            logical=dict(raw.get("logical", {})),
            physical=dict(raw.get("physical", {})),
            stats=dict(raw.get("stats", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical": dict(self.logical),
            "physical": dict(self.physical),
            "stats": dict(self.stats),
        }


@dataclass
class ComparisonSummary:
    type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> Optional["ComparisonSummary"]:
        if raw is None:
            return None
        payload = dict(raw)
        return cls(type=str(payload.get("type", "")), payload=payload)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.payload)


@dataclass
class ErrorSummary:
    message: str = ""

    @classmethod
    def from_message(cls, message: Optional[str]) -> Optional["ErrorSummary"]:
        if not message:
            return None
        return cls(message=str(message))

    def to_dict(self) -> Dict[str, Any]:
        return {"message": self.message}


@dataclass
class ExplanationResult:
    node_id: int
    nodes: List[int] = field(default_factory=list)
    factual: bool = False
    counterfactual: bool = False
    both: bool = False
    score: float = 0.0
    fidelity_plus: float = 0.0
    fidelity_minus: float = 0.0
    layer: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "ExplanationResult":
        return cls(
            node_id=int(raw.get("node_id", -1)),
            nodes=list(raw.get("nodes", [])),
            factual=bool(raw.get("factual", False)),
            counterfactual=bool(raw.get("counterfactual", False)),
            both=bool(raw.get("both", False)),
            score=float(raw.get("score", 0.0)),
            fidelity_plus=float(raw.get("Fid+", raw.get("fidelity_plus", 0.0))),
            fidelity_minus=float(raw.get("Fid-", raw.get("fidelity_minus", 0.0))),
            layer=raw.get("layer"),
            raw=dict(raw),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "node_id": self.node_id,
            "nodes": list(self.nodes),
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "both": self.both,
            "score": self.score,
            "fidelity_plus": self.fidelity_plus,
            "fidelity_minus": self.fidelity_minus,
        }
        if self.layer is not None:
            data["layer"] = self.layer
        return data


@dataclass
class QueryExecutionResult:
    query: Dict[str, Any]
    algorithm: str
    plan: PlanSummary = field(default_factory=PlanSummary)
    total_results: int = 0
    filtered_results: int = 0
    results: List[ExplanationResult] = field(default_factory=list)
    comparison: Optional[ComparisonSummary] = None
    time_seconds: float = 0.0
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorSummary] = None
    plan_only: bool = False
    analytics: Dict[str, Any] = field(default_factory=dict)
    materialized_as: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.plan, dict):
            self.plan = PlanSummary.from_raw(self.plan)
        if isinstance(self.comparison, dict):
            self.comparison = ComparisonSummary.from_raw(self.comparison)
        if isinstance(self.error, str):
            self.error = ErrorSummary.from_message(self.error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": dict(self.query),
            "algorithm": self.algorithm,
            "plan": self.plan.to_dict(),
            "total_results": self.total_results,
            "filtered_results": self.filtered_results,
            "results": [item.to_dict() for item in self.results],
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "time_seconds": self.time_seconds,
            "cache_stats": dict(self.cache_stats),
            "error": self.error.to_dict() if self.error else None,
            "plan_only": self.plan_only,
            "analytics": dict(self.analytics),
            "materialized_as": self.materialized_as,
        }
