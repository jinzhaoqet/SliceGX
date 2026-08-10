import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmark.workloads import BenchmarkCase


@dataclass
class BenchmarkRecord:
    case_id: str
    category: str
    repetition: int
    latency_seconds: float
    success: bool
    statement_count: int
    result_count: Optional[int] = None
    algorithm: Optional[str] = None
    estimated_cost: Optional[float] = None
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    records: List[BenchmarkRecord]

    def aggregate(self) -> Dict[str, Any]:
        grouped: Dict[str, List[BenchmarkRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.category, []).append(record)
        return {
            category: {
                "runs": len(records),
                "success_rate": sum(record.success for record in records) / len(records),
                "latency_mean": statistics.mean(record.latency_seconds for record in records),
                "latency_median": statistics.median(record.latency_seconds for record in records),
                "latency_p95": self._percentile(
                    [record.latency_seconds for record in records], 0.95
                ),
            }
            for category, records in grouped.items()
        }

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        ordered = sorted(values)
        index = min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1)
        return ordered[index]

    def write_jsonl(self, path: Path) -> None:
        Path(path).write_text(
            "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in self.records),
            encoding="utf-8",
        )


class BenchmarkRunner:
    def __init__(self, execute_statement: Callable[[str], Any]):
        self.execute_statement = execute_statement

    def run(
        self,
        cases: Sequence[BenchmarkCase],
        repetitions: int = 3,
        warmups: int = 1,
    ) -> BenchmarkSummary:
        if repetitions <= 0 or warmups < 0:
            raise ValueError("repetitions must be positive and warmups must be non-negative.")
        records = []
        for case in cases:
            for _ in range(warmups):
                self._execute_case(case)
            for repetition in range(repetitions):
                records.append(self._execute_case(case, repetition))
        return BenchmarkSummary(records)

    def _execute_case(self, case: BenchmarkCase, repetition: int = -1) -> BenchmarkRecord:
        start = time.perf_counter()
        final_result = None
        try:
            for statement in case.statements:
                final_result = self.execute_statement(statement)
            latency = time.perf_counter() - start
            physical = getattr(getattr(final_result, "plan", None), "physical", {}) or {}
            return BenchmarkRecord(
                case_id=case.case_id,
                category=case.category,
                repetition=repetition,
                latency_seconds=latency,
                success=True,
                statement_count=len(case.statements),
                result_count=getattr(final_result, "filtered_results", None),
                algorithm=getattr(final_result, "algorithm", None),
                estimated_cost=physical.get("estimated_cost"),
                cache_stats=dict(getattr(final_result, "cache_stats", {}) or {}),
            )
        except Exception as error:
            return BenchmarkRecord(
                case_id=case.case_id,
                category=case.category,
                repetition=repetition,
                latency_seconds=time.perf_counter() - start,
                success=False,
                statement_count=len(case.statements),
                error=str(error),
            )
