from dataclasses import dataclass, field
from typing import Dict, Optional

from result_schema import QueryExecutionResult


@dataclass
class QuerySessionStore:
    named_results: Dict[str, QueryExecutionResult] = field(default_factory=dict)

    def save(self, name: str, result: QueryExecutionResult) -> None:
        self.named_results[name.upper()] = result

    def get(self, name: str) -> Optional[QueryExecutionResult]:
        return self.named_results.get(name.upper())

    def list_names(self):
        return sorted(self.named_results.keys())
