import re
from typing import Any, Optional

from query_parser import QueryParser
from query_session import QuerySessionStore
from query_validator import QueryValidationError


class SessionStatementExecutor:
    """Execute benchmark statements with the same named-result semantics as the REPL."""

    def __init__(
        self,
        executor: Any,
        parser: Optional[QueryParser] = None,
        store: Optional[QuerySessionStore] = None,
    ):
        self.executor = executor
        self.parser = parser or QueryParser()
        self.store = store or QuerySessionStore()

    def __call__(self, statement: str):
        let_match = re.match(r"^\s*LET\s+(\w+)\s*=\s*(.+)$", statement, flags=re.IGNORECASE)
        if let_match:
            result = self._execute_expression(let_match.group(2))
            self.store.save(let_match.group(1), result)
            return result
        return self._execute_expression(statement)

    def _execute_expression(self, statement: str):
        filter_match = re.match(
            r"^\s*FILTER\s+(\w+)\s+WHERE\s+(.+)$",
            statement,
            flags=re.IGNORECASE,
        )
        rank_match = re.match(
            r"^\s*RANK\s+(\w+)\s+BY\s+(FIDELITY_PLUS)\s*$",
            statement,
            flags=re.IGNORECASE,
        )
        compare_match = re.match(
            r"^\s*COMPARE\s+(\w+)\s+BY\s+(FIDELITY_PLUS|COMMON_NODES)\s*$",
            statement,
            flags=re.IGNORECASE,
        )
        if filter_match:
            source_name = filter_match.group(1)
            source = self._require_result(source_name)
            filter_query = self.parser.parse(f"EXPLAIN NODE 0 WHERE {filter_match.group(2)}")
            return self.executor.filter_saved_result(source, filter_query, source_name.upper())
        if rank_match:
            source_name = rank_match.group(1)
            source = self._require_result(source_name)
            return self.executor.rank_saved_result(source, rank_match.group(2).lower(), source_name.upper())
        if compare_match:
            source_name = compare_match.group(1)
            source = self._require_result(source_name)
            return self.executor.compare_saved_result(source, compare_match.group(2).lower(), source_name.upper())
        return self.executor.execute(self.parser.parse(statement))

    def _require_result(self, name: str):
        result = self.store.get(name)
        if result is None:
            raise QueryValidationError(f"named result {name.upper()} not found.")
        return result
