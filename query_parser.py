from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExplainQuery:
    """解析后的查询表示"""
    target: str = 'node'  # 'node' | 'all' | 'class'
    node_ids: List[int] = field(default_factory=list)
    class_label: int = -1

    layer: int = 0  # 默认完整模型, -1 = 全部层
    K: Optional[int] = None
    h: Optional[float] = None
    theta: Optional[float] = None
    gamma: Optional[float] = None

    require_factual: Optional[bool] = None
    require_counterfactual: Optional[bool] = None
    fid_plus_threshold: Optional[float] = None
    fid_minus_threshold: Optional[float] = None
    max_subgraph_size: Optional[int] = None

    include_nodes: List[int] = field(default_factory=list)
    exclude_nodes: List[int] = field(default_factory=list)

    compare_by: Optional[str] = None  # 'fidelity_plus' | 'common_nodes'
    rank_by: Optional[str] = None  # 'fidelity_plus'
    project_fields: List[str] = field(default_factory=list)
    group_by: Optional[str] = None
    pattern_min_support: Optional[float] = None
    materialize_as: Optional[str] = None

    approximate: bool = False
    sample_ratio: float = 0.3
    max_error: Optional[float] = None
    min_confidence: Optional[float] = None
    time_budget_seconds: Optional[float] = None

    algorithm: Optional[str] = None  # 'SS' | 'MS' | 'MM'
    plan_only: bool = False


class QueryParser:
    """将查询字符串解析为 ExplainQuery 对象。"""

    def parse(self, query_str: str) -> ExplainQuery:
        stripped = query_str.strip()
        if not stripped:
            raise ValueError("Query cannot be empty.")
        plan_only = False
        upper = stripped.upper()
        if upper.startswith('EXPLAIN PLAN FOR '):
            stripped = stripped[len('EXPLAIN PLAN FOR '):]
            plan_only = True
        if not stripped.upper().startswith('EXPLAIN '):
            raise ValueError("A SliceGX query must start with EXPLAIN.")

        q = ExplainQuery()
        q.plan_only = plan_only
        tokens = stripped.split()
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i].upper()

            if tok == 'EXPLAIN' and i + 1 < n:
                i += 1
                next_tok = tokens[i].upper()
                if next_tok == 'NODE' and i + 1 < n:
                    i += 1
                    q.target = 'node'
                    q.node_ids = [int(tokens[i])]
                elif next_tok == 'NODES' and i + 1 < n:
                    i += 1
                    q.target = 'node'
                    q.node_ids = [int(x) for x in tokens[i].split(',')]
                elif next_tok == 'ALL':
                    q.target = 'all'
                elif next_tok == 'CLASS' and i + 1 < n:
                    i += 1
                    q.target = 'class'
                    q.class_label = int(tokens[i])
                else:
                    q.target = 'node'
                    q.node_ids = [int(next_tok)]

            elif tok in ('WHERE', 'AND') and i + 1 < n:
                i += 1
                cond = tokens[i].upper()
                if cond == 'FACTUAL' and i + 2 < n:
                    if tokens[i + 1] != '=':
                        raise ValueError("FACTUAL only supports the = comparator.")
                    i += 2
                    if tokens[i].upper() not in ('TRUE', 'FALSE'):
                        raise ValueError("FACTUAL value must be TRUE or FALSE.")
                    q.require_factual = tokens[i].upper() == 'TRUE'
                elif cond == 'COUNTERFACTUAL' and i + 2 < n:
                    if tokens[i + 1] != '=':
                        raise ValueError("COUNTERFACTUAL only supports the = comparator.")
                    i += 2
                    if tokens[i].upper() not in ('TRUE', 'FALSE'):
                        raise ValueError("COUNTERFACTUAL value must be TRUE or FALSE.")
                    q.require_counterfactual = tokens[i].upper() == 'TRUE'
                elif cond == 'FIDELITY_PLUS' and i + 2 < n:
                    if tokens[i + 1] != '>':
                        raise ValueError("FIDELITY_PLUS only supports the > comparator.")
                    i += 2
                    q.fid_plus_threshold = float(tokens[i])
                elif cond == 'FIDELITY_MINUS' and i + 2 < n:
                    if tokens[i + 1] != '<':
                        raise ValueError("FIDELITY_MINUS only supports the < comparator.")
                    i += 2
                    q.fid_minus_threshold = float(tokens[i])
                elif cond == 'SUBGRAPH_SIZE' and i + 2 < n:
                    if tokens[i + 1] != '<=':
                        raise ValueError("SUBGRAPH_SIZE only supports the <= comparator.")
                    i += 2
                    q.max_subgraph_size = int(tokens[i])
                else:
                    raise ValueError(f"Unsupported or incomplete WHERE predicate: {cond}")

            elif tok == 'AT' and i + 1 < n:
                i += 1
                next_tok = tokens[i].upper()
                if next_tok == 'ALL':
                    q.layer = -1
                    if i + 1 < n and tokens[i + 1].upper() == 'LAYERS':
                        i += 1
                elif next_tok == 'LAYER' and i + 1 < n:
                    i += 1
                    q.layer = int(tokens[i])
                else:
                    q.layer = int(next_tok)

            elif tok == 'INCLUDE' and i + 1 < n:
                i += 1
                q.include_nodes = [int(x) for x in tokens[i].split(',')]

            elif tok == 'EXCLUDE' and i + 1 < n:
                i += 1
                q.exclude_nodes = [int(x) for x in tokens[i].split(',')]

            elif tok == 'COMPARE' and i + 1 < n:
                i += 1
                if tokens[i].upper() == 'BY' and i + 1 < n:
                    i += 1
                    q.compare_by = tokens[i].lower()

            elif tok == 'RANK' and i + 1 < n:
                i += 1
                if tokens[i].upper() == 'BY' and i + 1 < n:
                    i += 1
                    q.rank_by = tokens[i].lower()

            elif tok == 'PROJECT' and i + 1 < n:
                i += 1
                q.project_fields = [field_name.strip().lower() for field_name in tokens[i].split(',')]

            elif tok == 'GROUP' and i + 2 < n and tokens[i + 1].upper() == 'BY':
                i += 2
                q.group_by = tokens[i].lower()

            elif tok == 'PATTERN' and i + 2 < n and tokens[i + 1].upper() == 'MIN_SUPPORT':
                i += 2
                q.pattern_min_support = float(tokens[i])

            elif tok == 'MATERIALIZE' and i + 2 < n and tokens[i + 1].upper() == 'AS':
                i += 2
                q.materialize_as = tokens[i]

            elif tok == 'WITH' and i + 1 < n:
                i += 1
                param = tokens[i].upper()
                if param == 'APPROXIMATE':
                    q.approximate = True
                    if i + 1 < n:
                        try:
                            q.sample_ratio = float(tokens[i + 1])
                            i += 1
                        except ValueError:
                            pass
                elif param == 'K' and i + 1 < n:
                    i += 1
                    q.K = int(tokens[i])
                elif param == 'H' and i + 1 < n:
                    i += 1
                    q.h = float(tokens[i])
                elif param == 'THETA' and i + 1 < n:
                    i += 1
                    q.theta = float(tokens[i])
                elif param == 'GAMMA' and i + 1 < n:
                    i += 1
                    q.gamma = float(tokens[i])
                elif param == 'MAX_ERROR' and i + 1 < n:
                    i += 1
                    q.max_error = float(tokens[i])
                    q.approximate = True
                elif param == 'MIN_CONFIDENCE' and i + 1 < n:
                    i += 1
                    q.min_confidence = float(tokens[i])
                    q.approximate = True
                elif param == 'TIME_BUDGET' and i + 1 < n:
                    i += 1
                    q.time_budget_seconds = float(tokens[i])
                else:
                    raise ValueError(f"Unsupported or incomplete WITH parameter: {param}")

            else:
                raise ValueError(f"Unexpected token at position {i}: {tokens[i]}")

            i += 1

        return q
