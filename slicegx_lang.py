"""
SliceGX Declarative Query Language
===================================
声明式 GNN 解释查询语言，支持：
  Feature 1: 包含/排除指定节点 (INCLUDE / EXCLUDE)
  Feature 2: 结果对比 (COMPARE BY fidelity_plus / common_nodes)
  Feature 3: 自动路由 SS/MS/MM (auto_route)
  Feature 4: 中间状态缓存与复用 (ResultCache)
  Feature 5: 近似采样 (WITH APPROXIMATE)

用法:
  python slicegx_lang.py                       # 进入 REPL
  python slicegx_lang.py --query "EXPLAIN NODE 519"  # 单次执行
"""

import copy
import time
import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import Counter

import torch
import hydra
from omegaconf import OmegaConf
from warnings import simplefilter

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataset
from utils import get_logger
from SliceGX import (Slicedmodel, Subfunction, Declarative, GreedyAlgorithm,
                     layerwise_run, MIN_VALUE)


# ============================================================================
# Section 1: ExplainQuery 数据类
# ============================================================================

@dataclass
class ExplainQuery:
    """解析后的查询表示"""
    # 目标
    target: str = 'node'          # 'node' | 'all' | 'class'
    node_ids: List[int] = field(default_factory=list)
    class_label: int = -1

    # 执行参数 (None = 使用 config 默认值)
    layer: int = 0                # 默认最后一层(完整模型), -1 = 全部层 (AT ALL LAYERS)
    K: Optional[int] = None
    h: Optional[float] = None
    theta: Optional[float] = None
    gamma: Optional[float] = None

    # WHERE 约束
    require_factual: Optional[bool] = None
    require_counterfactual: Optional[bool] = None
    fid_plus_threshold: Optional[float] = None
    fid_minus_threshold: Optional[float] = None
    max_subgraph_size: Optional[int] = None

    # Feature 1: 结构约束
    include_nodes: List[int] = field(default_factory=list)
    exclude_nodes: List[int] = field(default_factory=list)

    # Feature 2: 结果对比
    compare_by: Optional[str] = None  # 'fidelity_plus' | 'common_nodes'

    # Feature 5: 近似
    approximate: bool = False
    sample_ratio: float = 0.3

    # 自动填充
    algorithm: Optional[str] = None   # 'SS' | 'MS' | 'MM'


# ============================================================================
# Section 2: QueryParser (关键字状态机)
# ============================================================================

class QueryParser:
    """将查询字符串解析为 ExplainQuery 对象

    支持语法:
        EXPLAIN NODE <id>
        EXPLAIN NODES <id1>,<id2>,...
        EXPLAIN ALL
        EXPLAIN CLASS <label>
        WHERE FACTUAL = TRUE/FALSE
        WHERE COUNTERFACTUAL = TRUE/FALSE
        WHERE FIDELITY_PLUS > <val>
        WHERE FIDELITY_MINUS < <val>
        WHERE SUBGRAPH_SIZE <= <val>
        AT LAYER <n>
        AT ALL LAYERS
        INCLUDE <id1>,<id2>,...
        EXCLUDE <id1>,<id2>,...
        COMPARE BY FIDELITY_PLUS
        COMPARE BY COMMON_NODES
        WITH APPROXIMATE [<ratio>]
        WITH K <val>
        WITH H <val>
        WITH THETA <val>
        WITH GAMMA <val>
    """

    def parse(self, query_str: str) -> ExplainQuery:
        q = ExplainQuery()
        tokens = query_str.strip().split()
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
                    # EXPLAIN <id> 简写
                    q.target = 'node'
                    q.node_ids = [int(next_tok)]

            elif tok == 'WHERE' and i + 1 < n:
                i += 1
                cond = tokens[i].upper()
                if cond == 'FACTUAL' and i + 2 < n:
                    i += 2  # skip '='
                    q.require_factual = tokens[i].upper() == 'TRUE'
                elif cond == 'COUNTERFACTUAL' and i + 2 < n:
                    i += 2
                    q.require_counterfactual = tokens[i].upper() == 'TRUE'
                elif cond == 'FIDELITY_PLUS' and i + 2 < n:
                    i += 2  # skip '>'
                    q.fid_plus_threshold = float(tokens[i])
                elif cond == 'FIDELITY_MINUS' and i + 2 < n:
                    i += 2  # skip '<'
                    q.fid_minus_threshold = float(tokens[i])
                elif cond == 'SUBGRAPH_SIZE' and i + 2 < n:
                    i += 2  # skip '<='
                    q.max_subgraph_size = int(tokens[i])

            elif tok == 'AT' and i + 1 < n:
                i += 1
                next_tok = tokens[i].upper()
                if next_tok == 'ALL':
                    q.layer = -1  # 全部层，路由到 MM
                    if i + 1 < n and tokens[i + 1].upper() == 'LAYERS':
                        i += 1
                elif next_tok == 'LAYER' and i + 1 < n:
                    i += 1
                    q.layer = int(tokens[i])  # 特定层
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
                            pass  # 没有指定 ratio，使用默认 0.3
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

            i += 1

        return q


# ============================================================================
# Section 3: ResultCache (Feature 4 中间状态复用)
# ============================================================================

class ResultCache:
    """缓存 Subfunction 和贪心中间状态，支持跨查询复用"""

    def __init__(self):
        self._sf_cache: Dict[tuple, Any] = {}      # (layer, h, theta, sample_ratio) -> Subfunction
        self._exp_cache: Dict[tuple, list] = {}     # (node, layer, h, theta, K) -> explanatory list
        self._slice_cache: Dict[tuple, Any] = {}    # (layer,) -> (modelslice,)

    def get_subfunction(self, layer, h, theta, sample_ratio):
        key = (layer, h, theta, sample_ratio)
        return self._sf_cache.get(key)

    def save_subfunction(self, layer, h, theta, sample_ratio, sf):
        key = (layer, h, theta, sample_ratio)
        self._sf_cache[key] = sf

    def get_modelslice(self, layer):
        return self._slice_cache.get((layer,))

    def save_modelslice(self, layer, ms):
        self._slice_cache[(layer,)] = ms

    def get_initial_explanatory(self, node, layer, h, theta, target_K):
        """找最大的 K' < target_K 的缓存结果作为增量起点"""
        best_k = -1
        best_exp = None
        for key, exp in self._exp_cache.items():
            n, l, hh, th, kk = key
            if n == node and l == layer and hh == h and th == theta and kk < target_K:
                if kk > best_k:
                    best_k = kk
                    best_exp = exp
        return best_exp

    def save_explanatory(self, node, layer, h, theta, K, explanatory):
        key = (node, layer, h, theta, K)
        self._exp_cache[key] = list(explanatory)

    def stats(self):
        return {
            'subfunction_entries': len(self._sf_cache),
            'explanatory_entries': len(self._exp_cache),
            'modelslice_entries': len(self._slice_cache),
        }


# ============================================================================
# Section 4: auto_route (Feature 3 自动路由)
# ============================================================================

def auto_route(query: ExplainQuery) -> str:
    """根据查询特征自动选择最优算法

    决策规则:
      - 多层 (AT ALL LAYERS / layer == -1) → MM (层间 hop jumping)
      - 单层多节点 (ALL / CLASS / NODES) → MS (共享候选集)
      - 单层单节点 → SS (最简单)
    """
    multi_node = len(query.node_ids) > 1 or query.target in ('all', 'class')
    multi_layer = (query.layer == -1)

    if multi_layer:
        return 'MM'
    elif multi_node:
        return 'MS'
    else:
        return 'SS'


# ============================================================================
# Section 5: SliceGXExecutor (执行引擎)
# ============================================================================

class SliceGXExecutor:
    """查询执行引擎：解析 → 路由 → 执行 → 过滤 → 对比"""

    def __init__(self, config, dataset, state_dict, device, logger):
        self.config = config
        self.dataset = dataset
        self.state_dict = state_dict
        self.device = device
        self.logger = logger
        self.cache = ResultCache()
        self.layer_nums = len(config.models.param.gnn_latent_dim)

    def execute(self, query: ExplainQuery) -> dict:
        """执行一条查询，返回结果字典"""
        start_time = time.time()

        # 填充默认参数
        self._fill_defaults(query)

        # 解析目标节点
        test_nodes = self._resolve_nodes(query)
        if not test_nodes:
            return {'error': f'No test nodes found for target={query.target}'}

        # Feature 3: 自动路由
        query.algorithm = auto_route(query)
        self.logger.info(f'[Router] algorithm={query.algorithm}, '
                         f'nodes={len(test_nodes)}, layer={query.layer}')

        # 分发执行
        if query.algorithm == 'SS':
            raw_results = self._run_ss(query, test_nodes)
        elif query.algorithm == 'MS':
            raw_results = self._run_ms(query, test_nodes)
        else:
            raw_results = self._run_mm(query, test_nodes)

        # WHERE 过滤
        filtered = self._apply_filters(raw_results, query)

        # Feature 2: 结果对比
        if query.compare_by and filtered:
            compared = self._compare(filtered, query.compare_by)
        else:
            compared = None

        elapsed = time.time() - start_time
        self.logger.info(f'[Done] {len(filtered)}/{len(raw_results)} results passed filters, '
                         f'time={elapsed:.3f}s')

        return {
            'query': self._query_summary(query),
            'algorithm': query.algorithm,
            'total_results': len(raw_results),
            'filtered_results': len(filtered),
            'results': filtered,
            'comparison': compared,
            'time': round(elapsed, 3),
            'cache_stats': self.cache.stats(),
        }

    # ---------- 节点解析 ----------

    def _resolve_nodes(self, query: ExplainQuery) -> List[int]:
        data = self.dataset.data
        if query.target == 'node':
            return query.node_ids
        elif query.target == 'all':
            if self.config.datasets.dataset_name in ['tree_grid', 'tree_cycle']:
                return torch.where(data.test_mask * data.y != 0)[0].tolist()
            return torch.where(data.test_mask)[0].tolist()
        elif query.target == 'class':
            if self.config.datasets.dataset_name in ['tree_grid', 'tree_cycle']:
                mask = data.test_mask * data.y != 0
            else:
                mask = data.test_mask
            indices = torch.where(mask)[0]
            return [idx.item() for idx in indices if data.y[idx].item() == query.class_label]
        return []

    # ---------- 默认值填充 ----------

    @staticmethod
    def _first_or_val(v, default):
        """从 config 值中提取第一个元素（处理 ListConfig / list / scalar）"""
        try:
            return v[0]
        except (TypeError, IndexError, KeyError):
            return v if v is not None else default

    def _fill_defaults(self, query: ExplainQuery):
        cfg = self.config.datasets
        if query.K is None:
            query.K = int(self._first_or_val(getattr(cfg, 'K', 10), 10))
        if query.h is None:
            query.h = float(self._first_or_val(getattr(cfg, 'h', 0.3), 0.3))
        if query.theta is None:
            query.theta = float(self._first_or_val(getattr(cfg, 'theta', 0.2), 0.2))
        if query.gamma is None:
            query.gamma = float(getattr(cfg, 'gamma', 0.5))
        # 确保类型正确
        query.K = int(query.K)
        query.h = float(query.h)
        query.theta = float(query.theta)
        query.gamma = float(query.gamma)
        if query.max_subgraph_size is not None:
            query.K = min(query.K, query.max_subgraph_size)

    # ---------- SS 执行 ----------

    def _run_ss(self, query: ExplainQuery, test_nodes: List[int]) -> List[dict]:
        layer = query.layer if query.layer >= 0 else 0
        cut_layer = self.layer_nums - 1 - layer  # 转换为 cut_layer
        # 实际上 cut_layer=0 表示完整模型 (最后一层)
        # layer=0 对应 cut_layer = layer_nums - 1 (只有第一层)
        # 这里 layer 语义: 用户指定的是 GNN 层编号 (0=第1层, 2=第3层)
        # cut_layer 语义: 从顶部切掉几层。num_hop = layer_nums - cut_layer
        # 所以 layer=2 (第3层, 完整) -> cut_layer=0
        # 简化: 默认 cut_layer=0 即完整模型
        # layer 语义: 用户视角 0=完整模型(默认), 1=只看第1层, 2=只看前2层
        # cut_layer 语义: SliceGX.py 中 num_hop = layer_nums - cut_layer
        # 默认 layer=0 → cut_layer=0 → num_hop=layer_nums (完整模型)
        cut_layer = query.layer if query.layer >= 0 else 0
        cut_layer = max(0, min(cut_layer, self.layer_nums - 1))

        sample_ratio = query.sample_ratio if query.approximate else 1.0

        # Feature 4: 尝试复用缓存的 modelslice
        modelslice = self.cache.get_modelslice(cut_layer)
        if modelslice is None:
            modelslice = Slicedmodel(
                self.config, self.device, self.layer_nums,
                self.layer_nums - cut_layer, self.logger,
                self.dataset, self.state_dict)
            self.cache.save_modelslice(cut_layer, modelslice)

        # Feature 4: 尝试复用缓存的 Subfunction
        sf_key = (cut_layer, query.h, query.theta, sample_ratio)
        quality = self.cache.get_subfunction(*sf_key)
        if quality is None:
            dec = Declarative(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
            quality = Subfunction(test_nodes, dec, modelslice, self.logger, self.device,
                                  sample_ratio=sample_ratio)
            self.cache.save_subfunction(*sf_key, quality)
            self.logger.info(f'[Cache MISS] Subfunction computed for layer={cut_layer}')
        else:
            self.logger.info(f'[Cache HIT] Subfunction reused for layer={cut_layer}')

        dec = Declarative(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        results = []
        # 有 include/exclude 约束时，不复用 greedy 缓存（约束改变了解空间）
        has_constraints = bool(query.include_nodes or query.exclude_nodes)

        for node in test_nodes:
            # Feature 4: 尝试从缓存恢复贪心中间状态（无约束时才复用）
            init_exp = None
            if not has_constraints:
                init_exp = self.cache.get_initial_explanatory(
                    node, cut_layer, query.h, query.theta, query.K)
                if init_exp:
                    self.logger.info(f'[Cache HIT] Resuming greedy for node {node} from K={len(init_exp)}')

            self.dataset.data.to(self.device)
            algorithm = GreedyAlgorithm(
                dec, modelslice, node, self.logger, quality,
                include_nodes=query.include_nodes,
                exclude_nodes=query.exclude_nodes,
                initial_explanatory=init_exp)
            optimal = algorithm.get_solution()

            # Feature 4: 无约束时才保存缓存
            if not has_constraints:
                self.cache.save_explanatory(
                    node, cut_layer, query.h, query.theta, query.K,
                    algorithm.explanatory)

            if optimal is not None:
                optimal['node_id'] = node
                results.append(optimal)

        return results

    # ---------- MS 执行 ----------

    def _run_ms(self, query: ExplainQuery, test_nodes: List[int]) -> List[dict]:
        """Multi-Start: 多节点共享候选集"""
        from Slice_MS import (GreedyAlgorithm as GreedyMS,
                              Subfunction as SubMS,
                              Slicedmodel as SliceMS,
                              Declarative as DecMS)

        cut_layer = query.layer if query.layer >= 0 else 0
        cut_layer = max(0, min(cut_layer, self.layer_nums - 1))
        num_hop = self.layer_nums - cut_layer

        dec = DecMS(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMS(self.config, self.device, num_hop,
                             self.logger, self.dataset, self.state_dict)
        quality = SubMS(test_nodes, dec, modelslice, self.logger, self.device)

        self.dataset.data.to(self.device)
        algorithm = GreedyMS(dec, modelslice, test_nodes, self.logger, quality)
        optimal_list = algorithm.get_solution()

        results = []
        for i, opt in enumerate(optimal_list):
            if opt is not None:
                opt['node_id'] = test_nodes[i] if i < len(test_nodes) else -1
                results.append(opt)
        return results

    # ---------- MM 执行 ----------

    def _run_mm(self, query: ExplainQuery, test_nodes: List[int]) -> List[dict]:
        """Multi-Model: 多层 + hop jumping"""
        from Slice_MM import (GreedyAlgorithm as GreedyMM,
                              Subfunction as SubMM,
                              Slicedmodel as SliceMM,
                              Declarative as DecMM)

        dec = DecMM(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMM(self.config, self.device, self.layer_nums,
                             self.logger, self.dataset, self.state_dict)
        quality = SubMM(test_nodes, dec, modelslice, self.logger, self.device)

        self.dataset.data.to(self.device)
        algorithm = GreedyMM(dec, modelslice, test_nodes, self.logger, quality)
        all_optimal = algorithm.get_all_solution()

        # all_optimal 是 [layer][node_index] 的二维结构，展平
        results = []
        for layer_idx in range(len(all_optimal)):
            for node_idx, opt in enumerate(all_optimal[layer_idx]):
                if opt is not None:
                    opt['node_id'] = test_nodes[node_idx] if node_idx < len(test_nodes) else -1
                    opt['layer'] = layer_idx
                    results.append(opt)
        return results

    # ---------- WHERE 过滤 ----------

    def _apply_filters(self, results: List[dict], query: ExplainQuery) -> List[dict]:
        filtered = []
        for r in results:
            if query.require_factual is not None:
                if r.get('factual') != query.require_factual:
                    continue
            if query.require_counterfactual is not None:
                if r.get('counterfactual') != query.require_counterfactual:
                    continue
            if query.fid_plus_threshold is not None:
                if r.get('Fid+', -999) <= query.fid_plus_threshold:
                    continue
            if query.fid_minus_threshold is not None:
                if r.get('Fid-', 999) >= query.fid_minus_threshold:
                    continue
            filtered.append(r)
        return filtered

    # ---------- Feature 2: 结果对比 ----------

    def _compare(self, results: List[dict], compare_by: str) -> dict:
        if compare_by == 'fidelity_plus':
            best = max(results, key=lambda r: r.get('Fid+', -999))
            return {
                'type': 'best_fidelity_plus',
                'best_node': best.get('node_id'),
                'best_fid_plus': best.get('Fid+'),
                'best_nodes': best.get('nodes'),
            }
        elif compare_by == 'common_nodes':
            cnt = Counter()
            for r in results:
                cnt.update(r.get('nodes', []))
            n = len(results)
            common = [node for node, c in cnt.most_common() if c >= 0.5 * n]
            return {
                'type': 'common_nodes',
                'common_nodes': common,
                'total_explanations': n,
                'support': {str(k): round(v / n, 3) for k, v in cnt.most_common(20)},
            }
        return {}

    # ---------- 辅助 ----------

    def _query_summary(self, query: ExplainQuery) -> dict:
        return {
            'target': query.target,
            'node_ids': query.node_ids,
            'class_label': query.class_label,
            'layer': query.layer,
            'K': query.K,
            'h': query.h,
            'theta': query.theta,
            'gamma': query.gamma,
            'include_nodes': query.include_nodes,
            'exclude_nodes': query.exclude_nodes,
            'compare_by': query.compare_by,
            'approximate': query.approximate,
            'sample_ratio': query.sample_ratio if query.approximate else 1.0,
        }


# ============================================================================
# Section 6: 格式化输出
# ============================================================================

def format_result(result: dict) -> str:
    """将执行结果格式化为易读文本"""
    lines = []
    lines.append(f"=== SliceGX Query Result ===")
    lines.append(f"Algorithm: {result.get('algorithm', '?')}")
    lines.append(f"Results: {result.get('filtered_results', 0)}/{result.get('total_results', 0)} "
                 f"(passed filters/total)")
    lines.append(f"Time: {result.get('time', 0)}s")

    cache = result.get('cache_stats', {})
    if cache:
        lines.append(f"Cache: sf={cache.get('subfunction_entries', 0)}, "
                     f"exp={cache.get('explanatory_entries', 0)}, "
                     f"slice={cache.get('modelslice_entries', 0)}")

    comparison = result.get('comparison')
    if comparison:
        lines.append(f"\n--- Comparison ({comparison.get('type', '')}) ---")
        if comparison.get('type') == 'best_fidelity_plus':
            lines.append(f"  Best node: {comparison.get('best_node')}")
            lines.append(f"  Fid+: {comparison.get('best_fid_plus', 0):.4f}")
            lines.append(f"  Subgraph: {comparison.get('best_nodes')}")
        elif comparison.get('type') == 'common_nodes':
            lines.append(f"  Common nodes (>=50% support): {comparison.get('common_nodes')}")
            lines.append(f"  Top support: {comparison.get('support')}")

    for i, r in enumerate(result.get('results', [])[:10]):  # 最多显示 10 个
        node_id = r.get('node_id', '?')
        fid_p = r.get('Fid+', 0)
        fid_m = r.get('Fid-', 0)
        factual = r.get('factual', False)
        counter = r.get('counterfactual', False)
        nodes = r.get('nodes', [])
        layer_info = f" layer={r.get('layer')}" if 'layer' in r else ""
        lines.append(f"\n  [{i}] node={node_id}{layer_info} | "
                     f"factual={factual} counter={counter} | "
                     f"Fid+={fid_p:.4f} Fid-={fid_m:.4f} | "
                     f"subgraph({len(nodes)})={nodes}")

    if result.get('filtered_results', 0) > 10:
        lines.append(f"\n  ... and {result['filtered_results'] - 10} more results")

    return '\n'.join(lines)


# ============================================================================
# Section 7: main / REPL 入口
# ============================================================================

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]

    # 加载数据集
    if config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        dataset, _, _, _ = get_dataset(config.datasets.dataset_root,
                                       config.datasets.dataset_name)
    else:
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    if config.datasets.dataset_name in ['products']:
        dataset.data.y = torch.argmax(dataset.data.y, dim=1)
    dataset.data.y = dataset.data.y.squeeze().long()

    # Logger
    log_file = f"{config.datasets.dataset_name}_lang.log"
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')

    # 加载模型权重
    state_dict = torch.load(
        os.path.join(config.models.gnn_savedir,
                     config.datasets.dataset_name,
                     f'{config.models.gnn_name}_'
                     f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']

    # 初始化执行器
    executor = SliceGXExecutor(config, dataset, state_dict, device, logger)
    parser = QueryParser()

    # 检查是否有命令行 --query 参数
    query_arg = None
    for arg in sys.argv:
        if arg.startswith('--query='):
            query_arg = arg[len('--query='):]

    if query_arg:
        # 单次执行模式
        query = parser.parse(query_arg)
        result = executor.execute(query)
        print(format_result(result))
    else:
        # REPL 模式
        print(f"SliceGX Query Language (dataset={config.datasets.dataset_name}, device={device})")
        print(f"Type 'help' for syntax, 'exit' to quit.\n")

        while True:
            try:
                query_str = input("SliceGX>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not query_str:
                continue
            if query_str.lower() in ('exit', 'quit', 'q'):
                break
            if query_str.lower() == 'help':
                print_help()
                continue
            if query_str.lower() == 'cache':
                print(json.dumps(executor.cache.stats(), indent=2))
                continue

            try:
                query = parser.parse(query_str)
                result = executor.execute(query)
                print(format_result(result))
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            print()


def print_help():
    print("""
SliceGX Query Language Syntax
==============================

Basic:
  EXPLAIN NODE <id>                  Explain a single node
  EXPLAIN NODES <id1>,<id2>,...      Explain multiple nodes
  EXPLAIN ALL                        Explain all test nodes
  EXPLAIN CLASS <label>              Explain nodes of a class

Filters (WHERE):
  WHERE FACTUAL = TRUE               Only factual explanations
  WHERE COUNTERFACTUAL = TRUE         Only counterfactual ones
  WHERE FIDELITY_PLUS > 0.5          Fid+ threshold
  WHERE SUBGRAPH_SIZE <= 6           Max subgraph size

Layer:
  AT LAYER 2                         Specific layer (0-indexed)
  AT ALL LAYERS                      All layers (uses MM algorithm)

Structural Constraints:
  INCLUDE 15,23                      Force include nodes
  EXCLUDE 207                        Force exclude nodes

Comparison (Feature 2):
  COMPARE BY FIDELITY_PLUS           Find best explanation by Fid+
  COMPARE BY COMMON_NODES            Find common pattern (>=50% support)

Parameters:
  WITH K 6                           Override subgraph size
  WITH H 0.2                         Override influence threshold
  WITH THETA 0.1                     Override diversity threshold
  WITH APPROXIMATE 0.3               Approximate mode (30% sampling)

Examples:
  EXPLAIN NODE 519
  EXPLAIN ALL WHERE FACTUAL = TRUE COMPARE BY FIDELITY_PLUS
  EXPLAIN NODE 519 INCLUDE 518,517 WITH K 6
  EXPLAIN CLASS 1 COMPARE BY COMMON_NODES
  EXPLAIN NODE 519 WITH K 4
  EXPLAIN NODE 519 WITH K 6          (reuses K=4 cache)
  EXPLAIN ALL WITH APPROXIMATE 0.3

Special commands:
  help    Show this help
  cache   Show cache statistics
  exit    Quit
""")


if __name__ == '__main__':
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
