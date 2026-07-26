import time
from collections import Counter
from typing import Any, Dict, List

import torch

from SliceGX import Slicedmodel, Subfunction, Declarative, GreedyAlgorithm
from planner.optimizer import QueryOptimizer
from query_validator import QueryValidator
from result_schema import ComparisonSummary, ExplanationResult, QueryExecutionResult


class ResultCache:
    """缓存 Subfunction 和贪心中间状态，支持跨查询复用"""

    def __init__(self):
        self._sf_cache: Dict[tuple, Any] = {}
        self._exp_cache: Dict[tuple, list] = {}
        self._slice_cache: Dict[tuple, Any] = {}

    def get_subfunction(self, layer, h, theta, sample_ratio):
        return self._sf_cache.get((layer, h, theta, sample_ratio))

    def save_subfunction(self, layer, h, theta, sample_ratio, sf):
        self._sf_cache[(layer, h, theta, sample_ratio)] = sf

    def get_modelslice(self, layer):
        return self._slice_cache.get((layer,))

    def save_modelslice(self, layer, ms):
        self._slice_cache[(layer,)] = ms

    def get_initial_explanatory(self, node, layer, h, theta, target_K):
        best_k = -1
        best_exp = None
        for key, exp in self._exp_cache.items():
            n, l, hh, th, kk = key
            if n == node and l == layer and hh == h and th == theta and kk < target_K and kk > best_k:
                best_k = kk
                best_exp = exp
        return best_exp

    def save_explanatory(self, node, layer, h, theta, K, explanatory):
        self._exp_cache[(node, layer, h, theta, K)] = list(explanatory)

    def stats(self):
        return {
            'subfunction_entries': len(self._sf_cache),
            'explanatory_entries': len(self._exp_cache),
            'modelslice_entries': len(self._slice_cache),
        }


class SliceGXExecutor:
    """查询执行引擎：解析 → 规划 → 执行 → 过滤 → 对比"""

    def __init__(self, config, dataset, state_dict, device, logger):
        self.config = config
        self.dataset = dataset
        self.state_dict = state_dict
        self.device = device
        self.logger = logger
        self.cache = ResultCache()
        self.optimizer = QueryOptimizer()
        self.validator = QueryValidator()
        self.layer_nums = len(config.models.param.gnn_latent_dim)

    def execute(self, query) -> QueryExecutionResult:
        start_time = time.time()

        self.validator.validate(query)
        self._fill_defaults(query)

        test_nodes = self._resolve_nodes(query)
        if not test_nodes:
            return QueryExecutionResult(
                query=self._query_summary(query),
                algorithm="unknown",
                plan={},
                total_results=0,
                filtered_results=0,
                results=[],
                time_seconds=0.0,
                cache_stats=self.cache.stats(),
                error=f'No test nodes found for target={query.target}',
            )

        plan = self.optimizer.plan(query, test_nodes, self.cache.stats())
        query.algorithm = plan.physical.algorithm
        self.logger.info(
            f'[Planner] algorithm={query.algorithm}, nodes={len(test_nodes)}, '
            f'layer={query.layer}, cost={plan.physical.estimated_cost}'
        )
        for reason in plan.physical.reasons:
            self.logger.info(f'[Planner] reason={reason}')

        if query.plan_only:
            elapsed = time.time() - start_time
            self.logger.info(f'[Done] plan-only request, time={elapsed:.3f}s')
            return QueryExecutionResult(
                query=self._query_summary(query),
                algorithm=query.algorithm,
                plan=plan.to_dict(),
                total_results=0,
                filtered_results=0,
                results=[],
                comparison=None,
                time_seconds=round(elapsed, 3),
                cache_stats=self.cache.stats(),
                plan_only=True,
            )

        if query.algorithm == 'SS':
            raw_results = self._run_ss(query, test_nodes)
        elif query.algorithm == 'MS':
            raw_results = self._run_ms(query, test_nodes)
        else:
            raw_results = self._run_mm(query, test_nodes)

        filtered = self._apply_filters(raw_results, query)
        ranked = self._rank(filtered, query.rank_by) if query.rank_by and filtered else filtered
        compared = self._compare(ranked, query.compare_by) if query.compare_by and ranked else None

        elapsed = time.time() - start_time
        self.logger.info(f'[Done] {len(ranked)}/{len(raw_results)} results passed filters, time={elapsed:.3f}s')

        return QueryExecutionResult(
            query=self._query_summary(query),
            algorithm=query.algorithm,
            plan=plan.to_dict(),
            total_results=len(raw_results),
            filtered_results=len(ranked),
            results=[ExplanationResult.from_raw(item) for item in ranked],
            comparison=compared,
            time_seconds=round(elapsed, 3),
            cache_stats=self.cache.stats(),
            plan_only=False,
        )

    @staticmethod
    def compare_saved_result(saved_result: QueryExecutionResult, compare_by: str, source_name: str) -> QueryExecutionResult:
        """Run comparison operators over a previously materialized result set."""
        raw_results = [item.to_dict() for item in saved_result.results]
        compared = SliceGXExecutor._compare(raw_results, compare_by) if raw_results else None
        return QueryExecutionResult(
            query={
                "target": "saved_result",
                "source_name": source_name,
                "compare_by": compare_by,
                "plan_only": False,
            },
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.total_results,
            filtered_results=saved_result.filtered_results,
            results=list(saved_result.results),
            comparison=ComparisonSummary.from_raw(compared) if compared else None,
            time_seconds=0.0,
            cache_stats=dict(saved_result.cache_stats),
            plan_only=False,
        )

    @staticmethod
    def filter_saved_result(saved_result: QueryExecutionResult, filter_query, source_name: str) -> QueryExecutionResult:
        """Apply WHERE-style filters over a previously materialized result set."""
        raw_results = [item.raw if item.raw else item.to_dict() for item in saved_result.results]
        filtered = SliceGXExecutor._apply_filters(raw_results, filter_query)
        return QueryExecutionResult(
            query={
                "target": "saved_result",
                "source_name": source_name,
                "filter_applied": {
                    "require_factual": filter_query.require_factual,
                    "require_counterfactual": filter_query.require_counterfactual,
                    "fid_plus_threshold": filter_query.fid_plus_threshold,
                    "fid_minus_threshold": filter_query.fid_minus_threshold,
                },
                "plan_only": False,
            },
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.filtered_results,
            filtered_results=len(filtered),
            results=[ExplanationResult.from_raw(item) for item in filtered],
            comparison=None,
            time_seconds=0.0,
            cache_stats=dict(saved_result.cache_stats),
            plan_only=False,
        )

    @staticmethod
    def rank_saved_result(saved_result: QueryExecutionResult, rank_by: str, source_name: str) -> QueryExecutionResult:
        """Rank a previously materialized result set by a supported metric."""
        metric = rank_by.lower()
        sorted_results = SliceGXExecutor._rank(list(saved_result.results), metric)
        return QueryExecutionResult(
            query={
                "target": "saved_result",
                "source_name": source_name,
                "rank_by": metric,
                "plan_only": False,
            },
            algorithm=saved_result.algorithm,
            plan=saved_result.plan,
            total_results=saved_result.filtered_results,
            filtered_results=len(sorted_results),
            results=sorted_results,
            comparison=None,
            time_seconds=0.0,
            cache_stats=dict(saved_result.cache_stats),
            plan_only=False,
        )

    def _resolve_nodes(self, query) -> List[int]:
        data = self.dataset.data
        if query.target == 'node':
            return query.node_ids
        if query.target == 'all':
            if self.config.datasets.dataset_name in ['tree_grid', 'tree_cycle']:
                return torch.where(data.test_mask * data.y != 0)[0].tolist()
            return torch.where(data.test_mask)[0].tolist()
        if query.target == 'class':
            mask = data.test_mask * data.y != 0 if self.config.datasets.dataset_name in ['tree_grid', 'tree_cycle'] else data.test_mask
            indices = torch.where(mask)[0]
            return [idx.item() for idx in indices if data.y[idx].item() == query.class_label]
        return []

    @staticmethod
    def _first_or_val(v, default):
        try:
            return v[0]
        except (TypeError, IndexError, KeyError):
            return v if v is not None else default

    def _fill_defaults(self, query):
        cfg = self.config.datasets
        if query.K is None:
            query.K = int(self._first_or_val(getattr(cfg, 'K', 10), 10))
        if query.h is None:
            query.h = float(self._first_or_val(getattr(cfg, 'h', 0.3), 0.3))
        if query.theta is None:
            query.theta = float(self._first_or_val(getattr(cfg, 'theta', 0.2), 0.2))
        if query.gamma is None:
            query.gamma = float(getattr(cfg, 'gamma', 0.5))
        query.K = int(query.K)
        query.h = float(query.h)
        query.theta = float(query.theta)
        query.gamma = float(query.gamma)
        if query.max_subgraph_size is not None:
            query.K = min(query.K, query.max_subgraph_size)

    def _run_ss(self, query, test_nodes: List[int]) -> List[dict]:
        cut_layer = max(0, min(query.layer if query.layer >= 0 else 0, self.layer_nums - 1))
        sample_ratio = query.sample_ratio if query.approximate else 1.0

        modelslice = self.cache.get_modelslice(cut_layer)
        if modelslice is None:
            modelslice = Slicedmodel(
                self.config, self.device, self.layer_nums,
                self.layer_nums - cut_layer, self.logger,
                self.dataset, self.state_dict)
            self.cache.save_modelslice(cut_layer, modelslice)

        sf_key = (cut_layer, query.h, query.theta, sample_ratio)
        quality = self.cache.get_subfunction(*sf_key)
        if quality is None:
            dec = Declarative(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
            quality = Subfunction(test_nodes, dec, modelslice, self.logger, self.device, sample_ratio=sample_ratio)
            self.cache.save_subfunction(*sf_key, quality)
            self.logger.info(f'[Cache MISS] Subfunction computed for layer={cut_layer}')
        else:
            self.logger.info(f'[Cache HIT] Subfunction reused for layer={cut_layer}')

        dec = Declarative(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        results = []
        has_constraints = bool(query.include_nodes or query.exclude_nodes)

        for node in test_nodes:
            init_exp = None
            if not has_constraints:
                init_exp = self.cache.get_initial_explanatory(node, cut_layer, query.h, query.theta, query.K)
                if init_exp:
                    self.logger.info(f'[Cache HIT] Resuming greedy for node {node} from K={len(init_exp)}')

            self.dataset.data.to(self.device)
            algorithm = GreedyAlgorithm(
                dec, modelslice, node, self.logger, quality,
                include_nodes=query.include_nodes,
                exclude_nodes=query.exclude_nodes,
                initial_explanatory=init_exp)
            optimal = algorithm.get_solution()

            if not has_constraints:
                self.cache.save_explanatory(node, cut_layer, query.h, query.theta, query.K, algorithm.explanatory)

            if optimal is not None:
                optimal['node_id'] = node
                results.append(optimal)

        return results

    def _run_ms(self, query, test_nodes: List[int]) -> List[dict]:
        from Slice_MS import GreedyAlgorithm as GreedyMS, Subfunction as SubMS, Slicedmodel as SliceMS, Declarative as DecMS

        cut_layer = max(0, min(query.layer if query.layer >= 0 else 0, self.layer_nums - 1))
        num_hop = self.layer_nums - cut_layer

        dec = DecMS(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMS(self.config, self.device, num_hop, self.logger, self.dataset, self.state_dict)
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

    def _run_mm(self, query, test_nodes: List[int]) -> List[dict]:
        from Slice_MM import GreedyAlgorithm as GreedyMM, Subfunction as SubMM, Slicedmodel as SliceMM, Declarative as DecMM

        dec = DecMM(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMM(self.config, self.device, self.layer_nums, self.logger, self.dataset, self.state_dict)
        quality = SubMM(test_nodes, dec, modelslice, self.logger, self.device)

        self.dataset.data.to(self.device)
        algorithm = GreedyMM(dec, modelslice, test_nodes, self.logger, quality)
        all_optimal = algorithm.get_all_solution()

        results = []
        for layer_idx in range(len(all_optimal)):
            for node_idx, opt in enumerate(all_optimal[layer_idx]):
                if opt is not None:
                    opt['node_id'] = test_nodes[node_idx] if node_idx < len(test_nodes) else -1
                    opt['layer'] = layer_idx
                    results.append(opt)
        return results

    @staticmethod
    def _apply_filters(results: List[dict], query) -> List[dict]:
        filtered = []
        for r in results:
            if query.require_factual is not None and r.get('factual') != query.require_factual:
                continue
            if query.require_counterfactual is not None and r.get('counterfactual') != query.require_counterfactual:
                continue
            if query.fid_plus_threshold is not None and r.get('Fid+', -999) <= query.fid_plus_threshold:
                continue
            if query.fid_minus_threshold is not None and r.get('Fid-', 999) >= query.fid_minus_threshold:
                continue
            filtered.append(r)
        return filtered

    @staticmethod
    def _compare(results: List[dict], compare_by: str) -> dict:
        if compare_by == 'fidelity_plus':
            best = max(results, key=lambda r: r.get('Fid+', r.get('fidelity_plus', -999)))
            return {
                'type': 'best_fidelity_plus',
                'best_node': best.get('node_id'),
                'best_fid_plus': best.get('Fid+', best.get('fidelity_plus')),
                'best_nodes': best.get('nodes'),
            }
        if compare_by == 'common_nodes':
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

    @staticmethod
    def _rank(results, rank_by: str):
        if rank_by == 'fidelity_plus':
            return sorted(
                list(results),
                key=lambda item: item.get('Fid+', item.get('fidelity_plus', -999))
                if isinstance(item, dict) else item.fidelity_plus,
                reverse=True,
            )
        raise ValueError(f"Unsupported rank metric: {rank_by}")

    @staticmethod
    def _query_summary(query) -> dict:
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
            'rank_by': query.rank_by,
            'approximate': query.approximate,
            'sample_ratio': query.sample_ratio if query.approximate else 1.0,
            'plan_only': query.plan_only,
        }
