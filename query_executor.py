import time
import copy
from collections import Counter
from typing import Any, Dict, List

import torch

from SliceGX import Slicedmodel, Subfunction, Declarative, GreedyAlgorithm
from analytics.algebra import AlgebraExecutor, GroupPattern, Project
from analytics.data_model import ExplanationSet
from planner.optimizer import QueryOptimizer
from query_validator import QueryValidator
from result_operations import ResultOperations
from result_schema import ExplanationResult, QueryExecutionResult


class ResultCache:
    """缓存 Subfunction 和贪心中间状态，支持跨查询复用"""

    def __init__(self):
        self._sf_cache: Dict[tuple, Any] = {}
        self._exp_cache: Dict[tuple, list] = {}
        self._slice_cache: Dict[tuple, Any] = {}
        self._generation_cache: Dict[tuple, List[dict]] = {}
        self._materialized_results: Dict[str, QueryExecutionResult] = {}

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

    @staticmethod
    def generation_key(query, test_nodes):
        return (
            tuple(test_nodes),
            query.layer,
            query.K,
            query.h,
            query.theta,
            query.gamma,
            tuple(query.include_nodes),
            tuple(query.exclude_nodes),
            query.sample_ratio if query.approximate else 1.0,
        )

    def get_generation(self, key):
        value = self._generation_cache.get(key)
        return copy.deepcopy(value) if value is not None else None

    def save_generation(self, key, results):
        self._generation_cache[key] = copy.deepcopy(results)

    def inspect(self, query, test_nodes):
        key = self.generation_key(query, test_nodes)
        cut_layer = max(query.layer, 0)
        resumable_nodes = 0
        for node in test_nodes:
            for cache_key in self._exp_cache:
                cached_node, layer, h, theta, cached_k = cache_key
                if (
                    cached_node == node
                    and layer == cut_layer
                    and h == query.h
                    and theta == query.theta
                    and cached_k < query.K
                ):
                    resumable_nodes += 1
                    break
        return {
            "exact_generation_hit": key in self._generation_cache,
            "resumable_nodes": resumable_nodes,
            "generation_key": key,
        }

    def save_materialized_result(self, name, result):
        self._materialized_results[name.upper()] = copy.deepcopy(result)

    def get_materialized_result(self, name):
        value = self._materialized_results.get(name.upper())
        return copy.deepcopy(value) if value is not None else None

    def stats(self):
        return {
            'subfunction_entries': len(self._sf_cache),
            'explanatory_entries': len(self._exp_cache),
            'modelslice_entries': len(self._slice_cache),
            'generation_entries': len(self._generation_cache),
            'materialized_entries': len(self._materialized_results),
        }


class SliceGXExecutor:
    """查询执行引擎：解析 → 规划 → 执行 → 过滤 → 对比"""

    def __init__(self, config, dataset, state_dict, device, logger):
        self.config = config
        self.dataset = dataset
        self.state_dict = state_dict
        self.device = device
        self.logger = logger
        self.layer_nums = len(config.models.param.gnn_latent_dim)
        self.cache = ResultCache()
        self.optimizer = QueryOptimizer(layer_count=self.layer_nums)
        self.validator = QueryValidator()
        self.algebra = AlgebraExecutor()

    def execute(self, query) -> QueryExecutionResult:
        start_time = time.time()

        self.validator.validate(query)
        self._fill_defaults(query)
        if query.approximate:
            query.sample_ratio = self.optimizer.resolve_sample_ratio(query)[0]

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

        cache_context = self.cache.stats()
        cache_context.update(self.cache.inspect(query, test_nodes))
        plan = self.optimizer.plan(query, test_nodes, cache_context)
        query.algorithm = plan.physical.algorithm
        if query.approximate:
            query.sample_ratio = plan.physical.sample_ratio
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

        generation_key = self.cache.generation_key(query, test_nodes)
        if query.algorithm == 'CACHE':
            raw_results = self.cache.get_generation(generation_key) or []
        elif query.algorithm == 'SS':
            raw_results = self._run_ss(query, test_nodes)
        elif query.algorithm == 'SS_LAYERED':
            raw_results = self._run_ss_layered(query, test_nodes)
        elif query.algorithm == 'MS':
            raw_results = self._run_ms(query, test_nodes)
        else:
            raw_results = self._run_mm(query, test_nodes)
        if query.algorithm != 'CACHE':
            self.cache.save_generation(generation_key, raw_results)

        filtered = self._apply_filters(raw_results, query)
        ranked = self._rank(filtered, query.rank_by) if query.rank_by and filtered else filtered
        compared = self._compare(ranked, query.compare_by) if query.compare_by and ranked else None

        elapsed = time.time() - start_time
        if query.algorithm != 'CACHE':
            self.optimizer.record_execution(query.algorithm, plan.physical.estimated_cost, elapsed)
        self.logger.info(f'[Done] {len(ranked)}/{len(raw_results)} results passed filters, time={elapsed:.3f}s')

        result = QueryExecutionResult(
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
            analytics=self._derive_analytics(ranked, query),
            materialized_as=query.materialize_as.upper() if query.materialize_as else None,
        )
        if query.materialize_as:
            self.cache.save_materialized_result(query.materialize_as, result)
        return result

    @staticmethod
    def compare_saved_result(saved_result: QueryExecutionResult, compare_by: str, source_name: str) -> QueryExecutionResult:
        """Run comparison operators over a previously materialized result set."""
        return ResultOperations.compare(saved_result, compare_by, source_name)

    @staticmethod
    def filter_saved_result(saved_result: QueryExecutionResult, filter_query, source_name: str) -> QueryExecutionResult:
        """Apply WHERE-style filters over a previously materialized result set."""
        return ResultOperations.filter(saved_result, filter_query, source_name)

    @staticmethod
    def rank_saved_result(saved_result: QueryExecutionResult, rank_by: str, source_name: str) -> QueryExecutionResult:
        """Rank a previously materialized result set by a supported metric."""
        return ResultOperations.rank(saved_result, rank_by, source_name)

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
                optimal['layer'] = cut_layer
                results.append(optimal)

        return results

    def _run_ms(self, query, test_nodes: List[int]) -> List[dict]:
        from Slice_MS import GreedyAlgorithm as GreedyMS, Subfunction as SubMS, Slicedmodel as SliceMS, Declarative as DecMS

        cut_layer = max(0, min(query.layer if query.layer >= 0 else 0, self.layer_nums - 1))
        num_hop = self.layer_nums - cut_layer

        dec = DecMS(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMS(self.config, self.device, num_hop, self.logger, self.dataset, self.state_dict)
        quality = SubMS(
            test_nodes,
            dec,
            modelslice,
            self.logger,
            self.device,
            sample_ratio=query.sample_ratio if query.approximate else 1.0,
        )

        self.dataset.data.to(self.device)
        algorithm = GreedyMS(dec, modelslice, test_nodes, self.logger, quality)
        optimal_list = algorithm.get_solution()

        results = []
        for i, opt in enumerate(optimal_list):
            if opt is not None:
                opt['node_id'] = test_nodes[i] if i < len(test_nodes) else -1
                opt['layer'] = cut_layer
                results.append(opt)
        return results

    def _run_mm(self, query, test_nodes: List[int]) -> List[dict]:
        from Slice_MM import GreedyAlgorithm as GreedyMM, Subfunction as SubMM, Slicedmodel as SliceMM, Declarative as DecMM

        dec = DecMM(self.config, self.dataset, query.K, query.theta, query.h, query.gamma)
        modelslice = SliceMM(self.config, self.device, self.layer_nums, self.logger, self.dataset, self.state_dict)
        quality = SubMM(
            test_nodes,
            dec,
            modelslice,
            self.logger,
            self.device,
            sample_ratio=query.sample_ratio if query.approximate else 1.0,
        )

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

    def _run_ss_layered(self, query, test_nodes: List[int]) -> List[dict]:
        original_layer = query.layer
        results = []
        try:
            for layer in range(self.layer_nums):
                query.layer = layer
                layer_results = self._run_ss(query, test_nodes)
                for result in layer_results:
                    result['layer'] = layer
                results.extend(layer_results)
        finally:
            query.layer = original_layer
        return results

    @staticmethod
    def _apply_filters(results: List[dict], query) -> List[dict]:
        filtered = []
        for r in results:
            if getattr(query, 'require_factual', None) is not None and r.get('factual') != query.require_factual:
                continue
            if getattr(query, 'require_counterfactual', None) is not None and r.get('counterfactual') != query.require_counterfactual:
                continue
            if getattr(query, 'fid_plus_threshold', None) is not None and r.get('Fid+', r.get('fidelity_plus', -999)) <= query.fid_plus_threshold:
                continue
            if getattr(query, 'fid_minus_threshold', None) is not None and r.get('Fid-', r.get('fidelity_minus', 999)) >= query.fid_minus_threshold:
                continue
            if getattr(query, 'max_subgraph_size', None) is not None and len(r.get('nodes', [])) > query.max_subgraph_size:
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
                cnt.update(set(r.get('nodes', [])))
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
            'project_fields': list(query.project_fields),
            'group_by': query.group_by,
            'pattern_min_support': query.pattern_min_support,
            'materialize_as': query.materialize_as,
            'max_error': query.max_error,
            'min_confidence': query.min_confidence,
            'time_budget_seconds': query.time_budget_seconds,
        }

    def _derive_analytics(self, ranked_results, query) -> Dict[str, Any]:
        if not query.project_fields and query.pattern_min_support is None:
            return {}
        typed_results = [ExplanationResult.from_raw(item) if isinstance(item, dict) else item for item in ranked_results]
        explanation_set = ExplanationSet.from_results(
            typed_results,
            set_id=query.materialize_as or "query-result",
            graph_id=str(self.config.datasets.dataset_name),
            model_id=str(self.config.models.gnn_name),
        )
        analytics: Dict[str, Any] = {}
        if query.project_fields:
            projected = self.algebra.execute(explanation_set, Project(tuple(query.project_fields)))
            analytics["projection"] = {
                "fields": list(projected.fields),
                "rows": projected.rows,
            }
        if query.pattern_min_support is not None:
            pattern_set = self.algebra.execute(
                explanation_set,
                GroupPattern(group_by=query.group_by, min_support=query.pattern_min_support),
            )
            analytics["patterns"] = {
                "group_by": pattern_set.group_by,
                "items": [
                    {
                        "pattern_id": pattern.pattern_id,
                        "nodes": list(pattern.nodes),
                        "support": pattern.support,
                        "source_explanation_ids": list(pattern.source_explanation_ids),
                        "group_key": pattern.group_key,
                    }
                    for pattern in pattern_set.patterns
                ],
            }
        return analytics
