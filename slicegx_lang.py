"""
SliceGX Declarative Query Language
===================================
声明式 GNN 解释查询语言入口。
"""

import os
import sys
import json
import re
from typing import List
from warnings import simplefilter

import hydra
import torch

from dataset import get_dataset
from llm.nl2query import NL2QueryService
from llm.provider import OpenAICompatibleProvider
from llm.result2nl import Result2NLService
from query_executor import SliceGXExecutor
from query_formatter import format_result, result_to_json
from query_parser import QueryParser
from query_session import QuerySessionStore
from query_validator import QueryValidationError
from utils import get_logger


CLI_QUERY_ARG = None
CLI_NL_QUERY_ARG = None
CLI_OUTPUT_FORMAT = 'text'
CLI_PLAN_ONLY = False
CLI_NARRATE = False


def _extract_cli_args(argv: List[str]) -> List[str]:
    """Strip custom CLI flags before Hydra parses argv."""
    global CLI_QUERY_ARG, CLI_NL_QUERY_ARG, CLI_OUTPUT_FORMAT, CLI_PLAN_ONLY, CLI_NARRATE
    filtered = [argv[0]]
    for arg in argv[1:]:
        if arg.startswith('--query='):
            CLI_QUERY_ARG = arg[len('--query='):]
        elif arg.startswith('--nl-query='):
            CLI_NL_QUERY_ARG = arg[len('--nl-query='):]
        elif arg == '--query':
            continue
        elif arg == '--nl-query':
            continue
        elif arg == '--plan-only':
            CLI_PLAN_ONLY = True
        elif arg == '--narrate':
            CLI_NARRATE = True
        elif arg.startswith('--output-format='):
            CLI_OUTPUT_FORMAT = arg[len('--output-format='):].strip().lower() or 'text'
        elif arg == '--output-format':
            continue
        else:
            filtered.append(arg)
    return filtered


def _build_llm_services():
    provider = OpenAICompatibleProvider.from_env()
    return NL2QueryService(provider), Result2NLService(provider)


def _execute_session_expression(expr: str, parser: QueryParser, executor: SliceGXExecutor, session_store: QuerySessionStore):
    compare_match = re.match(
        r'^\s*COMPARE\s+([A-Za-z_][A-Za-z0-9_]*)\s+BY\s+(FIDELITY_PLUS|COMMON_NODES)\s*$',
        expr,
        flags=re.IGNORECASE,
    )
    rank_match = re.match(
        r'^\s*RANK\s+([A-Za-z_][A-Za-z0-9_]*)\s+BY\s+(FIDELITY_PLUS)\s*$',
        expr,
        flags=re.IGNORECASE,
    )
    filter_match = re.match(
        r'^\s*FILTER\s+([A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+(.+)$',
        expr,
        flags=re.IGNORECASE,
    )

    if compare_match:
        name = compare_match.group(1)
        metric = compare_match.group(2).lower()
        saved = session_store.get(name)
        if saved is None:
            raise QueryValidationError(f"named result {name.upper()} not found.")
        return SliceGXExecutor.compare_saved_result(saved, metric, name.upper())

    if rank_match:
        name = rank_match.group(1)
        metric = rank_match.group(2).lower()
        saved = session_store.get(name)
        if saved is None:
            raise QueryValidationError(f"named result {name.upper()} not found.")
        return SliceGXExecutor.rank_saved_result(saved, metric, name.upper())

    if filter_match:
        name = filter_match.group(1)
        where_clause = filter_match.group(2).strip()
        saved = session_store.get(name)
        if saved is None:
            raise QueryValidationError(f"named result {name.upper()} not found.")
        filter_query = parser.parse(f"EXPLAIN NODE 0 WHERE {where_clause}")
        return SliceGXExecutor.filter_saved_result(saved, filter_query, name.upper())

    query = parser.parse(expr)
    return executor.execute(query)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]

    if config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        dataset, _, _, _ = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    else:
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    if config.datasets.dataset_name in ['products']:
        dataset.data.y = torch.argmax(dataset.data.y, dim=1)
    dataset.data.y = dataset.data.y.squeeze().long()

    log_file = f"{config.datasets.dataset_name}_lang.log"
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_savedir,
            config.datasets.dataset_name,
            f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l_best.pth',
        )
    )['net']

    executor = SliceGXExecutor(config, dataset, state_dict, device, logger)
    parser = QueryParser()
    session_store = QuerySessionStore()

    if CLI_QUERY_ARG or CLI_NL_QUERY_ARG:
        try:
            narrator = None
            if CLI_NL_QUERY_ARG:
                translator, narrator = _build_llm_services()
                translation = translator.translate(CLI_NL_QUERY_ARG)
                if translation.needs_clarification:
                    print(f"Clarification required: {translation.clarification_question}")
                    return
                print(f"Generated query: {translation.query_text}")
                query = translation.query
            else:
                query = parser.parse(CLI_QUERY_ARG)
            if CLI_PLAN_ONLY:
                query.plan_only = True
            result = executor.execute(query)
            if CLI_OUTPUT_FORMAT == 'json':
                print(result_to_json(result))
            else:
                print(format_result(result))
            if CLI_NARRATE:
                if narrator is None:
                    _, narrator = _build_llm_services()
                print(f"\n=== LLM Grounded Summary ===\n{narrator.narrate(result)}")
        except QueryValidationError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"SliceGX Query Language (dataset={config.datasets.dataset_name}, device={device})")
        print("Type 'help' for syntax, 'exit' to quit.\n")

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
            if query_str.lower() == 'list':
                names = session_store.list_names()
                print(json.dumps(names, indent=2, ensure_ascii=False))
                continue
            if query_str.upper().startswith('ASK '):
                try:
                    translator, _ = _build_llm_services()
                    translation = translator.translate(query_str[4:].strip())
                    if translation.needs_clarification:
                        print(f"Clarification required: {translation.clarification_question}")
                        continue
                    print(f"Generated query: {translation.query_text}")
                    result = executor.execute(translation.query)
                    print(format_result(result))
                except Exception as e:
                    print(f"LLM error: {e}")
                continue
            summarize_match = re.match(
                r'^\s*SUMMARIZE\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
                query_str,
                flags=re.IGNORECASE,
            )
            if summarize_match:
                saved = session_store.get(summarize_match.group(1))
                if saved is None:
                    print(f"Validation error: named result {summarize_match.group(1).upper()} not found.")
                    continue
                try:
                    _, narrator = _build_llm_services()
                    print(narrator.narrate(saved))
                except Exception as e:
                    print(f"LLM error: {e}")
                continue
            try:
                let_match = re.match(r'^\s*LET\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$', query_str, flags=re.IGNORECASE)

                if let_match:
                    name = let_match.group(1)
                    rhs = let_match.group(2)
                    result = _execute_session_expression(rhs, parser, executor, session_store)
                    if result.error:
                        print(format_result(result))
                    elif result.plan_only:
                        print("Validation error: LET cannot store a plan-only query result.")
                    else:
                        session_store.save(name, result)
                        if result.materialized_as:
                            session_store.save(result.materialized_as, result)
                        print(f"Stored result as {name.upper()} ({result.filtered_results} rows).")
                else:
                    result = _execute_session_expression(query_str, parser, executor, session_store)
                    if result.materialized_as:
                        session_store.save(result.materialized_as, result)
                    print(format_result(result))
            except QueryValidationError as e:
                print(f"Validation error: {e}")
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
  RANK <name> BY FIDELITY_PLUS       Rank stored results by Fid+

Result algebra:
  PROJECT NODE_ID,LAYER,FIDELITY_PLUS
  GROUP BY LAYER PATTERN MIN_SUPPORT 0.5
  MATERIALIZE AS Q1

Parameters:
  WITH K 6                           Override subgraph size
  WITH H 0.2                         Override influence threshold
  WITH THETA 0.1                     Override diversity threshold
  WITH APPROXIMATE 0.3               Approximate mode (30% sampling)
  WITH MAX_ERROR 0.1                 Quality-aware approximate policy
  WITH MIN_CONFIDENCE 0.9            Minimum estimated confidence
  WITH TIME_BUDGET 10                Requested execution-time budget

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
  list    List stored LET results
  exit    Quit

LLM interaction:
  ASK <natural language>             Translate, validate, and execute an NL request
  SUMMARIZE <name>                   Grounded NL summary of a stored result

Composition:
  LET Q1 = EXPLAIN ALL WHERE FACTUAL = TRUE
  LET Q2 = FILTER Q1 WHERE FACTUAL = TRUE
  LET Q3 = RANK Q2 BY FIDELITY_PLUS
  COMPARE Q1 BY COMMON_NODES
  COMPARE Q1 BY FIDELITY_PLUS
  FILTER Q1 WHERE FIDELITY_PLUS > 0.5
  RANK Q1 BY FIDELITY_PLUS
""")


if __name__ == '__main__':
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv = _extract_cli_args(sys.argv)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
