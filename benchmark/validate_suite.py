import argparse
import json
import re
from dataclasses import asdict

from benchmark.workloads import ExplanationAnalyticsWorkloads
from query_parser import QueryParser
from query_validator import QueryValidator


def extract_explain_statement(statement: str):
    let_match = re.match(r"^LET\s+\w+\s*=\s*(EXPLAIN.+)$", statement, flags=re.IGNORECASE)
    if let_match:
        return let_match.group(1)
    if statement.upper().startswith("EXPLAIN"):
        return statement
    return None


def main():
    argument_parser = argparse.ArgumentParser(description="Validate the canonical SliceGX TKDE workload suite.")
    argument_parser.add_argument("--nodes", default="519,537")
    argument_parser.add_argument("--class-label", type=int, default=1)
    args = argument_parser.parse_args()

    nodes = [int(value) for value in args.nodes.split(",")]
    cases = ExplanationAnalyticsWorkloads.build(nodes, args.class_label)
    query_parser = QueryParser()
    validator = QueryValidator()
    validated = 0
    for case in cases:
        for statement in case.statements:
            explain_statement = extract_explain_statement(statement)
            if explain_statement is None:
                continue
            validator.validate(query_parser.parse(explain_statement))
            validated += 1
    print(json.dumps({"cases": [asdict(case) for case in cases], "validated_queries": validated}, indent=2))


if __name__ == "__main__":
    main()
