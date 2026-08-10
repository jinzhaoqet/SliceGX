import json
from typing import Any, Dict

from llm.provider import LLMProvider
from result_schema import QueryExecutionResult


RESULT_SYSTEM_PROMPT = """You explain a SliceGX query result using only the supplied JSON evidence.
Never invent node ids, layers, metrics, support values, causes, or optimizer behavior.
Distinguish observed association from causality. If the evidence is insufficient, say so explicitly.
Keep node ids and numerical metrics in the response. Mention query errors before any interpretation.
Return JSON only: {"summary": "..."}.
"""


class Result2NLError(RuntimeError):
    pass


class Result2NLService:
    def __init__(self, provider: LLMProvider, max_results: int = 20):
        self.provider = provider
        self.max_results = max_results

    def narrate(self, result: QueryExecutionResult) -> str:
        evidence = self.build_evidence(result)
        raw_output = self.provider.complete(
            [
                {"role": "system", "content": RESULT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, sort_keys=True)},
            ],
            temperature=0.0,
        )
        try:
            payload = json.loads(raw_output.strip())
            summary = payload.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("summary must be a non-empty string")
            return summary.strip()
        except (json.JSONDecodeError, AttributeError, ValueError) as error:
            raise Result2NLError(f"Invalid Result2NL response: {error}") from error

    def build_evidence(self, result: QueryExecutionResult) -> Dict[str, Any]:
        payload = result.to_dict()
        payload["results"] = payload["results"][: self.max_results]
        payload["evidence_policy"] = {
            "returned_results": len(payload["results"]),
            "total_filtered_results": result.filtered_results,
            "results_truncated": result.filtered_results > len(payload["results"]),
            "causal_claims_supported": False,
        }
        return payload
