import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm.intent_schema import IntentValidationError, QueryIntent, QueryIntentCompiler
from llm.provider import LLMProvider
from query_parser import ExplainQuery, QueryParser
from query_validator import QueryValidationError, QueryValidator


SYSTEM_PROMPT = """You translate natural-language GNN explanation requests into SliceGX QueryIntent JSON.
Return one JSON object only. Do not invent node ids, class labels, thresholds, or layers.
If essential information is ambiguous or missing, set needs_clarification=true and provide one concise clarification_question.

Schema:
{
  "needs_clarification": false,
  "clarification_question": null,
  "intent": {
    "target": {"type": "node|all|class", "node_ids": [], "class_label": null},
    "filters": {"factual": null, "counterfactual": null, "fidelity_plus_gt": null,
                "fidelity_minus_lt": null, "subgraph_size_lte": null},
    "layer": {"mode": "default|single|all", "index": null},
    "structure": {"include_nodes": [], "exclude_nodes": []},
    "compare_by": null,
    "rank_by": null,
    "project_fields": [],
    "group_by": null,
    "pattern_min_support": null,
    "materialize_as": null,
    "parameters": {"K": null, "h": null, "theta": null, "gamma": null,
                   "approximate_ratio": null, "max_error": null,
                   "min_confidence": null, "time_budget_seconds": null}
  }
}

Allowed compare_by: fidelity_plus, common_nodes, or null.
Allowed rank_by: fidelity_plus or null.
Allowed group_by: layer, factual, counterfactual, or null.
Allowed project_fields: explanation_id, node_id, nodes, factual, counterfactual,
fidelity_plus, fidelity_minus, score, subgraph_size, layer.
Fidelity filters are result filters. INCLUDE/EXCLUDE are generation constraints.
Use target.type=all only when the user explicitly requests all test nodes.
"""


@dataclass
class TranslationResult:
    natural_language: str
    needs_clarification: bool
    clarification_question: Optional[str] = None
    intent: Optional[QueryIntent] = None
    query_text: Optional[str] = None
    query: Optional[ExplainQuery] = None
    attempts: int = 0


class NL2QueryError(RuntimeError):
    pass


class NL2QueryService:
    def __init__(self, provider: LLMProvider, max_repairs: int = 2):
        self.provider = provider
        self.max_repairs = max_repairs
        self.compiler = QueryIntentCompiler()
        self.parser = QueryParser()
        self.validator = QueryValidator()

    def translate(self, natural_language: str) -> TranslationResult:
        if not natural_language.strip():
            raise NL2QueryError("Natural-language query cannot be empty.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": natural_language.strip()},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_repairs + 2):
            raw_output = self.provider.complete(messages, temperature=0.0)
            try:
                payload = self._parse_json(raw_output)
                if bool(payload.get("needs_clarification", False)):
                    question = str(payload.get("clarification_question") or "Please clarify the query intent.")
                    return TranslationResult(
                        natural_language=natural_language,
                        needs_clarification=True,
                        clarification_question=question,
                        attempts=attempt,
                    )
                intent_payload = payload.get("intent")
                intent = QueryIntent.from_dict(intent_payload)
                query_text = self.compiler.compile(intent)
                query = self.parser.parse(query_text)
                self.validator.validate(query)
                return TranslationResult(
                    natural_language=natural_language,
                    needs_clarification=False,
                    intent=intent,
                    query_text=query_text,
                    query=query,
                    attempts=attempt,
                )
            except (json.JSONDecodeError, IntentValidationError, QueryValidationError, TypeError) as error:
                last_error = error
                if attempt > self.max_repairs:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_output},
                        {
                            "role": "user",
                            "content": (
                                "The previous output failed deterministic validation with this error: "
                                f"{error}. Return a corrected JSON object only."
                            ),
                        },
                    ]
                )
        raise NL2QueryError(f"Unable to produce a valid query after validation and repair: {last_error}")

    @staticmethod
    def _parse_json(raw_output: str) -> Dict[str, Any]:
        stripped = raw_output.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            stripped = fenced.group(1)
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise TypeError("LLM output must be a JSON object.")
        return payload
