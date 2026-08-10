import os
from typing import Dict, List, Protocol

import requests


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        ...


class OpenAICompatibleProvider:
    """Minimal provider for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout_seconds: float = 60.0):
        if not api_key:
            raise LLMProviderError("SLICEGX_LLM_API_KEY is required.")
        if not model:
            raise LLMProviderError("SLICEGX_LLM_MODEL is required.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            api_key=os.environ.get("SLICEGX_LLM_API_KEY", ""),
            model=os.environ.get("SLICEGX_LLM_MODEL", ""),
            base_url=os.environ.get("SLICEGX_LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=float(os.environ.get("SLICEGX_LLM_TIMEOUT", "60")),
        )

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"])
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMProviderError(f"LLM request failed: {error}") from error
