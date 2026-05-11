from __future__ import annotations

from typing import Any, Optional

from ..response import AIResponse, AITokens
from .base import BaseProvider, TextResult, text_messages


class CohereProvider(BaseProvider):
    """Provider adapter for Cohere chat and embeddings."""

    name = "cohere"
    base_url = "https://api.cohere.com"

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Call Cohere v2 chat."""

        source = text_messages(prompt, system, messages)
        cohere_messages = [
            {
                "role": "assistant" if item["role"] == "assistant" else "user",
                "content": str(item["content"]),
            }
            for item in source
            if item.get("role") != "system"
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": cohere_messages,
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }
        system_text = system or next(
            (str(item["content"]) for item in source if item.get("role") == "system"), None
        )
        if system_text:
            payload["preamble"] = system_text
        raw, latency_ms = self._measure(lambda: self._post_json("/v2/chat", payload))
        message = raw.get("message") or {}
        content = message.get("content") or []
        text = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
        usage = raw.get("usage") or {}
        tokens = usage.get("tokens") or {}
        input_tokens = int(tokens.get("input_tokens") or 0)
        output_tokens = int(tokens.get("output_tokens") or 0)
        return self._build_response(
            model=model,
            mode="text",
            text=text,
            tokens=AITokens(
                input=input_tokens, output=output_tokens, total=input_tokens + output_tokens
            ),
            latency_ms=latency_ms,
            raw=raw,
            finish_reason=str(raw.get("finish_reason") or ""),
            request_id=raw.get("id"),
        )

    def embed(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create Cohere embeddings."""

        payload = {
            "model": model,
            "texts": kwargs.pop("texts", [prompt]),
            "input_type": kwargs.pop("input_type", "search_document"),
            **kwargs,
        }
        raw, latency_ms = self._measure(lambda: self._post_json("/v2/embed", payload))
        embeddings = raw.get("embeddings") or {}
        vectors = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
        first = (vectors or [[]])[0]
        usage = raw.get("meta", {}).get("billed_units", {})
        input_tokens = int(usage.get("input_tokens") or 0)
        return self._build_response(
            model=model,
            mode="embed",
            embeddings=[float(value) for value in first],
            tokens=AITokens(input=input_tokens, output=0, total=input_tokens),
            latency_ms=latency_ms,
            raw=raw,
            request_id=raw.get("id"),
        )
