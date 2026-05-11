from __future__ import annotations

from typing import Any, Optional

from ..response import AIResponse
from .base import BaseProvider, TextResult, text_messages, usage_from_openai


class MistralProvider(BaseProvider):
    """Provider adapter for Mistral AI."""

    name = "mistral"
    base_url = "https://api.mistral.ai/v1"

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Call Mistral chat completions."""

        payload = {
            "model": model,
            "messages": text_messages(prompt, system, messages),
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }
        raw, latency_ms = self._measure(lambda: self._post_json("/chat/completions", payload))
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return self._build_response(
            model=model,
            mode="text",
            text=str(message.get("content") or ""),
            tokens=usage_from_openai(raw),
            latency_ms=latency_ms,
            raw=raw,
            finish_reason=str(choice.get("finish_reason") or ""),
            request_id=raw.get("id"),
        )

    def embed(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create Mistral embeddings."""

        payload = {"model": model, "input": kwargs.pop("input", [prompt]), **kwargs}
        raw, latency_ms = self._measure(lambda: self._post_json("/embeddings", payload))
        first = (raw.get("data") or [{}])[0]
        embedding = first.get("embedding") or []
        return self._build_response(
            model=model,
            mode="embed",
            embeddings=[float(value) for value in embedding],
            tokens=usage_from_openai(raw),
            latency_ms=latency_ms,
            raw=raw,
            request_id=raw.get("id"),
        )
