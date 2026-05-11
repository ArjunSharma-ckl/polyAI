from __future__ import annotations

from typing import Any, Optional

from ..response import AIResponse, AITokens
from .base import BaseProvider, TextResult, image_inputs, image_to_base64, is_url


class GoogleProvider(BaseProvider):
    """Provider adapter for Google Gemini via the Generative Language REST API."""

    name = "google"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Generate text with Gemini."""

        payload = self._payload(prompt=prompt, system=system, messages=messages, **kwargs)
        raw, latency_ms = self._measure(
            lambda: self._post_json(self._generate_endpoint(model), payload)
        )
        return self._response(model, raw, latency_ms, mode="text")

    def vision(
        self,
        model: str,
        prompt: str,
        image: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Generate text from images with Gemini."""

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for item in image_inputs(image, images):
            if is_url(item):
                parts.append({"file_data": {"mime_type": "image/png", "file_uri": item}})
            else:
                mime, data = image_to_base64(item)
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._generation_config(kwargs),
        }
        raw, latency_ms = self._measure(
            lambda: self._post_json(self._generate_endpoint(model), payload)
        )
        return self._response(model, raw, latency_ms, mode="vision")

    def _payload(
        self,
        *,
        prompt: Optional[str],
        system: Optional[str],
        messages: Optional[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        contents: list[dict[str, Any]]
        if messages:
            contents = []
            for message in messages:
                role = "model" if message.get("role") == "assistant" else "user"
                content = message.get("content", "")
                if message.get("role") == "system":
                    system = str(content)
                    continue
                contents.append({"role": role, "parts": [{"text": str(content)}]})
        else:
            if prompt is None:
                from ..exceptions import InvalidInputError

                raise InvalidInputError(
                    "Missing prompt for Gemini call.", fix='Pass prompt="..." to AI(...).'
                )
            contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": self._generation_config(kwargs),
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        return payload

    def _generation_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": kwargs.pop("temperature", 0.7),
            "maxOutputTokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }

    def _generate_endpoint(self, model: str) -> str:
        return f"/models/{model}:generateContent?key={self.api_key}"

    def _response(
        self, model: str, raw: dict[str, Any], latency_ms: int, *, mode: str
    ) -> AIResponse:
        candidate = (raw.get("candidates") or [{}])[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        usage = raw.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        total = int(usage.get("totalTokenCount") or input_tokens + output_tokens)
        return self._build_response(
            model=model,
            mode=mode,
            text=text,
            tokens=AITokens(input=input_tokens, output=output_tokens, total=total),
            latency_ms=latency_ms,
            raw=raw,
            finish_reason=str(candidate.get("finishReason") or ""),
        )
