from __future__ import annotations

from typing import Any, Optional

from ..response import AIResponse, AITokens
from .base import BaseProvider, TextResult, image_inputs, image_to_base64, is_url, text_messages


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic Claude."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": str(self.options.get("anthropic_version", "2023-06-01")),
            "Content-Type": "application/json",
        }

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Call Anthropic Messages API."""

        payload = self._payload(model, prompt, system, messages, **kwargs)
        raw, latency_ms = self._measure(lambda: self._post_json("/messages", payload))
        return self._response(model, raw, latency_ms, mode="text")

    def vision(
        self,
        model: str,
        prompt: str,
        image: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Call Anthropic vision models."""

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in image_inputs(image, images):
            if is_url(item):
                content.append({"type": "image", "source": {"type": "url", "url": item}})
            else:
                mime, data = image_to_base64(item)
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data},
                    }
                )
        payload = self._payload(
            model,
            None,
            kwargs.pop("system", None),
            [{"role": "user", "content": content}],
            **kwargs,
        )
        raw, latency_ms = self._measure(lambda: self._post_json("/messages", payload))
        return self._response(model, raw, latency_ms, mode="vision")

    def _payload(
        self,
        model: str,
        prompt: Optional[str],
        system: Optional[str],
        messages: Optional[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        source_messages = text_messages(prompt, None, messages)
        anthropic_messages = [
            message for message in source_messages if message.get("role") != "system"
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "temperature": kwargs.pop("temperature", 0.7),
            **kwargs,
        }
        system_text = system or next(
            (
                str(message.get("content"))
                for message in source_messages
                if message.get("role") == "system"
            ),
            None,
        )
        if system_text:
            payload["system"] = system_text
        return payload

    def _response(
        self, model: str, raw: dict[str, Any], latency_ms: int, *, mode: str
    ) -> AIResponse:
        content = raw.get("content") or []
        text = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return self._build_response(
            model=model,
            mode=mode,
            text=text,
            tokens=AITokens(
                input=input_tokens, output=output_tokens, total=input_tokens + output_tokens
            ),
            latency_ms=latency_ms,
            raw=raw,
            finish_reason=str(raw.get("stop_reason") or ""),
            request_id=raw.get("id"),
        )
