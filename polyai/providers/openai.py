from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from ..exceptions import InvalidInputError, TimeoutError
from ..response import AIResponse, AITokens
from ..utils.costs import estimate_token_count
from .base import (
    BaseProvider,
    TextResult,
    image_inputs,
    image_to_data_url,
    text_messages,
    usage_from_openai,
)


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI and Azure OpenAI."""

    name = "openai"
    base_url = "https://api.openai.com/v1"

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self.azure_endpoint = kwargs.pop("azure_endpoint", None)
        self.azure_deployment = kwargs.pop("azure_deployment", None)
        self.api_version = kwargs.pop("api_version", "2024-10-21")
        self.organization = kwargs.pop("organization", None)
        super().__init__(api_key, **kwargs)
        if self.azure_endpoint:
            self.base_url = str(self.azure_endpoint).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if self.azure_endpoint:
            return {"api-key": self.api_key, "Content-Type": "application/json"}
        headers = super()._headers()
        if self.organization:
            headers["OpenAI-Organization"] = str(self.organization)
        return headers

    def _endpoint(self, path: str, model: str) -> str:
        if not self.azure_endpoint:
            return path
        deployment = self.azure_deployment or model
        return f"/openai/deployments/{deployment}{path}?api-version={self.api_version}"

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Call OpenAI chat completions."""

        stream = bool(kwargs.pop("stream", False))
        payload = {
            "model": model,
            "messages": text_messages(prompt, system, messages),
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }
        if stream:
            payload["stream"] = True
            return self._stream_chat(model, payload)
        raw, latency_ms = self._measure(
            lambda: self._post_json(self._endpoint("/chat/completions", model), payload)
        )
        return self._chat_response(model, raw, latency_ms, mode="text")

    def vision(
        self,
        model: str,
        prompt: str,
        image: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Call OpenAI vision-capable chat models."""

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in image_inputs(image, images):
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(item)}})
        messages = text_messages(
            None, kwargs.pop("system", None), [{"role": "user", "content": content}]
        )
        raw, latency_ms = self._measure(
            lambda: self._post_json(
                self._endpoint("/chat/completions", model),
                {
                    "model": model,
                    "messages": messages,
                    "temperature": kwargs.pop("temperature", 0.7),
                    "max_tokens": kwargs.pop("max_tokens", 1024),
                    **kwargs,
                },
            )
        )
        return self._chat_response(model, raw, latency_ms, mode="vision")

    def image(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Generate images with OpenAI image models."""

        payload = {
            "model": model,
            "prompt": prompt,
            "size": kwargs.pop("size", "1024x1024"),
            "quality": kwargs.pop("quality", "standard"),
            "n": kwargs.pop("n", 1),
            **kwargs,
        }
        raw, latency_ms = self._measure(
            lambda: self._post_json(self._endpoint("/images/generations", model), payload)
        )
        first = (raw.get("data") or [{}])[0]
        image_url = first.get("url")
        image_data = None
        if first.get("b64_json"):
            image_data = base64.b64decode(first["b64_json"])
        return self._build_response(
            model=model,
            mode="image",
            text=image_url or "",
            latency_ms=latency_ms,
            raw=raw,
            image_url=image_url,
            image_data=image_data,
            request_id=raw.get("id"),
        )

    def embed(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create embeddings with OpenAI embedding models."""

        payload = {"model": model, "input": kwargs.pop("input", prompt), **kwargs}
        raw, latency_ms = self._measure(
            lambda: self._post_json(self._endpoint("/embeddings", model), payload)
        )
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

    def audio(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create speech with OpenAI text-to-speech models."""

        payload = {
            "model": model,
            "input": prompt,
            "voice": kwargs.pop("voice", "alloy"),
            "response_format": kwargs.pop("response_format", "mp3"),
            **kwargs,
        }
        headers = self._headers()
        (audio_data, response_headers), latency_ms = self._measure(
            lambda: self._post_bytes(
                self._endpoint("/audio/speech", model), payload, headers=headers
            )
        )
        tokens = AITokens(
            input=estimate_token_count(prompt), output=0, total=estimate_token_count(prompt)
        )
        return self._build_response(
            model=model,
            mode="audio",
            text="",
            tokens=tokens,
            latency_ms=latency_ms,
            raw={"headers": response_headers, "bytes": len(audio_data)},
            audio_data=audio_data,
        )

    def stt(self, model: str, audio: str, **kwargs: Any) -> AIResponse:
        """Transcribe audio with OpenAI speech-to-text models."""

        path = Path(audio)
        if not path.is_file():
            raise InvalidInputError(
                f"Audio file '{audio}' was not found.",
                fix="Pass a valid local audio file path to AI(..., mode='stt').",
            )
        headers = self._headers()
        headers.pop("Content-Type", None)
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, "application/octet-stream")}
            data = {"model": model, **kwargs}
            raw, latency_ms = self._measure(
                lambda: self._post_multipart(
                    self._endpoint("/audio/transcriptions", model),
                    data=data,
                    files=files,
                    headers=headers,
                )
            )
        text = str(raw.get("text") or "")
        return self._build_response(
            model=model, mode="stt", text=text, latency_ms=latency_ms, raw=raw
        )

    def _chat_response(
        self,
        model: str,
        raw: dict[str, Any],
        latency_ms: int,
        *,
        mode: str,
    ) -> AIResponse:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content)
        return self._build_response(
            model=model,
            mode=mode,
            text=text,
            tokens=usage_from_openai(raw),
            latency_ms=latency_ms,
            raw=raw,
            finish_reason=str(choice.get("finish_reason") or ""),
            request_id=raw.get("id"),
        )

    def _stream_chat(self, model: str, payload: dict[str, Any]) -> Iterator[str]:
        endpoint = self._endpoint("/chat/completions", model)
        try:
            with self._client() as client:
                with client.stream(
                    "POST", endpoint, json=payload, headers=self._headers()
                ) as response:
                    self._raise_for_status(response)
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        choice = (data.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield str(content)
        except httpx.TimeoutException as exc:
            raise TimeoutError(self.name, timeout=self.timeout) from exc


class OpenAICompatibleProvider(OpenAIProvider):
    """Base class for providers that expose OpenAI-compatible endpoints."""

    name = "openai-compatible"
    base_url = ""
