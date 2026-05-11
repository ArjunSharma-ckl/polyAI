from __future__ import annotations

import base64
import json
import mimetypes
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import httpx

from ..exceptions import (
    AuthError,
    InvalidInputError,
    ModeNotSupportedError,
    PolyAIError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from ..response import AIResponse, AITokens
from ..utils.costs import calculate_cost
from ..utils.logger import debug_log

TextResult = AIResponse | Iterator[str]


class BaseProvider(ABC):
    """Base class for all provider adapters.

    Args:
        api_key: Provider API key.
        timeout: HTTP timeout in seconds.
        debug: Whether to print request/response details.
        base_url: Optional base URL override.
        options: Provider-specific options.
    """

    name = "base"
    base_url = ""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        debug: bool = False,
        base_url: Optional[str] = None,
        **options: Any,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.debug = debug
        self.base_url = base_url or self.base_url
        self.options = options

    @abstractmethod
    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        """Call a provider text/chat endpoint."""

    def image(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Generate an image.

        Raises:
            ModeNotSupportedError: Always, unless overridden by a provider.
        """

        raise ModeNotSupportedError(
            self.name, "image", model=model, supported=self.supported_modes(model)
        )

    def vision(
        self,
        model: str,
        prompt: str,
        image: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Process images with a multimodal model.

        Raises:
            ModeNotSupportedError: Always, unless overridden by a provider.
        """

        raise ModeNotSupportedError(
            self.name, "vision", model=model, supported=self.supported_modes(model)
        )

    def embed(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create embeddings.

        Raises:
            ModeNotSupportedError: Always, unless overridden by a provider.
        """

        raise ModeNotSupportedError(
            self.name, "embed", model=model, supported=self.supported_modes(model)
        )

    def audio(self, model: str, prompt: str, **kwargs: Any) -> AIResponse:
        """Create speech/audio.

        Raises:
            ModeNotSupportedError: Always, unless overridden by a provider.
        """

        raise ModeNotSupportedError(
            self.name, "audio", model=model, supported=self.supported_modes(model)
        )

    def stt(self, model: str, audio: str, **kwargs: Any) -> AIResponse:
        """Transcribe speech to text.

        Raises:
            ModeNotSupportedError: Always, unless overridden by a provider.
        """

        raise ModeNotSupportedError(
            self.name, "stt", model=model, supported=self.supported_modes(model)
        )

    def supported_modes(self, model: str) -> list[str]:
        """Return known supported modes for this model."""

        try:
            from ..registry import model_modes

            return model_modes(self.name, model)
        except Exception:
            return []

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _measure(self, fn: Callable[[], Any]) -> tuple[Any, int]:
        """Call ``fn`` and return ``(result, latency_ms)``."""

        started = time.perf_counter()
        result = fn()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return result, latency_ms

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        debug_log(self.debug, f"{self.name} request", {"endpoint": endpoint, "payload": payload})
        try:
            with self._client() as client:
                response = client.post(
                    endpoint, json=payload, headers=headers or self._headers(), params=params
                )
        except httpx.TimeoutException as exc:
            raise TimeoutError(self.name, timeout=self.timeout) from exc
        except httpx.HTTPError as exc:
            raise PolyAIError(
                f"HTTP request to provider '{self.name}' failed.",
                fix="Check your network connection, timeout, and provider endpoint.",
            ) from exc
        self._raise_for_status(response)
        data = self._response_json(response)
        debug_log(self.debug, f"{self.name} response", data)
        return data

    def _post_bytes(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[bytes, dict[str, str]]:
        debug_log(self.debug, f"{self.name} request", {"endpoint": endpoint, "payload": payload})
        try:
            with self._client() as client:
                response = client.post(endpoint, json=payload, headers=headers or self._headers())
        except httpx.TimeoutException as exc:
            raise TimeoutError(self.name, timeout=self.timeout) from exc
        self._raise_for_status(response)
        return response.content, dict(response.headers)

    def _post_multipart(
        self,
        endpoint: str,
        *,
        data: dict[str, Any],
        files: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        debug_log(
            self.debug, f"{self.name} multipart request", {"endpoint": endpoint, "data": data}
        )
        try:
            with self._client() as client:
                response = client.post(
                    endpoint, data=data, files=files, headers=headers or self._headers()
                )
        except httpx.TimeoutException as exc:
            raise TimeoutError(self.name, timeout=self.timeout) from exc
        self._raise_for_status(response)
        return self._response_json(response)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = self._error_detail(response)
        if response.status_code in {401, 403}:
            raise AuthError(self.name, detail=detail or "Invalid API key")
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            raise RateLimitError(
                self.name, retry_after=retry_after, detail=detail or "Rate limit reached"
            )
        if response.status_code >= 500:
            raise ServerError(
                self.name, status_code=response.status_code, detail=detail or "Server error"
            )
        raise InvalidInputError(
            f"Provider '{self.name}' rejected the request (HTTP {response.status_code}).",
            fix=detail or "Check the request parameters and model name.",
            provider=self.name,
        )

    def _response_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise PolyAIError(
                f"Provider '{self.name}' returned a non-JSON response.",
                fix="Enable debug=True to inspect the response, or try again later.",
                provider=self.name,
            ) from exc
        if not isinstance(data, dict):
            raise PolyAIError(
                f"Provider '{self.name}' returned an unexpected response shape.",
                fix="Open an issue with the raw response and provider/model used.",
                provider=self.name,
            )
        return data

    def _error_detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except json.JSONDecodeError:
            return response.text[:300]
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("type") or error)
            if isinstance(error, str):
                return error
            message = data.get("message")
            if message:
                return str(message)
        return response.text[:300]

    def _build_response(
        self,
        *,
        model: str,
        mode: str,
        text: str = "",
        tokens: Optional[AITokens] = None,
        latency_ms: int = 0,
        raw: Optional[dict[str, Any]] = None,
        finish_reason: str = "",
        request_id: Optional[str] = None,
        image_url: Optional[str] = None,
        image_data: Optional[bytes] = None,
        embeddings: Optional[list[float]] = None,
        audio_data: Optional[bytes] = None,
    ) -> AIResponse:
        """Construct a normalized AIResponse."""

        usage = tokens or AITokens()
        return AIResponse(
            text=text,
            provider=self.name,
            model=model,
            mode=mode,
            tokens=usage,
            cost=calculate_cost(self.name, model, usage),
            latency_ms=latency_ms,
            image_url=image_url,
            image_data=image_data,
            embeddings=embeddings,
            embedding_dim=len(embeddings) if embeddings is not None else None,
            audio_data=audio_data,
            raw=raw or {},
            finish_reason=finish_reason,
            request_id=request_id,
        )


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def text_messages(
    prompt: Optional[str],
    system: Optional[str] = None,
    messages: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-style messages from prompt/system/messages."""

    if messages is not None:
        return messages
    if prompt is None:
        raise InvalidInputError(
            "Missing prompt for text call.",
            fix='Pass prompt="..." or messages=[...] to AI(...).',
        )
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": prompt})
    return result


def image_inputs(image: Optional[str] = None, images: Optional[list[str]] = None) -> list[str]:
    """Normalize single and multiple image inputs."""

    result: list[str] = []
    if image:
        result.append(image)
    if images:
        result.extend(images)
    if not result:
        raise InvalidInputError(
            "Missing image input for vision call.",
            fix='Pass image="path-or-url" or images=[...] with mode="vision".',
        )
    return result


def is_url(value: str) -> bool:
    """Return whether a value looks like an HTTP URL."""

    return value.startswith("http://") or value.startswith("https://")


def image_to_data_url(value: str) -> str:
    """Convert a local image path to a data URL; URLs are returned unchanged."""

    if is_url(value):
        return value
    path = Path(value)
    if not path.is_file():
        raise InvalidInputError(
            f"Image file '{value}' was not found.",
            fix="Pass a valid local file path or an HTTPS image URL.",
        )
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_to_base64(value: str) -> tuple[str, str]:
    """Return ``(mime_type, base64_data)`` for Anthropic/Gemini style image blocks."""

    if is_url(value):
        raise InvalidInputError(
            "This provider requires local image files for base64 vision input.",
            fix="Download the image first or use OpenAI/Gemini for direct image URLs.",
        )
    path = Path(value)
    if not path.is_file():
        raise InvalidInputError(
            f"Image file '{value}' was not found.",
            fix="Pass a valid local image file path.",
        )
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def usage_from_openai(raw: dict[str, Any]) -> AITokens:
    """Extract token usage from an OpenAI-compatible response."""

    usage = raw.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return AITokens(input=input_tokens, output=output_tokens, total=total)
