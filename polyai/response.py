from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .exceptions import InvalidInputError


@dataclass
class AITokens:
    """Token usage returned by a provider.

    Attributes:
        input: Prompt/input token count.
        output: Completion/output token count.
        total: Combined token count.
    """

    input: int = 0
    output: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return token usage as a serializable dictionary."""

        return {"input": self.input, "output": self.output, "total": self.total}


@dataclass
class AIResponse:
    """Unified response returned by every PolyAI provider call.

    Attributes:
        text: Main text output.
        provider: Resolved provider name.
        model: Resolved model name.
        mode: Call mode such as ``text``, ``image``, ``vision``, ``embed``, or ``audio``.
        tokens: Token usage.
        cost: Estimated USD cost.
        latency_ms: Wall-clock request latency in milliseconds.
        created_at: Response creation timestamp.
        image_url: Generated image URL, when applicable.
        image_data: Raw image bytes, when available.
        embeddings: Embedding vector, when applicable.
        embedding_dim: Embedding vector length.
        audio_data: Raw audio bytes, when applicable.
        raw: Untouched provider response data.
        finish_reason: Provider finish reason.
        request_id: Provider request ID for debugging.
        provider_used: Provider selected by a fallback chain, when applicable.
    """

    text: str
    provider: str
    model: str
    mode: str
    tokens: AITokens = field(default_factory=AITokens)
    cost: float = 0.0
    latency_ms: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    embeddings: Optional[list[float]] = None
    embedding_dim: Optional[int] = None
    audio_data: Optional[bytes] = None
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    request_id: Optional[str] = None
    provider_used: Optional[str] = None
    _retry_call: Optional[Callable[[], "AIResponse"]] = field(
        default=None, repr=False, compare=False
    )

    def save_image(self, path: str) -> None:
        """Save response image data to ``path``.

        If the response only contains an image URL, PolyAI downloads it first.

        Args:
            path: Destination file path.

        Raises:
            InvalidInputError: If the response does not contain image data or a URL.
        """

        data = self.image_data
        if data is None and self.image_url:
            with httpx.Client(timeout=30) as client:
                result = client.get(self.image_url)
                result.raise_for_status()
                data = result.content
                self.image_data = data
        if data is None:
            raise InvalidInputError(
                "This response does not contain an image to save.",
                fix="Call AI(..., mode='image') and save the returned image response.",
            )
        Path(path).write_bytes(data)

    def save_audio(self, path: str) -> None:
        """Save response audio bytes to ``path``.

        Args:
            path: Destination file path.

        Raises:
            InvalidInputError: If the response does not contain audio data.
        """

        if self.audio_data is None:
            raise InvalidInputError(
                "This response does not contain audio to save.",
                fix="Call AI(..., mode='audio') and save the returned audio response.",
            )
        Path(path).write_bytes(self.audio_data)

    def retry(self) -> "AIResponse":
        """Re-run the exact call that produced this response.

        Returns:
            A fresh AIResponse from the same request arguments.

        Raises:
            InvalidInputError: If the response was not created by the PolyAI dispatcher.
        """

        if self._retry_call is None:
            raise InvalidInputError(
                "This response cannot be retried because it has no saved request context.",
                fix="Create responses through AI(...) to enable response.retry().",
            )
        return self._retry_call()

    def with_retry(self, retry_call: Callable[[], "AIResponse"]) -> "AIResponse":
        """Attach a retry callback and return ``self`` for fluent internal use."""

        self._retry_call = retry_call
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response dictionary."""

        image_data = base64.b64encode(self.image_data).decode("ascii") if self.image_data else None
        audio_data = base64.b64encode(self.audio_data).decode("ascii") if self.audio_data else None
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat(),
            "image_url": self.image_url,
            "image_data": image_data,
            "embeddings": self.embeddings,
            "embedding_dim": self.embedding_dim,
            "audio_data": audio_data,
            "raw": self.raw,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "provider_used": self.provider_used,
        }

    def __str__(self) -> str:
        """Return the main response text."""

        return self.text

    def __repr__(self) -> str:
        """Return a concise developer representation."""

        return (
            "AIResponse("
            f"provider={self.provider!r}, model={self.model!r}, mode={self.mode!r}, "
            f"text={self.text[:60]!r}, tokens={self.tokens.total}, cost={self.cost:.6f})"
        )
