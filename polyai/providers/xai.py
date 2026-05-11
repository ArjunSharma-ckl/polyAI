from __future__ import annotations

from .openai import OpenAICompatibleProvider


class XAIProvider(OpenAICompatibleProvider):
    """OpenAI-compatible adapter for xAI Grok."""

    name = "xai"
    base_url = "https://api.x.ai/v1"
