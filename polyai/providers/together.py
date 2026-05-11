from __future__ import annotations

from .openai import OpenAICompatibleProvider


class TogetherProvider(OpenAICompatibleProvider):
    """OpenAI-compatible adapter for Together AI."""

    name = "together"
    base_url = "https://api.together.xyz/v1"
