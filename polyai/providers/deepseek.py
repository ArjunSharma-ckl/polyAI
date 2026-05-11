from __future__ import annotations

from .openai import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """OpenAI-compatible adapter for DeepSeek."""

    name = "deepseek"
    base_url = "https://api.deepseek.com"
