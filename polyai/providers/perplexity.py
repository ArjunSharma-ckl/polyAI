from __future__ import annotations

from .openai import OpenAICompatibleProvider


class PerplexityProvider(OpenAICompatibleProvider):
    """OpenAI-compatible adapter for Perplexity."""

    name = "perplexity"
    base_url = "https://api.perplexity.ai"
