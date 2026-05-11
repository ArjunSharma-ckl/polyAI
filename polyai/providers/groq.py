from __future__ import annotations

from .openai import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """OpenAI-compatible adapter for Groq."""

    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
