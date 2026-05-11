from __future__ import annotations

from typing import Type

from .anthropic import AnthropicProvider
from .base import BaseProvider
from .cohere import CohereProvider
from .deepseek import DeepSeekProvider
from .google import GoogleProvider
from .groq import GroqProvider
from .mistral import MistralProvider
from .openai import OpenAICompatibleProvider, OpenAIProvider
from .perplexity import PerplexityProvider
from .together import TogetherProvider
from .xai import XAIProvider

PROVIDER_CLASSES: dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "xai": XAIProvider,
    "mistral": MistralProvider,
    "groq": GroqProvider,
    "cohere": CohereProvider,
    "together": TogetherProvider,
    "perplexity": PerplexityProvider,
    "deepseek": DeepSeekProvider,
}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a custom provider class."""

    PROVIDER_CLASSES[name.lower()] = provider_class


def get_provider_class(name: str) -> Type[BaseProvider]:
    """Return the provider class for a canonical provider name."""

    return PROVIDER_CLASSES[name]


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "CohereProvider",
    "XAIProvider",
    "GroqProvider",
    "TogetherProvider",
    "PerplexityProvider",
    "DeepSeekProvider",
    "PROVIDER_CLASSES",
    "register_provider",
    "get_provider_class",
]
