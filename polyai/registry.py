from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from typing import Any, Optional

from .exceptions import ModelNotFoundError, ProviderNotFoundError

PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "dall-e-3",
            "dall-e-2",
            "tts-1",
            "tts-1-hd",
            "whisper-1",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
        "aliases": {"best": "gpt-4o", "fast": "gpt-4o-mini", "cheap": "gpt-4o-mini"},
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        "aliases": {
            "best": "claude-opus-4-5",
            "fast": "claude-haiku-4-5",
            "balanced": "claude-sonnet-4-5",
        },
        "env_key": "ANTHROPIC_API_KEY",
    },
    "google": {
        "models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-ultra",
        ],
        "aliases": {"best": "gemini-1.5-pro", "fast": "gemini-2.0-flash"},
        "env_key": "GOOGLE_API_KEY",
    },
    "xai": {
        "models": ["grok-2", "grok-2-mini", "grok-3", "grok-3-mini", "grok-2-vision"],
        "aliases": {"best": "grok-3", "fast": "grok-3-mini"},
        "env_key": "XAI_API_KEY",
    },
    "mistral": {
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "codestral-latest",
            "mistral-embed",
        ],
        "aliases": {"best": "mistral-large-latest", "fast": "mistral-small-latest"},
        "env_key": "MISTRAL_API_KEY",
    },
    "groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "llama-3.3-70b-specdec",
        ],
        "aliases": {"best": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant"},
        "env_key": "GROQ_API_KEY",
    },
    "cohere": {
        "models": [
            "command-r-plus",
            "command-r",
            "command-light",
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
        ],
        "aliases": {"best": "command-r-plus", "fast": "command-r"},
        "env_key": "COHERE_API_KEY",
    },
    "together": {
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-2-27b-it",
            "deepseek-ai/DeepSeek-R1",
            "black-forest-labs/FLUX.1-schnell",
        ],
        "aliases": {"best": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "env_key": "TOGETHER_API_KEY",
    },
    "perplexity": {
        "models": [
            "sonar-pro",
            "sonar",
            "sonar-reasoning-pro",
            "sonar-reasoning",
            "sonar-deep-research",
        ],
        "aliases": {"best": "sonar-pro", "fast": "sonar"},
        "env_key": "PERPLEXITY_API_KEY",
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "aliases": {"best": "deepseek-chat", "reasoning": "deepseek-reasoner"},
        "env_key": "DEEPSEEK_API_KEY",
    },
}


PROVIDER_ALIASES: dict[str, str] = {
    "gpt": "openai",
    "chatgpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "bard": "google",
    "grok": "xai",
    "llama": "together",
    "meta": "together",
    "r1": "deepseek",
}


MODEL_MODES: dict[str, dict[str, set[str]]] = {
    "openai": {
        "text": {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
        },
        "vision": {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"},
        "image": {"dall-e-3", "dall-e-2"},
        "audio": {"tts-1", "tts-1-hd"},
        "stt": {"whisper-1"},
        "embed": {"text-embedding-3-small", "text-embedding-3-large"},
    },
    "anthropic": {
        "text": {
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        },
        "vision": {
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        },
    },
    "google": {
        "text": {
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-ultra",
        },
        "vision": {
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-ultra",
        },
        "embed": {"gemini-embedding-exp", "text-embedding-004"},
    },
    "xai": {
        "text": {"grok-2", "grok-2-mini", "grok-3", "grok-3-mini"},
        "vision": {"grok-2-vision"},
    },
    "mistral": {
        "text": {
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "codestral-latest",
        },
        "embed": {"mistral-embed"},
    },
    "groq": {
        "text": {
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "llama-3.3-70b-specdec",
        }
    },
    "cohere": {
        "text": {"command-r-plus", "command-r", "command-light"},
        "embed": {"embed-english-v3.0", "embed-multilingual-v3.0"},
    },
    "together": {
        "text": {
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-2-27b-it",
            "deepseek-ai/DeepSeek-R1",
        },
        "image": {"black-forest-labs/FLUX.1-schnell"},
    },
    "perplexity": {
        "text": {
            "sonar-pro",
            "sonar",
            "sonar-reasoning-pro",
            "sonar-reasoning",
            "sonar-deep-research",
        }
    },
    "deepseek": {"text": {"deepseek-chat", "deepseek-reasoner"}},
}


def normalize_provider(provider: str) -> str:
    """Resolve a provider name or alias to its canonical provider name."""

    normalized = provider.strip().lower()
    normalized = PROVIDER_ALIASES.get(normalized, normalized)
    if normalized not in PROVIDERS:
        suggestions = get_close_matches(normalized, list(PROVIDERS) + list(PROVIDER_ALIASES), n=3)
        raise ProviderNotFoundError(provider, suggestions=suggestions)
    return normalized


def provider_env_key(provider: str) -> str:
    """Return the primary environment variable for a provider."""

    canonical = normalize_provider(provider)
    return str(PROVIDERS[canonical]["env_key"])


def resolve_model(provider: str, model: str) -> str:
    """Resolve a model alias and validate it against the provider registry."""

    canonical_provider = normalize_provider(provider)
    aliases = PROVIDERS[canonical_provider].get("aliases", {})
    resolved = aliases.get(model, model)
    models = list(PROVIDERS[canonical_provider]["models"])
    if "*" in models:
        return str(resolved)
    if resolved not in models:
        suggestions = get_close_matches(resolved, models + list(aliases), n=4)
        raise ModelNotFoundError(canonical_provider, model, suggestions=suggestions)
    return str(resolved)


def parse_provider_model(
    provider: Optional[str], model: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Parse ``provider/model`` shorthand while preserving explicit values."""

    if provider and "/" in provider and model is None:
        left, right = provider.split("/", 1)
        return left, right
    return provider, model


def providers() -> list[str]:
    """Return all canonical provider names."""

    return list(PROVIDERS)


def models(
    provider: Optional[str] = None, *, mode: Optional[str] = None
) -> dict[str, list[str]] | list[str]:
    """Return registered models, optionally filtered by provider or mode."""

    if provider is not None:
        canonical = normalize_provider(provider)
        provider_models = list(PROVIDERS[canonical]["models"])
        if mode is None:
            return provider_models
        allowed = MODEL_MODES.get(canonical, {}).get(mode, set())
        return [item for item in provider_models if item in allowed]

    result: dict[str, list[str]] = {}
    for name in PROVIDERS:
        result[name] = list(models(name, mode=mode))  # type: ignore[arg-type]
    return result


def model_modes(provider: str, model: str) -> list[str]:
    """Return modes supported by a provider/model pair."""

    canonical = normalize_provider(provider)
    resolved = resolve_model(canonical, model)
    result = []
    for mode, mode_models in MODEL_MODES.get(canonical, {}).items():
        if "*" in mode_models or resolved in mode_models:
            result.append(mode)
    return result


def supports_mode(provider: str, model: str, mode: str) -> bool:
    """Return whether a provider/model supports ``mode``."""

    return mode in model_modes(provider, model)


def registry_snapshot() -> dict[str, dict[str, Any]]:
    """Return a defensive copy of provider registry metadata."""

    return deepcopy(PROVIDERS)
