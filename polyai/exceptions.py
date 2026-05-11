from __future__ import annotations

from typing import Optional

DOCS_BASE = "README.md"


class PolyAIError(Exception):
    """Base exception for all PolyAI errors.

    Args:
        message: Plain-English explanation of the failure.
        fix: Action the user can take to resolve the failure.
        docs_url: Optional documentation URL.
        provider: Optional provider name related to the failure.
    """

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        fix: Optional[str] = None,
        docs_url: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.message = message
        self.fix = fix
        self.docs_url = docs_url
        self.provider = provider
        parts = [message]
        if fix:
            parts.append(f"  -> {fix}")
        if docs_url:
            parts.append(f"  -> Docs: {docs_url}")
        super().__init__("\n".join(parts))


class AuthError(PolyAIError):
    """Raised when an API key is missing, invalid, or rejected by a provider."""

    def __init__(
        self,
        provider: str,
        *,
        env_var: Optional[str] = None,
        detail: str = "Invalid or missing API key",
    ) -> None:
        env_hint = env_var or f"{provider.upper()}_API_KEY"
        super().__init__(
            f"{detail} for provider '{provider}'.",
            fix=f'Set {env_hint} in your .env file, or pass api_key="..." directly.',
            docs_url=f"{DOCS_BASE}#provider-setup",
            provider=provider,
        )


class RateLimitError(PolyAIError):
    """Raised when a provider rate-limits a request."""

    retryable = True

    def __init__(
        self,
        provider: str,
        *,
        retry_after: Optional[float] = None,
        detail: str = "Rate limit reached",
    ) -> None:
        self.retry_after = retry_after
        wait_hint = (
            f"Wait {retry_after:g} seconds and retry, or reduce request volume."
            if retry_after is not None
            else "Wait briefly and retry, or reduce request volume."
        )
        super().__init__(
            f"{detail} for provider '{provider}'.",
            fix=wait_hint,
            docs_url=f"{DOCS_BASE}#retries-timeouts-and-rate-limits",
            provider=provider,
        )


class ModelNotFoundError(PolyAIError):
    """Raised when a model name is unknown for the selected provider."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        suggestions: Optional[list[str]] = None,
    ) -> None:
        hint = ""
        if suggestions:
            hint = f" Try one of: {', '.join(suggestions)}."
        super().__init__(
            f"Model '{model}' is not registered for provider '{provider}'.",
            fix=(
                f"Check the model name or call AI.models('{provider}') "
                f"to list supported models.{hint}"
            ),
            docs_url=f"{DOCS_BASE}#provider-setup",
            provider=provider,
        )
        self.model = model
        self.suggestions = suggestions or []


class ProviderNotFoundError(PolyAIError):
    """Raised when a provider name or alias is unknown."""

    def __init__(self, provider: str, *, suggestions: Optional[list[str]] = None) -> None:
        hint = ""
        if suggestions:
            hint = f" Did you mean: {', '.join(suggestions)}?"
        super().__init__(
            f"Provider '{provider}' is not supported by PolyAI.",
            fix=f"Use AI.providers() to list supported providers.{hint}",
            docs_url=f"{DOCS_BASE}#provider-setup",
        )
        self.provider_name = provider
        self.suggestions = suggestions or []


class ModeNotSupportedError(PolyAIError):
    """Raised when a provider or model does not support a requested mode."""

    def __init__(
        self,
        provider: str,
        mode: str,
        *,
        model: Optional[str] = None,
        supported: Optional[list[str]] = None,
    ) -> None:
        target = f"{provider}/{model}" if model else provider
        supported_hint = f" Supported modes: {', '.join(supported)}." if supported else ""
        super().__init__(
            f"{target} does not support mode '{mode}'.",
            fix=(
                f"Pick a model that supports '{mode}', or call "
                f"AI.models(mode='{mode}').{supported_hint}"
            ),
            docs_url=f"{DOCS_BASE}#modes",
            provider=provider,
        )
        self.mode = mode
        self.model = model
        self.supported = supported or []


class TimeoutError(PolyAIError):
    """Raised when a provider request times out."""

    retryable = True

    def __init__(self, provider: str, *, timeout: Optional[float] = None) -> None:
        detail = f"Request to provider '{provider}' timed out"
        if timeout is not None:
            detail += f" after {timeout:g} seconds"
        detail += "."
        super().__init__(
            detail,
            fix="Increase timeout=..., retry later, or choose a faster model.",
            docs_url=f"{DOCS_BASE}#retries-timeouts-and-rate-limits",
            provider=provider,
        )


class InvalidInputError(PolyAIError):
    """Raised when user input cannot be sent to a provider."""

    def __init__(
        self,
        message: str,
        *,
        fix: str = "Check the input parameters and try again.",
        docs_url: str = f"{DOCS_BASE}#api-reference",
        provider: Optional[str] = None,
    ) -> None:
        super().__init__(message, fix=fix, docs_url=docs_url, provider=provider)


class CostLimitError(PolyAIError):
    """Raised when a request exceeds the user-provided max_cost_usd limit."""

    def __init__(self, estimated: float, limit: float, *, provider: str, model: str) -> None:
        super().__init__(
            (
                f"Estimated cost ${estimated:.6f} exceeds "
                f"max_cost_usd=${limit:.6f} for {provider}/{model}."
            ),
            fix=(
                "Raise max_cost_usd, reduce max_tokens, shorten the prompt, "
                "or choose a cheaper model."
            ),
            docs_url=f"{DOCS_BASE}#cost-tracking",
            provider=provider,
        )
        self.estimated = estimated
        self.limit = limit
        self.model = model


class ServerError(PolyAIError):
    """Raised for retryable provider-side 5xx failures."""

    retryable = True

    def __init__(self, provider: str, *, status_code: int, detail: str = "Server error") -> None:
        self.status_code = status_code
        super().__init__(
            f"{detail} from provider '{provider}' (HTTP {status_code}).",
            fix="Retry later, or configure a fallback chain with AI.fallback(...).",
            docs_url=f"{DOCS_BASE}#fallbacks",
            provider=provider,
        )
