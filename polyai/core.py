from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional, Type

from .batch import run_batch
from .chat import ChatSession
from .compare import CompareResult, compare_models
from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    get_api_key,
    get_default,
    load_env,
)
from .config import (
    configure as configure_keys,
)
from .config import (
    set_default as set_default_config,
)
from .exceptions import AuthError, CostLimitError, InvalidInputError, ModeNotSupportedError
from .fallback import run_fallback
from .providers import (
    BaseProvider,
    get_provider_class,
)
from .providers import (
    register_provider as register_provider_class,
)
from .registry import (
    MODEL_MODES,
    PROVIDERS,
    normalize_provider,
    parse_provider_model,
    provider_env_key,
    resolve_model,
    supports_mode,
)
from .registry import (
    models as registry_models,
)
from .registry import (
    providers as registry_providers,
)
from .response import AIResponse
from .utils.costs import CostTracker
from .utils.costs import estimate_cost as estimate_request_cost
from .utils.retry import run_with_retries


class _AIClient:
    """Universal AI client exposed as ``polyai.AI``.

    The object is callable, so the shortest useful call is:

    ``AI("openai", "gpt-4o", api_key="sk-...", prompt="Hello")``.
    """

    def __call__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        mode: str = "text",
        stream: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        debug: bool = False,
        max_cost_usd: Optional[float] = None,
        **kwargs: Any,
    ) -> AIResponse | Iterator[str]:
        """Call any supported AI provider and return a normalized response.

        Args:
            provider: Provider name, alias, or ``provider/model`` shorthand.
            model: Model name or alias.
            api_key: Optional API key. If omitted, configured keys and env vars are used.
            prompt: User prompt.
            system: Optional system instruction.
            messages: OpenAI-style conversation messages.
            mode: ``text``, ``vision``, ``image``, ``audio``, ``stt``, or ``embed``.
            stream: Whether to return a text chunk iterator.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens where supported.
            timeout: HTTP timeout in seconds.
            retries: Retry count for retryable failures.
            debug: Whether to print request/response details to stderr.
            max_cost_usd: Optional pre-flight estimated cost limit.
            kwargs: Provider or mode specific options.

        Returns:
            AIResponse for normal calls, or an iterator of string chunks for streaming calls.
        """

        original_provider = provider
        original_model = model
        provider, model = parse_provider_model(provider, model)
        provider, model = self._resolve_defaults(provider, model)
        if provider is None:
            raise InvalidInputError(
                "Missing provider.",
                fix='Pass provider="openai" or call AI.set_default(provider=..., model=...).',
            )
        if model is None:
            raise InvalidInputError(
                f"Missing model for provider '{provider}'.",
                fix='Pass model="gpt-4o" or use shorthand like AI("openai/gpt-4o", ...).',
            )
        provider = normalize_provider(provider)
        model = resolve_model(provider, model)
        mode = mode.lower()
        if not supports_mode(provider, model, mode):
            raise ModeNotSupportedError(provider, mode, model=model)
        key = get_api_key(provider, api_key)
        if not key:
            raise AuthError(provider, env_var=provider_env_key(provider), detail="Missing API key")
        if max_cost_usd is not None:
            estimated = estimate_request_cost(
                provider,
                model,
                prompt=prompt or self._messages_text(messages),
                max_tokens=max_tokens,
            )
            if estimated > max_cost_usd:
                raise CostLimitError(estimated, max_cost_usd, provider=provider, model=model)

        provider_instance = self._provider_instance(
            provider,
            key,
            timeout=timeout,
            debug=debug,
            **self._provider_options(kwargs),
        )
        text_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        generation_kwargs = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}
        passthrough_kwargs = dict(kwargs)

        def invoke() -> AIResponse | Iterator[str]:
            if mode == "text":
                return provider_instance.text(
                    model,
                    prompt=prompt,
                    system=system,
                    messages=messages,
                    **text_kwargs,
                )
            if mode == "vision":
                if prompt is None:
                    raise InvalidInputError(
                        "Missing prompt for vision call.", fix='Pass prompt="...".'
                    )
                return provider_instance.vision(
                    model, prompt=prompt, system=system, **generation_kwargs
                )
            if mode == "image":
                if prompt is None:
                    raise InvalidInputError(
                        "Missing prompt for image call.", fix='Pass prompt="...".'
                    )
                return provider_instance.image(model, prompt=prompt, **passthrough_kwargs)
            if mode == "embed":
                if prompt is None:
                    raise InvalidInputError(
                        "Missing prompt for embedding call.", fix='Pass prompt="...".'
                    )
                return provider_instance.embed(model, prompt=prompt, **passthrough_kwargs)
            if mode == "audio":
                if prompt is None:
                    raise InvalidInputError(
                        "Missing prompt for audio call.", fix='Pass prompt="...".'
                    )
                return provider_instance.audio(model, prompt=prompt, **passthrough_kwargs)
            if mode == "stt":
                audio = kwargs.get("audio")
                if audio is None:
                    raise InvalidInputError(
                        "Missing audio file for STT call.", fix='Pass audio="recording.mp3".'
                    )
                stt_kwargs = {
                    key: value for key, value in passthrough_kwargs.items() if key != "audio"
                }
                return provider_instance.stt(model, audio=str(audio), **stt_kwargs)
            raise InvalidInputError(
                f"Unknown mode '{mode}'.", fix="Use text, vision, image, audio, stt, or embed."
            )

        if stream:
            return invoke()
        response = run_with_retries(lambda: self._ensure_response(invoke()), retries=retries)

        def retry_call() -> AIResponse:
            retry_result = self(
                original_provider,
                original_model,
                api_key=api_key,
                prompt=prompt,
                system=system,
                messages=messages,
                mode=mode,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                debug=debug,
                max_cost_usd=max_cost_usd,
                **kwargs,
            )
            return self._ensure_response(lambda: retry_result)

        response.with_retry(retry_call)
        return response

    def configure(self, keys: Optional[dict[str, str]] = None, **kwargs: str) -> None:
        """Store API keys globally."""

        configure_keys(keys, **kwargs)

    def load_env(
        self, path: Optional[str | Path] = None, *, override: bool = False
    ) -> Optional[Path]:
        """Load API keys from a ``.env`` file."""

        return load_env(path, override=override)

    def set_default(self, *, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Set default provider/model values for future calls."""

        set_default_config(provider=provider, model=model)

    def from_env(self) -> "_AIClient":
        """Load the nearest ``.env`` file and return this client."""

        self.load_env()
        return self

    def chat(self, provider: str, model: str, **kwargs: Any) -> ChatSession:
        """Create a multi-turn chat session."""

        return ChatSession(self, provider, model, **kwargs)

    def fallback(self, chain: list[str], **kwargs: Any) -> AIResponse:
        """Try provider/model specs in order until one succeeds."""

        return run_fallback(self, chain, **kwargs)

    def batch(self, calls: list[dict[str, Any]], *, max_parallel: int = 5) -> list[AIResponse]:
        """Run multiple AI calls concurrently."""

        return run_batch(self, calls, max_parallel=max_parallel)

    def compare(self, *, models: list[str], prompt: str, **kwargs: Any) -> CompareResult:
        """Run the same prompt against several models."""

        return compare_models(self, models=models, prompt=prompt, **kwargs)

    async def async_call(self, *args: Any, **kwargs: Any) -> AIResponse:
        """Async wrapper for one AI call."""

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(self, *args, **kwargs))
        return self._ensure_response(lambda: result)

    async def async_batch(
        self,
        calls: list[dict[str, Any]],
        *,
        max_parallel: int = 5,
    ) -> list[AIResponse]:
        """Run multiple AI calls concurrently from async code."""

        semaphore = asyncio.Semaphore(max(1, max_parallel))

        async def run_one(call: dict[str, Any]) -> AIResponse:
            async with semaphore:
                return await self.async_call(**call)

        return list(await asyncio.gather(*(run_one(call) for call in calls)))

    def cost_tracker(self) -> CostTracker:
        """Return a new cost tracker."""

        return CostTracker()

    def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> float:
        """Estimate request cost before sending it."""

        provider = normalize_provider(provider)
        model = resolve_model(provider, model)
        return estimate_request_cost(provider, model, prompt=prompt, max_tokens=max_tokens)

    def ping(self, provider: str, *, api_key: Optional[str] = None, model: str = "fast") -> bool:
        """Return whether a provider key can make a small test request."""

        try:
            self(provider, model, api_key=api_key, prompt="ping", max_tokens=1, retries=0)
        except Exception:
            return False
        return True

    def providers(self) -> list[str]:
        """Return supported provider names."""

        return registry_providers()

    def models(
        self,
        provider: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        pretty: bool = True,
    ) -> dict[str, list[str]] | list[str]:
        """Return supported models, optionally filtered by provider or mode."""

        result = registry_models(provider, mode=mode)
        if pretty:
            self._print_models(result)
        return result

    def register_provider(
        self,
        name: str,
        provider_class: Type[BaseProvider],
        *,
        models: Optional[list[str]] = None,
        aliases: Optional[dict[str, str]] = None,
        env_key: Optional[str] = None,
        modes: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Register a custom provider class.

        Args:
            name: Provider name users will pass to ``AI(...)``.
            provider_class: Subclass of ``BaseProvider``.
            models: Supported model names. If omitted, any model is accepted.
            aliases: Optional model aliases.
            env_key: Environment variable for API keys.
            modes: Optional mode mapping, e.g. ``{"text": ["model-a"]}``.
        """

        canonical = name.lower()
        register_provider_class(canonical, provider_class)
        PROVIDERS[canonical] = {
            "models": models or ["*"],
            "aliases": aliases or {},
            "env_key": env_key or f"{canonical.upper()}_API_KEY",
        }
        MODEL_MODES[canonical] = {
            mode_name: set(mode_models)
            for mode_name, mode_models in (modes or {"text": models or ["*"]}).items()
        }

    def _provider_instance(self, provider: str, api_key: str, **kwargs: Any) -> BaseProvider:
        provider_class = get_provider_class(provider)
        return provider_class(api_key, **kwargs)

    def _provider_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        names = {
            "base_url",
            "azure_endpoint",
            "azure_deployment",
            "api_version",
            "organization",
            "anthropic_version",
        }
        return {name: kwargs.pop(name) for name in list(kwargs) if name in names}

    def _resolve_defaults(
        self,
        provider: Optional[str],
        model: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        default = get_default()
        return provider or default.provider, model or default.model

    def _messages_text(self, messages: Optional[list[dict[str, Any]]]) -> str:
        if not messages:
            return ""
        return "\n".join(str(message.get("content", "")) for message in messages)

    def _ensure_response(self, fn: Any) -> AIResponse:
        result = fn() if callable(fn) else fn
        if isinstance(result, AIResponse):
            return result
        raise InvalidInputError(
            "Streaming calls return an iterator, not AIResponse.",
            fix="Use stream=False when you need response metadata.",
        )

    def _print_models(self, result: dict[str, list[str]] | list[str]) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="polyai models")
            table.add_column("Provider")
            table.add_column("Models")
            if isinstance(result, list):
                table.add_row("", "\n".join(result))
            else:
                for provider, provider_models in result.items():
                    table.add_row(provider, "\n".join(provider_models))
            Console().print(table)
        except Exception:
            print(result)


AI = _AIClient()
