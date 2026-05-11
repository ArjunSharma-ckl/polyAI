from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from polyai import AI
from polyai.config import CONFIG
from polyai.exceptions import CostLimitError, InvalidInputError, RateLimitError
from polyai.providers.base import BaseProvider, TextResult
from polyai.response import AIResponse, AITokens


def reset_config() -> None:
    with CONFIG._lock:
        CONFIG._keys.clear()
        CONFIG._default.provider = None
        CONFIG._default.model = None
        CONFIG._env_loaded = True


class FakeProvider(BaseProvider):
    name = "openai"

    def text(
        self,
        model: str,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TextResult:
        if kwargs.get("stream"):
            return iter(["he", "llo"])
        if model == "gpt-4o" and prompt == "fail":
            raise RateLimitError(self.name, retry_after=0)
        return AIResponse(
            text=prompt or "from messages",
            provider=self.name,
            model=model,
            mode="text",
            tokens=AITokens(1, 2, 3),
            raw={"api_key": self.api_key, "messages": messages, "system": system},
        )


class GroqFakeProvider(FakeProvider):
    name = "groq"


def test_dispatcher_routes_to_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    response = AI("openai", "gpt-4o", api_key="key", prompt="hello")

    assert response.text == "hello"
    assert response.provider == "openai"
    assert response.model == "gpt-4o"


def test_aliases_and_provider_model_shorthand(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    response = AI("gpt", "fast", api_key="key", prompt="hello")
    shorthand = AI("openai/gpt-4o", api_key="key", prompt="hello")

    assert response.model == "gpt-4o-mini"
    assert shorthand.model == "gpt-4o"


def test_provider_model_shorthand_ignores_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    AI.configure({"openai": "key"})
    AI.set_default(provider="openai", model="gpt-4o-mini")
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    response = AI("openai/gpt-4o", prompt="hello")

    assert response.model == "gpt-4o"


def test_configure_stores_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    AI.configure({"openai": "configured-key"})
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    response = AI("openai", "gpt-4o", prompt="hello")

    assert response.raw["api_key"] == "configured-key"


def test_default_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    AI.configure({"openai": "key"})
    AI.set_default(provider="openai", model="gpt-4o")
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    response = AI(prompt="hello")

    assert response.provider == "openai"
    assert response.model == "gpt-4o"


def test_cost_limit_is_checked_before_call(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    with pytest.raises(CostLimitError):
        AI("openai", "gpt-4o", api_key="key", prompt="hello", max_tokens=1000, max_cost_usd=0.0)


def test_streaming_yields_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    chunks = AI("openai", "gpt-4o", api_key="key", prompt="hello", stream=True)

    assert list(chunks) == ["he", "llo"]


def test_batch_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)
    calls = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "key", "prompt": "one"},
        {"provider": "openai", "model": "gpt-4o", "api_key": "key", "prompt": "two"},
    ]

    responses = AI.batch(calls, max_parallel=2)

    assert [response.text for response in responses] == ["one", "two"]


def test_fallback_tries_next_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    AI.configure({"openai": "key", "groq": "key"})

    def provider_class(name: str):
        return FakeProvider if name == "openai" else GroqFakeProvider

    monkeypatch.setattr("polyai.core.get_provider_class", provider_class)

    response = AI.fallback(["openai/gpt-4o", "groq/fast"], prompt="fail", retries=0)

    assert response.provider == "groq"
    assert response.provider_used == "groq"


def test_async_call(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    async def run() -> AIResponse:
        return await AI.async_call("openai", "gpt-4o", api_key="key", prompt="async")

    import asyncio

    response = asyncio.run(run())
    assert response.text == "async"


def test_compare_result(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: FakeProvider)

    result = AI.compare(
        models=["openai/gpt-4o", "openai/gpt-4o-mini"], prompt="same", api_key="key"
    )

    assert result.best_by_length().text == "same"
    assert len(result.to_dict()) == 2


def test_missing_provider_raises_helpful_error() -> None:
    reset_config()
    with pytest.raises(InvalidInputError):
        AI(prompt="hello")


def test_batch_runs_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()

    class SlowProvider(FakeProvider):
        def text(self, *args: Any, **kwargs: Any) -> TextResult:
            time.sleep(0.05)
            return super().text(*args, **kwargs)

    monkeypatch.setattr("polyai.core.get_provider_class", lambda name: SlowProvider)
    calls = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "key", "prompt": "a"},
        {"provider": "openai", "model": "gpt-4o", "api_key": "key", "prompt": "b"},
    ]

    started = time.perf_counter()
    AI.batch(calls, max_parallel=2)

    assert time.perf_counter() - started < 0.095
