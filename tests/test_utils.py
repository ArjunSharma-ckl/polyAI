from __future__ import annotations

import os

import pytest

from polyai.config import CONFIG, get_api_key
from polyai.exceptions import (
    AuthError,
    CostLimitError,
    InvalidInputError,
    ModelNotFoundError,
    ModeNotSupportedError,
    ProviderNotFoundError,
    RateLimitError,
    TimeoutError,
)
from polyai.registry import normalize_provider, resolve_model
from polyai.response import AITokens
from polyai.utils.costs import CostTracker, calculate_cost
from polyai.utils.env import load_env_file
from polyai.utils.retry import run_with_retries


def test_cost_calculation_and_tracker() -> None:
    cost = calculate_cost(
        "openai", "gpt-4o", AITokens(input=1_000_000, output=1_000_000, total=2_000_000)
    )

    assert cost == 12.5
    assert CostTracker().total_cost == 0


def test_retry_logic_retries_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    monkeypatch.setattr("time.sleep", lambda _: None)

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("openai", retry_after=0)
        return "ok"

    assert run_with_retries(flaky, retries=2) == "ok"
    assert attempts["count"] == 2


def test_retry_logic_does_not_retry_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    monkeypatch.setattr("time.sleep", lambda _: None)

    def bad_key() -> str:
        attempts["count"] += 1
        raise AuthError("openai")

    with pytest.raises(AuthError):
        run_with_retries(bad_key, retries=2)
    assert attempts["count"] == 1


def test_env_file_loading(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-env\nPOLYAI_GROQ_KEY=groq-env\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POLYAI_GROQ_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with CONFIG._lock:
        CONFIG._env_loaded = False

    loaded = load_env_file()

    assert loaded == env_file
    assert os.environ["OPENAI_API_KEY"] == "from-env"
    assert get_api_key("groq") == "groq-env"


def test_registry_aliases() -> None:
    assert normalize_provider("chatgpt") == "openai"
    assert resolve_model("openai", "cheap") == "gpt-4o-mini"


def test_registry_errors_are_specific() -> None:
    with pytest.raises(ProviderNotFoundError):
        normalize_provider("not-a-provider")

    with pytest.raises(ModelNotFoundError):
        resolve_model("openai", "not-a-model")


def test_exception_types_render_helpful_messages() -> None:
    errors = [
        AuthError("openai"),
        RateLimitError("openai", retry_after=1),
        ModeNotSupportedError("openai", "image", model="gpt-4o"),
        TimeoutError("openai", timeout=1),
        InvalidInputError("Bad input", fix="Fix it"),
        CostLimitError(0.02, 0.01, provider="openai", model="gpt-4o"),
    ]

    for error in errors:
        message = str(error)
        assert "->" in message
        assert "Docs:" in message or isinstance(error, InvalidInputError)
