from __future__ import annotations

import json
from typing import Any

import httpx

from polyai.providers.anthropic import AnthropicProvider
from polyai.providers.cohere import CohereProvider
from polyai.providers.deepseek import DeepSeekProvider
from polyai.providers.google import GoogleProvider
from polyai.providers.groq import GroqProvider
from polyai.providers.mistral import MistralProvider
from polyai.providers.openai import OpenAIProvider
from polyai.providers.perplexity import PerplexityProvider
from polyai.providers.together import TogetherProvider
from polyai.providers.xai import XAIProvider
from tests.fixtures.mock_responses import (
    ANTHROPIC_MESSAGE,
    GOOGLE_GENERATE,
    OPENAI_CHAT,
    OPENAI_EMBED,
)


class TransportProviderMixin:
    transport: httpx.MockTransport

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport)


def make_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data)


def test_openai_text_builds_chat_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        captured["authorization"] = request.headers["authorization"]
        return make_response(OPENAI_CHAT)

    class Provider(TransportProviderMixin, OpenAIProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("key").text("gpt-4o", prompt="hello", system="brief")

    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["authorization"] == "Bearer key"
    assert response.text == "hello"
    assert response.tokens.total == 5


def test_openai_embed_builds_embedding_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return make_response(OPENAI_EMBED)

    class Provider(TransportProviderMixin, OpenAIProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("key").embed("text-embedding-3-small", prompt="hello")

    assert captured["body"]["input"] == "hello"
    assert response.embeddings == [0.1, 0.2, 0.3]
    assert response.embedding_dim == 3


def test_anthropic_text_builds_messages_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        captured["api_key"] = request.headers["x-api-key"]
        return make_response(ANTHROPIC_MESSAGE)

    class Provider(TransportProviderMixin, AnthropicProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("ant-key").text("claude-sonnet-4-5", prompt="hello", system="brief")

    assert captured["path"] == "/v1/messages"
    assert captured["body"]["system"] == "brief"
    assert captured["api_key"] == "ant-key"
    assert response.text == "hello"


def test_google_text_builds_generate_content_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = request.url.query.decode()
        captured["body"] = json.loads(request.content.decode())
        return make_response(GOOGLE_GENERATE)

    class Provider(TransportProviderMixin, GoogleProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("google-key").text("gemini-2.0-flash", prompt="hello", system="brief")

    assert captured["path"] == "/v1beta/models/gemini-2.0-flash:generateContent"
    assert "key=google-key" in captured["query"]
    assert captured["body"]["system_instruction"]["parts"][0]["text"] == "brief"
    assert response.text == "hello"


def test_mistral_text_builds_chat_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return make_response(OPENAI_CHAT)

    class Provider(TransportProviderMixin, MistralProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("mistral-key").text("mistral-small-latest", prompt="hello")

    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["model"] == "mistral-small-latest"
    assert response.text == "hello"


def test_cohere_text_builds_v2_chat_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return make_response(
            {
                "id": "co_test",
                "message": {"content": [{"type": "text", "text": "hello"}]},
                "usage": {"tokens": {"input_tokens": 2, "output_tokens": 3}},
            }
        )

    class Provider(TransportProviderMixin, CohereProvider):
        transport = httpx.MockTransport(handler)

    response = Provider("co-key").text("command-r", prompt="hello", system="brief")

    assert captured["path"] == "/v2/chat"
    assert captured["body"]["preamble"] == "brief"
    assert response.text == "hello"


def test_openai_compatible_provider_base_urls() -> None:
    providers = [
        (XAIProvider, "https://api.x.ai/v1"),
        (GroqProvider, "https://api.groq.com/openai/v1"),
        (TogetherProvider, "https://api.together.xyz/v1"),
        (PerplexityProvider, "https://api.perplexity.ai"),
        (DeepSeekProvider, "https://api.deepseek.com"),
    ]

    for provider_class, base_url in providers:
        assert provider_class("key").base_url == base_url
