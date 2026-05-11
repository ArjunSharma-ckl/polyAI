# polyai

[![PyPI](https://img.shields.io/pypi/v/polyai.svg)](https://pypi.org/project/polyai/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

One import. One call. Any major AI provider.

`polyai` is a small Python client for calling OpenAI, Anthropic, Gemini, Grok,
Mistral, Groq, Cohere, Together, Perplexity, DeepSeek, and OpenAI-compatible
APIs without rewriting the same glue code every time.

```bash
pip install polyai
```

## 30-second quickstart

```python
from polyai import AI

response = AI("openai", "gpt-4o", api_key="sk-...", prompt="Say hello in one sentence")
print(response.text)
```

Or put this in `.env`:

```bash
OPENAI_API_KEY=sk-...
```

Then:

```python
from polyai import AI

print(AI("openai", "fast", prompt="Write a haiku about SQLite").text)
```

## Why use it

- Same return object for every provider: `AIResponse`
- Model aliases like `best`, `fast`, and `cheap`
- `.env` loading with no setup ceremony
- Text, vision, image generation, audio, speech-to-text, and embeddings
- Batch calls, model comparison, fallback chains, async wrappers, and cost estimates
- Helpful exceptions instead of raw provider payloads
- No provider SDKs required for normal use; `httpx` does the transport

## Common patterns

### Set keys once

```python
from polyai import AI

AI.configure({
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
    "groq": "gsk_...",
})

response = AI("anthropic", "balanced", prompt="Explain cache invalidation")
print(response.text)
```

### Use provider/model shorthand

```python
from polyai import AI

response = AI("openai/gpt-4o-mini", prompt="Give me three commit message options")
print(response.text)
```

### Stream text

```python
from polyai import AI

for chunk in AI("openai", "gpt-4o", prompt="Tell a short story", stream=True):
    print(chunk, end="", flush=True)
```

### Keep a chat session

```python
from polyai import AI

chat = AI.chat("anthropic", "claude-sonnet-4-5")
chat.send("My name is Alex")
reply = chat.send("What is my name?")
print(reply.text)
```

### Try a fallback chain

```python
from polyai import AI

response = AI.fallback(
    ["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "groq/fast"],
    prompt="Summarize this note in five bullets",
)

print(response.provider_used, response.text)
```

### Compare models

```python
from polyai import AI

result = AI.compare(
    models=["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "google/gemini-2.0-flash"],
    prompt="Explain the trolley problem without jargon",
)

result.print_table()
```

## Provider setup

| Provider | Env var | Good first model | Modes | API key |
| --- | --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | text, vision, image, audio, stt, embed | <https://platform.openai.com/api-keys> |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` | text, vision | <https://console.anthropic.com/settings/keys> |
| Google | `GOOGLE_API_KEY` | `gemini-2.0-flash` | text, vision | <https://aistudio.google.com/app/apikey> |
| xAI | `XAI_API_KEY` | `grok-3-mini` | text, vision | <https://console.x.ai/> |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` | text, embed | <https://console.mistral.ai/api-keys/> |
| Groq | `GROQ_API_KEY` | `llama-3.1-8b-instant` | text | <https://console.groq.com/keys> |
| Cohere | `COHERE_API_KEY` | `command-r` | text, embed | <https://dashboard.cohere.com/api-keys> |
| Together | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | text, image | <https://api.together.xyz/settings/api-keys> |
| Perplexity | `PERPLEXITY_API_KEY` | `sonar` | text | <https://www.perplexity.ai/settings/api> |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` | text | <https://platform.deepseek.com/api_keys> |

Alternative env names also work, for example `POLYAI_OPENAI_KEY` and
`POLYAI_OPENAI_API_KEY`.

## Models and aliases

```python
AI.providers()
AI.models()                  # prints a Rich table and returns the model data
AI.models("openai")
AI.models(mode="vision")
AI.models(pretty=False)      # return only, no table
```

Aliases are resolved before a request is sent:

```python
AI("openai", "best", prompt="...")      # gpt-4o
AI("openai", "fast", prompt="...")      # gpt-4o-mini
AI("anthropic", "balanced", prompt="...")  # claude-sonnet-4-5
AI("groq", "fast", prompt="...")        # llama-3.1-8b-instant
```

## Modes

### Text

```python
response = AI(
    "openai",
    "gpt-4o",
    prompt="Explain DNS like I am new to backend work",
    system="Be direct and practical",
    temperature=0.4,
    max_tokens=800,
)
```

### Vision

```python
response = AI(
    "openai",
    "gpt-4o",
    prompt="What is in this image?",
    image="./photo.jpg",
    mode="vision",
)
```

### Image generation

```python
response = AI(
    "openai",
    "dall-e-3",
    prompt="A clean product photo of a mechanical keyboard on a white desk",
    mode="image",
    size="1024x1024",
    quality="hd",
)

response.save_image("keyboard.png")
```

### Embeddings

```python
response = AI("openai", "text-embedding-3-small", prompt="Text to embed", mode="embed")
print(response.embedding_dim)
```

### Audio and speech-to-text

```python
speech = AI("openai", "tts-1", prompt="Build logs are green", mode="audio", voice="alloy")
speech.save_audio("status.mp3")

transcript = AI("openai", "whisper-1", audio="meeting.mp3", mode="stt")
print(transcript.text)
```

## AIResponse

Every normal call returns the same shape:

```python
response.text
response.provider
response.model
response.mode
response.tokens.input
response.tokens.output
response.tokens.total
response.cost
response.latency_ms
response.raw
response.request_id
response.to_dict()
```

Modality fields are filled only when relevant:

```python
response.image_url
response.image_data
response.embeddings
response.embedding_dim
response.audio_data
```

`print(response)` prints `response.text`.

## Cost tracking

```python
tracker = AI.cost_tracker()

r1 = AI("openai", "gpt-4o-mini", prompt="Short answer: what is Redis?")
r2 = AI("groq", "fast", prompt="Short answer: what is Postgres?")

tracker.add(r1)
tracker.add(r2)

print(tracker.total_cost)
print(tracker.breakdown())
```

You can block a request before it is sent:

```python
AI("openai", "gpt-4o", prompt="...", max_tokens=4000, max_cost_usd=0.01)
```

## Retries, timeouts, and rate limits

Defaults:

- timeout: 30 seconds
- retries: 3
- backoff: 1s, 2s, 4s plus small jitter
- retryable: rate limits, timeouts, provider 5xx responses
- not retried: auth errors, invalid input, unknown models

```python
response = AI("openai", "gpt-4o-mini", prompt="Hello", timeout=15, retries=2)
```

## Async

```python
import asyncio
from polyai import AI

async def main():
    response = await AI.async_call("openai", "gpt-4o-mini", prompt="Hello async")
    print(response.text)

asyncio.run(main())
```

```python
results = await AI.async_batch([
    {"provider": "openai", "model": "gpt-4o-mini", "prompt": "One"},
    {"provider": "groq", "model": "fast", "prompt": "Two"},
])
```

## Custom providers

Custom providers subclass `BaseProvider` and return `AIResponse`.

```python
from polyai import AI
from polyai.providers import BaseProvider
from polyai.response import AIResponse

class LocalProvider(BaseProvider):
    name = "local"

    def text(self, model, prompt=None, system=None, messages=None, **kwargs):
        return AIResponse(text=f"local: {prompt}", provider="local", model=model, mode="text")

AI.register_provider("local", LocalProvider, modes={"text": ["my-model"]})
print(AI("local", "my-model", api_key="unused", prompt="hi").text)
```

## Azure OpenAI

```python
response = AI(
    "openai",
    "gpt-4o",
    api_key="azure-key",
    prompt="Hello from Azure",
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="my-deployment",
    api_version="2024-10-21",
)
```

## Debugging

```python
AI("openai", "gpt-4o-mini", prompt="Hi", debug=True)
```

Debug output goes to stderr and includes request/response payloads. Do not enable it
when prompts contain secrets.

## API reference

### `AI(...)`

Main dispatcher. Important parameters:

- `provider`, `model`: provider and model, or `provider/model` shorthand
- `api_key`: explicit key; otherwise configured keys and env vars are used
- `prompt`, `system`, `messages`: text inputs
- `mode`: `text`, `vision`, `image`, `audio`, `stt`, or `embed`
- `stream`: return text chunks instead of `AIResponse`
- `temperature`, `max_tokens`, `timeout`, `retries`
- `max_cost_usd`: pre-flight cost guard
- provider options: `base_url`, `azure_endpoint`, `azure_deployment`, `api_version`

### Helpers

- `AI.configure({...})`
- `AI.load_env()`
- `AI.set_default(provider="openai", model="gpt-4o-mini")`
- `AI.chat(provider, model, **options)`
- `AI.batch(calls, max_parallel=5)`
- `AI.compare(models=[...], prompt="...")`
- `AI.fallback([...], prompt="...")`
- `AI.async_call(...)`
- `AI.async_batch([...])`
- `AI.cost_tracker()`
- `AI.estimate_cost(provider, model, prompt)`
- `AI.ping(provider, api_key="...")`
- `AI.register_provider(name, ProviderClass, ...)`
- `AI.providers()`
- `AI.models(provider=None, mode=None, pretty=True)`

## Contributing

Keep the public API boring and stable. Provider-specific weirdness should stay in
provider adapters, not leak into user code. Add tests for request bodies and error
translation before changing provider behavior.

Local checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
black --check .
pytest
```

## License

MIT. See [LICENSE](LICENSE).
