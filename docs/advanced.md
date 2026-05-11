# Advanced Usage

## Batch calls

`AI.batch()` runs calls in a thread pool and preserves input order.

```python
from polyai import AI

results = AI.batch(
    [
        {"provider": "openai", "model": "gpt-4o-mini", "prompt": "Capital of France?"},
        {"provider": "groq", "model": "fast", "prompt": "Capital of Japan?"},
    ],
    max_parallel=5,
)
```

## Compare models

```python
from polyai import AI

result = AI.compare(
    models=["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5", "groq/fast"],
    prompt="Explain durable queues",
)

result.print_table()
print(result.best_by_speed().model)
```

## Fallbacks

```python
from polyai import AI

response = AI.fallback(
    ["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "groq/fast"],
    prompt="Summarize this document",
)
print(response.provider_used)
```

## Async

The async methods use `asyncio` and run the sync provider calls in an executor,
which keeps the public API consistent.

```python
import asyncio
from polyai import AI

async def main():
    response = await AI.async_call("openai", "gpt-4o-mini", prompt="Hello")
    print(response.text)

asyncio.run(main())
```

## Cost tracking

```python
from polyai import AI

tracker = AI.cost_tracker()
response = AI("openai", "gpt-4o-mini", prompt="What is a token?")
tracker.add(response)

print(response.cost)
print(tracker.total_cost)
print(tracker.breakdown())
```

Use `max_cost_usd` to block an expensive request before it is sent:

```python
AI("openai", "gpt-4o", prompt="...", max_tokens=5000, max_cost_usd=0.02)
```

## Chat sessions

```python
from polyai import AI

chat = AI.chat("anthropic", "claude-sonnet-4-5", system="Be concise")
chat.send("My name is Alex")
print(chat.send("What is my name?").text)
chat.export_markdown("chat.md")
```

## Streaming

```python
from polyai import AI

for chunk in AI("openai", "gpt-4o-mini", prompt="Write a story", stream=True):
    print(chunk, end="", flush=True)
```

## HTTP options

Every call accepts `timeout`. OpenAI also accepts `base_url`, `organization`,
`azure_endpoint`, `azure_deployment`, and `api_version`.

```python
AI("openai", "gpt-4o", prompt="Hello", timeout=10)
```

## Debug logging

```python
AI("openai", "gpt-4o-mini", prompt="Hi", debug=True)
```

Debug logs include request and response payloads. Avoid this when prompts contain
secrets.

## Custom providers

```python
from polyai import AI
from polyai.providers import BaseProvider
from polyai.response import AIResponse

class LocalProvider(BaseProvider):
    name = "local"

    def text(self, model, prompt=None, system=None, messages=None, **kwargs):
        return AIResponse(text=f"local: {prompt}", provider="local", model=model, mode="text")

AI.register_provider("local", LocalProvider, modes={"text": ["my-model"]})
print(AI("local", "my-model", api_key="unused", prompt="hello").text)
```

