# Getting Started

Most users only need the README. This page is the same quick path in a smaller
form, useful when browsing the `docs/` folder locally.

## Install

```bash
pip install polyai
```

## 30-second quickstart

```python
from polyai import AI

response = AI("openai", "gpt-4o", api_key="sk-...", prompt="Say hello in one sentence")
print(response.text)
```

With a `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

```python
from polyai import AI

print(AI("openai", "fast", prompt="Write one useful sentence about Redis").text)
```

## Five common patterns

### Configure keys once

```python
from polyai import AI

AI.configure({
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
    "groq": "gsk_...",
})

print(AI("anthropic", "balanced", prompt="Explain DNS").text)
```

### Use aliases

```python
from polyai import AI

print(AI("openai", "cheap", prompt="Draft a concise changelog entry").text)
```

### Use shorthand

```python
from polyai import AI

print(AI("groq/fast", prompt="Capital of Germany?").text)
```

### Keep context

```python
from polyai import AI

chat = AI.chat("openai", "gpt-4o-mini")
chat.send("My name is Alex")
print(chat.send("What is my name?").text)
```

### Fall back when a provider is down

```python
from polyai import AI

response = AI.fallback(
    ["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "groq/fast"],
    prompt="Summarize this note",
)

print(response.provider_used, response.text)
```

## Provider keys

| Provider | Env var |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| xAI | `XAI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

More detail:

- [Provider guide](providers.md)
- [Modes](modes.md)
- [Advanced usage](advanced.md)
- [API reference](api_reference.md)
- [README](../README.md)

