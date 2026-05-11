# Providers

Provider-specific behavior stays inside provider adapters. User code should look
the same whether the request goes to OpenAI, Claude, Gemini, Groq, or a custom
OpenAI-compatible API.

## OpenAI

- API key: <https://platform.openai.com/api-keys>
- Env var: `OPENAI_API_KEY`
- Modes: text, vision, image, audio, speech-to-text, embeddings
- Models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1`,
  `o1-mini`, `o3`, `o3-mini`, `dall-e-3`, `dall-e-2`, `tts-1`, `tts-1-hd`,
  `whisper-1`, `text-embedding-3-small`, `text-embedding-3-large`
- Aliases: `best`, `fast`, `cheap`

```python
from polyai import AI

response = AI("openai", "fast", prompt="Explain vector search")
print(response.text)
```

Azure OpenAI uses the OpenAI adapter with deployment options:

```python
AI(
    "openai",
    "gpt-4o",
    api_key="azure-key",
    prompt="Hello",
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="my-deployment",
)
```

## Anthropic

- API key: <https://console.anthropic.com/settings/keys>
- Env var: `ANTHROPIC_API_KEY`
- Modes: text, vision
- Models: `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`,
  `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Aliases: `best`, `balanced`, `fast`

```python
from polyai import AI

response = AI("anthropic", "balanced", prompt="Review this API design")
print(response.text)
```

## Google Gemini

- API key: <https://aistudio.google.com/app/apikey>
- Env var: `GOOGLE_API_KEY`
- Modes: text, vision
- Models: `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-1.5-pro`,
  `gemini-1.5-flash`, `gemini-ultra`
- Aliases: `best`, `fast`
- Quirk: Google’s REST API uses `generationConfig`; PolyAI maps `max_tokens`
  and `temperature` for you.

```python
from polyai import AI

response = AI("google", "fast", prompt="Summarize this in two sentences")
print(response.text)
```

## xAI

- API key: <https://console.x.ai/>
- Env var: `XAI_API_KEY`
- Modes: text, vision
- Models: `grok-2`, `grok-2-mini`, `grok-3`, `grok-3-mini`, `grok-2-vision`
- Aliases: `best`, `fast`
- Quirk: xAI is handled by the OpenAI-compatible adapter.

```python
from polyai import AI

print(AI("xai", "fast", prompt="Give me a punchy headline").text)
```

## Mistral

- API key: <https://console.mistral.ai/api-keys/>
- Env var: `MISTRAL_API_KEY`
- Modes: text, embeddings
- Models: `mistral-large-latest`, `mistral-medium-latest`,
  `mistral-small-latest`, `codestral-latest`, `mistral-embed`
- Aliases: `best`, `fast`

```python
from polyai import AI

print(AI("mistral", "fast", prompt="Write a Python docstring").text)
```

## Groq

- API key: <https://console.groq.com/keys>
- Env var: `GROQ_API_KEY`
- Modes: text
- Models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gemma2-9b-it`,
  `mixtral-8x7b-32768`, `llama-3.3-70b-specdec`
- Aliases: `best`, `fast`
- Quirk: Groq is OpenAI-compatible and usually very fast.

```python
from polyai import AI

print(AI("groq", "fast", prompt="Capital of Japan?").text)
```

## Cohere

- API key: <https://dashboard.cohere.com/api-keys>
- Env var: `COHERE_API_KEY`
- Modes: text, embeddings
- Models: `command-r-plus`, `command-r`, `command-light`,
  `embed-english-v3.0`, `embed-multilingual-v3.0`
- Aliases: `best`, `fast`

```python
from polyai import AI

print(AI("cohere", "fast", prompt="Draft a customer support reply").text)
```

## Together

- API key: <https://api.together.xyz/settings/api-keys>
- Env var: `TOGETHER_API_KEY`
- Modes: text, image
- Models: `meta-llama/Llama-3.3-70B-Instruct-Turbo`,
  `mistralai/Mixtral-8x7B-Instruct-v0.1`, `google/gemma-2-27b-it`,
  `deepseek-ai/DeepSeek-R1`, `black-forest-labs/FLUX.1-schnell`
- Alias: `best`
- Quirk: Together uses full organization/model names.

```python
from polyai import AI

print(AI("together", "best", prompt="Explain queue backpressure").text)
```

## Perplexity

- API key: <https://www.perplexity.ai/settings/api>
- Env var: `PERPLEXITY_API_KEY`
- Modes: text
- Models: `sonar-pro`, `sonar`, `sonar-reasoning-pro`, `sonar-reasoning`,
  `sonar-deep-research`
- Aliases: `best`, `fast`
- Quirk: Perplexity is OpenAI-compatible but may include search-grounded
  behavior depending on the selected model.

```python
from polyai import AI

print(AI("perplexity", "fast", prompt="What is changing in Python packaging?").text)
```

## DeepSeek

- API key: <https://platform.deepseek.com/api_keys>
- Env var: `DEEPSEEK_API_KEY`
- Modes: text
- Models: `deepseek-chat`, `deepseek-reasoner`
- Aliases: `best`, `reasoning`
- Quirk: DeepSeek is handled by the OpenAI-compatible adapter.

```python
from polyai import AI

print(AI("deepseek", "reasoning", prompt="Solve this step by step: 19 * 27").text)
```

