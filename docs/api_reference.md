# API Reference

This is the public surface of `polyai`. Provider-specific request details live in
the provider classes.

## `AI(...)`

```python
AI(
    provider=None,
    model=None,
    *,
    api_key=None,
    prompt=None,
    system=None,
    messages=None,
    mode="text",
    stream=False,
    temperature=0.7,
    max_tokens=1024,
    timeout=30,
    retries=3,
    debug=False,
    max_cost_usd=None,
    **kwargs,
)
```

Returns `AIResponse`, except streaming text calls return an iterator of string
chunks.

## Configuration

- `AI.configure({"openai": "sk-..."})`: store keys globally.
- `AI.load_env(path=None, override=False)`: load `.env`.
- `AI.from_env()`: load env and return the client.
- `AI.set_default(provider=None, model=None)`: set default provider/model.

## Calls

- `AI.chat(provider, model, **kwargs)`: create `ChatSession`.
- `AI.batch(calls, max_parallel=5)`: run calls in parallel.
- `AI.compare(models, prompt, **kwargs)`: compare several models.
- `AI.fallback(chain, **kwargs)`: try provider/model specs in order.
- `AI.async_call(*args, **kwargs)`: async wrapper for one call.
- `AI.async_batch(calls, max_parallel=5)`: async wrapper for batch calls.

## Discovery and cost

- `AI.providers()`: list provider names.
- `AI.models(provider=None, mode=None, pretty=True)`: print and return models.
- `AI.estimate_cost(provider, model, prompt, max_tokens=1024)`: estimate cost.
- `AI.cost_tracker()`: return a `CostTracker`.
- `AI.ping(provider, api_key=None, model="fast")`: make a tiny test call.
- `AI.register_provider(name, provider_class, ...)`: register a provider.

## `AIResponse`

Fields:

- `text`
- `provider`
- `model`
- `mode`
- `tokens`
- `cost`
- `latency_ms`
- `created_at`
- `image_url`
- `image_data`
- `embeddings`
- `embedding_dim`
- `audio_data`
- `raw`
- `finish_reason`
- `request_id`
- `provider_used`

Methods:

- `save_image(path)`
- `save_audio(path)`
- `retry()`
- `to_dict()`

## `ChatSession`

- `send(prompt, **kwargs)`
- `reset()`
- `export_json(path=None)`
- `export_markdown(path=None)`
- `history`

## Exceptions

- `PolyAIError`
- `AuthError`
- `RateLimitError`
- `ModelNotFoundError`
- `ProviderNotFoundError`
- `ModeNotSupportedError`
- `TimeoutError`
- `InvalidInputError`
- `CostLimitError`

