from polyai import AI

# Batch runs independent calls concurrently and keeps the result order.
batch = AI.batch(
    [
        {"provider": "openai", "model": "gpt-4o-mini", "prompt": "Capital of France?"},
        {"provider": "groq", "model": "fast", "prompt": "Capital of Japan?"},
    ]
)

for response in batch:
    print(response.provider, response.text)

# Compare sends the same prompt to several models.
comparison = AI.compare(
    models=["openai/gpt-4o-mini", "groq/fast"],
    prompt="Explain DNS in two sentences.",
)
comparison.print_table()
