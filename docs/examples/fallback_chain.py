from polyai import AI

# Fallback tries each provider/model pair until one succeeds.
response = AI.fallback(
    ["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "groq/fast"],
    prompt="Summarize this in one sentence: polyai wraps several AI APIs.",
)

# provider_used tells you which model actually handled the request.
print(response.provider_used)
print(response.text)
