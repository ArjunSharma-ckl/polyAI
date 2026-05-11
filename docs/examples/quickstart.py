from polyai import AI

# Uses OPENAI_API_KEY from your environment or .env file.
response = AI("openai", "gpt-4o-mini", prompt="Say hello in one sentence")

# Every provider returns AIResponse, so text is always in the same place.
print(response.text)
