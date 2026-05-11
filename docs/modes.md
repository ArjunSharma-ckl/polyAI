# Modes

The `mode` parameter chooses the provider method. Text is the default.

## Text

Use text for normal chat and completion-style work.

Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Groq, Cohere,
Together, Perplexity, DeepSeek.

Common parameters: `prompt`, `system`, `messages`, `temperature`, `max_tokens`,
`stream`, `timeout`, `retries`.

```python
from polyai import AI

response = AI(
    "openai",
    "gpt-4o-mini",
    prompt="Explain optimistic locking",
    system="Use practical backend examples",
)
print(response.text)
```

## Vision

Use vision when the model needs to read one or more images.

Supported providers: OpenAI, Anthropic, Google, xAI.

Parameters: `prompt`, `image`, `images`, plus the usual generation options.

```python
from polyai import AI

response = AI(
    "openai",
    "gpt-4o",
    prompt="Describe this image for alt text",
    image="./photo.jpg",
    mode="vision",
)
print(response.text)
```

## Image

Use image mode for image generation.

Supported providers: OpenAI, Together.

Common parameters: `prompt`, `size`, `quality`, `n`.

```python
from polyai import AI

response = AI(
    "openai",
    "dall-e-3",
    prompt="A simple app icon for a note-taking tool",
    mode="image",
    size="1024x1024",
)
response.save_image("icon.png")
```

## Audio

Use audio mode for text-to-speech.

Supported providers: OpenAI.

Common parameters: `prompt`, `voice`, `response_format`.

```python
from polyai import AI

response = AI("openai", "tts-1", prompt="Deploy finished", mode="audio", voice="alloy")
response.save_audio("deploy.mp3")
```

## Speech to text

Use `mode="stt"` to transcribe a local audio file.

Supported providers: OpenAI.

Parameters: `audio`, plus provider transcription options.

```python
from polyai import AI

response = AI("openai", "whisper-1", audio="meeting.mp3", mode="stt")
print(response.text)
```

## Embeddings

Use embeddings when you need vectors for search, clustering, ranking, or
retrieval.

Supported providers: OpenAI, Mistral, Cohere.

Parameters: `prompt`; providers may also accept batch input through their native
options.

```python
from polyai import AI

response = AI("openai", "text-embedding-3-small", prompt="Text to embed", mode="embed")
print(response.embedding_dim)
```

