from __future__ import annotations

from typing import Any


def audio(provider: str, model: str, prompt: str, **kwargs: Any):
    """Call a text-to-speech model through the public AI singleton."""

    from ..core import AI

    return AI(provider, model, prompt=prompt, mode="audio", **kwargs)


def stt(provider: str, model: str, audio: str, **kwargs: Any):
    """Call a speech-to-text model through the public AI singleton."""

    from ..core import AI

    return AI(provider, model, audio=audio, mode="stt", **kwargs)
