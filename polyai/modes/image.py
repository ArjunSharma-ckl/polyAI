from __future__ import annotations

from typing import Any


def image(provider: str, model: str, prompt: str, **kwargs: Any):
    """Call an image generation model through the public AI singleton."""

    from ..core import AI

    return AI(provider, model, prompt=prompt, mode="image", **kwargs)
