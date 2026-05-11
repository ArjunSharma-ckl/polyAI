from __future__ import annotations

from typing import Any


def text(provider: str, model: str, prompt: str, **kwargs: Any):
    """Call a text model through the public AI singleton."""

    from ..core import AI

    return AI(provider, model, prompt=prompt, mode="text", **kwargs)
