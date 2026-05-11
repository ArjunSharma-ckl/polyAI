from __future__ import annotations

from typing import Any, Callable

from .exceptions import PolyAIError
from .response import AIResponse


def run_fallback(
    caller: Callable[..., AIResponse],
    chain: list[str],
    **kwargs: Any,
) -> AIResponse:
    """Try provider/model specs in order until one succeeds."""

    if not chain:
        raise PolyAIError(
            "Fallback chain is empty.",
            fix='Pass at least one provider/model string, such as ["openai/gpt-4o"].',
        )
    errors: list[BaseException] = []
    for item in chain:
        provider, model = item.split("/", 1) if "/" in item else (item, None)
        try:
            response = caller(provider=provider, model=model, **kwargs)
            response.provider_used = response.provider
            return response
        except Exception as exc:
            errors.append(exc)
            continue
    details = "; ".join(str(error).splitlines()[0] for error in errors)
    raise PolyAIError(
        "Every provider in the fallback chain failed.",
        fix=f"Check API keys, model names, and provider status. Failures: {details}",
    )
