from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from ..exceptions import (
    AuthError,
    InvalidInputError,
    ModelNotFoundError,
    PolyAIError,
    RateLimitError,
)

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (AuthError, InvalidInputError, ModelNotFoundError)):
        return False
    if isinstance(exc, PolyAIError):
        return bool(getattr(exc, "retryable", False))
    return False


def run_with_retries(fn: Callable[[], T], *, retries: int = 3, base_delay: float = 1.0) -> T:
    """Run ``fn`` with exponential backoff on retryable PolyAI errors."""

    attempts = max(0, retries) + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as exc:
            last_error = exc
            if attempt >= attempts - 1 or not _is_retryable(exc):
                raise
            retry_after = exc.retry_after if isinstance(exc, RateLimitError) else None
            delay = retry_after if retry_after is not None else base_delay * (2**attempt)
            delay += random.uniform(0, min(0.25, delay * 0.1))
            time.sleep(delay)
    raise last_error  # type: ignore[misc]
