from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .response import AIResponse


def run_batch(
    caller: Callable[..., AIResponse],
    calls: list[dict[str, Any]],
    *,
    max_parallel: int = 5,
) -> list[AIResponse]:
    """Run multiple AI calls concurrently while preserving input order.

    Args:
        caller: Callable compatible with ``AI(...)``.
        calls: List of call dictionaries.
        max_parallel: Maximum number of worker threads.

    Returns:
        Responses in the same order as ``calls``.
    """

    if not calls:
        return []
    results: list[AIResponse | None] = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as executor:
        future_map = {executor.submit(caller, **call): index for index, call in enumerate(calls)}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    return [item for item in results if item is not None]
