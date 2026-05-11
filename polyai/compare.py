from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .response import AIResponse


@dataclass
class CompareResult:
    """Results from ``AI.compare(...)``."""

    responses: list[AIResponse] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return comparison results as dictionaries."""

        return [response.to_dict() for response in self.responses]

    def best_by_speed(self) -> AIResponse:
        """Return the fastest response."""

        return min(self.responses, key=lambda item: item.latency_ms)

    def best_by_length(self) -> AIResponse:
        """Return the response with the longest text output."""

        return max(self.responses, key=lambda item: len(item.text or ""))

    def print_table(self) -> None:
        """Print a compact terminal table using Rich when available."""

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="polyai comparison")
            table.add_column("Provider")
            table.add_column("Model")
            table.add_column("Latency")
            table.add_column("Cost")
            table.add_column("Preview")
            for response in self.responses:
                table.add_row(
                    response.provider,
                    response.model,
                    f"{response.latency_ms} ms",
                    f"${response.cost:.6f}",
                    response.text.replace("\n", " ")[:80],
                )
            Console().print(table)
        except Exception:
            for response in self.responses:
                preview = response.text.replace("\n", " ")[:80]
                label = f"{response.provider}/{response.model}"
                print(f"{label} {response.latency_ms} ms " f"${response.cost:.6f} {preview}")


def compare_models(
    caller: Callable[..., AIResponse],
    *,
    models: list[str],
    prompt: str,
    **kwargs: Any,
) -> CompareResult:
    """Run the same prompt against multiple provider/model specs."""

    responses: list[AIResponse] = []
    for item in models:
        provider, model = item.split("/", 1)
        responses.append(caller(provider=provider, model=model, prompt=prompt, **kwargs))
    return CompareResult(responses=responses)
