from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..response import AIResponse, AITokens

COSTS: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o3": {"input": 10.00, "output": 40.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "text-embedding-3-small": {"input": 0.02, "output": 0.00},
        "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    },
    "anthropic": {
        "claude-opus-4-5": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    },
    "google": {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-ultra": {"input": 7.00, "output": 21.00},
    },
    "xai": {
        "grok-2": {"input": 2.00, "output": 10.00},
        "grok-2-mini": {"input": 0.30, "output": 0.50},
        "grok-3": {"input": 3.00, "output": 15.00},
        "grok-3-mini": {"input": 0.30, "output": 0.50},
        "grok-2-vision": {"input": 2.00, "output": 10.00},
    },
    "mistral": {
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-medium-latest": {"input": 0.40, "output": 2.00},
        "mistral-small-latest": {"input": 0.10, "output": 0.30},
        "codestral-latest": {"input": 0.30, "output": 0.90},
        "mistral-embed": {"input": 0.10, "output": 0.00},
    },
    "groq": {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "llama-3.3-70b-specdec": {"input": 0.59, "output": 0.99},
    },
    "cohere": {
        "command-r-plus": {"input": 2.50, "output": 10.00},
        "command-r": {"input": 0.15, "output": 0.60},
        "command-light": {"input": 0.30, "output": 0.60},
        "embed-english-v3.0": {"input": 0.10, "output": 0.00},
        "embed-multilingual-v3.0": {"input": 0.10, "output": 0.00},
    },
    "together": {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.60, "output": 0.60},
        "google/gemma-2-27b-it": {"input": 0.80, "output": 0.80},
        "deepseek-ai/DeepSeek-R1": {"input": 3.00, "output": 7.00},
    },
    "perplexity": {
        "sonar-pro": {"input": 3.00, "output": 15.00},
        "sonar": {"input": 1.00, "output": 1.00},
        "sonar-reasoning-pro": {"input": 2.00, "output": 8.00},
        "sonar-reasoning": {"input": 1.00, "output": 5.00},
        "sonar-deep-research": {"input": 2.00, "output": 8.00},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    },
}


def calculate_cost(provider: str, model: str, tokens: AITokens) -> float:
    """Calculate estimated USD cost from token usage.

    Costs are stored as USD per 1M tokens. Unknown models return ``0.0``.
    """

    rates = COSTS.get(provider, {}).get(model)
    if not rates:
        return 0.0
    input_cost = tokens.input / 1_000_000 * rates["input"]
    output_cost = tokens.output / 1_000_000 * rates["output"]
    return round(input_cost + output_cost, 10)


def estimate_token_count(text: str) -> int:
    """Estimate token count from text using a conservative character heuristic."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(
    provider: str,
    model: str,
    *,
    prompt: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> float:
    """Estimate request cost before sending it."""

    estimated_input = (
        input_tokens if input_tokens is not None else estimate_token_count(prompt or "")
    )
    estimated_output = output_tokens if output_tokens is not None else (max_tokens or 0)
    return calculate_cost(
        provider,
        model,
        AITokens(estimated_input, estimated_output, estimated_input + estimated_output),
    )


@dataclass
class CostTracker:
    """Accumulate cost across multiple AIResponse objects."""

    responses: list[AIResponse] = field(default_factory=list)

    def add(self, response: AIResponse) -> None:
        """Add a response to the tracker."""

        self.responses.append(response)

    @property
    def total_cost(self) -> float:
        """Return cumulative estimated USD cost."""

        return round(sum(item.cost for item in self.responses), 10)

    def breakdown(self) -> dict[str, float]:
        """Return cost totals grouped by provider."""

        totals: dict[str, float] = {}
        for response in self.responses:
            totals[response.provider] = totals.get(response.provider, 0.0) + response.cost
        return {key: round(value, 10) for key, value in sorted(totals.items())}

    def reset(self) -> None:
        """Clear tracked responses."""

        self.responses.clear()
