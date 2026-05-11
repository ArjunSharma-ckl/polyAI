from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .response import AIResponse


class ChatSession:
    """Stateful multi-turn chat session.

    Args:
        caller: Callable compatible with ``AI(...)``.
        provider: Provider name.
        model: Model name or alias.
        system: Optional system message.
        kwargs: Default call options.
    """

    def __init__(
        self,
        caller: Callable[..., AIResponse],
        provider: str,
        model: str,
        *,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._caller = caller
        self.provider = provider
        self.model = model
        self.system = system
        self.kwargs = kwargs
        self.history: list[dict[str, str]] = []
        if system:
            self.history.append({"role": "system", "content": system})

    def send(self, prompt: str, **kwargs: Any) -> AIResponse:
        """Send a user message and append the assistant response to history."""

        self.history.append({"role": "user", "content": prompt})
        options = {**self.kwargs, **kwargs}
        response = self._caller(self.provider, self.model, messages=self.history, **options)
        self.history.append({"role": "assistant", "content": response.text})
        return response

    def reset(self) -> None:
        """Clear conversation history, preserving the system message if set."""

        self.history.clear()
        if self.system:
            self.history.append({"role": "system", "content": self.system})

    def export_json(self, path: Optional[str] = None) -> str:
        """Return or save the chat history as JSON."""

        rendered = json.dumps(self.history, indent=2)
        if path:
            Path(path).write_text(rendered + "\n", encoding="utf-8")
        return rendered

    def export_markdown(self, path: Optional[str] = None) -> str:
        """Return or save the chat history as readable Markdown."""

        lines = [f"# Chat: {self.provider}/{self.model}", ""]
        for message in self.history:
            role = message["role"].title()
            lines.extend([f"## {role}", "", message["content"], ""])
        rendered = "\n".join(lines).rstrip() + "\n"
        if path:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered
