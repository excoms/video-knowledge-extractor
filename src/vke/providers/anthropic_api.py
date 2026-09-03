from __future__ import annotations

import os

from .base import ProviderError


class AnthropicProvider:
    name = "anthropic"
    DEFAULT_MODEL = "claude-opus-5"

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "Run: pip install 'video-knowledge-extractor[anthropic]'"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        self.model = model or self.DEFAULT_MODEL
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
