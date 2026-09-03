from __future__ import annotations

import os

from .base import ProviderError


class OpenAIProvider:
    name = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(
                "Run: pip install 'video-knowledge-extractor[openai]'"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set.")
        self.model = model or self.DEFAULT_MODEL
        self._client = OpenAI()

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        r = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""
