"""Fully local and free — the lowest barrier for someone trying this out.
Output quality is noticeably weaker, especially at holding a JSON schema."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import ProviderError


class OllamaProvider:
    name = "ollama"
    DEFAULT_MODEL = "llama3.1"

    def __init__(self, model: str | None = None):
        self.model = model or self.DEFAULT_MODEL
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        payload = json.dumps({
            "model": self.model, "system": system, "prompt": user,
            "stream": False, "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read()).get("response", "")
        except urllib.error.URLError as e:
            raise ProviderError(
                f"Could not reach Ollama at {self.host}. Is it running? ({e})"
            ) from e
