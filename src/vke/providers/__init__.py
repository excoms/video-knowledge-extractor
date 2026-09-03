"""LLM backends.

A public tool that supports one API key promotes nothing, because most people
who find it cannot run it. So: bring your own key, run it locally for free, or
use a Claude subscription you already pay for.
"""
from __future__ import annotations

from .base import Provider, ProviderError

REGISTRY = {
    "claude-cli": ("claude_cli", "ClaudeCliProvider"),
    "anthropic": ("anthropic_api", "AnthropicProvider"),
    "openai": ("openai_api", "OpenAIProvider"),
    "ollama": ("ollama", "OllamaProvider"),
}


def get_provider(name: str, model: str | None = None) -> Provider:
    if name not in REGISTRY:
        raise ProviderError(
            f"Unknown provider {name!r}. Available: {', '.join(REGISTRY)}"
        )
    module_name, cls_name = REGISTRY[name]
    module = __import__(f"vke.providers.{module_name}", fromlist=[cls_name])
    return getattr(module, cls_name)(model=model)


def autodetect() -> str:
    """Pick whatever the user already has, cheapest first.

    A subscription they are already paying for beats spending their API credit.
    """
    import os
    import shutil

    if shutil.which("claude"):
        return "claude-cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


__all__ = ["Provider", "ProviderError", "get_provider", "autodetect", "REGISTRY"]
