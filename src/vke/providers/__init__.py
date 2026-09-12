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


# Which environment variable each backend reads its credential from. A
# backend absent from this map needs no key: claude-cli uses a subscription
# the user has already signed into, ollama runs on their own machine.
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def set_key(name: str, key: str) -> None:
    """Hold a credential for the life of this process, and no longer.

    Deliberately not written to disk. A local tool that quietly persists an
    API key leaves it somewhere the user did not choose and will not remember
    to clean up; the cost of retyping it is smaller than that.
    """
    import os

    env = KEY_ENV.get(name)
    if not env:
        raise ProviderError(f"{name} does not take an API key.")
    key = (key or "").strip()
    if not key:
        raise ProviderError("No key given.")
    os.environ[env] = key


def describe() -> dict:
    """What every backend is, and what it would need to become usable."""
    import os

    out = {}
    for n in REGISTRY:
        entry = {"key_env": KEY_ENV.get(n), "has_key": False, "models": []}
        if KEY_ENV.get(n):
            entry["has_key"] = bool(os.environ.get(KEY_ENV[n]))
        try:
            get_provider(n)
            entry["status"] = "ready"
        except Exception as e:
            entry["status"] = str(e).splitlines()[0][:120]
        if n == "ollama":
            entry["models"] = ollama_models()
        out[n] = entry
    return out


def ollama_models() -> list[str]:
    """Models already pulled locally. Empty list if Ollama is not running —
    which is not an error, just nothing to offer."""
    import json
    import os
    import urllib.error
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


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


__all__ = ["Provider", "ProviderError", "get_provider", "autodetect",
           "REGISTRY", "KEY_ENV", "set_key", "describe", "ollama_models"]
