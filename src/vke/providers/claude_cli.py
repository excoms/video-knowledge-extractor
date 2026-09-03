"""Runs on an existing Claude subscription via the `claude` CLI — no API key,
no per-run cost. Usually the cheapest option for someone who already has one."""
from __future__ import annotations

import shutil
import subprocess

from .base import ProviderError


class ClaudeCliProvider:
    name = "claude-cli"

    def __init__(self, model: str | None = None):
        self.model = model
        if not shutil.which("claude"):
            raise ProviderError(
                "The `claude` CLI is not on PATH. Install Claude Code, or pick "
                "another provider with --provider."
            )

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        cmd = ["claude", "-p", f"{system}\n\n---\n\n{user}"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired as e:
            raise ProviderError("claude CLI timed out after 15 minutes.") from e
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip()[:300] or "no error output"
            raise ProviderError(
                f"claude CLI exited {r.returncode}: {detail}\n"
                f"If you are running vke from inside a Claude Code session, the "
                f"nested call can hang or fail — run it from a normal terminal, "
                f"or use --provider anthropic/openai/ollama."
            )
        out = r.stdout.strip()
        if not out:
            raise ProviderError("claude CLI returned nothing. Try --provider anthropic.")
        return out
