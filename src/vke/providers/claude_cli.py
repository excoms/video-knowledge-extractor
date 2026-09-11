"""Runs on an existing Claude subscription via the `claude` CLI — no API key,
no per-run cost. Usually the cheapest option for someone who already has one."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .base import ProviderError

TIMEOUT = 900


class ClaudeCliProvider:
    name = "claude-cli"

    def __init__(self, model: str | None = None):
        self.model = model
        self.candidates = self._find_installs()
        if not self.candidates:
            raise ProviderError(
                "The `claude` CLI is not on PATH. Install Claude Code, or pick "
                "another provider with --provider."
            )
        self.exe = self.candidates[0]
        self._tried: set[str] = set()

    @staticmethod
    def _find_installs() -> list[str]:
        """Every claude on this machine, PATH first.

        More than one install is common, and they are not interchangeable:
        a 2.1.83 npm-global build was observed returning an empty result in
        -p mode while the native 2.1.81 alongside it worked. When the first
        choice comes back empty we move down this list rather than failing.
        """
        found, seen = [], set()
        for path in (shutil.which("claude"),
                     os.path.expanduser("~/.local/bin/claude"),
                     "/usr/local/bin/claude",
                     "/opt/homebrew/bin/claude"):
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                real = os.path.realpath(path)
                if real not in seen:
                    seen.add(real)
                    found.append(path)
        return found

    # -- helpers -----------------------------------------------------------
    def _run(self, prompt: str, output_format: str,
             on_stdin: bool = False) -> subprocess.CompletedProcess:
        """`claude -p <prompt>` — the prompt is an argument, not stdin.

        Piping it in instead produces a *successful* call with an empty result
        and an input-token count of about 3: the prompt simply never arrives,
        and nothing reports an error. argv is safe here — macOS allows about a
        megabyte, far beyond any chunk we send.
        """
        cmd = [self.exe, "-p"]
        if not on_stdin:
            cmd.append(prompt)
        cmd += ["--output-format", output_format]
        if self.model:
            cmd += ["--model", self.model]
        return subprocess.run(cmd, input=prompt if on_stdin else None,
                              capture_output=True, text=True, timeout=TIMEOUT)

    @staticmethod
    def _unwrap(out: str) -> str:
        """`--output-format json` returns a result envelope, not the answer."""
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return out
        if isinstance(data, dict):
            if data.get("is_error") or data.get("subtype", "").startswith("error"):
                raise ProviderError(
                    "claude CLI reported an error: "
                    f"{data.get('subtype') or data.get('error') or out[:200]}"
                )
            for key in ("result", "text", "content"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return data[key]
            if "result" in data:
                # A successful call with an empty result means the prompt did
                # not reach the model. Say that, rather than handing the caller
                # the envelope to choke on.
                used = (data.get("usage") or {}).get("input_tokens")
                raise ProviderError(
                    "claude CLI completed but returned an empty response"
                    + (f" (it received only {used} input tokens, so the prompt "
                       f"did not reach it)" if isinstance(used, int) and used < 50
                       else "")
                    + ".\n  Check `claude -p 'say OK'` works in a terminal, "
                      "or use --provider anthropic."
                )
        return out

    def _fail(self, stage: str, r: subprocess.CompletedProcess) -> ProviderError:
        parts = [f"The claude CLI {stage}.", ""]
        parts.append(f"  command  : {' '.join(self.exe.split('/')[-1:])} -p --output-format …")
        parts.append(f"  exit code: {r.returncode}")
        parts.append(f"  stdout   : {(r.stdout or '').strip()[:300] or '(empty)'}")
        parts.append(f"  stderr   : {(r.stderr or '').strip()[:300] or '(empty)'}")
        parts.append("")
        blob = f"{r.stdout} {r.stderr}"
        if "authenticate" in blob or "authentication_error" in blob or "401" in blob:
            parts.append("  It is signed out. Run `claude` once in a terminal to sign in,")
            parts.append("  then start this again.")
        else:
            parts.append("  Check it works on its own:   claude -p 'say OK'")
            parts.append("  Or use another provider:     --provider anthropic | openai | ollama")
        return ProviderError("\n".join(parts))

    # -- interface ---------------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        prompt = f"{system}\n\n---\n\n{user}"
        try:
            r = self._run(prompt, "text")
        except subprocess.TimeoutExpired as e:
            raise ProviderError(
                f"claude CLI timed out after {TIMEOUT // 60} minutes. "
                f"A very long transcript can do this; try --provider anthropic."
            ) from e

        if r.returncode != 0:
            raise self._fail(f"exited with status {r.returncode}", r)

        out = (r.stdout or "").strip()
        if out:
            return self._unwrap(out)

        # Empty stdout on success is odd but recoverable: some versions only
        # emit on the json format. Try once more before giving up.
        try:
            r2 = self._run(prompt, "json")
        except subprocess.TimeoutExpired as e:
            raise ProviderError("claude CLI timed out on the retry.") from e
        out2 = (r2.stdout or "").strip()
        if r2.returncode == 0 and out2:
            try:
                return self._unwrap(out2)
            except ProviderError:
                pass                      # empty result — try another install

        self._tried.add(self.exe)
        for candidate in self.candidates:
            if candidate in self._tried:
                continue
            print(f"      {self.exe} returned nothing; trying {candidate}", flush=True)
            self.exe = candidate
            return self.complete(system, user, max_tokens)

        raise self._fail(
            "returned an empty result from every claude installation on this "
            f"machine ({', '.join(self.candidates)})", r2)
