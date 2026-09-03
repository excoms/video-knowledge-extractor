from __future__ import annotations

import json
import re
from typing import Protocol


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str: ...


def extract_json(raw: str):
    """Models wrap JSON in prose or fences however they like. Be forgiving on
    input, strict about the result."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ProviderError(
        "The model did not return usable JSON. Smaller local models often "
        "struggle with this — try a larger model, or a different provider."
    )
