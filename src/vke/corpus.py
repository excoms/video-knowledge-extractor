"""The cross-video pass.

Per-video output is homework: a stack of reports you still have to read and
compare. The corpus pass reads every record together and says what recurs —
which is the thing you actually act on, and the reason the output is JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from .providers.base import extract_json

CORPUS_SYSTEM = """You are given every record extracted from a set of videos by
the same source.

Find what recurs. Return JSON only:

{"summary": "<a short paragraph: what this body of material is, and its
              single most important pattern>",
 "patterns": [{"pattern": "<what recurs>",
               "occurrences": <integer>,
               "severity": "high" | "medium" | "low",
               "examples": ["<video title — brief detail>"],
               "recommendation": "<what to do about it>"}],
 "contradictions": [{"claim_a": "...", "claim_b": "...",
                     "where": "<which videos>", "note": "<which is better supported>"}],
 "strongest": ["<the material that holds up best, and why>"]}

Rules:
- Rank patterns by how much they matter, not how often they appear.
- Report [] honestly if there are no real contradictions. Do not manufacture them.
- Be specific. "Some arguments are weak" is useless."""


def build_register(provider, records: list[dict], request: str,
                   output_language: str = "English") -> dict:
    if not records:
        return {}

    condensed = [{
        "video": r["video_title"][:80],
        "speaker": r.get("speaker"),
        "confidence": r.get("confidence"),
        **{k: (v[:280] if isinstance(v, str) else v)
           for k, v in (r.get("record") or {}).items()},
    } for r in records]

    payload = json.dumps(condensed, ensure_ascii=False)[:120000]
    user = (f"The user asked for:\n{request}\n\n"
            f"Write your answer in {output_language}.\n\n"
            f"RECORDS ({len(records)} from "
            f"{len({r['video_id'] for r in records})} videos):\n{payload}\n\n"
            f"Return the JSON object.")
    try:
        return extract_json(provider.complete(CORPUS_SYSTEM, user, max_tokens=6000))
    except Exception:
        return {}


def write_register(reg: dict, path: Path, meta: dict) -> Path | None:
    if not reg:
        return None
    L = ["# Patterns across the whole source\n"]
    L.append(f"**Source:** {meta.get('source', '')}  ")
    L.append(f"**Videos:** {meta.get('video_count', 0)} · "
             f"**Records:** {meta.get('record_count', 0)}\n")
    if reg.get("summary"):
        L.append(reg["summary"] + "\n")

    if reg.get("patterns"):
        L.append("## What recurs\n")
        for p in reg["patterns"]:
            L.append(f"### {p.get('pattern', '')}")
            L.append(f"*{p.get('occurrences', '?')} occurrences · "
                     f"severity: {p.get('severity', '?')}*\n")
            for ex in p.get("examples", []) or []:
                L.append(f"- {ex}")
            if p.get("recommendation"):
                L.append(f"\n**Do this:** {p['recommendation']}\n")

    if reg.get("contradictions"):
        L.append("## Contradictions\n")
        for c in reg["contradictions"]:
            L.append(f"- **{c.get('claim_a', '')}** vs **{c.get('claim_b', '')}**  ")
            L.append(f"  {c.get('where', '')} — {c.get('note', '')}")
        L.append("")

    if reg.get("strongest"):
        L.append("## What holds up best\n")
        for s in reg["strongest"]:
            L.append(f"- {s}")
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path
