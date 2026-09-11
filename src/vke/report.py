"""Render records into something a person reads.

JSON is canonical; Markdown is generated from it. That ordering is what makes
the corpus pass possible and lets other tools consume the output — prose cannot
be aggregated, counted or piped anywhere.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _fmt_ts(ts) -> str:
    if not ts:
        return ""
    m, s = divmod(int(ts[0]), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def write_json(records: list[dict], path: Path, meta: dict) -> Path:
    path.write_text(json.dumps(
        {"meta": meta, "records": records}, indent=2, ensure_ascii=False
    ), encoding="utf-8")
    return path


def _title_for(rec: dict) -> tuple[str, str | None]:
    """First text field doubles as the headline; return its key so the body
    does not repeat it verbatim under the heading."""
    for k, v in (rec.get("record") or {}).items():
        if isinstance(v, str) and v.strip():
            return v.strip(), k
    return rec.get("record_name", "record").title(), None


def write_markdown(records: list[dict], path: Path, meta: dict) -> Path:
    L: list[str] = []
    L.append(f"# {meta.get('title', 'Analysis')}\n")
    L.append(f"**Request:** {meta.get('request', '')}  ")
    L.append(f"**Source:** {meta.get('source', '')}  ")
    L.append(f"**Videos:** {meta.get('video_count', 0)} · "
             f"**Records:** {len(records)}  ")
    L.append(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}  ")
    L.append(f"**Model:** {meta.get('provider', '')}\n")
    L.append("---\n")

    by_video: dict[str, list[dict]] = {}
    for r in records:
        by_video.setdefault(r["video_id"], []).append(r)

    for vid, items in by_video.items():
        L.append(f"## {items[0]['video_title']}\n")
        L.append(f"<{items[0]['url']}>\n")
        for i, rec in enumerate(items, 1):
            ts = _fmt_ts(rec.get("timestamp"))
            title, title_key = _title_for(rec)
            head = f"### {i}. {title}"
            L.append(head + "\n")
            bits = []
            if rec.get("speaker") and rec["speaker"] != "unclear":
                bits.append(rec["speaker"])
            if ts:
                bits.append(f"at {ts}"
                            + (" (approx)" if rec.get("timestamp_source") == "estimated"
                               else ""))
            if rec.get("confidence"):
                bits.append(f"confidence: {rec['confidence']}")
            if bits:
                L.append("*" + " · ".join(bits) + "*\n")
            if rec.get("quote_original"):
                L.append(f"> {rec['quote_original']}\n")
            for k, v in (rec.get("record") or {}).items():
                if k == title_key:
                    continue
                label = k.replace("_", " ").capitalize()
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                L.append(f"**{label}** — {v}\n")
            L.append("")
        L.append("---\n")

    path.write_text("\n".join(L), encoding="utf-8")
    return path


def write_docx(markdown_path: Path, out_path: Path | None = None) -> Path | None:
    """Convert the Markdown report to Word, if pandoc is available.

    Markdown is a developer's format. Most people who want to read a report,
    mark it up, or send it to someone else want a document. pandoc does this
    offline and is the only extra needed; without it we simply skip, because
    the Markdown and JSON are already written.
    """
    if not shutil.which("pandoc"):
        return None
    out_path = out_path or markdown_path.with_suffix(".docx")
    try:
        subprocess.run(
            ["pandoc", str(markdown_path), "-o", str(out_path),
             "--toc", "--toc-depth=2",
             "--metadata", f"title={markdown_path.stem.replace('_', ' ').title()}",
             "-V", "lang=en"],
            check=True, capture_output=True, text=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out_path if out_path.exists() else None
