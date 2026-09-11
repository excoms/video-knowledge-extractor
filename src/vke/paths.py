"""Where things go.

One folder per run, holding everything about that run, plus a shared cache of
transcripts keyed by video id. Strictly, self-contained runs would mean
re-transcribing every time — 43 minutes for a two-hour video — so transcripts
are cached centrally and *copied* into each run. They are a few hundred
kilobytes, so the copy costs nothing and the run folder still stands alone.

    vke-data/
      cache/transcripts/<video id>.txt        reused across runs
                        <video id>.segments.json
      runs/<date>_<time>_<slug>/
          run.json          what was asked, and how it was produced
          transcripts/      copies, so the folder is portable
          analysis.json     canonical
          analysis.md       readable
          analysis.docx     readable, for people who do not read Markdown
          patterns.md|docx  only when more than one video
      archive/              superseded working material
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = "vke-data"


def data_root(base: Path | None = None) -> Path:
    return (base or Path.cwd()) / DATA_DIR


def cache_dir(base: Path | None = None) -> Path:
    p = data_root(base) / "cache" / "transcripts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def slug(text: str, limit: int = 48, fallback: str = "run") -> str:
    """A short, filesystem-safe, human-readable name.

    Keeps only ASCII word characters: a folder named in Urdu script is correct
    but awkward to type, quote in a shell, or read in a narrow Finder column.
    A title in a non-Latin script therefore falls back to the caller's id
    rather than to nothing.
    """
    text = (text or "").strip()

    # A YouTube URL reads better as its channel handle or video id than as
    # the path that contained it.
    m = re.search(r"youtube\.com/@([\w.-]+)", text)
    if m:
        return m.group(1).lower()[:limit]
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{6,})", text)
    if m:
        return m.group(1)[:limit]

    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned, flags=re.ASCII)
    cleaned = re.sub(r"[\s_-]+", "-", cleaned.strip()).strip("-").lower()
    cleaned = cleaned[:limit].rstrip("-")
    return cleaned or slug(fallback, limit) if cleaned or fallback != "run" else "run"


def new_run_dir(label: str, base: Path | None = None,
                when: datetime | None = None, fallback: str = "run") -> Path:
    """A dated folder for one run. Never reuses an existing one."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M")
    root = data_root(base) / "runs"
    name = slug(label, fallback=fallback)
    candidate = root / f"{stamp}_{name}"
    n = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{name}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def write_run_manifest(run_dir: Path, **fields) -> Path:
    """Record what was asked and how it was produced.

    A folder full of output is not self-explanatory six months later; the
    request, the settings and the versions are what make it readable.
    """
    from . import __version__

    manifest = {"vke_version": __version__,
                "created": datetime.now().isoformat(timespec="seconds"),
                **fields}
    p = run_dir / "run.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
