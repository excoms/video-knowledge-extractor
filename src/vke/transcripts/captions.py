"""The free path: use YouTube's own captions when they exist."""
from __future__ import annotations

import re
from pathlib import Path

from ..normalise import clean_text


def _parse_vtt(raw: str) -> tuple[str, list]:
    """YouTube auto-captions repeat lines as they scroll; collapse them."""
    from . import Segment

    segments, seen_lines = [], []
    lines = raw.splitlines()
    i = 0
    ts = re.compile(r"(\d{2}):(\d{2}):(\d{2}\.\d+)\s+-->\s+(\d{2}):(\d{2}):(\d{2}\.\d+)")
    while i < len(lines):
        m = ts.match(lines[i].strip())
        if not m:
            i += 1
            continue
        h1, m1, s1, h2, m2, s2 = m.groups()
        start = int(h1) * 3600 + int(m1) * 60 + float(s1)
        end = int(h2) * 3600 + int(m2) * 60 + float(s2)
        i += 1
        parts = []
        while i < len(lines) and lines[i].strip():
            t = re.sub(r"<[^>]+>", "", lines[i]).strip()
            if t and (not parts or parts[-1] != t):
                parts.append(t)
            i += 1
        if parts:
            line = " ".join(parts)
            if not seen_lines or seen_lines[-1] != line:
                seen_lines.append(line)
                segments.append(Segment(start, end, line))
    return clean_text(" ".join(seen_lines)), segments


def fetch_captions(ref, langs: list[str], tmp_dir: Path, limits=None):
    """Return a Transcript, or None when the video has no usable captions.

    Only requests the languages asked for. The inherited script pulled every
    available track then discarded all but one, which is slow and rude.
    """
    from . import Transcript
    import yt_dlp

    tmp_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "skip_download": True, "writeautomaticsub": True, "writesubtitles": True,
        "subtitleslangs": list(langs), "subtitlesformat": "vtt",
        "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "ignoreerrors": True,
    }
    if limits is not None:
        opts.update(limits.ytdlp_opts())

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(ref.url, download=True)
    except Exception:
        return None
    if not info:
        return None

    files = sorted(tmp_dir.glob(f"{ref.video_id}*.vtt"))
    if not files:
        return None

    chosen, lang = files[0], "auto"
    for want in langs:
        for f in files:
            if f".{want}." in f.name:
                chosen, lang = f, want
                break
        else:
            continue
        break

    text, segments = _parse_vtt(chosen.read_text("utf-8", errors="replace"))
    for f in files:
        f.unlink(missing_ok=True)
    if not text.strip():
        return None

    return Transcript(
        video_id=ref.video_id, title=info.get("title") or ref.title, url=ref.url,
        text=text, language=lang, source=f"captions:{lang}",
        duration=float(info.get("duration") or 0), segments=segments,
    )
