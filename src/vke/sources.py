"""Resolve a URL — channel, playlist or single video — into video references."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRef:
    video_id: str
    title: str
    url: str


def _ydl(opts: dict):
    try:
        import yt_dlp
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit("yt-dlp is not installed. Run: pip install yt-dlp") from e
    return yt_dlp.YoutubeDL(opts)


def resolve(url: str, limits=None) -> list[VideoRef]:
    """Flat-extract so listing a 500-video channel stays one cheap request."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True,
            "ignoreerrors": True}
    if limits is not None:
        opts.update(limits.ytdlp_opts())

    with _ydl(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        return []

    entries = info.get("entries")
    if entries is None:                                       # a single video
        entries = [info]

    refs, seen = [], set()
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        refs.append(VideoRef(
            video_id=vid,
            title=e.get("title") or "Untitled",
            url=f"https://www.youtube.com/watch?v={vid}",
        ))
    return refs


def resolve_many(urls: list[str], limits=None) -> list[VideoRef]:
    out, seen = [], set()
    for u in urls:
        for r in resolve(u, limits):
            if r.video_id not in seen:
                seen.add(r.video_id)
                out.append(r)
    return out
