"""Getting text for a video: captions when they exist, speech recognition when
they don't.

Captions are free and instant but cover only the dozen or so languages YouTube
auto-captions. Urdu is not among them — an Urdu channel returns zero caption
tracks of any kind, which is why ASR is a first-class path here and not a
fallback bolted on.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    video_id: str
    title: str
    url: str
    text: str
    language: str
    source: str                      # "captions:ur" | "asr:large-v3-turbo"
    duration: float = 0.0
    segments: list[Segment] = field(default_factory=list)

    @property
    def char_rate(self) -> float:
        """Characters per second of speech. Stable across a healthy corpus
        (7.7-8.6 for Urdu prose); a sharp deviation means the decode degraded."""
        return len(self.text) / self.duration if self.duration else 0.0

    def header(self) -> str:
        return (f"TITLE: {self.title}\nURL: {self.url}\n"
                f"DURATION: {self.duration:.0f}s\nLANGUAGE: {self.language}\n"
                f"SOURCE: {self.source}")


from .captions import fetch_captions          # noqa: E402
from .asr import transcribe, AsrUnavailable   # noqa: E402

__all__ = ["Segment", "Transcript", "fetch_captions", "transcribe",
           "AsrUnavailable"]
