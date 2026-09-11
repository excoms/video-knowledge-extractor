"""Per-video checkpointing. A crash must never cost completed work — this is
the single most valuable property of the whole pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Store:
    """Run outputs in one folder; transcripts cached centrally.

    Transcription is the expensive step and depends only on the video, so it
    is cached by video id and reused. Everything else belongs to the run. Each
    transcript is also copied into the run folder, which keeps that folder
    self-contained for a few hundred kilobytes.
    """

    def __init__(self, root: Path, cache: Path | None = None):
        self.root = Path(root)
        self.transcripts = self.root / "transcripts"
        self.transcripts.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache) if cache else self.transcripts
        self.cache.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state: dict[str, dict[str, Any]] = {}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text("utf-8"))
            except json.JSONDecodeError:
                self.state = {}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(self.state_path)          # atomic; a killed write cannot corrupt state

    def status(self, vid: str) -> str:
        return self.state.get(vid, {}).get("status", "")

    def done(self, vid: str) -> bool:
        """Already transcribed — in this run, or by any earlier one.

        The cache is what makes this true across runs; without it a new run
        folder would mean transcribing a two-hour video again.
        """
        if self.status(vid) == "ok":
            return True
        return (self.cache / f"{vid}.txt").exists()

    def settled(self, vid: str) -> bool:
        """Finished for good — do not retry."""
        return self.status(vid) in ("ok", "unavailable")

    def mark(self, vid: str, **fields: Any) -> None:
        self.state.setdefault(vid, {}).update(fields)
        self.save()

    def put_transcript(self, vid: str, header: str, text: str,
                       segments=None) -> Path:
        """Save the transcript, and the segment timings beside it.

        Timings are what let a quote be traced back to the second it was
        spoken. Without them a later analysis run — which always rebuilds the
        transcript from here — produces records with no timestamp at all.
        """
        body = f"{header}\n{'-' * 60}\n\n{text}\n"
        seg_json = (json.dumps([[round(s.start, 2), round(s.end, 2), s.text]
                                for s in segments], ensure_ascii=False)
                    if segments else None)
        for folder in {self.cache, self.transcripts}:
            (folder / f"{vid}.txt").write_text(body, encoding="utf-8")
            if seg_json:
                (folder / f"{vid}.segments.json").write_text(seg_json, encoding="utf-8")
        return self.transcripts / f"{vid}.txt"

    def get_segments(self, vid: str) -> list:
        """Segment timings for a saved transcript, or [] if none were kept."""
        from .transcripts import Segment

        f = self._locate(f"{vid}.segments.json")
        if not f:
            return []
        try:
            raw = json.loads(f.read_text("utf-8"))
        except json.JSONDecodeError:
            return []
        return [Segment(float(a), float(b), t) for a, b, t in raw]

    def _locate(self, name: str) -> Path | None:
        for folder in (self.transcripts, self.cache):
            p = folder / name
            if p.exists():
                return p
        return None

    def adopt_from_cache(self, vid: str) -> bool:
        """Copy a cached transcript into this run and record it in state.

        The state entry matters as much as the copy: callers read
        `state[vid]` for the title, duration and language, so a transcript
        that arrived from the cache without one raises KeyError later.
        """
        src = self.cache / f"{vid}.txt"
        if not src.exists():
            return False

        if self.cache != self.transcripts:
            for suffix in (".txt", ".segments.json"):
                f = self.cache / f"{vid}{suffix}"
                if f.exists():
                    (self.transcripts / f.name).write_text(f.read_text("utf-8"),
                                                           encoding="utf-8")

        if self.state.get(vid, {}).get("status") != "ok":
            head, _, body = src.read_text("utf-8").partition("-" * 60)
            fields = {}
            for line in head.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fields[k.strip().lower()] = v.strip()
            text = body.strip()
            self.mark(vid, status="ok",
                      title=fields.get("title", vid),
                      duration=float(fields.get("duration", "0").rstrip("s") or 0),
                      chars=len(text),
                      source=fields.get("source", "cache"),
                      language=fields.get("language", ""))
        return True

    def get_transcript(self, vid: str) -> str | None:
        p = self._locate(f"{vid}.txt")
        if not p:
            return None
        return p.read_text("utf-8").split("-" * 60, 1)[-1].strip()

    def ordered_todo(self, refs: list) -> list:
        """Never-attempted first, previously-failed after.

        Retry-first ordering starves untried work: two permanently-failing
        premieres consumed every batch until this was fixed.
        """
        never = [r for r in refs if r.video_id not in self.state]
        retry = [r for r in refs if r.video_id in self.state
                 and not self.settled(r.video_id)
                 and self.status(r.video_id) != "deferred"]
        return never + retry
