"""Per-video checkpointing. A crash must never cost completed work — this is
the single most valuable property of the whole pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.transcripts = self.root / "transcripts"
        self.transcripts.mkdir(parents=True, exist_ok=True)
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
        return self.status(vid) == "ok"

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
        p = self.transcripts / f"{vid}.txt"
        p.write_text(f"{header}\n{'-' * 60}\n\n{text}\n", encoding="utf-8")
        if segments:
            (self.transcripts / f"{vid}.segments.json").write_text(
                json.dumps([[round(s.start, 2), round(s.end, 2), s.text]
                            for s in segments], ensure_ascii=False),
                encoding="utf-8")
        return p

    def get_segments(self, vid: str) -> list:
        """Segment timings for a saved transcript, or [] if none were kept."""
        from .transcripts import Segment

        f = self.transcripts / f"{vid}.segments.json"
        if not f.exists():
            return []
        try:
            raw = json.loads(f.read_text("utf-8"))
        except json.JSONDecodeError:
            return []
        return [Segment(float(a), float(b), t) for a, b, t in raw]

    def get_transcript(self, vid: str) -> str | None:
        p = self.transcripts / f"{vid}.txt"
        if not p.exists():
            return None
        body = p.read_text("utf-8")
        return body.split("-" * 60, 1)[-1].strip()

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
