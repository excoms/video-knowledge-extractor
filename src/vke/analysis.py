"""Turn a transcript plus a plain-English request into structured records.

The user says what they want in their own words. Rather than forcing that into
a fixed shape, the model first proposes a schema for the request, then extracts
records to it. Free-text in, structured out — so a timeline gets timeline
fields and an argument audit gets argument fields, with no vocabulary to learn.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from .providers.base import extract_json

SCHEMA_VERSION = "1.0"
CHUNK_CHARS = 12000
CHUNK_OVERLAP = 800

PLANNER_SYSTEM = """You design compact JSON schemas for information extraction.

Given what a user wants from a video, decide the fields each extracted record
should carry. Return JSON only:

{"record_name": "<singular noun for one record, e.g. argument, event, claim>",
 "fields": [{"name": "<snake_case>", "description": "<what goes here>"}]}

Rules:
- 3 to 8 fields. Only what the request actually needs.
- Field names are snake_case, no spaces.
- Do NOT include: video id, url, speaker, quote, timestamp, or confidence.
  Those are attached automatically to every record.
- Prefer specific fields over one catch-all "notes" field."""

EXTRACTOR_SYSTEM = """You extract structured records from a video transcript.

Return JSON only: a list of objects. Each object must contain exactly these keys:

  "fields"    — an object with the schema fields you were given
  "quote"     — a SHORT verbatim extract from the transcript, in its ORIGINAL
                language, that this record is drawn from. Copy it exactly;
                do not translate, paraphrase or correct it.
  "speaker"   — who is speaking, or "unclear"
  "confidence"— "high", "medium" or "low"

Rules:
- Be honest. If something is weak, say so; do not inflate it.
- Flag factual and terminological errors explicitly rather than smoothing them.
- The transcript may come from imperfect speech recognition. Occasional wrong
  words are expected — read through them, and do not treat a mis-transcription
  as a claim the speaker made.
- Skip greetings, announcements and sign-offs. They are not content.
- Return [] if the transcript genuinely contains nothing matching the request."""


@dataclass
class Plan:
    record_name: str = "record"
    fields: list[dict] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f'  "{f["name"]}": {f.get("description", "")}' for f in self.fields]
        return "{\n" + ",\n".join(lines) + "\n}"


def _coerce_fields(data) -> list[dict]:
    """Accept the shapes models actually return, not just the one we asked for.

    Smaller models wrap the answer, rename the key, or return a bare list of
    field names. All of those are usable; refusing them wastes a call.
    """
    if isinstance(data, list):                       # bare list of fields
        candidates = data
    elif isinstance(data, dict):
        candidates = None
        for key in ("fields", "record_fields", "schema", "properties", "columns"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        if candidates is None:
            # a single nested object, e.g. {"schema": {"fields": [...]}}
            for value in data.values():
                if isinstance(value, dict) and isinstance(value.get("fields"), list):
                    candidates = value["fields"]
                    break
        if candidates is None:
            return []
    else:
        return []

    fields = []
    for item in candidates:
        if isinstance(item, dict):
            name = item.get("name") or item.get("field") or item.get("key")
            if name:
                fields.append({"name": str(name),
                               "description": item.get("description", "")})
        elif isinstance(item, str) and item.strip():
            fields.append({"name": item.strip(), "description": ""})
    return fields


def plan_schema(provider, request: str, output_shape: str = "") -> Plan:
    user = f"The user wants this from the video:\n\n{request}\n"
    if output_shape:
        user += f"\nThey want the final output to look like:\n{output_shape}\n"
    user += "\nDesign the record schema. Return only the JSON object, nothing else."

    raw = provider.complete(PLANNER_SYSTEM, user, max_tokens=900)
    try:
        data = extract_json(raw)
    except Exception as e:
        raise ValueError(
            f"Could not read the schema the model proposed ({e}).\n\n"
            f"  It returned:\n    {raw.strip()[:600] or '(nothing)'}"
        ) from e

    fields = _coerce_fields(data)
    if not fields:
        raise ValueError(
            "The model did not propose any fields for this request.\n\n"
            f"  It returned:\n    {str(data)[:600]}\n\n"
            "  Try rewording what you want, or use a stronger model."
        )

    name = "record"
    if isinstance(data, dict):
        name = data.get("record_name") or data.get("name") or "record"
    return Plan(record_name=str(name), fields=fields)


def chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long transcripts with overlap so an item spanning a join survives.

    Replaces silent truncation: the inherited script cut at 14,000 characters
    and never said so, losing two thirds of a long debate without a word.
    """
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                       # prefer a sentence boundary
            window = text.rfind(". ", start + size // 2, end)
            if window != -1:
                end = window + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def locate_quote(segments, quote: str) -> list[float] | None:
    """Find when a quote was spoken, by matching it against the audio segments.

    Derived from the timings we already have rather than asked of the model —
    a model will happily invent a plausible timestamp, and a wrong clip point
    is worse than none.
    """
    if not segments or not quote:
        return None
    words = quote.split()
    if len(words) < 3:
        return None
    probe = " ".join(words[:8])
    best, best_score = None, 0.0
    for i, seg in enumerate(segments):
        window = " ".join(s.text for s in segments[i:i + 12])[:400]
        score = difflib.SequenceMatcher(None, probe, window[:len(probe) * 2]).ratio()
        if score > best_score:
            best_score, best = score, (seg.start, segments[min(i + 11, len(segments) - 1)].end)
    return [round(best[0], 1), round(best[1], 1)] if best_score >= 0.55 else None


def _dedupe(records: list[dict]) -> list[dict]:
    """Overlapping chunks produce the same item twice. Keep the first."""
    seen: list[str] = []
    out: list[dict] = []
    for r in records:
        sig = json.dumps(r.get("record", {}), sort_keys=True, ensure_ascii=False)[:220]
        if any(difflib.SequenceMatcher(None, sig, s).ratio() > 0.82 for s in seen):
            continue
        seen.append(sig)
        out.append(r)
    return out


def analyse(provider, transcript, plan: Plan, request: str,
            output_language: str = "English", profile: str = "",
            instructions: str = "") -> list[dict]:
    """Extract records from one transcript. Returns envelope-wrapped records."""
    chunks = chunk(transcript.text)
    records: list[dict] = []

    for n, part in enumerate(chunks, 1):
        system = EXTRACTOR_SYSTEM
        if profile:
            system = f"{profile.strip()}\n\n{system}"

        user = (
            f"WHAT THE USER WANTS:\n{request}\n\n"
            f"SCHEMA — the 'fields' object of every record must have exactly "
            f"these keys:\n{plan.describe()}\n\n"
            f"Write the field values in {output_language}. "
            f"Keep 'quote' in the transcript's original language.\n"
        )
        if instructions:
            user += f"\nADDITIONAL INSTRUCTIONS FROM THE USER:\n{instructions}\n"
        if len(chunks) > 1:
            user += f"\n(Part {n} of {len(chunks)} of a long transcript.)\n"
        user += (
            f"\nVIDEO: {transcript.title}\n"
            f"LANGUAGE: {transcript.language}\n\n"
            f"TRANSCRIPT:\n{part}\n\nReturn the JSON list."
        )

        raw = provider.complete(system, user, max_tokens=8000)
        try:
            items = extract_json(raw)
        except Exception:
            continue
        if isinstance(items, dict):
            items = items.get("records") or items.get(plan.record_name + "s") or [items]

        for item in items:
            if not isinstance(item, dict):
                continue
            quote = (item.get("quote") or "").strip()
            records.append({
                "schema_version": SCHEMA_VERSION,
                "record_name": plan.record_name,
                "video_id": transcript.video_id,
                "video_title": transcript.title,
                "url": transcript.url,
                "source": transcript.source,
                "language": transcript.language,
                "speaker": item.get("speaker") or "unclear",
                "confidence": item.get("confidence") or "medium",
                "quote_original": quote,
                "timestamp": locate_quote(transcript.segments, quote),
                "record": item.get("fields") or {},
            })

    return _dedupe(records)
