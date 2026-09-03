"""Between recognition and analysis: strip what isn't speech.

Whisper narrates non-speech audio rather than ignoring it — an Urdu video's
outro music comes back as the literal word 'موسیقی'. Left in, these become
phantom claims for the analyser to reason about.
"""
from __future__ import annotations

import re

# Non-speech narration across the languages we have seen. Anchored so real
# words containing these substrings survive.
ARTEFACTS = re.compile(
    r"(?:^|(?<=\s))(?:\[(?:music|Music|MUSIC|applause|Applause|laughter|"
    r"silence|inaudible)\]|\((?:music|applause|laughter)\)|موسیقی|موسیقى|"
    r"音楽|음악)(?=\s|$)"
)


def strip_artefacts(text: str) -> str:
    return ARTEFACTS.sub(" ", text)


def collapse_repeats(text: str, threshold: int = 4) -> str:
    """Whisper can loop on long audio, emitting one phrase many times over.

    Only collapses immediate repetition beyond the threshold — natural
    rhetorical repetition ('no doubt, no doubt') stays, because it is speech.
    """
    words = text.split()
    if not words:
        return text
    out, run, prev = [], 0, None
    for w in words:
        run = run + 1 if w == prev else 1
        prev = w
        if run <= threshold:
            out.append(w)
    return " ".join(out)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", collapse_repeats(strip_artefacts(text))).strip()
