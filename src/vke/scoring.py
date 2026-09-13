"""Turn extracted records into a running score for each speaker.

The scorecard in a report grades the whole performance in one number. That
hides the shape of a debate: who was ahead at the half hour, which ten
minutes cost someone the round, whether a collapse was gradual or a single
bad move. This derives that shape from the records already extracted.

Every rule below is deliberately visible, because a score whose derivation
cannot be inspected is not evidence, it is decoration. The report prints
these weights next to the chart so a reader can disagree with them.
"""
from __future__ import annotations

import re

# What each observable in a record is worth to the speaker it belongs to.
POINTS = {
    "unanswered": (+2, "point went unanswered"),
    "dodged": (+2, "opponent dodged it"),
    "conceded": (+2, "opponent conceded"),
    "answered": (+1, "engaged and answered"),
    "contested": (0, "contested, neither side took it"),
}
FALLACY_COST = {"high": -3, "major": -3, "moderate": -2, "medium": -2,
                "minor": -1, "low": -1}
DEFAULT_FALLACY_COST = -2
SUBSTANCE_BONUS = (+1, "made a substantive argument")
GOALPOST_COST = (-1, "shifted a definition or a standard")


def _first_word(value) -> str:
    return re.split(r"[ ,;:—-]", str(value or "").strip().lower())[0]


def _severity(fallacy) -> int:
    """Fallacies arrive either as {'name','severity'} or as free text."""
    if isinstance(fallacy, dict):
        return FALLACY_COST.get(str(fallacy.get("severity", "")).lower(),
                                DEFAULT_FALLACY_COST)
    text = str(fallacy).lower()
    for word, cost in FALLACY_COST.items():
        if word in text:
            return cost
    return DEFAULT_FALLACY_COST


def _fallacy_name(fallacy) -> str:
    if isinstance(fallacy, dict):
        return str(fallacy.get("name") or "fallacy")
    return re.split(r"[(;.]", str(fallacy))[0].strip()[:60] or "fallacy"


def _has_substance(rec: dict) -> bool:
    for key in ("defensible_ground", "defensible_element", "defensible"):
        value = str(rec.get(key) or "").strip()
        if len(value) > 40 and value.lower() not in ("none", "none.", "n/a"):
            return True
    return False


def events(records: list[dict]) -> list[dict]:
    """One scoring event per observation, in the order they were said.

    A record can produce several: an argument that went unanswered *and*
    contained a fallacy earns and loses in the same breath, which is exactly
    the kind of moment worth seeing on a chart.
    """
    out: list[dict] = []
    for r in records:
        rec = r.get("record") or {}
        speaker = r.get("speaker") or "?"
        stamp = r.get("timestamp") or [0, 0]
        at = float(stamp[0] if isinstance(stamp, (list, tuple)) else stamp or 0)

        outcome = _first_word(rec.get("outcome"))
        if outcome in POINTS and POINTS[outcome][0]:
            pts, why = POINTS[outcome]
            out.append({"at": at, "speaker": speaker, "points": pts,
                        "reason": why, "kind": "outcome"})

        if _has_substance(rec):
            pts, why = SUBSTANCE_BONUS
            out.append({"at": at, "speaker": speaker, "points": pts,
                        "reason": why, "kind": "substance"})

        fallacy = rec.get("fallacy") or rec.get("fallacies_and_errors")
        if fallacy and str(fallacy).strip().lower() not in ("none", "null", ""):
            out.append({"at": at, "speaker": speaker, "points": _severity(fallacy),
                        "reason": _fallacy_name(fallacy), "kind": "fallacy"})

        if rec.get("definitional_shift") and \
                str(rec["definitional_shift"]).strip().lower() not in ("none", "null"):
            pts, why = GOALPOST_COST
            out.append({"at": at, "speaker": speaker, "points": pts,
                        "reason": why, "kind": "goalpost"})

    out.sort(key=lambda e: e["at"])
    return out


def running(evts: list[dict], speakers: list[str]) -> dict[str, list[tuple]]:
    """Cumulative total per speaker as (seconds, total) pairs, starting at 0."""
    series = {s: [(0.0, 0)] for s in speakers}
    totals = {s: 0 for s in speakers}
    for e in evts:
        if e["speaker"] not in totals:
            continue
        totals[e["speaker"]] += e["points"]
        series[e["speaker"]].append((e["at"], totals[e["speaker"]]))
    return series


def rules_table() -> str:
    """The weights, as markdown, for printing beside the chart."""
    rows = [
        "| Observation | Points |",
        "|---|---|",
        "| Argument went unanswered | +2 |",
        "| Opponent dodged it | +2 |",
        "| Opponent conceded | +2 |",
        "| Engaged and answered | +1 |",
        "| Made a substantive argument | +1 |",
        "| Contested, neither side took it | 0 |",
        "| Shifted a definition or standard | −1 |",
        "| Fallacy — minor | −1 |",
        "| Fallacy — moderate | −2 |",
        "| Fallacy — major | −3 |",
    ]
    return "\n".join(rows)


def summary(evts: list[dict], speakers: list[str]) -> list[dict]:
    """Per speaker: what they earned, what it cost them, and on what.

    The chart shows the shape; this says whose each mark was. Without it a
    reader can see that a fallacy happened at 0:34 and has no way to tell
    which of the two people in the room committed it.
    """
    rows = []
    for s in speakers:
        mine = [e for e in evts if e["speaker"] == s]
        won = sum(e["points"] for e in mine if e["points"] > 0)
        lost = sum(e["points"] for e in mine if e["points"] < 0)
        rows.append({
            "speaker": s,
            "won": won,
            "lost": lost,
            "net": won + lost,
            "fallacies": sum(1 for e in mine if e["kind"] == "fallacy"),
            "goalposts": sum(1 for e in mine if e["kind"] == "goalpost"),
            "unanswered": sum(1 for e in mine
                              if e["kind"] == "outcome" and e["points"] > 0),
        })
    return sorted(rows, key=lambda r: -r["net"])
