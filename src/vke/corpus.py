"""The synthesis pass.

Per-record output is homework: a stack of findings you still have to read and
compare. This pass reads every record together and says what recurs, who
answered what, where the reasoning broke, and how each speaker performed.

It runs on a single long video as readily as on a channel. Eleven chunks of one
debate have exactly the same repetition and attribution problem as eleven
separate videos.
"""
from __future__ import annotations

import json
from pathlib import Path

from .providers.base import extract_json

# Below this there is nothing to synthesise — the records speak for themselves.
MIN_RECORDS = 4


class SynthesisFailed(RuntimeError):
    """The grouping, scoring and profiling pass could not be produced.

    Raised rather than swallowed. This pass is the report — the per-record
    file is working-out. A run that loses it and still says "done" looks
    identical to a run that had nothing to say, and the records that cost
    an hour of transcription sit there with no explanation attached.
    """

SYNTHESIS_SYSTEM = """You are given every record extracted from one or more
videos, and the roster of who was speaking.

Produce a debrief. Return JSON only, with exactly these keys:

{"summary": "<what this material is, and the single most important thing about
             how it was argued>",

 "themes": [{"theme": "<the recurring subject, named once>",
             "raised_by": "<speaker>",
             "times_made": <integer>,
             "core_claim": "<the argument in one sentence>",
             "defensible": "<what is genuinely strong here — be specific, name
                             the real support it has, and do not inflate>",
             "weakest_flank": "<where a well-informed opponent attacks first —
                                the specific weak joint, not a disclaimer>",
             "better_framing": "<the same argument put at its strongest>",
             "counter_response": "<the best reply the OTHER side can make to that
                                   improved version — argued as well as the framing
                                   itself, not a token objection. End by naming the
                                   ground the responder should move the exchange to,
                                   where their position is genuinely stronger.>",
             "how_it_fared": "unanswered" | "answered" | "conceded" | "contested",
             "note": "<what happened to it across the discussion>"}],

 "fallacies": [{"speaker": "...", "fallacy": "<standard name>",
                "what_happened": "<the specific move, not a definition>",
                "quote": "<short, verbatim, original language>",
                "severity": "high" | "medium" | "low"}],

 "goalpost_shifts": [{"speaker": "...",
                      "original_standard": "<what they first said would settle it>",
                      "replacement": "<what they demanded once it was met>",
                      "why_it_matters": "..."}],

 "unanswered": [{"question": "...", "put_by": "...", "put_to": "...",
                 "note": "<why it mattered>"}],

 "concessions": [{"speaker": "...", "conceded": "...", "to_whom": "..."}],

 "scorecard": [{"speaker": "...",
                "argument_quality": <0-10>, "evidence": <0-10>,
                "responsiveness": <0-10>, "clarity": <0-10>,
                "intellectual_honesty": <0-10>,
                "total": <0-50>,
                "one_line_verdict": "..."}],

 "profiles": [{"speaker": "...",
               "style": "<how they argue, in two or three sentences>",
               "strengths": ["..."], "weaknesses": ["..."],
               "characteristic_moves": ["<the tactics they return to>"],
               "advice": "<the single change that would most improve them>"}],

 "overall": "<who argued better, on what grounds — or say plainly that it was
             even, and why>"}

Rules on scoring, which is the part most often done badly:
- Judge the ARGUING, not the conclusion. A view you find mistaken, defended
  well, outscores one you share, defended badly.
- counter_response must be a real reply, of the same quality as the framing it
  answers. Steelman both sides: if the improved argument is genuinely hard to
  answer, say where its weight actually lies and what the responder's best
  available ground is instead. A weak counter is a failure of the analysis, not
  a finding about the argument.
- argument_quality: validity, and whether premises support conclusions.
- evidence: specificity and accuracy of what is cited. Vague gestures at
  "science says" score low however confident the delivery.
- responsiveness: did they engage with what was actually said, or with a
  version of it that was easier to answer?
- clarity: could a fair listener state their position afterwards?
- intellectual_honesty: conceding good points, representing opponents fairly,
  not overclaiming. Goalpost-shifting and straw men cost heavily here.
- Anchor the scale: 5 is competent. 8+ is genuinely strong and needs
  justifying. Below 3 means a serious failure you should be able to name.

Other rules:
- GROUP repetition. A point made six times is one theme with times_made 6, not
  six entries. This is the main reason this pass exists.
- Grouping must not lose detail. Each grouped argument still gets the full
  treatment — defensible, weakest flank, better framing. Collapsing six
  mentions into one line summary throws away the analysis that made the
  extraction worth doing.
- Use only the names on the roster.
- Return [] honestly for any section with nothing in it. Do not manufacture a
  fallacy to fill space — a clean argument is a finding.
- Quote in the original language; write everything else in the language asked for."""


def build_register(provider, records: list[dict], request: str,
                   output_language: str = "English",
                   speakers: list[dict] | None = None) -> dict:
    if len(records) < MIN_RECORDS:
        return {}

    condensed = [{
        "video": r["video_title"][:70],
        "speaker": r.get("speaker"),
        "confidence": r.get("confidence"),
        "quote": (r.get("quote_original") or "")[:160],
        **{k: (v[:300] if isinstance(v, str) else v)
           for k, v in (r.get("record") or {}).items()},
    } for r in records]

    from .analysis import roster_block
    payload = json.dumps(condensed, ensure_ascii=False)[:140000]
    user = (f"The user asked for:\n{request}\n\n"
            f"Write everything except quotes in {output_language}.\n\n"
            + (roster_block(speakers) + "\n" if speakers else "")
            + f"RECORDS ({len(records)} from "
              f"{len({r['video_id'] for r in records})} video(s)):\n{payload}\n\n"
              f"Return the JSON object.")
    try:
        raw = provider.complete(SYNTHESIS_SYSTEM, user, max_tokens=12000)
    except Exception as e:
        raise SynthesisFailed(
            f"{provider.name} could not complete the grouping and scoring pass "
            f"over {len(records)} records ({len(user):,} characters of prompt — "
            f"this pass sends every record in one call, so it is far larger than "
            f"the per-chunk calls that already succeeded). {e}") from e
    try:
        out = extract_json(raw)
    except Exception as e:
        raise SynthesisFailed(
            f"{provider.name} returned {len(raw):,} characters that were not "
            f"usable JSON. A truncated reply usually means the report outgrew "
            f"the reply limit. {e}") from e
    if not isinstance(out, dict) or not out:
        raise SynthesisFailed(
            f"{provider.name} returned {type(out).__name__} rather than the "
            f"report object, from {len(raw):,} characters of reply.")
    return out


def _bar(score, out_of=10) -> str:
    """A score you can scan down a column without re-reading the number."""
    try:
        n = max(0, min(int(round(float(score))), out_of))
    except (TypeError, ValueError):
        return "—"
    return f"{n}/{out_of}"


def write_register(reg: dict, path: Path, meta: dict) -> Path | None:
    if not reg:
        return None
    L: list[str] = [f"# Debrief — {meta.get('source_name', 'analysis')}\n"]
    L.append(f"**Source:** {meta.get('source', '')}  ")
    L.append(f"**Videos:** {meta.get('video_count', 0)} · "
             f"**Records analysed:** {meta.get('record_count', 0)}\n")
    if reg.get("summary"):
        L.append(reg["summary"] + "\n")

    if reg.get("scorecard"):
        L.append("## Scorecard\n")
        L.append("Judged on how the case was argued, not on whether the "
                 "conclusion is correct. 5 is competent; 8 or above needs "
                 "justifying.\n")
        L.append("| Speaker | Argument | Evidence | Responsive | Clarity | Honesty | Total |")
        L.append("|---|---|---|---|---|---|---|")
        for s in sorted(reg["scorecard"], key=lambda x: -(x.get("total") or 0)):
            L.append(f"| **{s.get('speaker','?')}** | {_bar(s.get('argument_quality'))} "
                     f"| {_bar(s.get('evidence'))} | {_bar(s.get('responsiveness'))} "
                     f"| {_bar(s.get('clarity'))} | {_bar(s.get('intellectual_honesty'))} "
                     f"| **{_bar(s.get('total'), 50)}** |")
        L.append("")
        for s in reg["scorecard"]:
            if s.get("one_line_verdict"):
                L.append(f"- **{s.get('speaker','?')}** — {s['one_line_verdict']}")
        L.append("")

    if reg.get("overall"):
        L.append("## Verdict\n")
        L.append(reg["overall"] + "\n")

    if reg.get("goalpost_shifts"):
        L.append("## Moving the goalposts\n")
        for g in reg["goalpost_shifts"]:
            L.append(f"### {g.get('speaker','?')}")
            L.append(f"- **First standard:** {g.get('original_standard','')}")
            L.append(f"- **Replaced with:** {g.get('replacement','')}")
            if g.get("why_it_matters"):
                L.append(f"- **Why it matters:** {g['why_it_matters']}")
            L.append("")

    if reg.get("fallacies"):
        L.append("## Fallacies\n")
        order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(reg["fallacies"], key=lambda x: order.get(x.get("severity"), 3)):
            L.append(f"### {f.get('fallacy','?')} — {f.get('speaker','?')}")
            L.append(f"*severity: {f.get('severity','?')}*\n")
            L.append(f"{f.get('what_happened','')}\n")
            if f.get("quote"):
                L.append(f"> {f['quote']}\n")

    if reg.get("unanswered"):
        L.append("## Questions never answered\n")
        L.append("In a scored debate an unanswered question costs more than a "
                 "poor answer.\n")
        for u in reg["unanswered"]:
            L.append(f"- **{u.get('put_by','?')} → {u.get('put_to','?')}:** "
                     f"{u.get('question','')}")
            if u.get("note"):
                L.append(f"  - {u['note']}")
        L.append("")

    if reg.get("concessions"):
        L.append("## Concessions\n")
        for c in reg["concessions"]:
            L.append(f"- **{c.get('speaker','?')}** conceded {c.get('conceded','')}"
                     + (f" to {c['to_whom']}" if c.get("to_whom") else ""))
        L.append("")

    if reg.get("themes"):
        L.append("## Arguments, grouped\n")
        L.append("Points made repeatedly are collapsed into one entry, with a "
                 "count.\n")
        for n, t in enumerate(sorted(reg["themes"],
                                     key=lambda x: -(x.get("times_made") or 0)), 1):
            times = t.get("times_made") or 1
            L.append(f"### {n}. {t.get('theme','?')}")
            L.append(f"*{t.get('raised_by','?')} · pressed {times}×"
                     f" · {t.get('how_it_fared','?')}*\n")
            if t.get("core_claim"):
                L.append(f"**The claim** — {t['core_claim']}\n")
            for label, key in (("What is genuinely defensible", "defensible"),
                               ("Weakest flank", "weakest_flank"),
                               ("How it should have been put", "better_framing"),
                               ("The reply that comes back", "counter_response"),
                               ("How it fared", "note")):
                if t.get(key):
                    L.append(f"**{label}** — {t[key]}\n")

    if reg.get("profiles"):
        L.append("## Speaker profiles\n")
        for pr in reg["profiles"]:
            L.append(f"### {pr.get('speaker','?')}\n")
            if pr.get("style"):
                L.append(pr["style"] + "\n")
            for label, key in (("Strengths", "strengths"),
                               ("Weaknesses", "weaknesses"),
                               ("Characteristic moves", "characteristic_moves")):
                items = pr.get(key) or []
                if items:
                    L.append(f"**{label}**\n")
                    for i in items:
                        L.append(f"- {i}")
                    L.append("")
            if pr.get("advice"):
                L.append(f"**The one change that would help most** — {pr['advice']}\n")

    path.write_text("\n".join(L), encoding="utf-8")
    return path
