from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from . import __version__
from .limits import Aborted, Limits
from .store import Store

PRESETS = {
    "argument-audit": (
        "Identify every distinct argument or claim. For each one, tell me what is "
        "genuinely defensible about it, where a well-informed opponent would attack "
        "first, and how it should be reframed to survive scrutiny. Flag any factual "
        "or terminological errors explicitly.",
        "One section per argument, ordered by how badly it needs fixing.",
    ),
    "timeline": (
        "Build a chronological timeline of every event described. Capture who was "
        "involved, where it happened, and the date as the speaker gives it. Flag any "
        "date stated inconsistently.",
        "A table, earliest first.",
    ),
    "contradictions": (
        "Find contradictions, both within a single video and across all of them. "
        "Show both statements and say which is better supported.",
        "A list, most significant first.",
    ),
    "key-points": (
        "Extract the key points actually made, with enough detail that someone who "
        "has not watched could use them.",
        "A bulleted brief.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vke",
        description="Turn any YouTube channel into structured, analysed knowledge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  vke run https://youtube.com/@SomeChannel --preset argument-audit
  vke run <url> --ask "Build a timeline of the events described"
  vke run <url> --ask "..." --output-language English --provider ollama
  vke run <url> --transcribe-only          # just the transcripts
  vke ui                                   # open the web interface
  vke providers                            # what can this machine use?
""")
    p.add_argument("--version", action="version", version=f"vke {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="transcribe and analyse")
    r.add_argument("urls", nargs="+", help="channel, playlist or video URLs")
    r.add_argument("--ask", default="", help="what you want from it, in plain English")
    r.add_argument("--shape", default="", help="how the output should look")
    r.add_argument("--preset", choices=sorted(PRESETS), help="a starting point for --ask")
    r.add_argument("--instructions", default="", help="extra steer for this run")
    r.add_argument("--profile", default="general",
                   help="expertise to bring; see `vke profiles`")
    r.add_argument("--provider", default="auto", help="claude-cli | anthropic | openai | ollama")
    r.add_argument("--model", default=None)
    r.add_argument("--asr", default="auto",
                   choices=["auto", "faster-whisper", "mlx"],
                   help="speech recognition backend (default: whichever is installed)")
    r.add_argument("--asr-model", default=None)
    r.add_argument("--lang", default=None, help="spoken language hint, e.g. ur")
    r.add_argument("--captions", default="", help="try these caption languages first, e.g. en,ur")
    r.add_argument("--output-language", default="English")
    r.add_argument("--limit", type=int, default=0, help="only the first N videos")
    r.add_argument("--outdir", default=None)
    r.add_argument("--pause", type=float, default=2.0)
    r.add_argument("--transcribe-only", action="store_true")
    r.add_argument("--no-corpus", action="store_true", help="skip the cross-video pass")
    r.add_argument("--yes", action="store_true", help="don't ask before large runs")

    u = sub.add_parser("ui", help="open the local web interface")
    u.add_argument("--port", type=int, default=7864)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--no-browser", action="store_true")

    sub.add_parser("profiles", help="list available expertise profiles")
    sub.add_parser("providers", help="show which analysis backends are usable here")
    return p


def _load_profile(name: str) -> str:
    """Bundled profiles ship inside the package; a ./profiles dir overrides them
    so users can add or edit expertise without touching the install."""
    for base in (Path.cwd() / "profiles",
                 Path(__file__).resolve().parent / "profiles"):
        f = base / f"{name}.txt"
        if f.exists():
            return f.read_text("utf-8")
    return ""


def list_profiles() -> list[str]:
    names = set()
    for base in (Path(__file__).resolve().parent / "profiles",
                 Path.cwd() / "profiles"):
        if base.is_dir():
            names.update(f.stem for f in base.glob("*.txt"))
    return sorted(names)


def cmd_ui(a) -> int:
    from .ui import serve
    serve(host=a.host, port=a.port, open_browser=not a.no_browser)
    return 0


def cmd_providers() -> int:
    from .providers import REGISTRY, autodetect, get_provider

    print("\n  Analysis backends\n")
    for name in REGISTRY:
        try:
            get_provider(name)
            print(f"    ready       {name}")
        except Exception as e:
            print(f"    unavailable {name}  — {str(e).splitlines()[0][:70]}")
    print(f"\n  Would use: {autodetect()}\n")

    from .transcripts.asr import available_backends, pick_backend
    print("  Speech recognition\n")
    for name, status in available_backends().items():
        print(f"    {'ready      ' if status == 'ready' else 'unavailable'} {name}"
              f"{'' if status == 'ready' else '  — ' + status}")
    try:
        print(f"\n  Would use: {pick_backend('auto')}\n")
    except Exception as e:
        print(f"\n  None usable: {str(e).splitlines()[0]}\n")
    return 0


def cmd_run(a) -> int:
    from . import corpus as corpus_mod
    from . import report
    from .analysis import analyse, plan_schema
    from .providers import autodetect, get_provider
    from .sources import resolve_many
    from .transcripts import fetch_captions, transcribe
    from .transcripts.asr import AsrUnavailable, pick_backend

    ask, shape = a.ask, a.shape
    if a.preset and not ask:
        ask, preset_shape = PRESETS[a.preset]
        shape = shape or preset_shape
    if not ask and not a.transcribe_only:
        print("Nothing to do: pass --ask \"what you want\" or --preset NAME.",
              file=sys.stderr)
        return 2

    limits = Limits(pause=a.pause, assume_yes=a.yes)
    outdir = Path(a.outdir) if a.outdir else Path("vke-out")
    store = Store(outdir)
    tmp = outdir / ".tmp"

    # Sweep orphaned audio from any previously killed run before we add more.
    for stale in tmp.glob("*.wav"):
        with contextlib.suppress(Exception):
            stale.unlink()

    print(f"\n  Listing {', '.join(a.urls)} ...", flush=True)
    refs = resolve_many(a.urls, limits)
    if not refs:
        print("  No videos found. Check the URL.", file=sys.stderr)
        return 1
    if a.limit:
        refs = refs[:a.limit]
    print(f"  {len(refs)} video(s).")
    limits.confirm_scale(len(refs))

    caption_langs = [s.strip() for s in a.captions.split(",") if s.strip()]

    # Resolve the backend before fetching anything: discovering that speech
    # recognition is unavailable after downloading audio wastes the download
    # and the user's time.
    asr_backend = None
    try:
        asr_backend = pick_backend(a.asr)
        print(f"  Speech recognition: {asr_backend}")
    except AsrUnavailable as e:
        if caption_langs:
            # Captions may still carry the whole job; carry on without ASR.
            print(f"  No speech recognition available; captions only.")
        else:
            print(f"\n  {e}\n", file=sys.stderr)
            return 2

    # ---- acquire -------------------------------------------------------
    todo = [r for r in store.ordered_todo(refs) if not store.done(r.video_id)]
    for i, ref in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {ref.title[:60]}", flush=True)
        try:
            limits.check_resources("this video")
            t = None
            if caption_langs:
                t = fetch_captions(ref, caption_langs, tmp, limits)
                if t:
                    print(f"        captions ({t.language})", flush=True)
            if t is None:
                t = transcribe(ref, backend=asr_backend, model=a.asr_model,
                               language=a.lang, tmp_dir=tmp, limits=limits)
                print(f"        transcribed  {len(t.text)} chars "
                      f"({t.char_rate:.1f} ch/s)", flush=True)
            store.put_transcript(ref.video_id, t.header(), t.text)
            store.mark(ref.video_id, status="ok", title=t.title,
                       duration=t.duration, chars=len(t.text),
                       source=t.source, language=t.language)
        except Aborted:
            raise
        except Exception as e:
            msg = str(e)
            deferred = "Premieres in" in msg or "will begin in" in msg
            store.mark(ref.video_id,
                       status="deferred" if deferred else f"error: {msg[:160]}",
                       title=ref.title)
            print(f"        {'not out yet — will retry' if deferred else 'failed: ' + msg[:80]}",
                  flush=True)
        limits.nap()

    ready = [r for r in refs if store.done(r.video_id)]
    print(f"\n  {len(ready)}/{len(refs)} transcribed -> {store.transcripts}")
    if a.transcribe_only or not ready:
        return 0

    # ---- analyse -------------------------------------------------------
    provider_name = autodetect() if a.provider == "auto" else a.provider
    provider = get_provider(provider_name, a.model)
    print(f"  Analysing with {provider_name} ...", flush=True)

    plan = plan_schema(provider, ask, shape)
    print(f"  Schema: {plan.record_name} "
          f"({', '.join(f['name'] for f in plan.fields)})", flush=True)

    profile_text = _load_profile(a.profile)
    from .transcripts import Transcript

    records: list[dict] = []
    for i, ref in enumerate(ready, 1):
        text = store.get_transcript(ref.video_id)
        if not text:
            continue
        st = store.state[ref.video_id]
        t = Transcript(video_id=ref.video_id, title=st.get("title", ref.title),
                       url=ref.url, text=text, language=st.get("language", ""),
                       source=st.get("source", ""), duration=st.get("duration", 0))
        print(f"  [{i}/{len(ready)}] {t.title[:60]}", flush=True)
        try:
            got = analyse(provider, t, plan, ask, a.output_language,
                          profile_text, a.instructions)
            records.extend(got)
            print(f"        {len(got)} {plan.record_name}(s)", flush=True)
        except Exception as e:
            print(f"        analysis failed: {str(e)[:90]}", flush=True)

    meta = {"title": f"{plan.record_name.title()} analysis",
            "request": ask, "source": ", ".join(a.urls),
            "video_count": len(ready), "record_count": len(records),
            "provider": provider_name}

    report.write_json(records, outdir / "analysis.json", meta)
    report.write_markdown(records, outdir / "analysis.md", meta)
    print(f"\n  {len(records)} records -> {outdir / 'analysis.md'}")

    if not a.no_corpus and len({r['video_id'] for r in records}) > 1:
        print("  Looking for patterns across all videos ...", flush=True)
        reg = corpus_mod.build_register(provider, records, ask, a.output_language)
        if corpus_mod.write_register(reg, outdir / "patterns.md", meta):
            print(f"  Patterns -> {outdir / 'patterns.md'}")

    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "providers":
            return cmd_providers()
        if args.command == "ui":
            return cmd_ui(args)
        if args.command == "profiles":
            for n in list_profiles():
                print(f"  {n}")
            return 0
        return cmd_run(args)
    except Aborted as e:
        print(f"\n  Stopped. {e}\n", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n  Interrupted. Finished videos are saved; rerun to continue.\n",
              file=sys.stderr)
        return 130
