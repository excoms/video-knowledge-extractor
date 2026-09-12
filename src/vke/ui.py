"""A small local web interface: `vke ui`.

A thin layer over the same core the CLI uses — never a second implementation,
or the limits chokepoint stops being one. Standard library only, because a
public tool that needs a web framework to show a form has already lost people
at the install step.
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

# One run at a time. This is a personal tool on someone's laptop, not a service.
JOB: dict = {
    "state": "idle",          # idle | listing | transcribing | analysing | done | error
    "message": "",
    "videos": [],             # [{id, title, state, chars, rate, note}]
    "records": [],
    "outdir": "",
    "started": None,
    "error": "",
}
LOCK = threading.Lock()


def _set(**fields) -> None:
    with LOCK:
        JOB.update(fields)


def _video(vid: str, **fields) -> None:
    with LOCK:
        for v in JOB["videos"]:
            if v["id"] == vid:
                v.update(fields)
                return
        JOB["videos"].append({"id": vid, **fields})


def run_job(cfg: dict) -> None:
    """Execute a run, reporting progress into JOB as it goes."""
    from . import corpus as corpus_mod
    from . import report
    from .analysis import analyse, identify_speakers, plan_schema
    from .cli import _load_profile
    from .limits import Aborted, Limits
    from .providers import autodetect, get_provider
    from .sources import resolve_many
    from .store import Store
    from .transcripts import Transcript, fetch_captions, transcribe
    from .transcripts.asr import AsrUnavailable, pick_backend

    try:
        urls = [u.strip() for u in cfg.get("urls", "").splitlines() if u.strip()]
        if not urls:
            raise ValueError("Add at least one link.")
        ask = (cfg.get("ask") or "").strip()
        if not ask and not cfg.get("transcribe_only"):
            raise ValueError("Say what you want from the videos.")

        limits = Limits(assume_yes=True)
        from .paths import cache_dir, new_run_dir, write_run_manifest
        outdir = (Path(cfg["outdir"]) if cfg.get("outdir")
                  else new_run_dir(urls[0], fallback="run"))
        store = Store(outdir, cache=Path(cfg["cache"]) if cfg.get("cache")
                      else cache_dir())
        tmp = outdir / ".tmp"
        for stale in tmp.glob("*.wav"):
            stale.unlink(missing_ok=True)

        _set(state="listing", message="Finding videos…", outdir=str(outdir.resolve()),
             started=datetime.now().isoformat(), videos=[], records=[], error="")
        refs = resolve_many(urls, limits)
        if not refs:
            raise ValueError("No videos found. Check the link.")
        if cfg.get("limit"):
            refs = refs[: int(cfg["limit"])]

        for r in refs:
            _video(r.video_id, title=r.title, state="waiting", chars=0, rate=0, note="")

        caption_langs = [s.strip() for s in (cfg.get("captions") or "").split(",") if s.strip()]
        # Fail before downloading anything, not after.
        asr_backend = pick_backend(cfg.get("asr") or "auto")

        _set(state="transcribing", message=f"{len(refs)} videos")
        for ref in refs:
            if store.done(ref.video_id):
                store.adopt_from_cache(ref.video_id)
                st = store.state[ref.video_id]
                _video(ref.video_id, state="done", chars=st.get("chars", 0),
                       rate=round(st.get("chars", 0) / (st.get("duration") or 1), 1),
                       note="already saved")
                continue
            _video(ref.video_id, state="running", note="transcribing")
            try:
                limits.check_resources("this video")
                t = None
                if caption_langs:
                    t = fetch_captions(ref, caption_langs, tmp, limits)
                if t is None:
                    started = time.monotonic()

                    def on_transcribe_progress(fraction, _ref=ref, _t0=started):
                        """Surface transcription progress on the page."""
                        pct = int(fraction * 100)
                        elapsed = time.monotonic() - _t0
                        note = f"transcribing {pct}%"
                        if fraction > 0.02:
                            remaining = elapsed / fraction - elapsed
                            note += f" · about {remaining / 60:.0f} min left"
                        _video(_ref.video_id, state="running",
                               progress=pct, note=note)

                    t = transcribe(ref, backend=asr_backend,
                                   language=cfg.get("lang") or None,
                                   tmp_dir=tmp, limits=limits,
                                   on_progress=on_transcribe_progress)
                store.put_transcript(ref.video_id, t.header(), t.text, t.segments)
                store.mark(ref.video_id, status="ok", title=t.title, duration=t.duration,
                           chars=len(t.text), source=t.source, language=t.language)
                _video(ref.video_id, state="done", title=t.title, chars=len(t.text),
                       rate=round(t.char_rate, 1), note=t.source.split(":")[0])
            except Aborted as e:
                _video(ref.video_id, state="failed", note=str(e)[:120])
                raise
            except Exception as e:
                msg = str(e)
                deferred = "Premieres in" in msg or "will begin in" in msg
                store.mark(ref.video_id,
                           status="deferred" if deferred else f"error: {msg[:160]}",
                           title=ref.title)
                _video(ref.video_id, state="deferred" if deferred else "failed",
                       note="not released yet" if deferred else msg[:110])
            limits.nap()

        ready = [r for r in refs if store.done(r.video_id)]
        if cfg.get("transcribe_only") or not ready:
            _set(state="done", message=f"{len(ready)} transcribed")
            return

        name = cfg.get("provider") or "auto"
        provider = get_provider(autodetect() if name == "auto" else name,
                                cfg.get("model") or None)
        _set(state="analysing", message=f"Working out what to extract…")

        plan = plan_schema(provider, ask, cfg.get("shape", ""))
        _set(message=f"Extracting {plan.record_name}s "
                     f"({', '.join(f['name'] for f in plan.fields)})")

        profile_text = _load_profile(cfg.get("profile") or "general")

        # One roster for the whole run. Chunks are analysed independently, so
        # without it the same person is named differently in each.
        roster: list[dict] = []
        if ready:
            first_text = store.get_transcript(ready[0].video_id) or ""
            if first_text:
                st0 = store.state[ready[0].video_id]
                probe = Transcript(ready[0].video_id, st0.get("title", ready[0].title),
                                   ready[0].url, first_text, st0.get("language", ""),
                                   st0.get("source", ""), st0.get("duration", 0))
                roster = identify_speakers(provider, probe)
                if roster:
                    _set(message="Speakers: " + ", ".join(r["name"] for r in roster))
        records: list[dict] = []
        for i, ref in enumerate(ready, 1):
            text = store.get_transcript(ref.video_id)
            if not text:
                continue
            st = store.state[ref.video_id]
            t = Transcript(ref.video_id, st.get("title", ref.title), ref.url, text,
                           st.get("language", ""), st.get("source", ""),
                           st.get("duration", 0),
                                segments=store.get_segments(ref.video_id))
            _video(ref.video_id, state="analysing", progress=0,
                   note=f"analysing ({i} of {len(ready)})")
            got = analyse(provider, t, plan, ask,
                          cfg.get("output_language") or "English", profile_text,
                          cfg.get("instructions", ""), speakers=roster)
            records.extend(got)
            _video(ref.video_id, state="done",
                   note=f"{len(got)} {plan.record_name}(s)")
            _set(records=records, message=f"{len(records)} found across {i} video(s)")

        meta = {"title": f"{plan.record_name.title()} analysis", "request": ask,
                "source": ", ".join(urls), "video_count": len(ready),
                "record_count": len(records), "provider": provider.name}
        write_run_manifest(outdir, request=ask, output_shape=cfg.get("shape", ""),
                           sources=urls, profile=cfg.get("profile"),
                           provider=provider.name, asr=asr_backend,
                           output_language=cfg.get("output_language"),
                           videos=len(ready), records=len(records))
        report.write_json(records, outdir / "analysis.json", meta)
        report.write_markdown(records, outdir / "analysis.md", meta)
        doc = report.write_docx(outdir / "analysis.md")
        if len(records) >= corpus_mod.MIN_RECORDS:
            _set(message="Grouping, scoring and profiling")
            meta["source_name"] = (ready[0].title if len(ready) == 1
                                   else ", ".join(urls))
            try:
                reg = corpus_mod.build_register(provider, records, ask,
                                                cfg.get("output_language") or "English",
                                                speakers=roster)
            except corpus_mod.SynthesisFailed as e:
                (outdir / "REPORT-NOT-WRITTEN.txt").write_text(
                    f"The report was not written.\n\n{e}\n\nThe {len(records)} "
                    f"records are intact in analysis.json. To retry just this "
                    f"pass:\n\n    vke report {outdir}\n", encoding="utf-8")
                _set(state="error", error=str(e)[:400],
                     message=f"{len(records)} records saved, but the report failed")
                return
            if corpus_mod.write_register(reg, outdir / "debrief.md", meta):
                report.write_docx(outdir / "debrief.md")

        _set(state="done", message=f"{len(records)} records from {len(ready)} videos")

    except Exception as e:
        _set(state="error", error=str(e)[:400], message="Stopped")


CHAT_SYSTEM = """You are the analyst inside Video Knowledge Extractor, talking
to the person who just ran it. You can see the run they are asking about.

Answer from the run when the run contains the answer, and say plainly when it
does not rather than inventing a plausible detail. If they ask why a speaker
scored as they did, or why an argument was graded a certain way, cite the
specific record — the quote, the timestamp, the fallacy named.

Two things you must never soften: attribution in this tool is inferred from
context rather than heard, because the transcription carries no speaker
labels; and a counter-argument you compose is your reasoning, not something
anybody said. Flag both whenever they bear on the answer.

Be brief. This is a chat box, not a report."""

CHAT_CONTEXT_CHARS = 30000


def recent_runs(limit: int = 20) -> list[dict]:
    """Finished runs, newest first, for the chat box to attach itself to."""
    from .paths import data_root

    runs = data_root() / "runs"
    if not runs.is_dir():
        return []
    out = []
    for d in sorted((x for x in runs.iterdir() if x.is_dir()),
                    key=lambda x: x.name, reverse=True)[:limit]:
        manifest = d / "run.json"
        title = d.name
        records = 0
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text("utf-8"))
                records = m.get("records", 0)
                srcs = m.get("sources") or []
                title = f"{d.name} — {records} records"
                if srcs:
                    title += f" — {srcs[0][:60]}"
            except (OSError, json.JSONDecodeError):
                pass
        out.append({"dir": d.name, "label": title, "records": records,
                    "has_report": (d / "debrief.md").exists()})
    return out


def _run_context(name: str) -> str:
    """The report if there is one, otherwise the records. Reports are already
    condensed, so they buy far more context per character."""
    from .paths import data_root

    if not name:
        return ""
    d = data_root() / "runs" / name
    if not d.is_dir() or d.resolve().parent != (data_root() / "runs").resolve():
        return ""            # never let a path escape the runs directory
    for candidate in ("debrief.md", "report.md", "analysis.md"):
        f = d / candidate
        if f.exists():
            body = f.read_text("utf-8", errors="replace")
            head = f"--- {candidate} from run {name} ---\n"
            if len(body) > CHAT_CONTEXT_CHARS:
                body = (body[:CHAT_CONTEXT_CHARS]
                        + f"\n\n[truncated at {CHAT_CONTEXT_CHARS:,} characters "
                          f"of {len(body):,}]")
            return head + body
    return ""


def chat_reply(body: dict) -> str:
    """One turn of conversation, with the chosen run attached."""
    from .providers import autodetect, get_provider

    message = (body.get("message") or "").strip()
    if not message:
        raise ValueError("Nothing to answer.")
    name = body.get("provider") or "auto"
    provider = get_provider(autodetect() if name == "auto" else name,
                            body.get("model") or None)

    parts = []
    context = _run_context(body.get("run") or "")
    if context:
        parts.append(context)
    for turn in (body.get("history") or [])[-12:]:
        who = "User" if turn.get("role") == "user" else "You"
        parts.append(f"{who}: {turn.get('content', '')}")
    parts.append(f"User: {message}")
    return provider.complete(CHAT_SYSTEM, "\n\n".join(parts), max_tokens=2000)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):        # keep the terminal readable
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Never cache. A stale page keeps sending stale choices long after the
        # server is fixed, which is indistinguishable from the fix not working.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (STATIC / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/status":
            with LOCK:
                return self._json(dict(JOB))
        if self.path == "/api/asr":
            from .transcripts.asr import available_backends
            return self._json(available_backends())
        if self.path == "/api/providers":
            from .providers import describe
            return self._json(describe())
        if self.path == "/api/runs":
            return self._json(recent_runs())
        self._send(404, b"not found", "text/plain")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_POST(self):
        if self.path == "/api/provider-key":
            from .providers import describe, set_key
            body = self._body()
            try:
                set_key(body.get("provider", ""), body.get("key", ""))
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            return self._json(describe())

        if self.path == "/api/chat":
            body = self._body()
            try:
                reply = chat_reply(body)
            except Exception as e:
                return self._json({"error": str(e)[:400]}, 400)
            return self._json({"reply": reply})

        if self.path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        with LOCK:
            if JOB["state"] in ("listing", "transcribing", "analysing"):
                return self._json({"error": "A run is already in progress."}, 409)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            cfg = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "Bad request."}, 400)
        threading.Thread(target=run_job, args=(cfg,), daemon=True).start()
        self._json({"ok": True})


def _port_holder(port: int) -> str:
    """Who is on this port? Best effort — purely to make the message useful."""
    import shutil
    import subprocess
    if not shutil.which("lsof"):
        return ""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return out[1] if len(out) > 1 else ""
    except Exception:
        return ""


def serve(host: str = "127.0.0.1", port: int = 7864, open_browser: bool = True,
          max_tries: int = 12) -> None:
    """Start the interface, stepping past a busy port rather than dying on it.

    An older copy of this tool left running is the common case, and it is a
    nasty one: it holds the port and keeps serving its own stale code, so every
    fix looks like it did not work. Falling forward to a free port means the
    browser lands on the new server instead.
    """
    httpd = None
    first = port
    for candidate in range(first, first + max_tries):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
            port = candidate
            break
        except OSError as e:
            if e.errno not in (48, 98):        # EADDRINUSE on macOS / Linux
                raise
            continue

    if httpd is None:
        holder = _port_holder(first)
        raise SystemExit(
            f"\n  Ports {first}-{first + max_tries - 1} are all in use.\n"
            + (f"  {first} is held by: {holder}\n" if holder else "")
            + f"\n  Stop whatever is on {first} and try again:\n"
              f"    lsof -ti:{first} | xargs kill\n"
        )

    if port != first:
        holder = _port_holder(first)
        print(f"\n  Port {first} was busy" + (f" (held by: {holder})" if holder else "")
              + f"\n  — probably an older copy of this tool still running.")
        print(f"  Using port {port} instead. To tidy up later:  "
              f"lsof -ti:{first} | xargs kill")

    url = f"http://{host}:{port}"
    print(f"\n  Video Knowledge Extractor")
    print(f"  {url}")
    print(f"  Ctrl-C to stop\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")
        httpd.shutdown()
