"""A small local web interface: `vke ui`.

A thin layer over the same core the CLI uses — never a second implementation,
or the limits chokepoint stops being one. Standard library only, because a
public tool that needs a web framework to show a form has already lost people
at the install step.
"""
from __future__ import annotations

import json
import threading
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
                    t = transcribe(ref, backend=asr_backend,
                                   language=cfg.get("lang") or None,
                                   tmp_dir=tmp, limits=limits)
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
            _video(ref.video_id, state="analysing", note="reading")
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
            reg = corpus_mod.build_register(provider, records, ask,
                                            cfg.get("output_language") or "English",
                                            speakers=roster)
            if corpus_mod.write_register(reg, outdir / "debrief.md", meta):
                report.write_docx(outdir / "debrief.md")

        _set(state="done", message=f"{len(records)} records from {len(ready)} videos")

    except Exception as e:
        _set(state="error", error=str(e)[:400], message="Stopped")


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
            from .providers import REGISTRY, get_provider
            out = {}
            for n in REGISTRY:
                try:
                    get_provider(n)
                    out[n] = "ready"
                except Exception as e:
                    out[n] = str(e).splitlines()[0][:90]
            return self._json(out)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
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
