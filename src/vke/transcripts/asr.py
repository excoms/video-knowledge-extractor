"""Speech recognition, for the ~86 languages YouTube never captions.

All Whisper variants run the same weights, so the backend choice is speed and
platform only, never accuracy:

  faster-whisper  cross-platform, pip-installable, CPU or CUDA   <- default
  mlx-whisper     Apple Silicon only, fastest on a Mac
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import platform
import re
from pathlib import Path

from ..normalise import clean_text

DEFAULT_MODEL = {"faster-whisper": "large-v3", "mlx": "mlx-community/whisper-large-v3-turbo"}


class AsrUnavailable(RuntimeError):
    """The requested backend is not installed."""


INSTALL_HINT = {
    "faster-whisper": "pip install 'video-knowledge-extractor[asr]'",
    "mlx": "pip install 'video-knowledge-extractor[mlx]'  (Apple Silicon only)",
}


def _installed(module: str) -> bool:
    """Is the package present, without importing it?

    Importing is not safe as a probe: mlx_whisper raises on import when no Metal
    device is reachable, which would make an installed backend look missing.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> dict[str, str]:
    """Backend -> "ready", or the reason it cannot be used."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    out = {}
    out["faster-whisper"] = ("ready" if _installed("faster_whisper")
                             else f"not installed — {INSTALL_HINT['faster-whisper']}")
    if not apple:
        out["mlx"] = "Apple Silicon only"
    else:
        out["mlx"] = ("ready" if _installed("mlx_whisper")
                      else f"not installed — {INSTALL_HINT['mlx']}")
    return out


def pick_backend(requested: str = "auto") -> str:
    """Choose a backend that actually exists.

    Prefers mlx on Apple Silicon, where it is several times faster than
    faster-whisper on the same weights.
    """
    available = available_backends()
    if requested != "auto":
        if available.get(requested) == "ready":
            return requested
        raise AsrUnavailable(
            _diagnosis(f"Speech recognition backend {requested!r} cannot be used: "
                       f"{available.get(requested, 'unknown backend')}")
        )
    for name in ("mlx", "faster-whisper"):
        if available.get(name) == "ready":
            return name
    raise AsrUnavailable(_diagnosis("No speech recognition backend is usable."))


def _diagnosis(headline: str) -> str:
    """Say what was checked and what to do, not just what failed."""
    import sys
    lines = [headline, ""]
    lines.append(f"  platform : {platform.system()} {platform.machine()}")
    lines.append(f"  python   : {sys.executable}")
    lines.append("  backends :")
    for name, status in available_backends().items():
        lines.append(f"    {name:16} {status}")
    lines.append("")
    lines.append("  Install one of:")
    for hint in INSTALL_HINT.values():
        lines.append(f"    {hint}")
    lines.append("")
    lines.append("  Then run `vke providers` to confirm, and restart `vke ui`")
    lines.append("  if it is running — a running server keeps the old code.")
    return "\n".join(lines)


class _ProgressTap(io.TextIOBase):
    """Reads the progress bar the transcriber prints, and reports the fraction.

    Whisper writes a tqdm bar to stderr. Left alone it lands in whichever
    terminal happens to be running the server, which is no use to someone
    watching the web page — so it is intercepted, parsed, and forwarded.
    """

    PERCENT = re.compile(r"(\d{1,3})%\|")

    def __init__(self, on_progress, passthrough=None):
        self.on_progress = on_progress
        self.passthrough = passthrough
        self._last = -1

    def write(self, text: str) -> int:
        if self.passthrough is not None:
            self.passthrough.write(text)
        m = None
        for m in self.PERCENT.finditer(text):
            pass                       # keep the last match in the chunk
        if m:
            pct = min(int(m.group(1)), 100)
            if pct != self._last:
                self._last = pct
                with contextlib.suppress(Exception):
                    self.on_progress(pct / 100.0)
        return len(text)

    def flush(self) -> None:
        if self.passthrough is not None:
            with contextlib.suppress(Exception):
                self.passthrough.flush()


def _download_audio(ref, tmp_dir: Path, limits=None) -> tuple[Path, dict]:
    """16 kHz mono wav — what Whisper wants, and the smallest thing that works."""
    import yt_dlp

    tmp_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"extractaudio": ["-ar", "16000", "-ac", "1"]},
    }
    if limits is not None:
        opts.update(limits.ytdlp_opts())
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(ref.url, download=True)
    return tmp_dir / f"{ref.video_id}.wav", (info or {})


def _run_faster_whisper(wav: Path, model_name: str, language: str | None,
                        on_progress=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise AsrUnavailable(
            "faster-whisper is not installed. Run: pip install 'video-knowledge-extractor[asr]'"
        ) from e
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segs, info = model.transcribe(str(wav), language=language, vad_filter=True)
    total = getattr(info, "duration", 0) or 0
    out = []
    for seg in segs:                       # a generator — position is known
        out.append((seg.start, seg.end, seg.text.strip()))
        if on_progress and total:
            with contextlib.suppress(Exception):
                on_progress(min(seg.end / total, 1.0))
    return out, getattr(info, "language", language or "unknown")


def _run_mlx(wav: Path, model_name: str, language: str | None, on_progress=None):
    import sys

    try:
        import mlx_whisper
    except ImportError as e:
        raise AsrUnavailable(
            "mlx-whisper is not installed (Apple Silicon only). "
            "Run: pip install 'video-knowledge-extractor[mlx]'"
        ) from e
    if on_progress:
        tap = _ProgressTap(on_progress, passthrough=sys.stderr)
        with contextlib.redirect_stderr(tap):
            r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model_name,
                                       language=language, verbose=False)
    else:
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model_name,
                                   language=language, verbose=False)
    out = [(s.get("start", 0.0), s.get("end", 0.0), s.get("text", "").strip())
           for s in r.get("segments", [])]
    return out, r.get("language", language or "unknown")


def transcribe(ref, backend: str = "auto", model: str | None = None,
               language: str | None = None, tmp_dir: Path | None = None,
               limits=None, on_progress=None):
    """Download audio, transcribe it, delete the audio. Always.

    The backend is resolved here as well as at the entry points: a guard that
    only sits on the callers is one refactor away from being bypassed, and the
    failure it prevents is a confusing error after a download has already run.
    """
    from . import Segment, Transcript

    backend = pick_backend(backend or "auto")
    tmp_dir = tmp_dir or Path(".vke-tmp")
    model = model or DEFAULT_MODEL.get(backend, DEFAULT_MODEL["faster-whisper"])

    if limits is not None:
        limits.check_resources("downloading audio")

    wav, info = _download_audio(ref, tmp_dir, limits)
    try:
        if limits is not None:
            limits.check_resources("transcription")
        runner = _run_mlx if backend == "mlx" else _run_faster_whisper
        raw, detected = runner(wav, model, language, on_progress)
    finally:
        # A killed process skips this; the caller sweeps orphans on the next run.
        with contextlib.suppress(Exception):
            wav.unlink(missing_ok=True)

    segments = [Segment(a, b, t) for a, b, t in raw if t]
    text = clean_text(" ".join(s.text for s in segments))
    return Transcript(
        video_id=ref.video_id, title=info.get("title") or ref.title, url=ref.url,
        text=text, language=detected or (language or "unknown"),
        source=f"asr:{model}", duration=float(info.get("duration") or 0),
        segments=segments,
    )
