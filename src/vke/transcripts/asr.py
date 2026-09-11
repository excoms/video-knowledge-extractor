"""Speech recognition, for the ~86 languages YouTube never captions.

All Whisper variants run the same weights, so the backend choice is speed and
platform only, never accuracy:

  faster-whisper  cross-platform, pip-installable, CPU or CUDA   <- default
  mlx-whisper     Apple Silicon only, fastest on a Mac
"""
from __future__ import annotations

import contextlib
import importlib.util
import platform
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
            f"{requested} cannot be used: {available.get(requested, 'unknown backend')}"
        )
    for name in ("mlx", "faster-whisper"):
        if available.get(name) == "ready":
            return name
    raise AsrUnavailable(
        "No speech recognition backend is installed.\n"
        f"  {INSTALL_HINT['faster-whisper']}\n"
        f"  {INSTALL_HINT['mlx']}"
    )


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


def _run_faster_whisper(wav: Path, model_name: str, language: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise AsrUnavailable(
            "faster-whisper is not installed. Run: pip install 'video-knowledge-extractor[asr]'"
        ) from e
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segs, info = model.transcribe(str(wav), language=language, vad_filter=True)
    out = [(s.start, s.end, s.text.strip()) for s in segs]
    return out, getattr(info, "language", language or "unknown")


def _run_mlx(wav: Path, model_name: str, language: str | None):
    try:
        import mlx_whisper
    except ImportError as e:
        raise AsrUnavailable(
            "mlx-whisper is not installed (Apple Silicon only). "
            "Run: pip install 'video-knowledge-extractor[mlx]'"
        ) from e
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model_name,
                               language=language, verbose=False)
    out = [(s.get("start", 0.0), s.get("end", 0.0), s.get("text", "").strip())
           for s in r.get("segments", [])]
    return out, r.get("language", language or "unknown")


def transcribe(ref, backend: str = "faster-whisper", model: str | None = None,
               language: str | None = None, tmp_dir: Path | None = None,
               limits=None):
    """Download audio, transcribe it, delete the audio. Always."""
    from . import Segment, Transcript

    tmp_dir = tmp_dir or Path(".vke-tmp")
    model = model or DEFAULT_MODEL.get(backend, DEFAULT_MODEL["faster-whisper"])

    if limits is not None:
        limits.check_resources("downloading audio")

    wav, info = _download_audio(ref, tmp_dir, limits)
    try:
        if limits is not None:
            limits.check_resources("transcription")
        runner = _run_mlx if backend == "mlx" else _run_faster_whisper
        raw, detected = runner(wav, model, language)
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
