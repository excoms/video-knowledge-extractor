"""Desktop entry point.

The app is the same local server the CLI serves; double-clicking it should
just open the page. Everything here exists to make the failure modes legible
to someone who has never seen a terminal, because that is the whole point of
shipping a download.
"""
from __future__ import annotations

import os
import sys
import traceback
import webbrowser
from pathlib import Path


def _bundled() -> bool:
    """True when running from inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _writable_home() -> Path:
    """Where vke keeps its data when launched by double-click.

    A bundled app's working directory is undefined — on macOS it can be "/" —
    so a relative vke-data/ would land somewhere unwritable or invisible.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "VideoKnowledgeExtractor"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VideoKnowledgeExtractor"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "vke"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A locked-down or managed machine may refuse the standard location.
        # Somewhere writable beats refusing to start.
        base = Path.home() / "VideoKnowledgeExtractor"
        base.mkdir(parents=True, exist_ok=True)
    return base


def _fail(title: str, detail: str) -> None:
    """Say what went wrong somewhere a double-click user will actually see.

    A traceback on a stdout nobody is watching is the same as no message.
    """
    sys.stderr.write(f"\n{title}\n\n{detail}\n")
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(title, detail)
        root.destroy()
    except Exception:
        pass


def main() -> int:
    if _bundled():
        os.environ.setdefault("VKE_DATA", str(_writable_home()))
        os.chdir(_writable_home())
    try:
        from vke.ui import serve
    except Exception:
        _fail("Video Knowledge Extractor could not start",
              "The application files look incomplete. Reinstalling usually "
              "fixes this.\n\n" + traceback.format_exc(limit=3))
        return 1

    try:
        serve(open_browser=True)
    except KeyboardInterrupt:
        return 0
    except OSError as e:
        _fail("Could not open the local page",
              f"Another copy may already be running.\n\n{e}\n\n"
              f"Close it and try again, or open http://127.0.0.1:7864 — the "
              f"copy that is already running will answer.")
        webbrowser.open("http://127.0.0.1:7864")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
