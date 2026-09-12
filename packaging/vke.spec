# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop builds.

Deliberately built against faster-whisper, not mlx-whisper. mlx is the faster
backend on Apple Silicon, but it depends on torch, which is 526MB on its own
and would roughly quadruple the download. faster-whisper reaches the same
Whisper weights through CTranslate2 with no torch at all — identical accuracy,
slower, and small enough that people will actually download it.

Anyone who wants the mlx speed installs from pip and gets it; `pick_backend`
already prefers mlx whenever it is importable.

Build:  pyinstaller packaging/vke.spec --noconfirm
"""
import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ROOT / "src" / "vke" / "static"), "vke/static"),
        (str(ROOT / "src" / "vke" / "profiles"), "vke/profiles"),
    ],
    hiddenimports=[
        "vke.providers.claude_cli", "vke.providers.anthropic_api",
        "vke.providers.openai_api", "vke.providers.ollama",
        "vke.transcripts.asr",
    ],
    hookspath=[],
    runtime_hooks=[],
    # torch arrives only through mlx-whisper and is not used by the bundled
    # backend. Excluding it is the difference between a 200MB download and
    # something closer to a gigabyte.
    excludes=["torch", "mlx", "mlx_whisper", "numba", "llvmlite",
              "tkinter.test", "test", "unittest"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VideoKnowledgeExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window on double-click
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="VideoKnowledgeExtractor",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Video Knowledge Extractor.app",
        icon=None,
        bundle_identifier="ai.excoms.vke",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            # Nothing here talks to the microphone or camera; the app only
            # reads files the user points it at and fetches public video.
            "LSApplicationCategoryType": "public.app-category.productivity",
        },
    )
