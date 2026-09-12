# Desktop builds

Two downloads, one per platform, both produced by CI on a runner of that
platform — a Windows executable cannot be built on macOS and the reverse is
also true.

```bash
pip install pyinstaller
pyinstaller packaging/vke.spec --noconfirm
```

Output lands in `dist/`: `Video Knowledge Extractor.app` on macOS, a
`VideoKnowledgeExtractor/` folder on Windows.

## Why the app ships faster-whisper and not mlx

`mlx-whisper` is several times quicker on Apple Silicon and it is what the
development machine uses. It is **not** in the bundle, because it depends on
`torch`:

| package | size |
|---|---|
| torch | 526 MB |
| mlx | 203 MB |
| llvmlite (via numba, via mlx-whisper) | 125 MB |

`faster-whisper` reaches the *same Whisper weights* through CTranslate2 and
needs none of them. All Whisper variants share identical weights, so this
costs speed and nothing else — a point worth making to anyone who assumes the
small download is the weaker one.

Someone who wants the mlx speed installs from pip:

```bash
pip install 'video-knowledge-extractor[mlx]'
```

`pick_backend()` prefers mlx whenever it is importable, so no setting changes.

## Model weights are not bundled either

Whisper weights are ~1.5 GB and are fetched on first transcription, then
cached under the user's home directory. Bundling them would make the download
larger than most people will accept for a tool they have not tried yet.

The consequence, which the first-run experience has to state plainly: **the
first transcription needs a network connection and a few minutes before
anything appears to happen.**

## Where a double-clicked app keeps its data

A bundled app has no meaningful working directory — on macOS it can be `/` —
so the launcher sets `VKE_DATA` and `data_root()` honours it:

| platform | location |
|---|---|
| macOS | `~/Library/Application Support/VideoKnowledgeExtractor/vke-data` |
| Windows | `%LOCALAPPDATA%\VideoKnowledgeExtractor\vke-data` |
| Linux | `$XDG_DATA_HOME/vke/vke-data` |

Running from a terminal is unaffected: without `VKE_DATA` it still uses
`./vke-data`, so the CLI keeps its existing behaviour.

## Signing — not yet done

Neither build is signed, so both show a warning on first launch:

- **macOS** — "cannot be opened because it is from an unidentified developer".
  Right-click → Open once, or notarise properly with an Apple Developer
  account (£79/year). Notarisation is the only way to remove the warning.
- **Windows** — SmartScreen "unrecognised app". An EV code-signing certificate
  removes it; reputation builds over time without one.

For a free tool that is arguably acceptable, and it will cost some downloads.
It is a cost worth knowing before the download page goes up rather than after.

## ffmpeg

Not a Python package, so pip cannot supply it. CI copies the platform binary
into the bundle. Without it, audio extraction fails at the first video and
looks like a bug in this tool.
