#!/bin/bash
# Double-click this file in Finder to start the tool.
# It opens your browser automatically. Close this window (or press Ctrl-C) to stop.

cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/vke ]; then
  echo
  echo "  The tool isn't installed in this folder yet."
  echo "  Run these two lines once, then double-click this file again:"
  echo
  echo "    python3 -m venv .venv"
  echo "    .venv/bin/pip install -e ."
  echo
  read -r -p "  Press Return to close. "
  exit 1
fi

clear
echo
echo "  Video Knowledge Extractor"
echo "  ─────────────────────────"
echo "  Your browser will open in a moment."
echo "  Leave this window open while you use it."
echo "  To stop: close this window, or press Ctrl-C."
echo

.venv/bin/vke ui

echo
read -r -p "  Stopped. Press Return to close. "
