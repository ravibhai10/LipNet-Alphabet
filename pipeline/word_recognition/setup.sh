#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -f .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe; else PYTHON=python3.11; fi
fi
"$PYTHON" pipeline/word_recognition/setup.py "$@"
