#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -f .venv-word-recognition/word_pipeline_setup.json ]]; then
  bash pipeline/word_recognition/setup.sh
fi
if [[ -f .venv-word-recognition/Scripts/python.exe ]]; then
  PYTHON=.venv-word-recognition/Scripts/python.exe
elif [[ -f .venv-word-recognition/bin/python ]]; then
  PYTHON=.venv-word-recognition/bin/python
else
  echo 'Dedicated environment marker exists, but Python is missing.' >&2
  exit 1
fi
"$PYTHON" -u -m pipeline.word_recognition.run_all "$@"
