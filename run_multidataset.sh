#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Optional Git Bash entry point; Windows PowerShell remains the native runner.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./run_multidataset.ps1 "$@"
