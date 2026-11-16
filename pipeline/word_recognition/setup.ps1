param([string]$Python = '.\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $projectRoot
try {
    & $Python pipeline/word_recognition/setup.py
    if ($LASTEXITCODE -ne 0) { throw "Word environment setup failed with exit $LASTEXITCODE" }
} finally { Pop-Location }
