param([int]$Port = 8001, [string]$Head = 'runs/word_accuracy/avhubert_miracl8_v1/head.pt')
$ErrorActionPreference = 'Stop'
$repositoryPath = $PSScriptRoot
$headPath = if ([System.IO.Path]::IsPathRooted($Head)) { $Head } else { Join-Path $repositoryPath $Head }
if (-not (Test-Path -LiteralPath $headPath)) { throw 'Run the supervised word accuracy pipeline first; no trained head exists.' }
$headPath = (Resolve-Path -LiteralPath $headPath).Path
$preservedHead = (Resolve-Path -LiteralPath (Join-Path $repositoryPath 'runs/word_accuracy/avhubert_miracl8_v1/head.pt')).Path
if ($headPath -ne $preservedHead) {
    $reportPath = Join-Path (Split-Path -Parent $headPath) 'evaluation.json'
    if (-not (Test-Path -LiteralPath $reportPath)) { throw 'A personal head needs its completed paired evaluation first.' }
    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $actualHeadHash = (Get-FileHash -LiteralPath $headPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($report.eligible_for_manual_research_activation -ne $true -or $report.head_sha256 -ne $actualHeadHash) { throw 'This personal head has not passed the measured research activation gate. The preserved model remains available.' }
}
$previousWordBackend = $env:LIPNET_WORD_BACKEND
$previousWordHead = $env:LIPNET_AVHUBERT_WORD_HEAD
$previousPreload = $env:LIPNET_PRELOAD_MODELS
Push-Location $repositoryPath
try {
    & (Join-Path $repositoryPath '.venv-avhubert-cpu/Scripts/python.exe') scripts/patch_fairseq_meta.py
    if ($LASTEXITCODE -ne 0) { throw 'Fairseq compatibility check failed.' }
    $env:LIPNET_WORD_BACKEND = 'avhubert'
    $env:LIPNET_AVHUBERT_WORD_HEAD = $headPath
    $env:LIPNET_PRELOAD_MODELS = '1'
    & (Join-Path $repositoryPath '.venv-avhubert-cpu/Scripts/python.exe') -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Word research server exited with code $LASTEXITCODE." }
} finally {
    $env:LIPNET_WORD_BACKEND = $previousWordBackend
    $env:LIPNET_AVHUBERT_WORD_HEAD = $previousWordHead
    $env:LIPNET_PRELOAD_MODELS = $previousPreload
    Pop-Location
}
