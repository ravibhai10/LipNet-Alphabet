param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
$repositoryPath = $PSScriptRoot
$pythonPath = Join-Path $repositoryPath '.venv-avhubert-cpu/Scripts/python.exe'
$headPath = Join-Path $repositoryPath 'runs/word_accuracy/avhubert_miracl8_v1/head.pt'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The verified AV-HuBERT CPU environment is missing.' }
if (-not (Test-Path -LiteralPath $headPath)) { throw 'The preserved eight-word fallback head is missing.' }
$savedBackend = $env:LIPNET_WORD_BACKEND
$savedHead = $env:LIPNET_AVHUBERT_WORD_HEAD
$savedOmp = $env:OMP_NUM_THREADS
$savedMkl = $env:MKL_NUM_THREADS
$savedPreload = $env:LIPNET_PRELOAD_MODELS
Push-Location $repositoryPath
try {
    & $pythonPath scripts/patch_fairseq_meta.py
    if ($LASTEXITCODE -ne 0) { throw 'Fairseq compatibility check failed.' }
    $env:LIPNET_WORD_BACKEND = 'avhubert'
    $env:LIPNET_AVHUBERT_WORD_HEAD = $headPath
    $env:LIPNET_PRELOAD_MODELS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    Write-Host "Private research demo: http://127.0.0.1:$Port/demo/live"
    & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port --workers 1
    if ($LASTEXITCODE -ne 0) { throw "Research server exited with code $LASTEXITCODE. Use an unused port; existing servers are not stopped." }
} finally {
    $env:LIPNET_WORD_BACKEND = $savedBackend
    $env:LIPNET_AVHUBERT_WORD_HEAD = $savedHead
    $env:OMP_NUM_THREADS = $savedOmp
    $env:MKL_NUM_THREADS = $savedMkl
    $env:LIPNET_PRELOAD_MODELS = $savedPreload
    Pop-Location
}
