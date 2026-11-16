param([switch]$Train, [switch]$Resume, [string]$Run = 'runs/multidataset/miracl8_cpu_fullmanifest_pilot_v1')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv-word-recognition/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Provision the isolated CPU environment first: python -m pipeline.word_recognition.setup'
}
function Invoke-PythonStage([string[]]$Arguments) {
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code $LASTEXITCODE. Inputs and partial outputs preserved." }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Invoke-PythonStage -Arguments @('scripts/multidataset.py','preflight','--output',".qa/multidataset_preflight_$stamp.json")
Invoke-PythonStage -Arguments @('scripts/multidataset.py','acquire-miracl','--minutes','10')
$manifestPath = 'data/processed/multidataset/miracl_v1.jsonl'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    Invoke-PythonStage -Arguments @('scripts/multidataset.py','ingest','--dataset','miracl','--root','data/external/miracl/original','--output',$manifestPath)
} else {
    Invoke-PythonStage -Arguments @('scripts/multidataset.py','validate','--manifest',$manifestPath,'--verify-sources')
}
if ($Train) {
    $stageArguments = @('-m','pipeline.multidataset.experiment','--manifest',$manifestPath,'--run',$Run,'--epochs','12','--minutes','8','--allow-training-exclusions')
    if ($Resume) { $stageArguments += '--resume' }
    Invoke-PythonStage -Arguments $stageArguments
}
