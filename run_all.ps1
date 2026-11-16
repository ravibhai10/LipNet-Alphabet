param(
    [string]$Config = 'pipeline/word_recognition/config.cpu.json',
    [string]$Run = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $wordPython = Join-Path $PSScriptRoot '.venv-word-recognition/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv-word-recognition/word_pipeline_setup.json'))) {
        & (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') pipeline/word_recognition/setup.py
        if ($LASTEXITCODE -ne 0) { throw "Dependency setup failed with exit $LASTEXITCODE" }
    }
    $pipelineArguments = @('-u', '-m', 'pipeline.word_recognition.run_all', '--config', $Config)
    if ($Run) { $pipelineArguments += @('--run', $Run) }
    if ($Resume) { $pipelineArguments += '--resume' }
    & $wordPython @pipelineArguments
    if ($LASTEXITCODE -ne 0) { throw "Word pipeline failed with exit $LASTEXITCODE" }
} finally { Pop-Location }
