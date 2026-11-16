param(
    [string]$Config = 'pipeline/word_recognition/config.cpu.json',
    [string]$Run = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repositoryPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
# Use the canonical isolated-environment runner, including protected outputs,
# the actual smoke module, diagnostic stage, and reproducible --resume support.
& (Join-Path $repositoryPath 'run_all.ps1') -Config $Config -Run $Run -Resume:$Resume
