param([int]$Port = 8002, [switch]$BuildFrontend)
$ErrorActionPreference = 'Stop'
$oldCollector = $env:LIPNET_FEEDBACK_TOKEN
$oldReview = $env:LIPNET_REVIEW_TOKEN
$oldAlphabet = $env:LIPNET_MODEL_BACKEND
function New-PrivateKey {
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    return [Convert]::ToBase64String($bytes)
}
Push-Location $PSScriptRoot
try {
    if ($BuildFrontend) {
        & npm --prefix frontend run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    }
    if (-not $env:LIPNET_FEEDBACK_TOKEN) { $env:LIPNET_FEEDBACK_TOKEN = New-PrivateKey }
    if (-not $env:LIPNET_REVIEW_TOKEN) { $env:LIPNET_REVIEW_TOKEN = New-PrivateKey }
    if ($env:LIPNET_FEEDBACK_TOKEN.Length -lt 32 -or $env:LIPNET_REVIEW_TOKEN.Length -lt 32 -or $env:LIPNET_FEEDBACK_TOKEN -eq $env:LIPNET_REVIEW_TOKEN) {
        throw 'Use different collector and reviewer keys of at least 32 characters.'
    }
    $env:LIPNET_MODEL_BACKEND = 'alphabet'
    Write-Host "Private local demo: http://127.0.0.1:$Port/demo/feedback"
    Write-Host 'Keep these access keys private. Enter a collector key to record, or a reviewer key to manage data.'
    Write-Host "Collector key: $env:LIPNET_FEEDBACK_TOKEN"
    Write-Host "Reviewer key: $env:LIPNET_REVIEW_TOKEN"
    Write-Host 'Set persistent private keys before launch to retain access to earlier recordings across restarts.'
    & (Join-Path $PSScriptRoot 'start_word_demo.ps1') -Port $Port
} finally {
    $env:LIPNET_FEEDBACK_TOKEN = $oldCollector
    $env:LIPNET_REVIEW_TOKEN = $oldReview
    $env:LIPNET_MODEL_BACKEND = $oldAlphabet
    Pop-Location
}
