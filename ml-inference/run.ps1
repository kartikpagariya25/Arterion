param(
    [switch]$Setup
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venvPython = ".\venv\Scripts\python.exe"
if ($Setup -or -not (Test-Path $venvPython)) {
    Write-Host "Setting up virtual environment..." -ForegroundColor Cyan
    if (Test-Path ".\venv") { Remove-Item -Recurse -Force ".\venv" }
    py -3.11 -m venv venv
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements.txt
    if (Test-Path ".\requirements-add.txt") {
        & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements-add.txt
    }
}
Write-Host "Checking fastapi/uvicorn are installed in this venv..." -ForegroundColor Cyan
& $venvPython -m pip show fastapi uvicorn
if ($LASTEXITCODE -ne 0) {
    Write-Host "fastapi/uvicorn missing, re-running setup..." -ForegroundColor Yellow
    & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements.txt
}
Write-Host "Starting Arterion backend on http://127.0.0.1:8000 using $venvPython ..." -ForegroundColor Green
& $venvPython -m uvicorn app.main:app --reload
