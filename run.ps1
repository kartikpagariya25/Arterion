param(
    [switch]$Setup
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$backendDir = Join-Path $PSScriptRoot "ml-inference"
$frontendDir = Join-Path $PSScriptRoot "frontend"
$venvPython = Join-Path $backendDir "venv\Scripts\python.exe"

if ($Setup -or -not (Test-Path $venvPython)) {
    Write-Host "Setting up backend virtual environment..." -ForegroundColor Cyan
    Push-Location $backendDir
    if (Test-Path ".\venv") { Remove-Item -Recurse -Force ".\venv" }
    py -3.11 -m venv venv
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements.txt
    if (Test-Path ".\requirements-add.txt") {
        & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements-add.txt
    }
    Pop-Location
}

Write-Host "Checking fastapi/uvicorn are installed in the backend venv..." -ForegroundColor Cyan
& $venvPython -m pip show fastapi uvicorn | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "fastapi/uvicorn missing, re-running backend setup..." -ForegroundColor Yellow
    Push-Location $backendDir
    & $venvPython -m pip install --default-timeout=120 --retries 5 -r requirements.txt
    Pop-Location
}

Write-Host "Starting Arterion backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    param($py, $dir)
    Set-Location $dir
    & $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
} -ArgumentList $venvPython, $backendDir

Write-Host "Waiting for backend to come up..." -ForegroundColor Cyan
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $backendReady = $true; break }
    } catch {
        if ($backendJob.State -eq "Failed") {
            Write-Host "Backend job failed to start:" -ForegroundColor Red
            Receive-Job $backendJob
            exit 1
        }
    }
}
if ($backendReady) {
    Write-Host "Backend is up." -ForegroundColor Green
} else {
    Write-Host "Backend did not respond in time, continuing anyway (check the job output with Receive-Job)." -ForegroundColor Yellow
}

$frontendPort = 5500
Write-Host "Starting frontend static server on http://127.0.0.1:$frontendPort ..." -ForegroundColor Green
$frontendJob = Start-Job -ScriptBlock {
    param($py, $dir, $port)
    Set-Location $dir
    & $py -m http.server $port --bind 127.0.0.1
} -ArgumentList $venvPython, $frontendDir, $frontendPort

Start-Sleep -Seconds 2

$url = "http://127.0.0.1:$frontendPort/index.html"
Write-Host "Opening $url in your default browser..." -ForegroundColor Green
Start-Process $url

Write-Host ""
Write-Host "Arterion is running:" -ForegroundColor Cyan
Write-Host "  Frontend : http://127.0.0.1:$frontendPort/index.html"
Write-Host "  Backend  : http://127.0.0.1:8000  (docs at /docs)"
Write-Host ""
Write-Host "Press Ctrl+C to stop this script. Background jobs (backend/frontend) will keep running" -ForegroundColor Yellow
Write-Host "until you stop them. To stop both: Get-Job | Stop-Job; Get-Job | Remove-Job" -ForegroundColor Yellow
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 2
        if ($backendJob.State -eq "Failed") {
            Write-Host "Backend stopped unexpectedly:" -ForegroundColor Red
            Receive-Job $backendJob
            break
        }
    }
} finally {
    Write-Host "Shutting down background servers..." -ForegroundColor Cyan
    Stop-Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
}
