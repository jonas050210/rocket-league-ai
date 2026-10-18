# Rocket League AI - Windows setup
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
#        powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Cuda cu128
param(
    [string]$Cuda = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== Rocket League AI setup ==" -ForegroundColor Green

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python not found. Install Python 3.11 or 3.12 from python.org and enable 'Add python.exe to PATH'."
}
$version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python $version detected"
if ($version -notin @("3.11", "3.12")) {
    Write-Warning "Python $version is not the tested version (3.11/3.12). Continuing anyway."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Created .venv"
}

$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

if ($Cuda -ne "") {
    Write-Host "Installing CUDA build of torch (cu$Cuda)..." -ForegroundColor Yellow
    & $venvPython -m pip install "torch==2.9.1" --index-url "https://download.pytorch.org/whl/cu$Cuda"
}

Write-Host "Installing project dependencies (this downloads ~2-3 GB for torch)..." -ForegroundColor Yellow
& $venvPython -m pip install -e ".[dev]"

Write-Host "Running tests..." -ForegroundColor Yellow
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Fix the environment before training."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start the app with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m rlai.cli serve --port 8000"
Write-Host "Then open http://localhost:8000"
