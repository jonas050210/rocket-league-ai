$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No .venv found. Run scripts/setup_windows.ps1 first."
}
& $python -m rlai.cli serve --port 8000
