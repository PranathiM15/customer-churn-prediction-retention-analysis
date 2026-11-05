$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating project environment..." -ForegroundColor Cyan
    py -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
Write-Host "Installing or updating required packages..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
Write-Host "Starting ChurnScope..." -ForegroundColor Green
& $python -m streamlit run app.py
