$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONNOUSERSITE = "1"
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot ".cache\pip"
$env:HF_HOME = Join-Path $ProjectRoot ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $ProjectRoot ".cache\huggingface"
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot ".cache\huggingface\datasets"
$env:TORCH_HOME = Join-Path $ProjectRoot ".cache\torch"
$env:TRITON_CACHE_DIR = Join-Path $ProjectRoot ".cache\triton"
& (Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1")
Write-Host "Activated isolated environment: $ProjectRoot\.venv"
