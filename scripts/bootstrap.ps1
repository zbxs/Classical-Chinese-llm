param(
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

. "$PSScriptRoot\activate.ps1"
$Requirements = if ($Dev) { "requirements\dev.txt" } else { "requirements\training.txt" }
& ".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

# Prefer the already downloaded CUDA wheel on this machine. On a fresh
# machine, fetch the same build directly into the project-local environment.
$ReusableTorch = "E:\研二学习\科研规划\project1\artifacts\torch-2.6.0+cu124-cp312-cp312-win_amd64.whl"
& ".venv\Scripts\python.exe" -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath $ReusableTorch) {
        & ".venv\Scripts\python.exe" -m pip install $ReusableTorch
    }
    else {
        & ".venv\Scripts\python.exe" -m pip install torch==2.6.0+cu124 `
            --index-url https://download.pytorch.org/whl/cu124
    }
}
& ".venv\Scripts\python.exe" -m pip install -r $Requirements
& ".venv\Scripts\python.exe" -m pip list --format=freeze |
    Where-Object { $_ -notmatch '^classical-chinese-llm-lab==' } |
    Set-Content -Encoding UTF8 "requirements\requirements.lock"
Write-Host "Environment ready. Verify with: classical-llm doctor"
