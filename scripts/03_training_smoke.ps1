$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"

classical-llm train cpt --config configs/smoke/cpt.yaml
classical-llm train sft --config configs/smoke/sft.yaml
classical-llm train reward --config configs/smoke/reward.yaml
classical-llm train dpo --config configs/smoke/dpo.yaml
classical-llm train grpo --config configs/smoke/grpo.yaml
classical-llm report --output reports/generated/results.json
