$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"

# These configs expect the reviewed full datasets under data/final. Each stage
# stops immediately on failure; later stages never silently fall back to a bad checkpoint.
classical-llm train cpt --config configs/experiments/cpt_local.yaml
classical-llm train sft --config configs/experiments/sft_local.yaml
classical-llm train reward --config configs/experiments/reward_local.yaml
classical-llm train dpo --config configs/experiments/dpo_local.yaml
classical-llm train grpo --config configs/experiments/grpo_local.yaml
classical-llm report --output reports/generated/results.json
