$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"

$BaseModel = ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
classical-llm build-eval data/examples/evaluation_candidates.jsonl `
    --output data/final/evaluation/smoke_domain_4.jsonl --per-task 1
classical-llm generate --model $BaseModel `
    --dataset data/final/evaluation/smoke_domain_4.jsonl `
    --output outputs/evaluation/smoke_base_generations.jsonl `
    --max-new-tokens 64
classical-llm score `
    --input outputs/evaluation/smoke_base_generations.jsonl `
    --output outputs/evaluation/smoke_base_scored.jsonl
classical-llm perplexity --model $BaseModel `
    --dataset data/cleaned/huggingface_sample/test.jsonl `
    --output outputs/evaluation/smoke_base_perplexity.json `
    --max-length 256 --max-documents 2
