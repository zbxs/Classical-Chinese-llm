param(
    [string]$Teacher = "E:\研二学习\科研规划\project1\models\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
    [int]$Limit = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"

classical-llm synthesize-sft `
    --input data/cleaned/huggingface_sample/train.jsonl `
    --output data/sft/teacher_smoke `
    --teacher $Teacher `
    --tasks appreciation,creation `
    --limit $Limit `
    --max-new-tokens 96
