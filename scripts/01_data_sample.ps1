param([int]$MaxDocuments = 100)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"

classical-llm acquire --config configs/data/sources.yaml --output data/raw/huggingface_sample --max-documents $MaxDocuments
classical-llm prepare `
    data/raw/huggingface_sample/gujilab-classical.jsonl `
    data/raw/huggingface_sample/gujilab-translate.jsonl `
    data/raw/huggingface_sample/gujilab-punctuate.jsonl `
    data/raw/huggingface_sample/fineweb2-chinese.jsonl `
    data/raw/huggingface_sample/fineweb-english.jsonl `
    --output data/cleaned/huggingface_sample
classical-llm mix `
    --input data/cleaned/huggingface_sample/train.jsonl `
    --output data/final/pretrain/sample_100k/train.jsonl `
    --tokenizer .cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 `
    --total-tokens 100000 `
    --fractions '{"classical":0.20,"general_zh":0.65,"general_en":0.15}'
classical-llm build-sft `
    data/cleaned/huggingface_sample/train.jsonl `
    data/cleaned/huggingface_sample/validation.jsonl `
    data/cleaned/huggingface_sample/test.jsonl `
    --output data/sft/huggingface_sample
classical-llm build-preferences `
    --input data/sft/huggingface_sample/train.jsonl `
    --output data/preferences/huggingface_sample `
    --limit 20000
