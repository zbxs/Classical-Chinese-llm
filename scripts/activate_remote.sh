#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  printf 'Remote virtual environment is missing. Run scripts/07_remote_setup.sh first.\n' >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"
