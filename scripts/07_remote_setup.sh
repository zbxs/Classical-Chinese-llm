#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p .cache/pip .cache/huggingface .cache/torch .cache/triton reports/generated outputs/logs

# Reuse the image's CUDA-enabled PyTorch without writing packages into its
# global Conda environment. All additional packages live in this project venv.
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[data,train,eval,dev]'
python -m pip check
python -m classical_llm.cli doctor | tee reports/generated/remote_environment.json
python -m pytest
python -m ruff check src tests scripts/select_cpt.py scripts/update_readme_results.py
