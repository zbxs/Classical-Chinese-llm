from __future__ import annotations

import os

from classical_llm.utils.config import project_root


def configure_project_environment() -> dict[str, str]:
    """Keep model, dataset and compiler caches inside the repository.

    ``setdefault`` respects an explicit caller override while preventing normal
    CLI use from writing into the user's global Hugging Face or Torch caches.
    """
    root = project_root()
    values = {
        "PYTHONNOUSERSITE": "1",
        "PIP_CACHE_DIR": str(root / ".cache" / "pip"),
        "HF_HOME": str(root / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(root / ".cache" / "huggingface"),
        "HF_DATASETS_CACHE": str(root / ".cache" / "huggingface" / "datasets"),
        "TORCH_HOME": str(root / ".cache" / "torch"),
        "TRITON_CACHE_DIR": str(root / ".cache" / "triton"),
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values

