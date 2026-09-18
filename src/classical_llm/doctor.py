from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from classical_llm.utils.config import project_root


def environment_report() -> dict[str, Any]:
    root = project_root()
    # On Linux, venv/bin/python is commonly a symlink to the base interpreter.
    # Resolving that symlink makes a valid project venv look non-isolated, so
    # judge the environment by sys.prefix and the executable path as invoked.
    venv_root = (root / ".venv").absolute()
    executable = Path(sys.executable).absolute()
    prefix = Path(sys.prefix).absolute()
    report: dict[str, Any] = {
        "project_root": str(root),
        "python": sys.version,
        "executable": sys.executable,
        "isolated_venv": executable.is_relative_to(venv_root) or prefix.is_relative_to(venv_root),
        "platform": platform.platform(),
        "disk_free_gb": round(shutil.disk_usage(root).free / 1024**3, 2),
        "cache_environment": {
            key: os.environ.get(key)
            for key in ("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "PYTHONNOUSERSITE")
        },
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None,
        }
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostics must survive
        report["torch_error"] = repr(exc)
    for package in ("transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes"):
        try:
            module = __import__(package)
            report[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostics must survive
            report[f"{package}_error"] = repr(exc)
    return report


def save_environment_report(path: str | Path | None = None) -> dict[str, Any]:
    report = environment_report()
    destination = Path(path) if path else project_root() / "reports" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
