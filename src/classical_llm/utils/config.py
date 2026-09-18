from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail early on malformed configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must contain a mapping: {config_path}")
    return value


def project_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve config paths relative to the repository, not the caller's CWD."""
    path = Path(value)
    return path if path.is_absolute() else project_root() / path
