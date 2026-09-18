from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classical_llm.utils.config import load_yaml, resolve_project_path
from classical_llm.utils.io import sha256_file


def audit_resources(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    results: list[dict[str, Any]] = []
    for resource in config.get("resources", []):
        path = Path(resource["path"])
        item = {"name": resource["name"], "path": str(path), "exists": path.exists()}
        expected = resource.get("sha256")
        if expected and path.is_file():
            actual = sha256_file(path)
            item.update(sha256=actual, hash_matches=actual.lower() == str(expected).lower())
        model_expected = resource.get("model_sha256")
        model_file = path / "model.safetensors"
        if model_expected and model_file.exists():
            actual = sha256_file(model_file)
            item.update(model_sha256=actual, model_hash_matches=actual.lower() == str(model_expected).lower())
        results.append(item)
    report = {"policy": config.get("policy"), "resources": results}
    output = resolve_project_path("data/manifests/reused_resources.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

