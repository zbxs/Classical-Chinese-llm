"""Fetch a pinned official instruction baseline into the project cache."""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    destination = ROOT / ".cache/huggingface" / "Qwen2.5-0.5B-Instruct" / REVISION
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "LICENSE", "README.md"],
    )
    files = {str(path.relative_to(destination)): path.stat().st_size for path in destination.rglob("*") if path.is_file()}
    result = {"repo": REPO, "revision": REVISION, "path": str(destination), "files": files,
              "bytes": sum(files.values())}
    output = ROOT / "reports/generated/instruct_baseline.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
