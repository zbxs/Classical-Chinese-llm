from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classical_llm.utils.config import project_root


def collect_results(output_path: str | Path) -> dict[str, Any]:
    """Collect run metadata without inventing results for stages not executed."""
    root = project_root()
    runs: list[dict[str, Any]] = []
    for metadata in sorted((root / "outputs").glob("**/run_metadata.json")):
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["metadata_path"] = str(metadata.relative_to(root))
        runs.append(payload)
    summaries: list[dict[str, Any]] = []
    for summary in sorted((root / "outputs" / "evaluation").glob("*.summary.json")):
        summaries.append(
            {"path": str(summary.relative_to(root)), "data": json.loads(summary.read_text(encoding="utf-8"))}
        )
    perplexity: list[dict[str, Any]] = []
    for result_path in sorted((root / "outputs" / "evaluation").glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "perplexity" in payload and "predicted_tokens" in payload:
            perplexity.append({"path": str(result_path.relative_to(root)), "data": payload})
    report = {
        "runs": runs,
        "evaluation_summaries": summaries,
        "perplexity_results": perplexity,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
