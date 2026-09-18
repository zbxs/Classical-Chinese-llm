from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from classical_llm.utils.io import read_jsonl, write_jsonl


def _reference_text(row: dict[str, Any]) -> str:
    direct = row.get("reference") or row.get("answer") or row.get("output")
    if direct:
        return str(direct)
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def _constraint_score(row: dict[str, Any]) -> float | None:
    constraints = row.get("constraints")
    if not isinstance(constraints, dict):
        return None
    response = str(row.get("response", ""))
    checks: list[bool] = []
    if "line_count" in constraints:
        lines = [part for part in re.split(r"[\n。！？]", response) if part.strip()]
        checks.append(len(lines) == int(constraints["line_count"]))
    if "keywords" in constraints:
        checks.extend(str(word) in response for word in constraints["keywords"])
    return sum(checks) / len(checks) if checks else None


def score_generation_file(path: str | Path, output_path: str | Path) -> dict[str, Any]:
    import sacrebleu

    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    scored: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        response = str(row.get("response", ""))
        reference = _reference_text(row)
        metrics: dict[str, float] = {
            "response_length": float(len(response)),
            "nonempty": float(bool(response.strip())),
        }
        if reference:
            metrics["chrf"] = float(sacrebleu.sentence_chrf(response, [reference]).score)
        constraint = _constraint_score(row)
        if constraint is not None:
            metrics["constraint_accuracy"] = constraint
        item = dict(row)
        item["automatic_metrics"] = metrics
        scored.append(item)
        by_task[str(row.get("task", "unknown"))].append(metrics)

    summary: dict[str, Any] = {"tasks": {}}
    for task, metrics in by_task.items():
        keys = sorted({key for row in metrics for key in row})
        summary["tasks"][task] = {
            key: mean(row[key] for row in metrics if key in row) for key in keys
        }
        summary["tasks"][task]["count"] = len(metrics)
    write_jsonl(output_path, scored)
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_blinded_review(
    generation_files: dict[str, str | Path], output_path: str | Path, seed: int = 20260912
) -> dict[str, Any]:
    """Shuffle model identities and emit empty rubric fields for human review."""
    rng = random.Random(seed)
    dimensions = {
        "old_to_modern": ["faithfulness", "completeness", "fluency"],
        "modern_to_old": ["meaning_preservation", "classical_style", "fluency"],
        "appreciation": ["factuality", "textual_evidence", "depth"],
        "creation": ["constraint_following", "style", "coherence", "originality"],
    }
    rows: list[dict[str, Any]] = []
    key: dict[str, str] = {}
    for model_name, path in generation_files.items():
        for row in read_jsonl(path):
            blind_id = hashlib.sha256(f"{seed}:{model_name}:{row['id']}".encode()).hexdigest()[:16]
            key[blind_id] = model_name
            rows.append(
                {
                    "blind_id": blind_id,
                    "question_id": row["id"],
                    "task": row.get("task"),
                    "prompt": row.get("prompt") or row.get("input") or [
                        message for message in row.get("messages", [])
                        if message.get("role") != "assistant"
                    ],
                    "response": row.get("response"),
                    "reference": _reference_text(row),
                    "scores": {name: None for name in dimensions.get(str(row.get("task")), [])},
                    "review_notes": "",
                }
            )
    rng.shuffle(rows)
    write_jsonl(output_path, rows)
    key_path = Path(output_path).with_suffix(".key.json")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": len(rows), "key_path": str(key_path)}


def summarize_blinded_review(
    review_path: str | Path, key_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Validate completed 1-5 scores and aggregate by hidden model and task."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    incomplete = 0
    for row in read_jsonl(review_path):
        model = key.get(str(row.get("blind_id")))
        if model is None:
            raise ValueError(f"Unknown blind_id: {row.get('blind_id')}")
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores or any(value is None for value in scores.values()):
            incomplete += 1
            continue
        values = [float(value) for value in scores.values()]
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("Every human rubric score must be between 1 and 5")
        grouped[str(model)][str(row.get("task", "unknown"))].append(mean(values))
    summary: dict[str, Any] = {"models": {}, "incomplete_rows": incomplete}
    for model, tasks in grouped.items():
        all_values = [value for values in tasks.values() for value in values]
        summary["models"][model] = {
            "overall_mean": mean(all_values),
            "count": len(all_values),
            "tasks": {
                task: {"mean": mean(values), "count": len(values)}
                for task, values in sorted(tasks.items())
            },
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
