from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classical_llm.utils.io import read_jsonl, sha256_text, write_jsonl


def build_domain_evaluation(
    candidate_paths: list[str | Path],
    output_path: str | Path,
    per_task: int = 100,
    seed: int = 20260912,
) -> dict[str, int]:
    """Select a balanced, source-held-out evaluation set.

    Candidates must have ``split=test`` and a reference/rubric. This function
    deliberately refuses to fabricate test answers from training data.
    """
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for path in candidate_paths:
        for row in read_jsonl(path):
            if row.get("split") != "test":
                continue
            task = str(row.get("task", ""))
            prompt = row.get("prompt") or row.get("input")
            reference = row.get("reference") or row.get("output") or row.get("answer")
            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                if prompt is None:
                    prompt = messages[:-1]
                if reference is None and isinstance(messages[-1], dict):
                    reference = messages[-1].get("content")
            if not task or not prompt or not reference:
                continue
            identity = sha256_text(f"{task}\0{prompt}")
            if identity in seen:
                continue
            seen.add(identity)
            item = dict(row)
            item["id"] = identity
            grouped[task].append(item)

    selected: list[dict[str, Any]] = []
    expected = {"old_to_modern", "modern_to_old", "appreciation", "creation"}
    for task in sorted(expected):
        rows = grouped.get(task, [])
        rng.shuffle(rows)
        if len(rows) < per_task:
            raise ValueError(f"Evaluation task {task} has {len(rows)} candidates, needs {per_task}")
        selected.extend(rows[:per_task])
    rng.shuffle(selected)
    write_jsonl(output_path, selected)
    return dict(Counter(str(row["task"]) for row in selected))


def build_perplexity_sets(
    input_path: str | Path,
    output_dir: str | Path,
    per_category: int = 200,
    seed: int = 20260912,
) -> dict[str, int]:
    """Freeze balanced held-out text sets for forgetting measurements."""
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_path):
        category = str(row.get("category", ""))
        if category in {"classical", "general_zh", "general_en"} and row.get("text"):
            grouped[category].append(row)
    target = Path(output_dir)
    counts: dict[str, int] = {}
    for category in ("classical", "general_zh", "general_en"):
        rows = grouped.get(category, [])
        rng.shuffle(rows)
        if not rows:
            raise ValueError(f"No held-out rows found for {category}")
        selected = rows[:per_category]
        counts[category] = write_jsonl(target / f"{category}.jsonl", selected)
    return counts
