from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from classical_llm.utils.io import read_jsonl, sha256_text, write_jsonl

_NEGATION_SWAPS = [("不", ""), ("未", "已"), ("无", "有"), ("莫", "皆")]


def _corrupt(answer: str, task: str, rng: random.Random) -> tuple[str, str]:
    """Create a fluent-ish hard negative with an auditable error label."""
    options: list[tuple[str, str]] = []
    for old, new in _NEGATION_SWAPS:
        if old in answer:
            options.append((answer.replace(old, new, 1), "wrong_negation"))
    number = re.search(r"[一二三四五六七八九十百千万两0-9]+", answer)
    if number:
        replacement = "三" if number.group() != "三" else "五"
        options.append((answer[: number.start()] + replacement + answer[number.end() :], "wrong_number"))
    clauses = [part for part in re.split(r"(?<=[。！？；])", answer) if part]
    if len(clauses) >= 2:
        options.append(("".join(clauses[:-1]), "omission"))
    if task in {"modern_to_old", "creation"}:
        options.append((answer + "总的来说，这个结果非常不错。", "modern_style_intrusion"))
    if task == "appreciation":
        options.append(("作品语言优美，感情真挚，具有很高的艺术价值。", "unsupported_appreciation"))
    if not options:
        cutoff = max(1, int(len(answer) * 0.62))
        options.append((answer[:cutoff] + "。", "truncation"))
    candidate, error = rng.choice(options)
    if candidate.strip() == answer.strip():
        candidate = answer[: max(1, len(answer) // 2)] + "。"
        error = "truncation"
    return candidate, error


def build_preferences(
    sft_path: str | Path,
    output_dir: str | Path,
    limit: int = 20_000,
    seed: int = 20260912,
) -> dict[str, Any]:
    """Create preference pairs from grounded SFT references.

    Model-sampled rejected answers can be supplied later in the ``rejected``
    field. Otherwise a controlled, labeled corruption is created.
    """
    rng = random.Random(seed)
    rows = list(read_jsonl(sft_path))
    rng.shuffle(rows)
    limit = min(limit, len(rows))
    output: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    for row in rows[:limit]:
        messages = list(row["messages"])
        if not messages or messages[-1].get("role") != "assistant":
            continue
        chosen = str(messages[-1]["content"])
        prompt = messages[:-1]
        rejected = row.get("rejected")
        error = str(row.get("rejected_error", "model_sample"))
        if not rejected:
            rejected, error = _corrupt(chosen, str(row.get("task", "")), rng)
        errors[error] += 1
        output.append(
            {
                "id": sha256_text(f"{row['id']}\0{chosen}\0{rejected}"),
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "task": row.get("task"),
                "source_id": row.get("source_id"),
                "error_types": [error],
            }
        )

    partitions: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in output:
        key = str(row.get("source_id") or row["id"])
        point = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
        partition = "train" if point < 0.80 else "validation" if point < 0.90 else "test"
        partitions[partition].append(row)
    target = Path(output_dir)
    counts = {name: write_jsonl(target / f"{name}.jsonl", data) for name, data in partitions.items()}
    return {"counts": counts, "error_types": dict(errors), "seed": seed}
