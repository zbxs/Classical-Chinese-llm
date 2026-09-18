from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from classical_llm.utils.io import read_jsonl, write_jsonl


def build_token_mixture(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer_name_or_path: str,
    total_tokens: int,
    fractions: dict[str, float],
    seed: int = 20260912,
    max_document_epochs: int = 2,
    max_chunk_tokens: int = 2048,
) -> dict[str, Any]:
    """Build an auditable mixture whose quotas are measured in model tokens."""
    from transformers import AutoTokenizer

    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("Mixture fractions must sum to 1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_path):
        category = str(row.get("category", ""))
        if category in fractions:
            token_ids = tokenizer.encode(str(row["text"]), add_special_tokens=False)
            for chunk_index, start in enumerate(range(0, len(token_ids), max_chunk_tokens)):
                chunk = token_ids[start : start + max_chunk_tokens]
                if not chunk:
                    continue
                item = dict(row)
                item["text"] = tokenizer.decode(chunk, skip_special_tokens=True)
                item["token_count"] = len(chunk) + 1
                item["parent_source_id"] = row.get("source_id") or row.get("id")
                item["chunk_index"] = chunk_index
                by_category[category].append(item)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    achieved: dict[str, int] = {}
    for category, fraction in fractions.items():
        quota = round(total_tokens * fraction)
        pool = by_category.get(category, [])
        if not pool:
            raise ValueError(f"No accepted documents for category {category}")
        rng.shuffle(pool)
        count = 0
        for epoch in range(max_document_epochs):
            for row in pool:
                if count >= quota:
                    break
                item = dict(row)
                remaining = quota - count
                if int(item["token_count"]) > remaining > 1:
                    ids = tokenizer.encode(str(item["text"]), add_special_tokens=False)
                    ids = ids[: remaining - 1]
                    item["text"] = tokenizer.decode(ids, skip_special_tokens=True)
                    item["token_count"] = len(ids) + 1
                item["mixture_epoch"] = epoch
                selected.append(item)
                count += int(item["token_count"])
            if count >= quota:
                break
        if count < quota:
            raise ValueError(
                f"Category {category} supplies {count:,} tokens after {max_document_epochs} epochs; "
                f"quota is {quota:,}. Add licensed data or lower the budget."
            )
        achieved[category] = count

    rng.shuffle(selected)
    count = write_jsonl(output_path, selected)
    report = {
        "rows": count,
        "requested_total_tokens": total_tokens,
        "achieved_tokens": achieved,
        "fractions": fractions,
        "tokenizer": tokenizer_name_or_path,
        "seed": seed,
        "max_chunk_tokens": max_chunk_tokens,
    }
    report_path = Path(output_path).with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
