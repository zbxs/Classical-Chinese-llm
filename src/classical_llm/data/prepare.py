from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from classical_llm.data.dedup import DuplicateDetector
from classical_llm.data.normalize import normalize_text
from classical_llm.data.quality import assess_quality, looks_stem_or_technical
from classical_llm.data.split import stable_split
from classical_llm.utils.io import read_jsonl, sha256_text, write_jsonl


def prepare_corpus(
    inputs: list[str | Path], output_dir: str | Path, seed: int = 20260912
) -> dict[str, Any]:
    """Normalize, audit, deduplicate and group-split unified JSONL corpora."""
    detector = DuplicateDetector()
    accepted: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rejection_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for input_path in inputs:
        for raw in read_jsonl(input_path):
            text = normalize_text(str(raw.get("text") or raw.get("content") or ""))
            category = str(raw.get("category", "general_zh"))
            decision = assess_quality(text, category)
            if not decision.accepted:
                rejection_counts.update(decision.reasons)
                continue
            duplicate, reason = detector.is_duplicate(text)
            if duplicate:
                rejection_counts[reason or "duplicate"] += 1
                continue

            record = dict(raw)
            record["text"] = text
            record["category"] = (
                "stem_technical"
                if category == "general_zh" and looks_stem_or_technical(text)
                else category
            )
            record["quality_score"] = decision.score
            record["content_hash"] = sha256_text(text)
            record["split"] = stable_split(record, seed=seed)
            accepted[record["split"]].append(record)
            source_counts[str(record.get("source", "unknown"))] += 1

    target = Path(output_dir)
    counts = {split: write_jsonl(target / f"{split}.jsonl", rows) for split, rows in accepted.items()}
    report = {
        "inputs": [str(Path(path)) for path in inputs],
        "accepted": counts,
        "rejections": dict(rejection_counts),
        "sources": dict(source_counts),
        "seed": seed,
    }
    (target / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

