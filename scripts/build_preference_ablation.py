"""Merge double judgments and build matched DPO/SFT causal-ablation datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from classical_llm.evaluation.preference_audit import (
    extract_source,
    merge_judgments,
    read_jsonl,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def band(row: dict) -> str:
    score = float(row["rejected_chrf"])
    return "low" if score < 20 else "medium" if score < 40 else "hard"


def stable(rows: list[dict], seed: int, namespace: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{namespace}:{row.get('id', row.get('review_id'))}".encode()
        ).hexdigest(),
    )


def match_bands(
    candidates: list[dict], target: list[dict], seed: int, namespace: str
) -> list[dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in stable(candidates, seed, namespace):
        pools[band(row)].append(row)
    needs = Counter(band(row) for row in target)
    selected: list[dict] = []
    for name, count in sorted(needs.items()):
        selected.extend(pools[name][:count])
    if len(selected) != len(target):
        raise ValueError(f"Could not match difficulty bands: target={needs}, selected={len(selected)}")
    return stable(selected, seed, f"{namespace}:final")


def sft_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "source_id": row.get("source_id"),
        "task": row.get("task", "old_to_modern"),
        "source_text": extract_source(row["prompt"]),
        "prompt_text": row["prompt"],
        "answer_text": row["chosen"],
    }


def grpo_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "source_id": row.get("source_id"),
        "task": row.get("task", "old_to_modern"),
        "source_text": extract_source(row["prompt"]),
        "prompt": row["prompt"],
        "reference": row["chosen"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/repair-dpo-v1")
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "data/preference-diagnosis-v1")
    parser.add_argument("--train-limit", type=int, default=768)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--min-train", type=int, default=256)
    parser.add_argument("--min-validation", type=int, default=64)
    parser.add_argument("--min-position-consistency", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    split_rows = {
        split: read_jsonl(args.pairs / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    all_rows = [row for rows in split_rows.values() for row in rows]
    by_id = {str(row["id"]): row for row in all_rows}
    presentations = read_jsonl(args.diagnosis / "judge_input.jsonl")
    judgments = read_jsonl(args.diagnosis / "judge_output.jsonl")
    merged, judge_report = merge_judgments(presentations, judgments)
    expected_pairs = sum(len(rows) for rows in split_rows.values())
    if judge_report["complete_pairs"] != expected_pairs:
        raise ValueError(
            f"Incomplete double judgments: {judge_report['complete_pairs']} != {expected_pairs}"
        )
    if judge_report["position_consistency_rate"] < args.min_position_consistency:
        raise ValueError(
            "Judge position consistency below gate: "
            f"{judge_report['position_consistency_rate']:.3f} < {args.min_position_consistency:.3f}"
        )
    merged_by_id = {str(row["pair_id"]): row for row in merged}

    selected: dict[str, list[dict]] = {}
    controls: dict[str, list[dict]] = {}
    limits = {"train": args.train_limit, "validation": args.validation_limit}
    minimums = {"train": args.min_train, "validation": args.min_validation}
    for split, rows in split_rows.items():
        high = [row for row in rows if merged_by_id.get(str(row["id"]), {}).get("high_confidence")]
        high = stable(high, args.seed, f"high:{split}")[: limits[split]]
        if len(high) < minimums[split]:
            raise ValueError(
                f"Not enough high-confidence {split} pairs: {len(high)} < {minimums[split]}"
            )
        selected[split] = high
        high_ids = {str(row["id"]) for row in high}
        candidate_controls = [
            row
            for row in rows
            if str(row["id"]) not in high_ids
            and 0.8 <= float(merged_by_id.get(str(row["id"]), {}).get("features", {}).get(
                "length_ratio", 0
            )) <= 1.25
        ]
        controls[split] = match_bands(
            candidate_controls, high, args.seed, f"control:{split}"
        )

    output = args.diagnosis / "ablation"
    for split in ("train", "validation"):
        write_jsonl(output / "dpo_high_confidence" / f"{split}.jsonl", selected[split])
        write_jsonl(output / "reward_high_confidence" / f"{split}.jsonl", selected[split])
        write_jsonl(output / "dpo_length_matched_unjudged" / f"{split}.jsonl", controls[split])
        write_jsonl(output / "sft_high_confidence" / f"{split}.jsonl", map(sft_row, selected[split]))
        write_jsonl(output / "grpo_high_confidence" / f"{split}.jsonl", map(grpo_row, selected[split]))

    review_pool = []
    for label, rows in (
        ("high_confidence", [*selected["train"], *selected["validation"]]),
        (
            "ambiguous_or_reversed",
            [
                by_id[item["pair_id"]]
                for item in merged
                if not item["high_confidence"] and item["pair_id"] in by_id
            ],
        ),
    ):
        for row in stable(rows, args.seed, f"review:{label}")[:50]:
            chosen_is_a = random.Random(f"{args.seed}:{row['id']}").random() < 0.5
            review_pool.append({
                "review_id": f"{label}:{row['id']}",
                "stratum": label,
                "source": extract_source(row["prompt"]),
                "option_a": row["chosen"] if chosen_is_a else row["rejected"],
                "option_b": row["rejected"] if chosen_is_a else row["chosen"],
                "human_winner": "",
                "human_fidelity_a": None,
                "human_fidelity_b": None,
                "notes": "",
            })
    review_pool = stable(review_pool, args.seed, "review-final")
    key = []
    for item in review_pool:
        pair_id = item["review_id"].split(":", 1)[1]
        row = by_id[pair_id]
        key.append({
            "review_id": item["review_id"],
            "chosen_is": "A" if item["option_a"] == row["chosen"] else "B",
            "judge": merged_by_id[pair_id],
        })
    write_jsonl(args.diagnosis / "blind_review.jsonl", review_pool)
    write_jsonl(args.diagnosis / "blind_review_key.jsonl", key)

    deterministic = json.loads(
        (args.diagnosis / "deterministic_report.json").read_text(encoding="utf-8")
    )
    report = {
        "seed": args.seed,
        "deterministic": deterministic,
        "judge": judge_report,
        "selection": {
            split: {
                "high_confidence": len(selected[split]),
                "difficulty_bands": dict(Counter(band(row) for row in selected[split])),
                "matched_control": len(controls[split]),
            }
            for split in ("train", "validation")
        },
        "blind_review_rows": len(review_pool),
    }
    (args.diagnosis / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
