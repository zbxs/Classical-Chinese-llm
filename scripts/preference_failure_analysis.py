"""Build deterministic diagnostics and position-swapped judge inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classical_llm.evaluation.preference_audit import (
    build_judge_presentations,
    read_jsonl,
    summarize_pairs,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/repair-dpo-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/preference-diagnosis-v1")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    splits = {
        split: read_jsonl(args.data / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    report = {
        "seed": args.seed,
        "source": str(args.data),
        "splits": {name: summarize_pairs(rows) for name, rows in splits.items()},
    }
    presentations = build_judge_presentations(splits, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "judge_input.jsonl", presentations)
    report["judge_presentations"] = len(presentations)
    (args.output / "deterministic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
