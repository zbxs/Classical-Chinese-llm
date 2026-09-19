"""Paired analysis of the preference-free Dr.GRPO frozen evaluation outputs."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from classical_llm.training.grpo import critical_token_reward, source_copy_penalty_reward
from classical_llm.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("base_plain", "drgrpo_reference_proxy", "drgrpo_constraint_aware")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    means = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def model_metrics(rows: list[dict]) -> dict:
    critical = critical_token_reward(
        [row["response"] for row in rows], source_text=[row["source"] for row in rows]
    )
    copy_penalties = source_copy_penalty_reward(
        [row["response"] for row in rows], source_text=[row["source"] for row in rows]
    )
    return {
        "count": len(rows),
        "mean_chrf": sum(float(row["chrf"]) for row in rows) / len(rows),
        "mean_critical_token_reward": sum(critical) / len(critical),
        "mean_source_copy_penalty": sum(copy_penalties) / len(copy_penalties),
        "exact_source_copy_rate": sum(
            normalized(row["response"]) == normalized(row["source"]) for row in rows
        ) / len(rows),
        "stop_rate": sum(bool(row["stopped"]) for row in rows) / len(rows),
        "replacement_rate": sum(bool(row["replacement"]) for row in rows) / len(rows),
        "mean_repetition": sum(float(row["repetition"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "reports/generated")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/generated/preference_grpo_analysis.json",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    models = {
        name: list(read_jsonl(args.input_dir / f"{name}.translation.jsonl"))
        for name in NAMES
    }
    by_id = {name: {row["id"]: row for row in rows} for name, rows in models.items()}
    ids = sorted(set.intersection(*(set(rows) for rows in by_id.values())))
    if len(ids) != 128:
        raise ValueError(f"Expected 128 shared frozen examples, found {len(ids)}")

    baseline = by_id["base_plain"]
    comparisons = {}
    for name in NAMES[1:]:
        candidate = by_id[name]
        deltas = [float(candidate[item]["chrf"]) - float(baseline[item]["chrf"]) for item in ids]
        comparisons[name] = {
            "paired_mean_chrf_delta": sum(deltas) / len(deltas),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(deltas, args.seed),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
            "positive_ci": bootstrap_mean_ci(deltas, args.seed)[0] > 0,
        }

    report = {
        "seed": args.seed,
        "frozen_examples": len(ids),
        "models": {name: model_metrics(models[name]) for name in NAMES},
        "paired_vs_base_plain": comparisons,
        "interpretation_rule": (
            "Acceptance uses the preregistered point thresholds; paired bootstrap intervals are "
            "reported as uncertainty and are not used to retroactively change those thresholds."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
