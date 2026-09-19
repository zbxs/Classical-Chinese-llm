"""Run Dr.GRPO without preference labels after the registered judge gate fails."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
import yaml
from repair_capability_v2 import REPORT as CAPABILITY_REPORT
from repair_capability_v2 import evaluate

from classical_llm.training.grpo import run_grpo
from classical_llm.utils.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "outputs/repair-capability-v2/base_plain/best"
SOURCE_DATA = ROOT / "data/repair-capability-v2"
DATA = ROOT / "data/preference-diagnosis-v1/grpo-reference"
RUN = ROOT / "outputs/preference-grpo-reference-v1"
REPORT = ROOT / "reports/generated/preference_grpo_v1.json"
SEED = 20260919

VARIANTS = {
    "drgrpo_reference_proxy": {
        "reward_functions": [
            "reference_chrf",
            "nonempty",
            "anti_repetition",
            "reference_length",
        ],
        "reward_weights": [0.65, 0.10, 0.10, 0.15],
        "hypothesis": "proxy-heavy control using reference overlap without preference labels",
    },
    "drgrpo_constraint_aware": {
        "reward_functions": [
            "reference_chrf",
            "critical_token",
            "source_copy_penalty",
            "nonempty",
            "anti_repetition",
            "reference_length",
        ],
        "reward_weights": [0.35, 0.20, 0.15, 0.10, 0.10, 0.10],
        "hypothesis": (
            "reduce reliance on a single reference and explicitly protect numbers, negation, "
            "termination, repetition, length, and source-copy failure"
        ),
    },
}


def save(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_rows(rows: list[dict], limit: int, split: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SEED}:grpo-reference:{split}:{row['id']}".encode()
        ).hexdigest(),
    )[:limit]


def prepare_data() -> dict[str, int]:
    limits = {"train": 2048, "validation": 192}
    counts: dict[str, int] = {}
    DATA.mkdir(parents=True, exist_ok=True)
    for split, limit in limits.items():
        output = DATA / f"{split}.jsonl"
        if not output.exists():
            source = stable_rows(list(read_jsonl(SOURCE_DATA / f"{split}.jsonl")), limit, split)
            rows = [
                {
                    "id": row["id"],
                    "source_id": row.get("source_id"),
                    "task": row.get("task", "old_to_modern"),
                    "source_text": row["source_text"],
                    "prompt": row["prompt_text"],
                    "reference": row["answer_text"],
                }
                for row in source
            ]
            write_jsonl(output, rows)
        counts[split] = len(list(read_jsonl(output)))
        if counts[split] != limit:
            raise ValueError(f"Unexpected frozen {split} count: {counts[split]} != {limit}")
    manifest = {
        "seed": SEED,
        "source": str(SOURCE_DATA),
        "selection": "sha256 deterministic order",
        "counts": counts,
        "contains_preference_labels": False,
    }
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts


def complete_output(path: Path) -> bool:
    metadata = path / "run_metadata.json"
    adapter = path / "best/adapter_config.json"
    return (
        metadata.exists()
        and adapter.exists()
        and json.loads(metadata.read_text(encoding="utf-8")).get("status") == "complete"
    )


def grpo_config(name: str, variant: dict) -> Path:
    config = {
        "stage": "grpo",
        "model_name_or_path": str(START),
        "dataset_path": str(DATA / "train.jsonl"),
        "validation_path": str(DATA / "validation.jsonl"),
        "output_dir": str(RUN / name),
        "seed": SEED,
        "max_completion_length": 128,
        "num_generations": 4,
        "generation_batch_size": 4,
        "temperature": 0.8,
        "repetition_penalty": 1.05,
        "beta": 0.01,
        "loss_type": "dr_grpo",
        "scale_rewards": False,
        "mask_truncated_completions": True,
        "reward_functions": variant["reward_functions"],
        "reward_weights": variant["reward_weights"],
        "use_qlora": True,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 5e-6,
        "max_steps": 64,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "logging_steps": 4,
        "save_steps": 32,
        "save_total_limit": 2,
        "warmup_steps": 4,
        "bf16": True,
        "resume_from_checkpoint": "auto",
    }
    path = RUN / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def accepted(model: dict, baseline: dict) -> bool:
    translation, general = model["translation"], model["general"]
    return (
        translation["mean_chrf"] >= baseline["translation"]["mean_chrf"]
        and translation["stop_rate"] == 1
        and translation["replacement_rate"] == 0
        and translation["mean_repetition"] <= 0.005
        and general["pass_rate"] >= baseline["general"]["pass_rate"]
    )


def main() -> None:
    baseline = json.loads(CAPABILITY_REPORT.read_text(encoding="utf-8"))["models"]["base_plain"]
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {
        "status": "preparing_reference_only_grpo",
        "experiment": "preference-label-free Dr.GRPO after judge gate failure",
        "baseline": baseline,
        "data": {},
        "judge_gate": {
            "registered_position_consistency": 0.5995535714285715,
            "registered_minimum": 0.70,
            "independent_v2_pilot_consistency": 0.38,
            "independent_v2_pilot_pairs": 200,
            "preference_labels_accepted": False,
        },
        "reward_model": {
            "trained": False,
            "reason": "judge quality gate failed; preference-derived RM prohibited",
        },
        "variants": VARIANTS,
        "training": {},
        "models": {},
        "decision": {},
    }
    payload["data"] = prepare_data()
    save(payload)

    for name, variant in VARIANTS.items():
        payload["status"] = f"training_{name}"
        save(payload)
        output = RUN / name
        if not complete_output(output):
            payload["training"][name] = run_grpo(str(grpo_config(name, variant)))
            save(payload)
        payload["status"] = f"evaluating_{name}"
        save(payload)
        if name not in payload["models"]:
            payload["models"][name] = evaluate(name, output / "best", "plain")
            save(payload)
        gc.collect()
        torch.cuda.empty_cache()

    decisions = {name: accepted(model, baseline) for name, model in payload["models"].items()}
    payload["decision"] = {
        "accepted": any(decisions.values()),
        "per_model": decisions,
        "criteria": (
            "chrF >= accepted base_plain, 100% stop, 0 replacement, repetition <= 0.5%, "
            "general pass rate >= accepted base_plain"
        ),
        "requires_human_review": True,
    }
    payload["status"] = "complete_accepted" if any(decisions.values()) else "complete_rejected"
    save(payload)
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
