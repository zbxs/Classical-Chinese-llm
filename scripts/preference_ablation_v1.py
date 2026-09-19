"""Causal SFT/DPO ablations on double-judged, length-balanced preference data."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import yaml
from repair_capability_v2 import REPORT as CAPABILITY_REPORT
from repair_capability_v2 import evaluate

from classical_llm.training.dpo import run_dpo
from classical_llm.training.sft import run_sft

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "outputs/repair-capability-v2/base_plain/best"
DATA = ROOT / "data/preference-diagnosis-v1/ablation"
RUN = ROOT / "outputs/preference-ablation-v1"
REPORT = ROOT / "reports/generated/preference_ablation_v1.json"
DEFAULT_SEEDS = (20260919, 20260920, 20260921)


VARIANTS = {
    "sft_high_confidence": {
        "stage": "sft",
        "data": "sft_high_confidence",
        "learning_rate": 5e-6,
    },
    "dpo_high_confidence_b005": {
        "stage": "dpo",
        "data": "dpo_high_confidence",
        "learning_rate": 5e-6,
        "beta": 0.05,
    },
    "dpo_high_confidence_b020": {
        "stage": "dpo",
        "data": "dpo_high_confidence",
        "learning_rate": 2e-6,
        "beta": 0.20,
    },
    "dpo_length_matched_unjudged": {
        "stage": "dpo",
        "data": "dpo_length_matched_unjudged",
        "learning_rate": 5e-6,
        "beta": 0.05,
    },
}


def save(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_config(name: str, variant: dict, seed: int) -> Path:
    output = RUN / name
    config = {
        "stage": variant["stage"],
        "model_name_or_path": str(START),
        "dataset_path": str(DATA / variant["data"] / "train.jsonl"),
        "validation_path": str(DATA / variant["data"] / "validation.jsonl"),
        "output_dir": str(output),
        "seed": seed,
        "use_qlora": True,
        "load_in_4bit": True,
        "learning_rate": variant["learning_rate"],
        "max_steps": 64,
        "gradient_checkpointing": True,
        "logging_steps": 8,
        "eval_steps": 32,
        "save_steps": 32,
        "save_total_limit": 2,
        "warmup_steps": 4,
        "bf16": True,
        "resume_from_checkpoint": "auto",
    }
    if variant["stage"] == "sft":
        config.update(
            format_style="plain",
            max_seq_length=512,
            eos_token="<|endoftext|>",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
        )
    else:
        config.update(
            reference_model_name_or_path=str(START),
            max_length=512,
            beta=variant["beta"],
            loss_type="sigmoid",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
        )
    path = RUN / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def complete_output(path: Path) -> bool:
    metadata = path / "run_metadata.json"
    if not metadata.exists() or not (path / "best/adapter_config.json").exists():
        return False
    return json.loads(metadata.read_text(encoding="utf-8")).get("status") == "complete"


def summarize(models: dict, variants: list[str], seeds: tuple[int, ...]) -> dict:
    summary = {}
    for variant in variants:
        rows = [models[f"{variant}_s{seed}"] for seed in seeds if f"{variant}_s{seed}" in models]
        if not rows:
            continue
        values = {
            "translation_chrf": [row["translation"]["mean_chrf"] for row in rows],
            "translation_stop_rate": [row["translation"]["stop_rate"] for row in rows],
            "translation_repetition": [row["translation"]["mean_repetition"] for row in rows],
            "general_pass_rate": [row["general"]["pass_rate"] for row in rows],
        }
        summary[variant] = {
            key: {
                "mean": statistics.fmean(items),
                "stdev": statistics.stdev(items) if len(items) > 1 else 0.0,
                "values": items,
            }
            for key, items in values.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    analysis_path = ROOT / "data/preference-diagnosis-v1/analysis_report.json"
    if not analysis_path.exists():
        raise FileNotFoundError("Run build_preference_ablation.py first")
    baseline = json.loads(CAPABILITY_REPORT.read_text(encoding="utf-8"))["models"]["base_plain"]
    if REPORT.exists():
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
    else:
        payload = {
            "status": "starting",
            "start_model": str(START),
            "seeds": list(seeds),
            "variants": args.variants,
            "data_analysis": json.loads(analysis_path.read_text(encoding="utf-8")),
            "baseline": baseline,
            "training": {},
            "models": {},
            "summary": {},
        }
        save(payload)

    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        for seed in seeds:
            name = f"{variant_name}_s{seed}"
            output = RUN / name
            payload["status"] = f"training_{name}"
            save(payload)
            config_path = make_config(name, variant, seed)
            if not complete_output(output):
                payload["training"][name] = (
                    run_sft(str(config_path))
                    if variant["stage"] == "sft"
                    else run_dpo(str(config_path))
                )
                save(payload)
            payload["status"] = f"evaluating_{name}"
            save(payload)
            if name not in payload["models"]:
                payload["models"][name] = evaluate(name, output / "best", "plain")
                save(payload)
            gc.collect()
            torch.cuda.empty_cache()

    payload["summary"] = summarize(payload["models"], args.variants, seeds)
    payload["status"] = "complete_needs_causal_interpretation"
    save(payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
