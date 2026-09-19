"""Train a calibrated reward model and compare two conservative Dr.GRPO reward mixes."""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from repair_capability_v2 import REPORT as CAPABILITY_REPORT
from repair_capability_v2 import evaluate

from classical_llm.training.common import load_tokenizer
from classical_llm.training.grpo import _load_reward_adapter, run_grpo
from classical_llm.training.reward import run_reward
from classical_llm.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "outputs/repair-capability-v2/base_plain/best"
DATA = ROOT / "data/preference-diagnosis-v1/ablation"
RUN = ROOT / "outputs/preference-grpo-v1"
REPORT = ROOT / "reports/generated/preference_grpo_v1.json"
SEED = 20260919


GRPO_VARIANTS = {
    "drgrpo_reference_only": {
        "reward_functions": [
            "reference_chrf",
            "critical_token",
            "nonempty",
            "anti_repetition",
            "reference_length",
        ],
        "reward_weights": [0.45, 0.20, 0.10, 0.15, 0.10],
    },
    "drgrpo_calibrated_rm": {
        "reward_functions": [
            "reward_model",
            "reference_chrf",
            "critical_token",
            "nonempty",
            "anti_repetition",
            "reference_length",
        ],
        "reward_weights": [0.30, 0.30, 0.15, 0.10, 0.10, 0.05],
    },
}


def save(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def complete_output(path: Path) -> bool:
    metadata = path / "run_metadata.json"
    if not metadata.exists() or not (path / "best/adapter_config.json").exists():
        return False
    return json.loads(metadata.read_text(encoding="utf-8")).get("status") == "complete"


def reward_config() -> Path:
    adapter = json.loads((START / "adapter_config.json").read_text(encoding="utf-8"))
    config = {
        "stage": "reward",
        "model_name_or_path": str(adapter["base_model_name_or_path"]),
        "dataset_path": str(DATA / "reward_high_confidence/train.jsonl"),
        "validation_path": str(DATA / "reward_high_confidence/validation.jsonl"),
        "output_dir": str(RUN / "reward"),
        "seed": SEED,
        "max_length": 512,
        "use_qlora": True,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-5,
        "max_steps": 128,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "logging_steps": 8,
        "eval_steps": 32,
        "save_steps": 32,
        "save_total_limit": 2,
        "warmup_steps": 8,
        "bf16": True,
        "resume_from_checkpoint": "auto",
    }
    path = RUN / "reward.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def calibrate_reward(model_path: Path) -> dict:
    rows = list(read_jsonl(DATA / "reward_high_confidence/validation.jsonl"))
    tokenizer = load_tokenizer(str(model_path))
    model = _load_reward_adapter(str(model_path), {"load_in_4bit": True})
    margins: list[float] = []
    length_gaps: list[int] = []
    batch_size = 16
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        texts = [row["prompt"] + row["chosen"] for row in batch] + [
            row["prompt"] + row["rejected"] for row in batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)
        with torch.inference_mode():
            scores = model(**inputs).logits.float().reshape(-1).cpu().numpy()
        count = len(batch)
        margins.extend((scores[:count] - scores[count:]).tolist())
        length_gaps.extend(
            len(re.sub(r"\s+", "", row["chosen"]))
            - len(re.sub(r"\s+", "", row["rejected"]))
            for row in batch
        )
    margin_array = np.asarray(margins)
    gap_array = np.asarray(length_gaps, dtype=float)
    correlation = float(np.corrcoef(margin_array, gap_array)[0, 1]) if len(rows) > 2 else 0.0
    result = {
        "count": len(rows),
        "preference_accuracy": float(np.mean(margin_array > 0)),
        "mean_margin": float(np.mean(margin_array)),
        "margin_length_gap_correlation": correlation,
        "gate": {
            "minimum_accuracy": 0.70,
            "maximum_abs_length_correlation": 0.35,
        },
    }
    result["accepted"] = (
        result["preference_accuracy"] >= result["gate"]["minimum_accuracy"]
        and abs(correlation) <= result["gate"]["maximum_abs_length_correlation"]
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def grpo_config(name: str, variant: dict) -> Path:
    config = {
        "stage": "grpo",
        "model_name_or_path": str(START),
        "dataset_path": str(DATA / "grpo_high_confidence/train.jsonl"),
        "validation_path": str(DATA / "grpo_high_confidence/validation.jsonl"),
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
    if "reward_model" in variant["reward_functions"]:
        config["reward_model_path"] = str(RUN / "reward/best")
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
    ablation_report = ROOT / "reports/generated/preference_ablation_v1.json"
    if not ablation_report.exists():
        raise FileNotFoundError("Preference causal ablation must complete before GRPO")
    ablation = json.loads(ablation_report.read_text(encoding="utf-8"))
    if ablation.get("status") != "complete_needs_causal_interpretation":
        raise ValueError(f"Preference ablation is incomplete: {ablation.get('status')}")
    baseline = json.loads(CAPABILITY_REPORT.read_text(encoding="utf-8"))["models"]["base_plain"]
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {
        "status": "training_reward_model",
        "baseline": baseline,
        "preference_ablation_summary": ablation["summary"],
        "reward_training": {},
        "reward_calibration": {},
        "grpo_training": {},
        "models": {},
        "decision": {},
    }
    save(payload)

    reward_output = RUN / "reward"
    if not complete_output(reward_output):
        payload["reward_training"] = run_reward(str(reward_config()))
        save(payload)
    payload["status"] = "calibrating_reward_model"
    save(payload)
    payload["reward_calibration"] = calibrate_reward(reward_output / "best")
    save(payload)
    if not payload["reward_calibration"]["accepted"]:
        payload["status"] = "stopped_reward_calibration_failed"
        payload["decision"] = {"accepted": False, "reason": "reward calibration gate failed"}
        save(payload)
        raise RuntimeError("Reward model failed calibration; GRPO was not started")

    for name, variant in GRPO_VARIANTS.items():
        payload["status"] = f"training_{name}"
        save(payload)
        output = RUN / name
        if not complete_output(output):
            payload["grpo_training"][name] = run_grpo(str(grpo_config(name, variant)))
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
        "criteria": "chrF >= baseline, 100% stop, 0 replacement, repetition <= 0.5%, general >= baseline",
    }
    payload["status"] = "complete_accepted" if any(decisions.values()) else "complete_rejected"
    save(payload)
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
