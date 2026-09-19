"""Conservative DPO repair using real errors sampled from the repaired SFT model."""
from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sacrebleu
import torch
import yaml
from repair_capability_v2 import DATA as EVAL_DATA
from repair_capability_v2 import REPORT as CAPABILITY_REPORT
from repair_capability_v2 import evaluate

from classical_llm.evaluation.generate import _load_model
from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_tokenizer
from classical_llm.training.dpo import run_dpo
from classical_llm.utils.io import read_jsonl, write_jsonl

configure_project_environment()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "outputs/repair-capability-v2/base_plain/best"
DATA = ROOT / "data/repair-dpo-v1"
RUN = ROOT / "outputs/repair-dpo-v1"
REPORT = ROOT / "reports/generated/repair_dpo_v1.json"
SEED = 20260918


def save(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _repetition(text: str) -> float:
    grams = [text[index:index + 4] for index in range(max(0, len(text) - 3))]
    return 1 - len(set(grams)) / max(1, len(grams))


def sample_errors(split: str, candidate_limit: int) -> list[dict]:
    rows = list(read_jsonl(EVAL_DATA / f"{split}.jsonl"))
    random.Random(f"{SEED}:{split}").shuffle(rows)
    rows = rows[:candidate_limit]
    tokenizer = load_tokenizer(str(START))
    model = _load_model(str(START))
    model.eval()
    stops = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
    output: list[dict] = []
    batch_size = 12
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        prompts = [row["prompt_text"] for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=384, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs.input_ids.shape[1]
        for row, sequence in zip(batch, generated, strict=True):
            tokens = sequence[prompt_width:]
            response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            reference = row["answer_text"].strip()
            chrf = sacrebleu.sentence_chrf(response, [reference]).score
            source_copy = sacrebleu.sentence_chrf(response, [row["source_text"]]).score
            output.append({
                "id": row["id"],
                "prompt": row["prompt_text"],
                "chosen": reference,
                "rejected": response,
                "task": "old_to_modern",
                "source_id": row.get("source_id"),
                "rejected_origin": "repaired_sft_greedy_bf16",
                "rejected_chrf": chrf,
                "source_copy_chrf": source_copy,
                "tokens": len(tokens),
                "stopped": bool(len(tokens) and int(tokens[-1]) in stops),
                "replacement": "�" in response,
                "repetition": _repetition(response),
            })
        print(f"generated {split}: {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def select_pairs(rows: list[dict], limit: int) -> tuple[list[dict], dict]:
    eligible = [row for row in rows if row["stopped"] and not row["replacement"]
                and row["rejected"] and row["chosen"] != row["rejected"]
                and row["repetition"] <= 0.10 and 3 <= row["rejected_chrf"] <= 68]
    bands = {
        "low": sorted((row for row in eligible if row["rejected_chrf"] < 20),
                      key=lambda row: row["rejected_chrf"]),
        "medium": sorted((row for row in eligible if 20 <= row["rejected_chrf"] < 40),
                         key=lambda row: row["rejected_chrf"]),
        "hard": sorted((row for row in eligible if row["rejected_chrf"] >= 40),
                       key=lambda row: row["rejected_chrf"]),
    }
    targets = {"low": int(limit * 0.30), "medium": int(limit * 0.50)}
    targets["hard"] = limit - targets["low"] - targets["medium"]
    chosen: list[dict] = []
    leftovers: list[dict] = []
    rng = random.Random(SEED)
    for name, band in bands.items():
        rng.shuffle(band)
        chosen.extend(band[:targets[name]])
        leftovers.extend(band[targets[name]:])
    if len(chosen) < limit:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:limit - len(chosen)])
    rng.shuffle(chosen)
    selected = chosen[:limit]
    stats = {
        "candidates": len(rows),
        "eligible": len(eligible),
        "selected": len(selected),
        "selected_bands": {
            "low": sum(row["rejected_chrf"] < 20 for row in selected),
            "medium": sum(20 <= row["rejected_chrf"] < 40 for row in selected),
            "hard": sum(row["rejected_chrf"] >= 40 for row in selected),
        },
        "mean_rejected_chrf": sum(row["rejected_chrf"] for row in selected) / max(1, len(selected)),
        "mean_source_copy_chrf": sum(row["source_copy_chrf"] for row in selected) / max(1, len(selected)),
    }
    return selected, stats


def prepare_preferences() -> dict:
    if DATA.exists() or RUN.exists():
        raise FileExistsError("Preserve repair-dpo-v1; use a new version for another run")
    DATA.mkdir(parents=True)
    RUN.mkdir(parents=True)
    train, train_stats = select_pairs(sample_errors("train", 3072), 2048)
    validation, validation_stats = select_pairs(sample_errors("validation", 256), 192)
    if len(train) != 2048 or len(validation) < 128:
        raise ValueError(f"Insufficient real-error pairs: train={len(train)}, validation={len(validation)}")
    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "validation.jsonl", validation)
    audit = {"seed": SEED, "generator": str(START), "train": train_stats,
             "validation": validation_stats, "sample": train[:50]}
    (DATA / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def train() -> dict:
    config = {
        "stage": "dpo",
        "model_name_or_path": str(START),
        "reference_model_name_or_path": str(START),
        "dataset_path": str(DATA / "train.jsonl"),
        "validation_path": str(DATA / "validation.jsonl"),
        "output_dir": str(RUN / "model"),
        "seed": SEED,
        "max_length": 512,
        "beta": 0.05,
        "loss_type": "sigmoid",
        "use_qlora": True,
        "load_in_4bit": True,
        "learning_rate": 5e-6,
        "max_steps": 128,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "logging_steps": 8,
        "eval_steps": 64,
        "save_steps": 64,
        "save_total_limit": 2,
        "warmup_steps": 8,
        "bf16": True,
        "resume_from_checkpoint": None,
    }
    path = RUN / "dpo.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return run_dpo(str(path))


def main() -> None:
    payload = {"status": "building_real_error_preferences", "data": {}, "training": {},
               "evaluation": {}, "decision": {}}
    save(payload)
    payload["data"] = prepare_preferences()
    payload["status"] = "training"
    save(payload)
    payload["training"] = train()
    payload["status"] = "evaluating"
    save(payload)
    payload["evaluation"] = evaluate("dpo_repair_v1", RUN / "model/best", "plain")
    baseline = json.loads(CAPABILITY_REPORT.read_text(encoding="utf-8"))["models"]["base_plain"]
    current = payload["evaluation"]
    t, g = current["translation"], current["general"]
    accepted = (t["stop_rate"] == 1 and t["replacement_rate"] == 0
                and t["mean_repetition"] <= 0.01
                and t["mean_chrf"] >= baseline["translation"]["mean_chrf"]
                and g["pass_rate"] >= baseline["general"]["pass_rate"] - 0.05)
    payload["decision"] = {"accepted": accepted, "baseline": baseline,
                           "reason": "must improve chrF without degrading stopping or general checks"}
    payload["status"] = "complete_accepted" if accepted else "complete_rejected"
    save(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
