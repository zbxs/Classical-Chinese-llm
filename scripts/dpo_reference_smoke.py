"""Two-step smoke test for explicit frozen-reference DPO loading."""
from __future__ import annotations

from pathlib import Path

import yaml

from classical_llm.training.dpo import run_dpo
from classical_llm.utils.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "outputs/repair-capability-v2/base_plain/best"
DATA = ROOT / "data/dpo-reference-smoke"
RUN = ROOT / "outputs/dpo-reference-smoke"


def main() -> None:
    if DATA.exists() or RUN.exists():
        raise FileExistsError("Preserve the smoke run; use a new name to repeat it")
    DATA.mkdir(parents=True)
    rows = []
    for row in list(read_jsonl(ROOT / "data/repair-capability-v2/train.jsonl"))[:16]:
        rows.append({"prompt": row["prompt_text"], "chosen": row["answer_text"],
                     "rejected": row["source_text"]})
    write_jsonl(DATA / "train.jsonl", rows)
    config = {
        "stage": "dpo",
        "model_name_or_path": str(START),
        "reference_model_name_or_path": str(START),
        "dataset_path": str(DATA / "train.jsonl"),
        "output_dir": str(RUN),
        "seed": 20260918,
        "max_length": 512,
        "beta": 0.05,
        "loss_type": "sigmoid",
        "use_qlora": True,
        "load_in_4bit": True,
        "learning_rate": 5e-6,
        "max_steps": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_steps": 2,
        "save_total_limit": 1,
        "warmup_steps": 0,
        "bf16": True,
    }
    path = DATA / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    print(run_dpo(str(path)))


if __name__ == "__main__":
    main()
