from __future__ import annotations

from typing import Any

from classical_llm.training.common import (
    base_training_args,
    finish_training,
    load_causal_model,
    load_json_dataset,
    load_tokenizer,
    lora_for_new_model,
    resolve_model,
    resolve_resume_checkpoint,
    save_run_metadata,
)
from classical_llm.utils.config import load_yaml, resolve_project_path


def run_sft(config_path: str) -> dict[str, Any]:
    from trl import SFTConfig, SFTTrainer

    config = load_yaml(config_path)
    model_path = resolve_model(config)
    output_dir = resolve_project_path(config["output_dir"])
    save_run_metadata(output_dir, config, model_path, "starting")
    dataset = load_json_dataset(config["dataset_path"], config.get("validation_path"))
    original_columns = dataset["train"].column_names
    dataset = dataset.map(
        lambda row: {
            "prompt": row["messages"][:-1],
            "completion": [row["messages"][-1]],
        },
        remove_columns=original_columns,
    )
    tokenizer = load_tokenizer(model_path)
    model = load_causal_model(model_path, config)
    has_eval = "validation" in dataset
    args = SFTConfig(
        **base_training_args(config, has_eval),
        max_length=int(config.get("max_seq_length", 1024)),
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=lora_for_new_model(model, config),
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    return finish_training(trainer, tokenizer, config, model_path, result)
