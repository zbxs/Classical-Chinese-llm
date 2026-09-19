from __future__ import annotations

from typing import Any

from classical_llm.training.common import (
    base_training_args,
    ensure_fsdp_module_compat,
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


def run_dpo(config_path: str) -> dict[str, Any]:
    ensure_fsdp_module_compat()
    from trl import DPOConfig, DPOTrainer

    config = load_yaml(config_path)
    model_path = resolve_model(config)
    output_dir = resolve_project_path(config["output_dir"])
    save_run_metadata(output_dir, config, model_path, "starting")
    dataset = load_json_dataset(config["dataset_path"], config.get("validation_path"))
    tokenizer = load_tokenizer(model_path)
    model = load_causal_model(model_path, config)
    reference_model = None
    if config.get("reference_model_name_or_path"):
        reference_config = dict(config)
        reference_config["model_name_or_path"] = config["reference_model_name_or_path"]
        reference_model_path = resolve_model(reference_config)
        reference_model = load_causal_model(reference_model_path, reference_config)
        reference_model.requires_grad_(False)
        reference_model.eval()
    has_eval = "validation" in dataset
    args = DPOConfig(
        **base_training_args(config, has_eval),
        max_length=int(config.get("max_length", 1024)),
        beta=float(config.get("beta", 0.1)),
        loss_type=[str(config.get("loss_type", "sigmoid"))],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=lora_for_new_model(model, config),
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    return finish_training(trainer, tokenizer, config, model_path, result)
