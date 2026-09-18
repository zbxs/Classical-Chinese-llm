from __future__ import annotations

from typing import Any

import numpy as np

from classical_llm.training.common import (
    base_training_args,
    finish_training,
    load_json_dataset,
    load_tokenizer,
    lora_config,
    quantization_config,
    resolve_model,
    resolve_resume_checkpoint,
    save_run_metadata,
)
from classical_llm.utils.config import load_yaml, resolve_project_path


def _accuracy(eval_prediction) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple) and len(predictions) >= 2:
        chosen, rejected = predictions[:2]
        return {"preference_accuracy": float(np.mean(np.asarray(chosen) > np.asarray(rejected)))}
    values = np.asarray(predictions)
    if values.ndim >= 2 and values.shape[-1] >= 2:
        return {"preference_accuracy": float(np.mean(values[..., 0] > values[..., 1]))}
    return {}


def run_reward(config_path: str) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    config = load_yaml(config_path)
    model_path = resolve_model(config)
    output_dir = resolve_project_path(config["output_dir"])
    save_run_metadata(output_dir, config, model_path, "starting")
    dataset = load_json_dataset(config["dataset_path"], config.get("validation_path"))
    tokenizer = load_tokenizer(model_path)
    quantization = quantization_config(config)
    kwargs: dict[str, Any] = {"num_labels": 1, "trust_remote_code": False}
    if quantization is not None:
        kwargs.update(quantization_config=quantization, device_map={"": 0})
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    has_eval = "validation" in dataset
    args = RewardConfig(
        **base_training_args(config, has_eval),
        max_length=int(config.get("max_length", 1024)),
    )
    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=lora_config(config, task_type="SEQ_CLS"),
        compute_metrics=_accuracy,
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    return finish_training(trainer, tokenizer, config, model_path, result)
