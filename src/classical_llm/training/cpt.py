from __future__ import annotations

from typing import Any

from classical_llm.training.common import (
    base_training_args,
    finish_training,
    load_causal_model,
    load_json_dataset,
    load_tokenizer,
    lora_config,
    resolve_model,
    resolve_resume_checkpoint,
    save_run_metadata,
)
from classical_llm.utils.config import load_yaml, resolve_project_path


def _tokenize_and_block(dataset, tokenizer, block_size: int):
    """Pack documents with EOS boundaries before collation.

    This avoids TRL padding-free packing, whose flattened attention requires a
    FlashAttention implementation that is unavailable in the Windows profile.
    """

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing continual-pretraining text",
    )
    eos_id = tokenizer.eos_token_id

    def group(batch):
        stream: list[int] = []
        for input_ids in batch["input_ids"]:
            stream.extend(input_ids)
            if eos_id is not None:
                stream.append(eos_id)
        blocks = [stream[index : index + block_size] for index in range(0, len(stream), block_size)]
        blocks = [block for block in blocks if len(block) >= 8]
        return {"input_ids": blocks, "attention_mask": [[1] * len(block) for block in blocks]}

    return tokenized.map(group, batched=True, desc="Building EOS-separated token blocks")


def run_cpt(config_path: str) -> dict[str, Any]:
    """Run parameter-efficient causal continual pretraining on raw text."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    config = load_yaml(config_path)
    model_path = resolve_model(config)
    output_dir = resolve_project_path(config["output_dir"])
    save_run_metadata(output_dir, config, model_path, "starting")
    dataset = load_json_dataset(config["dataset_path"], config.get("validation_path"))
    tokenizer = load_tokenizer(model_path)
    dataset = _tokenize_and_block(dataset, tokenizer, int(config.get("max_seq_length", 1024)))
    model = load_causal_model(model_path, config)
    peft_config = lora_config(config)
    if peft_config is not None:
        from peft import get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True))
        )
        model = get_peft_model(model, peft_config)
    has_eval = "validation" in dataset and len(dataset["validation"]) > 0
    args = TrainingArguments(**base_training_args(config, has_eval))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if has_eval else None,
        processing_class=tokenizer,
        data_collator=collator,
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    return finish_training(trainer, tokenizer, config, model_path, result)
