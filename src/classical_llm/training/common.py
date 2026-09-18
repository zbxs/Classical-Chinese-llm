from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from classical_llm.utils.config import resolve_project_path


def ensure_fsdp_module_compat() -> None:
    """Provide TRL's FSDP2 type name when running on PyTorch 2.5.

    TRL 0.29 imports ``FSDPModule`` unconditionally, although single-GPU
    LoRA training does not use FSDP. Alias the type name only to permit imports
    in this single-GPU environment; FSDP1 is NOT equivalent to FSDP2, and this
    workaround must not be used as distributed/FSDP2 support.
    """
    from torch.distributed import fsdp

    if not hasattr(fsdp, "FSDPModule"):
        fsdp.FSDPModule = fsdp.FullyShardedDataParallel


def resolve_model(config: dict[str, Any]) -> str:
    """Use an existing local stage output, otherwise an explicit fallback."""
    primary = str(config["model_name_or_path"])
    candidate = resolve_project_path(primary)
    if candidate.exists():
        return str(candidate)
    if "/" in primary and not primary.startswith(("outputs/", "checkpoints/")):
        return primary
    fallback = config.get("fallback_model_name_or_path")
    if fallback:
        return str(fallback)
    raise FileNotFoundError(f"Model path does not exist and no fallback is configured: {candidate}")


def resolve_resume_checkpoint(config: dict[str, Any]) -> str | None:
    """Resolve an explicit checkpoint or the latest numbered one for long runs."""
    value = config.get("resume_from_checkpoint")
    if value != "auto":
        return str(value) if value else None
    output_dir = resolve_project_path(config["output_dir"])
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return str(max(checkpoints)[1]) if checkpoints else None


def precision_flags(config: dict[str, Any]) -> tuple[bool, bool]:
    cuda = torch.cuda.is_available()
    bf16 = bool(config.get("bf16", True) and cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(config.get("fp16", False) and cuda and not bf16)
    return bf16, fp16


def quantization_config(config: dict[str, Any]):
    if not config.get("load_in_4bit", False):
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit training was requested but CUDA is unavailable")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )


def lora_config(config: dict[str, Any], task_type: str = "CAUSAL_LM"):
    if not config.get("use_qlora", False):
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=int(config.get("lora_r", 32)),
        lora_alpha=int(config.get("lora_alpha", 64)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        target_modules="all-linear",
        task_type=task_type,
    )


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
    tokenizer.padding_side = "left"
    return tokenizer


def set_assistant_end_token(tokenizer, token: str) -> None:
    """Plain-text ChatML template with an explicit learnable assistant terminator.

    Opt-in only. Existing saved runs retain their original templates. Prompts
    keep im_end; only assistant endings change. No tool-call support here.
    """
    if token not in tokenizer.get_vocab():
        raise ValueError(f"End token is not in the tokenizer vocabulary: {token}")
    if token not in {"<|endoftext|>", "<|im_end|>"}:
        raise ValueError("Unsupported assistant end token")
    tokenizer.eos_token = token
    tokenizer.chat_template = (
        "{% if messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}"
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] }}"
        "{% if message['role'] == 'assistant' %}{{ '" + token + "\\n' }}"
        "{% else %}{{ '<|im_end|>\\n' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )


def load_causal_model(model_path: str, config: dict[str, Any]):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"trust_remote_code": False}
    quantization = quantization_config(config)
    if quantization is not None:
        kwargs.update(quantization_config=quantization, device_map={"": 0})
    elif torch.cuda.is_available():
        bf16, _ = precision_flags(config)
        kwargs.update(device_map={"": 0}, dtype=torch.bfloat16 if bf16 else torch.float16)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training

        peft = PeftConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(peft.base_model_name_or_path, **kwargs)
        if config.get("load_in_4bit"):
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True))
            )
        model = PeftModel.from_pretrained(base, model_path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    return model


def lora_for_new_model(model, config: dict[str, Any], task_type: str = "CAUSAL_LM"):
    """Create LoRA only when the input is not already an adapter checkpoint."""
    if hasattr(model, "peft_config"):
        return None
    return lora_config(config, task_type=task_type)


def load_json_dataset(train_path: str, validation_path: str | None = None):
    from datasets import load_dataset

    files = {"train": str(resolve_project_path(train_path))}
    if validation_path:
        validation = resolve_project_path(validation_path)
        if validation.exists() and validation.stat().st_size:
            files["validation"] = str(validation)
    return load_dataset("json", data_files=files)


def base_training_args(config: dict[str, Any], has_eval: bool) -> dict[str, Any]:
    bf16, fp16 = precision_flags(config)
    args = {
        "output_dir": str(resolve_project_path(config["output_dir"])),
        "seed": int(config.get("seed", 20260912)),
        "data_seed": int(config.get("seed", 20260912)),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bf16,
        "fp16": fp16,
        "logging_steps": int(config.get("logging_steps", 10)),
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": int(config.get("eval_steps", 250)) if has_eval else None,
        "save_strategy": "steps",
        "save_steps": int(config.get("save_steps", 250)),
        "save_total_limit": int(config.get("save_total_limit", 3)),
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
        "report_to": "none",
        "optim": "paged_adamw_8bit" if config.get("load_in_4bit") else "adamw_torch",
        "dataloader_num_workers": 0,
    }
    if "max_steps" in config:
        args["max_steps"] = int(config["max_steps"])
    if "warmup_steps" in config:
        args["warmup_steps"] = int(config["warmup_steps"])
    return args


def save_run_metadata(
    output_dir: str | Path, config: dict[str, Any], model_path: str, status: str, metrics: dict | None = None
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "model_name_or_path": model_path,
        "config": config,
        "metrics": metrics or {},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def finish_training(trainer, tokenizer, config: dict[str, Any], model_path: str, result) -> dict[str, Any]:
    output_dir = resolve_project_path(config["output_dir"])
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    save_run_metadata(output_dir, config, model_path, "complete", metrics)
    return metrics
