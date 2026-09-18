from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from classical_llm.training.common import load_tokenizer
from classical_llm.utils.io import read_jsonl, write_jsonl


def _load_model(path: str):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "trust_remote_code": False,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "dtype": torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else None,
    }
    adapter_config = Path(path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **kwargs)
        return PeftModel.from_pretrained(base, path)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def _format_prompt(tokenizer, row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        # Evaluation rows retain the gold assistant answer for scoring. It must
        # never be included in the generation prompt. Keep any earlier turns,
        # but remove the trailing reference answer(s).
        prompt_messages = list(messages)
        while prompt_messages and prompt_messages[-1].get("role") == "assistant":
            prompt_messages.pop()
        if prompt_messages and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

    prompt = row.get("prompt") or row.get("input") or row.get("text")
    if isinstance(prompt, list) and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(prompt)


def generate_responses(
    model_path: str,
    dataset_path: str | Path,
    output_path: str | Path,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_path)
    model = _load_model(model_path)
    model.eval()
    output: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in read_jsonl(dataset_path):
        rendered = _format_prompt(tokenizer, row)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        item = dict(row)
        item["response"] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        item["model_path"] = model_path
        output.append(item)
    elapsed = time.perf_counter() - started
    count = write_jsonl(output_path, output)
    return {"rows": count, "elapsed_seconds": elapsed, "seconds_per_row": elapsed / max(1, count)}
