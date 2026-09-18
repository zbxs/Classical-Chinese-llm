"""Verify that the published translation adapter reloads and stops offline."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from classical_llm.evaluation.generate import _load_model
from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_tokenizer

configure_project_environment()
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "artifacts/translation-repaired"


def main() -> None:
    tokenizer = load_tokenizer(str(ADAPTER))
    model = _load_model(str(ADAPTER))
    prompt = "任务：请将下列文言文准确翻译为现代汉语，只输出译文。\n原文：学而时习之，不亦说乎\n译文："
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id)
    tokens = generated[0, inputs.input_ids.shape[1]:]
    result = {"response": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
              "tokens": len(tokens),
              "stopped": bool(len(tokens) and int(tokens[-1]) == tokenizer.eos_token_id)}
    if not result["response"] or not result["stopped"]:
        raise RuntimeError(f"Published adapter verification failed: {result}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
