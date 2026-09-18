from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from classical_llm.evaluation.generate import _load_model
from classical_llm.training.common import load_tokenizer
from classical_llm.utils.io import read_jsonl


def evaluate_perplexity(
    model_path: str,
    dataset_path: str | Path,
    output_path: str | Path,
    max_length: int = 1024,
    max_documents: int | None = None,
) -> dict[str, Any]:
    """Compute token-weighted causal-LM perplexity on a fixed JSONL corpus."""
    tokenizer = load_tokenizer(model_path)
    model = _load_model(model_path)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    documents = 0
    for row in read_jsonl(dataset_path):
        if max_documents is not None and documents >= max_documents:
            break
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        tokenized = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
        ids = tokenized["input_ids"][0]
        for start in range(0, ids.numel(), max_length):
            chunk = ids[start : start + max_length]
            if chunk.numel() < 2:
                continue
            input_ids = chunk.unsqueeze(0).to(model.device)
            with torch.inference_mode():
                loss = model(input_ids=input_ids, labels=input_ids).loss
            predicted = chunk.numel() - 1
            total_nll += float(loss) * predicted
            total_tokens += predicted
        documents += 1
    if total_tokens == 0:
        raise ValueError("No usable tokens found in perplexity dataset")
    mean_nll = total_nll / total_tokens
    result = {
        "model_path": model_path,
        "dataset_path": str(dataset_path),
        "documents": documents,
        "predicted_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
        "max_length": max_length,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
