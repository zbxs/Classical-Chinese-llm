"""Read-only tensor comparison; does not load a GPU or change training outputs."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def compare(left, right):
    assert left.keys() == right.keys(), "Tensor names differ"
    squared_delta = sum(float((left[k].float() - right[k].float()).square().sum()) for k in left)
    return {"equal_tensors": sum(torch.equal(left[k], right[k]) for k in left),
            "tensor_count": len(left), "delta_l2": squared_delta ** 0.5}


def main():
    root = Path(__file__).resolve().parents[1]
    sft = load_file(str(root / "outputs/sft/qwen2.5-0.5b-classical/best/adapter_model.safetensors"))
    result = {}
    for stage in ("sft", "dpo", "grpo"):
        path = root / f"outputs/{stage}/qwen2.5-0.5b-classical/best"
        tensors = load_file(str(path / "adapter_model.safetensors"))
        entry = {"all_finite": all(bool(torch.isfinite(v).all()) for v in tensors.values()),
                 "parameters": sum(v.numel() for v in tensors.values()),
                 "policy_vs_sft": compare(tensors, sft)}
        ref = path / "ref/adapter_model.safetensors"
        if ref.exists():
            reference = load_file(str(ref))
            entry["reference_vs_sft"] = compare(reference, sft)
        result[stage] = entry
    destination = root / "reports/generated/adapter_audit.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
