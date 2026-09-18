"""Load one local model, run a short CUDA generation, and report peak VRAM."""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="将“学而时习之，不亦说乎”翻译成现代汉语。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
    )
    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    completion = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    report = {
        "model": args.model,
        "completion": completion.strip(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

