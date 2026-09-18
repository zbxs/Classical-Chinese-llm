"""Diagnostic only: inspect frozen data/tokenizers and run bounded inference."""
import collections
import gc
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_causal_model, load_tokenizer
from classical_llm.utils.io import read_jsonl

configure_project_environment()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
PATHS = {"base": BASE, "cpt": ROOT / "outputs/cpt/qwen2.5-0.5b-classical05/best",
         **{n: ROOT / f"outputs/{n}/qwen2.5-0.5b-classical/best" for n in ("sft", "dpo")}}
report = {"tokenizers": {}, "data": {}, "inference": []}
dest = ROOT / ("reports/generated/quality_data_audit.json" if "--data-only" in sys.argv else "reports/generated/quality_diagnosis.json")


def save():
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


base_tok = load_tokenizer(str(BASE))
for name, path in PATHS.items():
    tok = load_tokenizer(str(path))
    report["tokenizers"][name] = {
        "vocabulary_equal_base": tok.get_vocab() == base_tok.get_vocab(),
        "template_equal_base": tok.chat_template == base_tok.chat_template,
        "eos": tok.eos_token_id, "pad": tok.pad_token_id,
        "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
    }
rows = list(read_jsonl(ROOT / "data/final/sft/train.jsonl"))
tok = load_tokenizer(str(PATHS["sft"]))
stats = collections.Counter()
bytask = collections.defaultdict(collections.Counter)
samples = collections.defaultdict(list)
# Deterministic spread across the actual frozen training set, no random redraw.
for row in rows[::20]:
    ms = row["messages"]
    prompt = tok.apply_chat_template(ms[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template(ms, tokenize=True, add_generation_prompt=False, return_dict=False)
    counts = {"rows": 1, "prompt_ge_1024": int(len(prompt) >= 1024),
              "full_gt_1024": int(len(full) > 1024),
              "completion_prefix_mismatch": int(full[:len(prompt)] != prompt),
              "answer_lt_10_char": int(len(ms[-1]["content"]) < 10),
              "im_end_in_retained_completion": int(tok.convert_tokens_to_ids("<|im_end|>") in full[len(prompt):1024])}
    stats.update(counts)
    bytask[row["task"]].update(counts)
    if len(samples[row["task"]]) < 3:
        samples[row["task"]].append({"source": row.get("provenance"), "prompt": ms[-2]["content"][:500],
                                    "answer": ms[-1]["content"][:500], "prompt_tokens": len(prompt), "total_tokens": len(full)})
report["data"] = {"total_train_rows": len(rows), "sample": dict(stats),
                  "by_task": {k: dict(v) for k, v in bytask.items()}, "examples": dict(samples)}
save()
print("DATA_AUDIT_READY", flush=True)
if "--data-only" in sys.argv:
    raise SystemExit(0)
prompts = [
    [{"role": "system", "content": "你是一名严谨的中国古典文学助手。回答必须忠于原文，不虚构作者、时代或典故。"},
     {"role": "user", "content": "请把下面的文言文准确翻译成现代汉语：\n学而时习之，不亦说乎？"}],
    rows[0]["messages"][:-1],
]
for name, path in PATHS.items():
    tokenizer = load_tokenizer(str(path))
    for precision in ("bf16", "nf4"):
        model = load_causal_model(str(path), {"load_in_4bit": precision == "nf4", "bf16": True,
                                            "gradient_checkpointing": False})
        model.eval()
        for i, messages in enumerate(prompts):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=48, repetition_penalty=1.05,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
            new = output[0, inputs.input_ids.shape[1]:]
            item = {"model": name, "precision": precision, "prompt_index": i,
                    "response": tokenizer.decode(new, skip_special_tokens=True), "tokens": len(new)}
            report["inference"].append(item)
            save()
            print(json.dumps(item, ensure_ascii=False), flush=True)
        del model, inputs, output, new
        gc.collect()
        torch.cuda.empty_cache()
print("DONE", flush=True)
