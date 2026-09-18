"""Bounded EOS A/B repair pilot. Preserves all previous datasets/checkpoints."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import yaml

from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_causal_model, load_tokenizer
from classical_llm.training.sft import run_sft
from classical_llm.utils.io import read_jsonl, write_jsonl

configure_project_environment()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/repair-eos-pilot-v1"
DATA = ROOT / "data/repair-eos-pilot-v1"
REPORT = ROOT / "reports/generated/repair_eos_pilot.json"
SOURCE = "outputs/sft/qwen2.5-0.5b-classical/best"


def save(value):
    REPORT.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare():
    if OUT.exists() or DATA.exists():
        raise FileExistsError("Pilot paths already exist: preserve them; use a new version")
    OUT.mkdir(parents=True)
    DATA.mkdir(parents=True)
    tokenizer = load_tokenizer(str(ROOT / SOURCE))
    seen = set()
    groups = set()
    chosen = {}
    for split, count in (("test", 16), ("validation", 64), ("train", 1024)):
        rows = list(read_jsonl(ROOT / f"data/final/sft/{split}.jsonl"))
        random.Random(20260918).shuffle(rows)
        selected = []
        for row in rows:
            if row["task"] != "old_to_modern" or row.get("provenance") != "historytrans-bidirectional":
                continue
            messages = row["messages"]
            answer = messages[-1]["content"]
            source = messages[-2]["content"]
            if not 12 <= len(answer) <= 180 or "�" in source + answer:
                continue
            if any(x in source + answer for x in ("<|", "http", "答案：")):
                continue
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
            if len(tokens) > 384:
                continue
            key = hashlib.sha256((source + "\0" + answer).encode()).hexdigest()
            group = row.get("source_id") or key
            if key in seen or group in groups:
                continue
            seen.add(key)
            groups.add(group)
            selected.append(row)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f"Insufficient {split}: {len(selected)}")
        write_jsonl(DATA / f"{split}.jsonl", selected)
        chosen[split] = len(selected)
    return chosen


def evaluate(model_path, label, precision="nf4"):
    import sacrebleu

    tokenizer = load_tokenizer(str(ROOT / model_path))
    model = load_causal_model(str(ROOT / model_path), {"load_in_4bit": precision == "nf4", "gradient_checkpointing": False})
    model.eval()
    rows = list(read_jsonl(DATA / "test.jsonl"))
    manual = ["请把下面的文言文准确翻译成现代汉语：\n学而时习之，不亦说乎？",
              "请把下面的文言文准确翻译成现代汉语：\n苟全性命于乱世，不求闻达于诸侯。",
              "请把下面的文言文准确翻译成现代汉语：\n己所不欲，勿施于人。",
              "请把下面的文言文准确翻译成现代汉语：\n知之为知之，不知为不知，是知也。"]
    for i, prompt in enumerate(manual):
        rows.append({"id": f"manual-{i}", "messages": [rows[0]["messages"][0], {"role": "user", "content": prompt}]})
    outputs = []
    stops = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
    for row in rows:
        messages = row["messages"]
        has_reference = messages[-1]["role"] == "assistant"
        prompt = messages[:-1] if has_reference else messages
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=True,
                                 repetition_penalty=1.05, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
        new = ids[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new, skip_special_tokens=True).strip()
        grams = [response[i:i+4] for i in range(max(0, len(response)-3))]
        item = {"id": row["id"], "prompt": prompt[-1]["content"], "response": response,
                "tokens": len(new), "stopped": int(new[-1]) in stops,
                "repetition": 1 - len(set(grams)) / max(1, len(grams)), "replacement_character": "�" in response}
        if has_reference:
            item["reference"] = messages[-1]["content"]
            item["chrf"] = sacrebleu.sentence_chrf(response, [item["reference"]]).score
        outputs.append(item)
    write_jsonl(OUT / f"{label}.jsonl", outputs)
    summary = {"count": len(outputs), "stop_rate": sum(r["stopped"] for r in outputs)/len(outputs),
               "mean_repetition": sum(r["repetition"] for r in outputs)/len(outputs),
               "replacement_rate": sum(r["replacement_character"] for r in outputs)/len(outputs),
               "chrf_16": sum(r.get("chrf", 0) for r in outputs)/16,
               "manual": outputs[-4:]}
    del model, inputs, ids, new
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    result = {"status": "preparing", "scope": "EOS-only controlled pilot, not final benchmark", "counts": prepare(), "results": {}}
    save(result)
    result["status"] = "evaluating_original"
    save(result)
    result["results"]["original"] = evaluate(SOURCE, "original")
    save(result)
    for name, eos in (("control", "<|im_end|>"), ("eos_fix", "<|endoftext|>")):
        result["status"] = f"training_{name}"
        save(result)
        config = {"stage": "sft", "model_name_or_path": SOURCE,
                  "dataset_path": str(DATA / "train.jsonl"), "output_dir": str(OUT / name),
                  "assistant_end_token": eos, "max_seq_length": 512, "load_in_4bit": True,
                  "use_qlora": True, "seed": 20260918, "learning_rate": 5e-5, "max_steps": 64,
                  "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4,
                  "gradient_checkpointing": True, "bf16": True, "logging_steps": 8,
                  "save_steps": 64, "save_total_limit": 2, "warmup_steps": 4}
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        result[f"train_{name}"] = run_sft(str(path))
        gc.collect()
        torch.cuda.empty_cache()
        result["status"] = f"evaluating_{name}"
        save(result)
        result["results"][name] = evaluate(str(OUT / name / "best"), name)
        save(result)
    result["status"] = "complete_needs_review"
    save(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
