"""Controlled Base/CPT/Instruct repair comparison; never overwrites old runs."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sacrebleu
import torch
import yaml

from classical_llm.evaluation.generate import _load_model
from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_tokenizer
from classical_llm.training.sft import run_sft
from classical_llm.utils.io import read_jsonl, write_jsonl

configure_project_environment()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/repair-capability-v2"
DATA = ROOT / "data/repair-capability-v2"
REPORT = ROOT / "reports/generated/repair_capability_v2.json"
BASE = ROOT / ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
CPT = ROOT / "outputs/cpt/qwen2.5-0.5b-classical05/best"
INSTRUCT_ROOT = ROOT / ".cache/huggingface/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "你是一名严谨的中文助手。回答应准确、简洁；翻译文言文时忠于原意，不虚构信息。"

GENERAL = [
    ("12乘以7等于多少？只回答数字。", ["84"], False),
    ("中国的首都是哪里？", ["北京"], False),
    ("水的化学式是什么？", ["h2o", "h₂o"], False),
    ("把英文 hello 翻译成中文。", ["你好"], False),
    ("“高”的反义词是什么？", ["低"], False),
    ("数列2、4、6的下一项是什么？只回答数字。", ["8"], False),
    ("红色和蓝色混合通常得到什么颜色？", ["紫"], False),
    ("我们生活在哪颗行星上？", ["地球"], False),
    ("100除以4等于多少？只回答数字。", ["25"], False),
    ("一年通常有多少个月？只回答数字。", ["12"], False),
    ("请用一句话概括：小明出门时看到乌云，于是带上雨伞。", ["乌云", "雨伞"], True),
    ("将“我喜欢读书”翻译成英语。", ["i like reading", "i like to read"], False),
    ("太阳从哪个方向升起？", ["东"], False),
    ("三角形有几条边？只回答数字。", ["3"], False),
    ("请给“快乐”写一个近义词。", ["高兴", "愉快", "欢乐"], False),
    ("如果今天是星期一，明天是星期几？", ["星期二", "周二"], False),
    ("5加9等于多少？只回答数字。", ["14"], False),
    ("请判断：鲸鱼是鱼类吗？先回答是或否。", ["否"], False),
    ("中国传统节日中赏月通常指哪个节日？", ["中秋"], False),
    ("请把“请保持安静”改成礼貌表达。", ["请", "安静"], True),
]


def save(payload):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cjk_ratio(text):
    chars = [ch for ch in text if not ch.isspace()]
    return sum("\u3400" <= ch <= "\u9fff" for ch in chars) / max(1, len(chars))


def clean_pair(row):
    source = str(row.get("classical") or row.get("input") or "").strip()
    answer = str(row.get("modern") or row.get("output") or "").strip()
    if row.get("task") != "old_to_modern" or not 8 <= len(source) <= 260 or not 10 <= len(answer) <= 360:
        return None
    ratio = len(answer) / len(source)
    if not 0.55 <= ratio <= 4.0 or cjk_ratio(source) < 0.55 or cjk_ratio(answer) < 0.55:
        return None
    bad = ("http", "<|", "答案：", "作为ai", "无法回答", "translation:")
    if any(part in (source + answer).lower() for part in bad):
        return None
    if re.search(r"(.)\1{5,}", source + answer):
        return None
    return source, answer


def prepare_data():
    if RUN.exists() or DATA.exists():
        raise FileExistsError("Preserve repair-capability-v2; use a new version for another run")
    RUN.mkdir(parents=True)
    DATA.mkdir(parents=True)
    buckets = {"train": [], "validation": [], "test": []}
    seen = set()
    for row in read_jsonl(ROOT / "data/raw/production_sft/historytrans-bidirectional.jsonl"):
        pair = clean_pair(row)
        split = row.get("split")
        if pair is None or split not in buckets:
            continue
        source, answer = pair
        digest = hashlib.sha256((source + "\0" + answer).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        plain = f"任务：请将下列文言文准确翻译为现代汉语，只输出译文。\n原文：{source}\n译文："
        item = {"id": digest, "task": "old_to_modern", "source_id": row.get("source_id"),
                "source_text": source, "answer_text": answer, "prompt_text": plain,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"请将下列文言文准确翻译为现代汉语，只输出译文：\n{source}"},
                             {"role": "assistant", "content": answer}]}
        buckets[split].append(item)
    limits = {"train": 8192, "validation": 256, "test": 128}
    counts = {}
    for split, rows in buckets.items():
        random.Random(20260918).shuffle(rows)
        selected = rows[: limits[split]]
        if len(selected) != limits[split]:
            raise ValueError(f"Insufficient clean {split}: {len(selected)}")
        counts[split] = write_jsonl(DATA / f"{split}.jsonl", selected)
    return counts


def train_config(name, model, style, learning_rate):
    config = {"stage": "sft", "model_name_or_path": str(model),
              "dataset_path": str(DATA / "train.jsonl"), "validation_path": str(DATA / "validation.jsonl"),
              "output_dir": str(RUN / name), "format_style": style, "max_seq_length": 512,
              "use_qlora": True, "load_in_4bit": True, "lora_r": 32, "lora_alpha": 64,
              "lora_dropout": 0.05, "learning_rate": learning_rate, "max_steps": 256,
              "per_device_train_batch_size": 4, "per_device_eval_batch_size": 4,
              "gradient_accumulation_steps": 4, "gradient_checkpointing": True, "bf16": True,
              "logging_steps": 16, "eval_steps": 128, "save_steps": 128, "save_total_limit": 2,
              "warmup_steps": 16, "seed": 20260918}
    if style == "plain":
        config["eos_token"] = "<|endoftext|>"
    path = RUN / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def model_prompt(tokenizer, kind, prompt, is_translation):
    if kind == "plain":
        if is_translation:
            return f"任务：请将下列文言文准确翻译为现代汉语，只输出译文。\n原文：{prompt}\n译文："
        return f"任务：请准确、简洁地回答问题。\n问题：{prompt}\n回答："
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, rendered, limit=128):
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
    stops = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=limit, do_sample=False, use_cache=True,
                                repetition_penalty=1.05, eos_token_id=stops,
                                pad_token_id=tokenizer.pad_token_id)
    new = output[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new, skip_special_tokens=True).strip()
    return response, len(new), bool(len(new) and int(new[-1]) in stops)


def evaluate(name, path, kind):
    tokenizer = load_tokenizer(str(path))
    model = _load_model(str(path))
    model.eval()
    translations = []
    for row in read_jsonl(DATA / "test.jsonl"):
        rendered = model_prompt(tokenizer, kind, row["source_text"], True)
        response, tokens, stopped = generate(model, tokenizer, rendered)
        grams = [response[i:i+4] for i in range(max(0, len(response)-3))]
        translations.append({"id": row["id"], "source": row["source_text"],
                             "reference": row["answer_text"], "response": response,
                             "tokens": tokens, "stopped": stopped, "replacement": "�" in response,
                             "repetition": 1-len(set(grams))/max(1, len(grams)),
                             "chrf": sacrebleu.sentence_chrf(response, [row["answer_text"]]).score})
    general = []
    for prompt, expected, require_all in GENERAL:
        rendered = model_prompt(tokenizer, kind, prompt, False)
        response, tokens, stopped = generate(model, tokenizer, rendered, 96)
        normalized = response.lower().replace(" ", "")
        matches = [term.lower().replace(" ", "") in normalized for term in expected]
        passed = all(matches) if require_all else any(matches)
        general.append({"prompt": prompt, "expected": expected, "response": response,
                        "require_all": require_all, "passed": passed,
                        "tokens": tokens, "stopped": stopped})
    write_jsonl(RUN / f"{name}.translation.jsonl", translations)
    write_jsonl(RUN / f"{name}.general.jsonl", general)
    summary = {"translation": {"count": len(translations),
        "stop_rate": sum(x["stopped"] for x in translations)/len(translations),
        "replacement_rate": sum(x["replacement"] for x in translations)/len(translations),
        "mean_repetition": sum(x["repetition"] for x in translations)/len(translations),
        "mean_chrf": sum(x["chrf"] for x in translations)/len(translations)},
        "general": {"count": len(general), "pass_rate": sum(x["passed"] for x in general)/len(general),
                    "stop_rate": sum(x["stopped"] for x in general)/len(general)},
        "examples": translations[:5], "general_failures": [x for x in general if not x["passed"]]}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    if not INSTRUCT_ROOT.exists():
        raise FileNotFoundError(f"Official Instruct baseline missing: {INSTRUCT_ROOT}")
    payload = {"status": "preparing", "scope": "256-step capability repair pilot", "data": prepare_data(),
               "models": {}, "training": {}}
    save(payload)
    candidates = [
        ("instruct_baseline", INSTRUCT_ROOT, "chat", None),
        ("base_plain", BASE, "plain", 5e-5),
        ("cpt_plain", CPT, "plain", 5e-5),
        ("instruct_sft", INSTRUCT_ROOT, "chat", 1e-5),
    ]
    for name, source, kind, learning_rate in candidates:
        if learning_rate is not None:
            payload["status"] = f"training_{name}"
            save(payload)
            payload["training"][name] = run_sft(str(train_config(name, source, kind, learning_rate)))
            source = RUN / name / "best"
        payload["status"] = f"evaluating_{name}"
        save(payload)
        payload["models"][name] = evaluate(name, source, kind)
        save(payload)
    instruct_general = payload["models"]["instruct_baseline"]["general"]["pass_rate"]
    eligible = []
    for name, metrics in payload["models"].items():
        t, g = metrics["translation"], metrics["general"]
        if t["stop_rate"] >= 0.95 and t["replacement_rate"] == 0 and t["mean_repetition"] <= 0.10 and g["pass_rate"] >= instruct_general - 0.10:
            eligible.append((t["mean_chrf"], g["pass_rate"], name))
    payload["selection"] = {"instruct_general_floor": instruct_general - 0.10,
                            "eligible": sorted(eligible, reverse=True),
                            "selected": max(eligible)[2] if eligible else None,
                            "requires_human_review": True}
    payload["status"] = "complete_needs_human_review"
    save(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
