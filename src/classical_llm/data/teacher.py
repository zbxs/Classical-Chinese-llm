from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch

from classical_llm.data.normalize import normalize_text
from classical_llm.training.common import load_tokenizer
from classical_llm.utils.io import read_jsonl, sha256_text, write_jsonl

_SYSTEM = (
    "你是中国古典文学数据标注员。只能依据给出的原文作答；不确定的作者、年代和典故不得猜测。"
    "输出应完整、具体，不要写与任务无关的开场白。"
)


def _segments(text: str, minimum: int = 48, maximum: int = 420) -> list[str]:
    """Split long works at sentence boundaries without changing historical characters."""
    pieces = [part.strip() for part in re.split(r"(?<=[。！？；])", text) if part.strip()]
    output: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) > maximum:
            if len(buffer) >= minimum:
                output.append(buffer)
            buffer = ""
        buffer += piece
    if len(buffer) >= minimum:
        output.append(buffer)
    return output


def _is_classical_source(row: dict[str, Any]) -> bool:
    """Keep general web text out even when all sources share one cleaned file."""
    if str(row.get("category", "")).lower() != "classical":
        return False
    text = str(row.get("text", ""))
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    return cjk / max(1, len(text)) >= 0.65


def _task_prompt(task: str, passage: str) -> str:
    if task == "appreciation":
        return (
            "请先准确翻译下列古文，再从内容、语言或写法中选择两点进行赏析。"
            "每一点都要引用原文中的词句作为依据，不得补写原文没有的背景：\n" + passage
        )
    if task == "creation":
        return (
            "请从下列古文中提取主题和意象，另写一段60至120字的文言短文。"
            "不可照抄连续八字，不得虚构真实历史人物的事迹。完成后只输出作品：\n" + passage
        )
    raise ValueError(f"Unsupported teacher task: {task}")


def _load_teacher(path: str):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if torch.cuda.is_available():
        kwargs.update(device_map={"": 0}, dtype=dtype or torch.float16)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def _passes_rules(task: str, answer: str, passage: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(answer.strip()) < 24:
        reasons.append("answer_too_short")
    if len(answer) > 1600:
        reasons.append("answer_too_long")
    evidence_spans = {
        passage[index : index + 4]
        for index in range(max(0, len(passage) - 3))
        if "\n" not in passage[index : index + 4]
    }
    if task == "appreciation" and not any(span in answer for span in evidence_spans):
        reasons.append("no_textual_overlap")
    if task == "creation" and any(passage[index : index + 8] in answer for index in range(max(0, len(passage) - 7))):
        reasons.append("copied_eight_char_span")
    return not reasons, reasons


def synthesize_sft(
    input_path: str | Path,
    output_dir: str | Path,
    teacher_model: str,
    tasks: list[str],
    limit: int,
    seed: int = 20260912,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    """Generate auditable SFT candidates; rejected rows remain in the audit file.

    This stage is intentionally separate from :func:`build_sft_dataset`: model
    outputs are candidates, not human ground truth. A full run should be followed
    by stratified human review before accepted rows enter the 100K mixture.
    """
    invalid = sorted(set(tasks) - {"appreciation", "creation"})
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for row in read_jsonl(input_path):
        if not _is_classical_source(row):
            continue
        for passage in _segments(normalize_text(str(row.get("text", "")))):
            candidates.append({"row": row, "passage": passage})
    rng.shuffle(candidates)
    candidates = candidates[:limit]

    tokenizer = load_tokenizer(teacher_model)
    model = _load_teacher(teacher_model)
    model.eval()
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        source = candidate["row"]
        task = tasks[index % len(tasks)]
        prompt = _task_prompt(task, candidate["passage"])
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        passed, reasons = _passes_rules(task, answer, candidate["passage"])
        identity = sha256_text(f"{teacher_model}\0{task}\0{prompt}\0{answer}")
        record = {
            "id": identity,
            "task": task,
            "input": prompt,
            "output": answer,
            "source_id": source.get("source_id") or source.get("id"),
            "work": source.get("work"),
            "author": source.get("author"),
            "license": source.get("license", "unknown"),
            "source": "teacher_synthetic",
            "split": source.get("split", "train"),
            "teacher_model": teacher_model,
            "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "accepted_by_rules": passed,
            "rejection_reasons": reasons,
            "requires_human_review": True,
        }
        audit.append(record)
        if passed:
            accepted.append(record)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "audit.jsonl", audit)
    write_jsonl(target / "accepted_candidates.jsonl", accepted)
    manifest = {
        "teacher_model": teacher_model,
        "requested": limit,
        "generated": len(audit),
        "rule_accepted": len(accepted),
        "human_review_required": True,
        "seed": seed,
        "tasks": tasks,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
