from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classical_llm.data.normalize import normalize_text
from classical_llm.utils.io import read_jsonl, sha256_text, write_jsonl

SYSTEM_PROMPT = "你是一名严谨的中国古典文学助手。回答必须忠于原文，不虚构作者、时代或典故。"

TASK_TEMPLATES: dict[str, list[str]] = {
    "old_to_modern": [
        "请把下面的文言文准确翻译成现代汉语：\n{text}",
        "请解释下列古文的现代汉语含义，注意保留人名、地名和否定关系：\n{text}",
        "将下列文言语段译为通顺、完整的白话文：\n{text}",
    ],
    "modern_to_old": [
        "请将下面的现代汉语改写为简洁自然的文言文：\n{text}",
        "在不改变原意的前提下，把下文译成文言：\n{text}",
        "请用符合古汉语习惯的表达重写下文：\n{text}",
    ],
    "appreciation": [
        "请结合原文翻译并赏析下列作品，分析必须给出文本依据：\n{text}",
        "请说明下文的主要内容、表达手法和思想感情，并引用原文作为依据：\n{text}",
        "请对下面的古诗文作准确解读与赏析，不要虚构背景知识：\n{text}",
    ],
    "creation": [
        "请按以下要求进行古典文学创作：\n{text}",
        "请严格遵守体裁、主题、字数和关键词要求完成创作：\n{text}",
        "完成下列仿写或语境运用任务，并保持古雅、连贯：\n{text}",
    ],
}


def _task_from_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("task", ""))
    aliases = {
        "classical_to_modern": "old_to_modern",
        "translation": "old_to_modern",
        "modern_to_classical": "modern_to_old",
        "analysis": "appreciation",
        "writing": "creation",
    }
    return aliases.get(explicit, explicit)


def _pair_from_row(row: dict[str, Any], task: str) -> tuple[str, str]:
    if task == "old_to_modern":
        return str(row.get("classical") or row.get("input") or ""), str(
            row.get("modern") or row.get("output") or ""
        )
    if task == "modern_to_old":
        return str(row.get("modern") or row.get("input") or ""), str(
            row.get("classical") or row.get("output") or ""
        )
    return str(row.get("input") or row.get("prompt") or ""), str(
        row.get("output") or row.get("answer") or ""
    )


def _valid_pair(source: str, target: str, task: str) -> bool:
    if not source.strip() or not target.strip():
        return False
    if task == "appreciation":
        # PoetryMTEB includes concise but valid subject/emotion facets for long
        # poems. Guard by absolute answer length and a looser relative ratio.
        ratio = len(target) / max(1, len(source))
        return len(target) >= 6 and 0.02 <= ratio <= 8.0
    if task == "modern_to_old":
        ratio = len(target) / max(1, len(source))
        return 0.12 <= ratio <= 1.5
    return 0.12 <= len(target) / max(1, len(source)) <= 8.0


def build_sft_dataset(
    inputs: list[str | Path],
    output_dir: str | Path,
    quotas: dict[str, int] | None = None,
    seed: int = 20260912,
) -> dict[str, int]:
    """Build source-grounded conversational SFT rows.

    The function never silently manufactures answers. A row must already carry
    an aligned reference answer, possibly produced by a separately audited
    teacher-generation stage.
    """
    rng = random.Random(seed)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for input_path in inputs:
        for row in read_jsonl(input_path):
            task = _task_from_row(row)
            if task not in TASK_TEMPLATES:
                continue
            source, target = map(normalize_text, _pair_from_row(row, task))
            if not _valid_pair(source, target, task):
                continue
            identity = sha256_text(f"{task}\0{source}\0{target}")
            if identity in seen:
                continue
            seen.add(identity)
            template_index = int(identity[:8], 16) % len(TASK_TEMPLATES[task])
            user_prompt = TASK_TEMPLATES[task][template_index].format(text=source)
            candidates[task].append(
                {
                    "id": identity,
                    "task": task,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": target},
                    ],
                    "source_id": row.get("source_id") or row.get("id"),
                    "work": row.get("work"),
                    "author": row.get("author"),
                    "license": row.get("license", "unknown"),
                    "provenance": row.get("source", str(input_path)),
                    "split": row.get("split", "train"),
                }
            )

    selected: list[dict[str, Any]] = []
    if quotas is None:
        selected = [row for rows in candidates.values() for row in rows]
    else:
        for task, quota in quotas.items():
            rows_by_split = {
                split: [row for row in candidates.get(task, []) if row["split"] == split]
                for split in ("train", "validation", "test")
            }
            for rows in rows_by_split.values():
                rng.shuffle(rows)
            if len(rows_by_split["train"]) < quota:
                raise ValueError(
                    f"Task {task!r} has {len(rows_by_split['train'])} unique training rows, "
                    f"below quota {quota}"
                )
            selected.extend(rows_by_split["train"][:quota])
            # Validation and test are additional to the exact training quota.
            auxiliary_quota = max(1, round(quota * 0.05))
            selected.extend(rows_by_split["validation"][:auxiliary_quota])
            selected.extend(rows_by_split["test"][:auxiliary_quota])
    rng.shuffle(selected)

    partitions: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    # Inputs should carry source-level splits from prepare_corpus. Unknown rows
    # default to training and are never used to create the final evaluation set.
    for row in selected:
        partition = str(row.get("split", "train"))
        if partition not in partitions:
            partition = "train"
        partitions[partition].append(row)

    target = Path(output_dir)
    counts = {name: write_jsonl(target / f"{name}.jsonl", rows) for name, rows in partitions.items()}
    counts.update({f"task_{task}": count for task, count in Counter(r["task"] for r in selected).items()})
    return counts
