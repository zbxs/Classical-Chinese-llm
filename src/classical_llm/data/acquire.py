from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classical_llm.data.split import stable_split
from classical_llm.utils.config import load_yaml
from classical_llm.utils.io import read_jsonl, write_jsonl


def _pick_text(row: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _task_from_instruction(instruction: str, source_tag: str) -> str | None:
    value = f"{instruction}\n{source_tag}".lower()
    if any(token in value for token in ("翻译成古文", "翻译成文言", "译成文言", "modern to classical")):
        return "modern_to_old"
    if any(token in value for token in ("翻译成现代", "翻译成白话", "译为白话", "文言文翻译", "classical to modern")):
        return "old_to_modern"
    if source_tag in {"firefly_AncientPoem", "firefly_Couplet", "firefly_JinYongGeneration"}:
        return "creation"
    if any(token in value for token in ("写诗", "作诗", "诗词创作", "对联", "仿写", "续写")):
        return "creation"
    # Mentioning classical language or an idiom is not an appreciation task.
    # Prefer dropping ambiguous candidates to silently satisfying a quota.
    if any(token in value for token in ("赏析", "鉴赏", "思想感情", "表达手法")):
        return "appreciation"
    return None


def _make_record(
    source: dict[str, Any],
    name: str,
    index: int,
    text: str,
    raw: dict[str, Any],
    suffix: str = "",
) -> dict[str, Any]:
    source_id = f"{name}:{index}"
    work_fields = source.get("work_fields", ["source", "work", "title", "chapter"])
    work = next((raw.get(field) for field in work_fields if raw.get(field)), None)
    record = {
        "id": f"{source_id}{suffix}",
        "source_id": source_id,
        "source": name,
        "dataset": source["dataset"],
        "license": source["license"],
        "category": source["category"],
        "text": text,
        "work": work,
        "author": raw.get("author"),
        "dynasty": raw.get("era") or raw.get("dynasty"),
        "metadata": {
            key: value
            for key, value in raw.items()
            if key not in source.get("text_fields", [])
            and isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    record["split"] = stable_split(record, seed=int(source.get("seed", 20260912)))
    return record


def _normalized_rows(dataset, source: dict[str, Any], name: str, max_documents: int | None):
    raw_count = 0
    for index, item in enumerate(dataset):
        if max_documents is not None and raw_count >= max_documents:
            break
        raw = dict(item)
        mode = source.get("mode", "corpus")
        if mode == "translation_pair":
            classical = str(raw.get(source.get("classical_field", "inputs"), "")).strip()
            modern = str(raw.get(source.get("modern_field", "truth"), "")).strip()
            if not classical or not modern:
                continue
            base = _make_record(source, name, index, f"{classical}\n{modern}", raw, ":o2m")
            yield {
                **base,
                "task": "old_to_modern",
                "input": classical,
                "output": modern,
                "classical": classical,
                "modern": modern,
            }
            yield {
                **base,
                "id": f"{name}:{index}:m2o",
                "task": "modern_to_old",
                "input": modern,
                "output": classical,
                "classical": classical,
                "modern": modern,
            }
            raw_count += 1
            continue
        if mode == "generic_sft":
            instruction = str(raw.get("instruction", "")).strip()
            input_text = str(raw.get("input", "")).strip()
            output_text = str(raw.get("output", "")).strip()
            source_tag = str(raw.get("source", ""))
            task = _task_from_instruction(instruction, source_tag)
            if not task or not output_text or not (instruction or input_text):
                continue
            prompt = "\n".join(value for value in (instruction, input_text) if value)
            record = _make_record(source, name, index, f"{prompt}\n{output_text}", raw)
            yield {
                **record,
                "task": task,
                "instruction": instruction,
                "input": prompt,
                "output": output_text,
            }
            raw_count += 1
            continue
        if mode == "instruction":
            task_aliases = {
                "c2m": "old_to_modern",
                "m2c": "modern_to_old",
                "punctuate": "punctuate",
            }
            task = task_aliases.get(str(raw.get("task", "")), str(raw.get("task", "")))
            input_text = str(raw.get("input", ""))
            output_text = str(raw.get("output", ""))
            text = output_text if task == "punctuate" else f"{input_text}\n{output_text}"
        else:
            task = None
            input_text = ""
            output_text = ""
            text = _pick_text(raw, list(source.get("text_fields", ["text"])))
        if not text:
            continue
        record = _make_record(source, name, index, text, raw)
        if mode == "instruction":
            record.update(
                task=task,
                instruction=raw.get("instruction"),
                input=input_text,
                output=output_text,
            )
            if task == "old_to_modern":
                record.update(classical=input_text, modern=output_text)
            elif task == "modern_to_old":
                record.update(modern=input_text, classical=output_text)
        yield record
        raw_count += 1


def _poetry_appreciation_rows(
    corpus, analysis, source: dict[str, Any], name: str, max_documents: int | None
):
    analyses = {str(row["id"]): dict(row) for row in analysis}
    aspects = {
        "intent": "请分析这首诗的创作意图，并结合诗句说明。",
        "subject": "请判断这首诗的题材，并给出文本依据。",
        "theme": "请概括这首诗的主题，并结合原文解释。",
        "thought": "请分析这首诗表达的思想内涵，并引用原文作为依据。",
        "emotion": "请分析这首诗的情感，并指出承载这种情感的诗句。",
    }
    for index, item in enumerate(corpus):
        if max_documents is not None and index >= max_documents:
            break
        raw = dict(item)
        paired = analyses.get(str(raw.get("id")))
        poem = str(raw.get("text", "")).strip()
        if not paired or not poem:
            continue
        for aspect, instruction in aspects.items():
            answer = str(paired.get(aspect, "")).strip()
            if not answer:
                continue
            prompt = f"{instruction}\n{poem}"
            record = _make_record(source, name, index, f"{prompt}\n{answer}", raw, f":{aspect}")
            yield {
                **record,
                "task": "appreciation",
                "instruction": instruction,
                "input": prompt,
                "output": answer,
                "analysis_aspect": aspect,
            }


def inspect_huggingface_source(dataset_name: str, config_name: str | None = None) -> dict[str, Any]:
    """Read one streamed row to document a dataset's actual schema."""
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    row = dict(next(iter(dataset)))
    return {
        "dataset": dataset_name,
        "config": config_name,
        "fields": {key: type(value).__name__ for key, value in row.items()},
        "preview": {
            key: (value[:500] if isinstance(value, str) else value)
            for key, value in row.items()
        },
    }


def acquire_huggingface_sources(
    config_path: str | Path,
    output_dir: str | Path,
    max_documents_per_source: int | None = None,
) -> dict[str, Any]:
    """Stream enabled Hugging Face sources into traceable local JSONL files.

    Remote dataset code is never enabled. Gated sources remain disabled until
    their user agreements are explicitly reviewed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data dependencies before acquisition") from exc

    config = load_yaml(config_path)
    destination = Path(output_dir)
    summary: dict[str, Any] = {"sources": {}}
    for source in config.get("sources", []):
        if not source.get("enabled", False) or source.get("kind") not in {
            "huggingface",
            "huggingface_file",
        }:
            continue
        name = str(source["name"])
        try:
            source_limit = max_documents_per_source
            if source_limit is None and source.get("max_documents") is not None:
                source_limit = int(source["max_documents"])
            if source.get("kind") == "huggingface_file":
                from huggingface_hub import hf_hub_download

                archive_dir = destination / "source_archives" / name
                downloaded = hf_hub_download(
                    repo_id=source["dataset"],
                    filename=source["filename"],
                    repo_type="dataset",
                    revision=source.get("revision"),
                    local_dir=archive_dir,
                )
                rows = _normalized_rows(read_jsonl(downloaded), dict(source), name, source_limit)
            elif source.get("mode") == "poetry_appreciation":
                corpus = load_dataset(
                    source["dataset"],
                    source.get("corpus_config", "corpus"),
                    split=source.get("split", "test"),
                    streaming=bool(source.get("streaming", True)),
                    revision=source.get("revision"),
                    trust_remote_code=False,
                )
                analysis = load_dataset(
                    source["dataset"],
                    source.get("analysis_config", "analysis"),
                    split=source.get("split", "test"),
                    streaming=bool(source.get("streaming", True)),
                    revision=source.get("revision"),
                    trust_remote_code=False,
                )
                rows = _poetry_appreciation_rows(corpus, analysis, dict(source), name, source_limit)
            else:
                dataset = load_dataset(
                    source["dataset"],
                    source.get("config"),
                    split=source.get("split", "train"),
                    streaming=bool(source.get("streaming", True)),
                    revision=source.get("revision"),
                    trust_remote_code=False,
                )
                rows = _normalized_rows(dataset, dict(source), name, source_limit)
            output = destination / f"{name}.jsonl"
            count = write_jsonl(output, rows)
            summary["sources"][name] = {
                "rows": count,
                "path": str(output),
                "license": source["license"],
                "revision": source.get("revision"),
            }
        except Exception as exc:  # noqa: BLE001 - acquisition must report and continue
            summary["sources"][name] = {
                "rows": 0,
                "error": repr(exc),
                "license": source["license"],
                "revision": source.get("revision"),
            }

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "acquisition.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
