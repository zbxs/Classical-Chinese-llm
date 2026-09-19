"""Judge preference pairs by scoring each candidate independently under two rubrics."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from classical_llm.evaluation.preference_audit import (
    candidate_utility,
    compare_candidate_scores,
    extract_source,
    merge_judgments,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = (
    """你是古汉语译文质量评分员。每次只评一个译文，不猜测也不比较任何未展示的答案。以忠实原文为首要标准：增添原文没有的事实属于幻觉，漏掉关键动作、否定、数量或关系会降低完整度。只输出一个紧凑 JSON 对象，不要 markdown。""",
    """你是独立的古文翻译核验员。先检查译文是否捏造，再检查是否漏译，最后判断语义忠实和行文通顺。当前译文必须单独评分，不能假设存在另一个候选。只输出一个紧凑 JSON 对象，不要 markdown。""",
)
PROMPTS = (
    """原文：{source}
待评译文：{candidate}
严格输出：{{"faithfulness":0到4,"completeness":0到4,"fluency":0到2,"hallucination":0到2}}""",
    """待核验译文：{candidate}
对照原文：{source}
严格输出：{{"hallucination":0到2,"fluency":0到2,"completeness":0到4,"faithfulness":0到4}}""",
)


def extract_scores(text: str) -> dict:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict) or candidate_utility(value) is None:
        raise ValueError("invalid rubric scores")
    return value


def stable_pairs(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"independent-v2:{row['id']}".encode()).hexdigest(),
    )
    return ordered if limit <= 0 else ordered[:limit]


def label_for_presentation(presentation: dict, winner_role: str) -> str:
    if winner_role in {"tie", "invalid"}:
        return winner_role.upper()
    if presentation["option_a_role"] == winner_role:
        return "A"
    if presentation["option_b_role"] == winner_role:
        return "B"
    return "INVALID"


def scores_for_label(presentation: dict, by_role: dict[str, dict]) -> tuple[dict, dict]:
    return by_role[presentation["option_a_role"]], by_role[presentation["option_b_role"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/repair-dpo-v1")
    parser.add_argument(
        "--presentations",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge_input.jsonl",
    )
    parser.add_argument(
        "--task-output",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge_independent_v2.jsonl",
    )
    parser.add_argument(
        "--judge-output",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge_output.v2.jsonl",
    )
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--tie-margin", type=float, default=1.0)
    parser.add_argument("--min-consistency", type=float, default=0.70)
    args = parser.parse_args()

    pairs = []
    for split in ("train", "validation"):
        for row in read_jsonl(args.pairs / f"{split}.jsonl"):
            row = dict(row)
            row["split"] = split
            pairs.append(row)
    pairs = stable_pairs(pairs, args.pair_limit)
    pair_by_id = {str(row["id"]): row for row in pairs}
    presentations = [
        row
        for row in read_jsonl(args.presentations)
        if str(row["pair_id"]) in pair_by_id
    ]

    tasks = []
    for row in pairs:
        for protocol in range(2):
            for role in ("chosen", "rejected"):
                tasks.append({
                    "task_id": f"{row['id']}:{protocol}:{role}",
                    "pair_id": str(row["id"]),
                    "protocol": protocol,
                    "role": role,
                    "source": extract_source(row["prompt"]),
                    "candidate": row[role],
                })

    completed: dict[str, dict] = {}
    if args.task_output.exists():
        completed = {row["task_id"]: row for row in read_jsonl(args.task_output)}
    pending = [row for row in tasks if row["task_id"] not in completed]
    print(f"independent judge pending={len(pending)} completed={len(completed)}", flush=True)

    manifest_path = args.task_output.with_suffix(".run.json")
    manifest = {
        "status": "loading_model",
        "protocol": "independent-v2",
        "model": args.model,
        "pairs": len(pairs),
        "tasks": len(tasks),
        "already_complete": len(completed),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "tie_margin": args.tie_margin,
        "min_consistency": args.min_consistency,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
        model.eval()
        manifest["status"] = "judging"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        args.task_output.parent.mkdir(parents=True, exist_ok=True)
        with args.task_output.open("a", encoding="utf-8") as handle:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompts = []
                for task in batch:
                    protocol = int(task["protocol"])
                    user = PROMPTS[protocol].format(
                        source=task["source"], candidate=task["candidate"]
                    )
                    prompts.append(tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEMS[protocol]},
                            {"role": "user", "content": user},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    ))
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).to(model.device)
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                width = inputs.input_ids.shape[1]
                for task, sequence in zip(batch, sequences, strict=True):
                    raw = tokenizer.decode(sequence[width:], skip_special_tokens=True).strip()
                    try:
                        scores = extract_scores(raw)
                        error = None
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        scores = {}
                        error = f"{type(exc).__name__}: {exc}"
                    result = {
                        "task_id": task["task_id"],
                        "pair_id": task["pair_id"],
                        "protocol": task["protocol"],
                        "role": task["role"],
                        "scores": scores,
                        "parse_error": error,
                        "raw": raw,
                    }
                    completed[result["task_id"]] = result
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    f"independent judged {min(offset + len(batch), len(pending))}/{len(pending)}",
                    flush=True,
                )

    by_pair_protocol: dict[tuple[str, int], dict[str, dict]] = {}
    for result in completed.values():
        key = (str(result["pair_id"]), int(result["protocol"]))
        by_pair_protocol.setdefault(key, {})[str(result["role"])] = result["scores"]

    presentation_by_pair_order = {
        (str(row["pair_id"]), int(row["order"])): row for row in presentations
    }
    compatible = []
    for row in pairs:
        pair_id = str(row["id"])
        for protocol in range(2):
            presentation = presentation_by_pair_order[(pair_id, protocol)]
            role_scores = by_pair_protocol.get((pair_id, protocol), {})
            if set(role_scores) != {"chosen", "rejected"}:
                winner_role = "invalid"
                role_scores = {"chosen": {}, "rejected": {}}
            else:
                comparison = compare_candidate_scores(
                    role_scores["chosen"], role_scores["rejected"], args.tie_margin
                )
                winner_role = {
                    "first": "chosen",
                    "second": "rejected",
                    "tie": "tie",
                    "invalid": "invalid",
                }[comparison]
            score_a, score_b = scores_for_label(presentation, role_scores)
            compatible.append({
                "presentation_id": presentation["presentation_id"],
                "pair_id": pair_id,
                "judge": {
                    "winner": label_for_presentation(presentation, winner_role),
                    "a": score_a,
                    "b": score_b,
                    "reason": (
                        f"independent-v2 protocol={protocol}; "
                        f"chosen={candidate_utility(role_scores['chosen'])}; "
                        f"rejected={candidate_utility(role_scores['rejected'])}"
                    ),
                },
                "parse_error": None if winner_role != "invalid" else "invalid independent scores",
                "raw": "",
            })

    args.judge_output.parent.mkdir(parents=True, exist_ok=True)
    args.judge_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compatible),
        encoding="utf-8",
    )
    _, report = merge_judgments(presentations, compatible)
    report.update({
        "protocol": "independent-v2",
        "pairs_requested": len(pairs),
        "parse_error_tasks": sum(bool(row.get("parse_error")) for row in completed.values()),
        "utility_formula": "4*faithfulness + 2*completeness + fluency - 4*hallucination",
        "tie_margin": args.tie_margin,
        "min_consistency": args.min_consistency,
        "passed": report["position_consistency_rate"] >= args.min_consistency,
    })
    report_path = args.judge_output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.update(
        status="complete" if report["passed"] else "quality_gate_failed",
        completed_tasks=len(completed),
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
        report=str(report_path),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
