"""Judge each preference twice with reversed answer order using a stronger local model."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from classical_llm.evaluation.preference_audit import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = """你是严格的古汉语翻译审校员。比较两个现代汉语译文，首要标准是忠实于原文，其次是信息完整，最后才是语言流畅。不能因答案更长、更详细或更像参考写法而偏爱它。新增原文没有的人物、官职、事件、因果、数字属于幻觉；漏译关键动作、否定、数量和关系属于不完整。若原文本身或两个译文无法可靠判断，选择 INVALID；质量相当选择 TIE。只输出一个 JSON 对象，不要 markdown。"""
SCHEMA = """字段必须是：
{{"winner":"A|B|TIE|INVALID","a":{{"faithfulness":0到4,"completeness":0到4,"fluency":0到2,"hallucination":0到2}},"b":{{"faithfulness":0到4,"completeness":0到4,"fluency":0到2,"hallucination":0到2}},"reason":"不超过40字"}}
原文：{source}
译文A：{a}
译文B：{b}"""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise TypeError("judge output is not an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/preference-diagnosis-v1/judge_input.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/preference-diagnosis-v1/judge_output.jsonl"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    completed: dict[str, dict] = {}
    if args.output.exists():
        completed = {row["presentation_id"]: row for row in read_jsonl(args.output)}
    pending = [row for row in rows if row["presentation_id"] not in completed]
    print(f"judge pending={len(pending)} completed={len(completed)}", flush=True)

    manifest_path = args.output.with_name("judge_run.json")
    manifest = {
        "status": "loading_model",
        "model": args.model,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "presentations": len(rows),
        "already_complete": len(completed),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

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
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            prompts = []
            for row in batch:
                user = SCHEMA.format(source=row["source"], a=row["option_a"], b=row["option_b"])
                prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
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
            for row, sequence in zip(batch, sequences, strict=True):
                raw = tokenizer.decode(sequence[width:], skip_special_tokens=True).strip()
                try:
                    judge = extract_json(raw)
                    error = None
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    judge = {"winner": "INVALID"}
                    error = f"{type(exc).__name__}: {exc}"
                result = {
                    "presentation_id": row["presentation_id"],
                    "pair_id": row["pair_id"],
                    "judge": judge,
                    "parse_error": error,
                    "raw": raw,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"judged {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)
    manifest.update(
        status="complete",
        completed_presentations=len(rows),
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
