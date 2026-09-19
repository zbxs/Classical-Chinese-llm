"""Read-only special-token and loss-boundary audit, no optimizer or training."""
import json
import os
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from classical_llm.runtime import configure_project_environment
from classical_llm.training.common import load_causal_model, load_tokenizer
from classical_llm.utils.io import read_jsonl

configure_project_environment()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
path = root / "outputs/sft/qwen2.5-0.5b-classical/best"
tok = load_tokenizer(str(path))
rows = list(read_jsonl(root / "data/final/sft/train.jsonl"))
counts = Counter()
for row in rows:
    counts[f"task/{row['task']}"] += 1
    counts[f"source/{row.get('provenance')}"] += 1
    if row["task"] == "appreciation" and "文言文翻译：" in row["messages"][-2]["content"]:
        counts["appreciation_contains_translation_instruction"] += 1
    if row["task"] == "appreciation" and len(row["messages"][-1]["content"]) < 20:
        counts["appreciation_answer_under20_chars"] += 1
raw = json.loads((path / "tokenizer.json").read_text())
vocab = raw["model"]["vocab"]
report = {"data_full_counts": dict(counts), "special_tokens": {}, "supervision": [], "boundary_logits": []}
for text in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
    report["special_tokens"][text] = {"encode": tok.encode(text, add_special_tokens=False),
                                      "runtime_id": tok.convert_tokens_to_ids(text),
                                      "json_vocab_id": vocab.get(text),
                                      "added": [v for v in raw["added_tokens"] if v["content"] == text]}
collator = DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id, completion_only_loss=True)
examples = []
for row in rows[:4]:
    ms = row["messages"]
    prompt = tok.apply_chat_template(ms[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template(ms, tokenize=True, return_dict=False)
    mask = [0] * len(prompt) + [1] * (len(full) - len(prompt))
    examples.append({"input_ids": full[:1024], "completion_mask": mask[:1024]})
batch = collator(examples)
for labels in batch["labels"]:
    supervised = labels[labels != -100].tolist()
    report["supervision"].append({"text": tok.decode(supervised),
                                  "im_end_supervised": 151645 in supervised,
                                  "tokens": len(supervised)})
model = load_causal_model(str(path), {"load_in_4bit": True, "gradient_checkpointing": False})
model.eval()
report["embedding_rows"] = {}
for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "。", "的"):
    idx = tok.encode(token, add_special_tokens=False)[0]
    emb = model.get_input_embeddings().weight[idx].detach().float()
    head = model.get_output_embeddings().weight[idx].detach().float()
    report["embedding_rows"][token] = {"id": idx, "input_norm": float(emb.norm()),
        "output_norm": float(head.norm()), "output_nonzero": int(torch.count_nonzero(head)),
        "output_trainable": model.get_output_embeddings().weight.requires_grad}
report["embedding_modules"] = [n for n, _ in model.named_modules() if "lm_head" in n or "embed_tokens" in n]
head = model.get_output_embeddings().weight.detach().float()
target = head[151645]
distances = (head - target).norm(dim=1)
close_ids = (distances < 0.001).nonzero().flatten()
report["end_token_neighbors"] = {"rows_within_l2_0.001": len(close_ids),
    "start_end_l2": float(distances[151644]), "examples": close_ids[:20].tolist(),
    "head_has_adapter": hasattr(model.get_output_embeddings(), "lora_A")}
for row in rows[:3]:
    ms = row["messages"]
    ids = tok.apply_chat_template(ms, tokenize=True, return_dict=False)
    end = len(ids) - 1 - ids[::-1].index(151645)
    inp = torch.tensor([ids[:end]], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=inp).logits[0, -1].float()
    probs = logits.softmax(-1)
    top = logits.topk(5).indices.tolist()
    report["boundary_logits"].append({"gold_answer": ms[-1]["content"],
        "im_end_probability": float(probs[151645]),
        "im_end_rank": int((logits > logits[151645]).sum()) + 1,
        "top": [{"id": i, "text": tok.decode([i]), "p": float(probs[i])} for i in top]})
(root / "reports/generated/boundary_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
