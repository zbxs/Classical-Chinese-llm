"""Private localhost-only model comparison UI; forward port 17860 over SSH."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from classical_llm.runtime import configure_project_environment

configure_project_environment()

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
INSTRUCT = ROOT / ".cache/huggingface/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775"
PATHS = {
    "base": BASE,
    "cpt": ROOT / "outputs/cpt/qwen2.5-0.5b-classical05/best",
    **{name: ROOT / f"outputs/{name}/qwen2.5-0.5b-classical/best"
       for name in ("sft", "dpo", "grpo")},
}
PATHS["official_instruct"] = INSTRUCT
PATHS["translation_repaired"] = ROOT / "outputs/repair-capability-v2/base_plain/best"
PATHS["drgrpo_reference_proxy"] = (
    ROOT / "outputs/preference-grpo-reference-v1/drgrpo_reference_proxy/best"
)
PATHS["drgrpo_constraint_aware"] = (
    ROOT / "outputs/preference-grpo-reference-v1/drgrpo_constraint_aware/best"
)
verification = ROOT / "reports/generated/repair_chat_verification.json"
if verification.exists() and json.loads(verification.read_text(encoding="utf-8"))["stop_rate"] >= 0.9:
    PATHS["sft_repaired"] = ROOT / "outputs/repair-eos-pilot-v1/eos_fix/best"
CACHE = {}
LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


def _translation_request(text: str) -> bool:
    return any(term in text.lower() for term in ("翻译", "译成", "译为", "今译", "白话", "现代汉语"))


def _translation_source(text: str) -> str:
    quoted = re.findall(r"[“「『\"]([^”」』\"]{2,})[”」』\"]", text)
    if quoted:
        return max(quoted, key=len).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [line for line in lines if not _translation_request(line)]
    if candidates:
        return candidates[-1]
    return re.sub(r"^.*?(?:[:：])", "", text, count=1).strip() or text.strip()


def _backend(requested: str, messages: list[dict]) -> str:
    if requested != "smart_repaired":
        return requested
    return "translation_repaired" if _translation_request(messages[-1]["content"]) else "official_instruct"


def reply(payload):
    import torch

    from classical_llm.evaluation.generate import _load_model
    from classical_llm.training.common import load_tokenizer

    requested = payload.get("model")
    if requested not in {*PATHS, "smart_repaired"}:
        raise ValueError("未知模型")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 41:
        raise ValueError("每段对话最多 20 轮，请清空后继续")
    if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("消息格式不正确")
    if messages[-1]["role"] != "user":
        raise ValueError("最后一条必须是用户提问")
    limit = int(payload.get("max_tokens", 192))
    if not 32 <= limit <= 512:
        raise ValueError("输出长度应在 32–512 token 之间")
    started = time.monotonic()
    name = _backend(requested, messages)
    if name not in CACHE:
        tokenizer = load_tokenizer(str(PATHS[name]))
        model = _load_model(str(PATHS[name]))
        model.eval()
        CACHE[name] = (tokenizer, model)
    tokenizer, model = CACHE[name]
    if name in {
        "translation_repaired",
        "drgrpo_reference_proxy",
        "drgrpo_constraint_aware",
    }:
        source = _translation_source(messages[-1]["content"])
        rendered = f"任务：请将下列文言文准确翻译为现代汉语，只输出译文。\n原文：{source}\n译文："
    else:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    if inputs.input_ids.shape[1] > 2048:
        raise ValueError("上下文超过 2048 token，请清空对话或缩短输入；本页面不静默截断")
    inputs = inputs.to(model.device)
    stops = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        stops.append(im_end)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                repetition_penalty=1.05, use_cache=True,
                                eos_token_id=list(set(stops)), pad_token_id=tokenizer.pad_token_id)
    tokens = output[0, inputs.input_ids.shape[1]:]
    return {"response": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "tokens": len(tokens), "seconds": round(time.monotonic() - started, 2),
            "hit_limit": len(tokens) >= limit, "model": requested, "backend": name}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Do not persist user prompts or query strings.

    def send(self, status, body, mime="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Host") not in {"127.0.0.1:17860", "localhost:17860"}:
            return self.send(403, {"error": "Localhost only"})
        if self.path == "/health":
            return self.send(200, {"ready": True, "models": ["smart_repaired", *PATHS],
                                   "loaded": list(CACHE)})
        if self.path != "/":
            return self.send(404, {"error": "Not found"})
        return self.send(200, (ROOT / "scripts/model_chat.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        if (self.path != "/chat" or self.headers.get("Host") not in {"127.0.0.1:17860", "localhost:17860"}
            or self.headers.get("X-Local-Chat") != "1"
            or self.headers.get("Origin", "http://127.0.0.1:17860") not in
                {"http://127.0.0.1:17860", "http://localhost:17860"}):
            return self.send(403, {"error": "仅允许本机页面请求"})
        if not LOCK.acquire(blocking=False):
            return self.send(429, {"error": "服务器正在生成，请稍后重试"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 100000:
                raise ValueError("请求过大或为空")
            payload = json.loads(self.rfile.read(size))
            self.send(200, reply(payload))
        except (ValueError, TypeError, KeyError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            LOGGER.exception("Chat inference failed: %s", type(exc).__name__)
            self.send(500, {"error": "生成失败，请检查服务器日志；未修改模型"})
        finally:
            LOCK.release()


if __name__ == "__main__":
    assert all(path.exists() for path in PATHS.values()), "Model files missing"
    print("Private chat listening on 127.0.0.1:17860", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 17860), Handler).serve_forever()
