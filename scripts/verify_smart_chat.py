"""Smoke-test both backends of the localhost-only smart chat deployment."""
from __future__ import annotations

import json
import urllib.request


def ask(content: str) -> dict:
    body = json.dumps({"model": "smart_repaired", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 64}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:17860/chat",
        data=body,
        headers={"Content-Type": "application/json", "X-Local-Chat": "1",
                 "Origin": "http://127.0.0.1:17860"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    general = ask("12乘以7等于多少？")
    translation = ask("请把“学而时习之，不亦说乎”翻译成现代汉语。")
    assert general["backend"] == "official_instruct"
    assert translation["backend"] == "translation_repaired"
    assert general["response"] and translation["response"]
    print(json.dumps({"general": general, "translation": translation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
