"""Verify the saved repair adapter under the chat server's BF16 load mode."""
# ruff: noqa: I001
import json
from repair_pilot import OUT, ROOT, evaluate

if __name__ == "__main__":
    report = json.loads((ROOT / "reports/generated/repair_eos_pilot.json").read_text())
    if report["status"] != "complete_needs_review":
        raise RuntimeError("Wait for the controlled experiment to complete")
    result = evaluate(str(OUT / "eos_fix/best"), "eos_fix_bf16", precision="bf16")
    (ROOT / "reports/generated/repair_chat_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
