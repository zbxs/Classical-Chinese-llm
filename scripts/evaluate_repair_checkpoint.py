"""Evaluate an existing capability-repair checkpoint without retraining it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repair_capability_v2 import REPORT, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("chat", "plain"), required=True)
    args = parser.parse_args()

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if args.name in payload["models"]:
        raise ValueError(f"Refusing to overwrite existing result: {args.name}")
    payload["status"] = f"evaluating_{args.name}"
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["models"][args.name] = evaluate(args.name, args.model, args.kind)
    payload["status"] = "complete_needs_human_review"
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["models"][args.name], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
