from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"


def main() -> None:
    report_path = ROOT / "reports/generated/results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    production = [
        run
        for run in report.get("runs", [])
        if run.get("status") == "complete" and "/smoke/" not in run.get("metadata_path", "").replace("\\", "/")
    ]
    lines = [
        START,
        "### 正式运行自动摘要",
        "",
        "| 阶段输出 | 训练损失 | 运行秒数 | GPU |",
        "|---|---:|---:|---|",
    ]
    for run in production:
        metrics = run.get("metrics", {})
        env = run.get("environment", {})
        lines.append(
            f"| `{run['config']['output_dir']}` | {metrics.get('train_loss', '—')} | "
            f"{metrics.get('train_runtime', '—')} | {env.get('gpu', '—')} |"
        )
    lines.extend(
        [
            "",
            "完整训练元数据、自动指标和困惑度见 `reports/generated/results.json`；",
            "匿名人工评分表见 `outputs/evaluation/blinded_review.jsonl`。未完成人工评分时不生成或暗示人工结论。",
            END,
        ]
    )
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    block = "\n".join(lines)
    if START in text and END in text:
        prefix, remainder = text.split(START, 1)
        _, suffix = remainder.split(END, 1)
        text = f"{prefix}{block}{suffix}"
    else:
        text = f"{text.rstrip()}\n\n{block}\n"
    readme_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
