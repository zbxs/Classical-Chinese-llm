from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "classical_05": ROOT / "outputs/cpt/qwen2.5-0.5b-classical05",
    "classical_20": ROOT / "outputs/cpt/qwen2.5-0.5b-classical20",
    "classical_50": ROOT / "outputs/cpt/qwen2.5-0.5b-classical50",
}


def main() -> None:
    scores = []
    for name, output in CANDIDATES.items():
        state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
        metric = state.get("best_metric")
        if metric is None:
            raise RuntimeError(f"{name} has no best validation metric")
        scores.append(
            {
                "name": name,
                "output": str(output.relative_to(ROOT)).replace("\\", "/"),
                "best_eval_loss": float(metric),
                "best_global_step": int(state["best_global_step"]),
            }
        )
    scores.sort(key=lambda item: item["best_eval_loss"])
    selected = scores[0]

    report_dir = ROOT / "reports/generated"
    generated_dir = ROOT / "configs/generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cpt_selection.json").write_text(
        json.dumps(
            {
                "criterion": "minimum fixed-validation eval_loss",
                "selected": selected,
                "candidates": scores,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = yaml.safe_load((ROOT / "configs/remote/sft_4090d.yaml").read_text(encoding="utf-8"))
    config["model_name_or_path"] = f"{selected['output']}/best"
    (generated_dir / "sft_selected_4090d.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False))


if __name__ == "__main__":
    main()
