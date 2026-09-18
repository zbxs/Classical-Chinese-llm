from __future__ import annotations

from classical_llm.cli import build_parser
from classical_llm.training.common import precision_flags
from classical_llm.utils.config import load_yaml


def test_registered_configs_load() -> None:
    for path in (
        "configs/experiments/cpt_local.yaml",
        "configs/experiments/sft_local.yaml",
        "configs/experiments/reward_local.yaml",
        "configs/experiments/dpo_local.yaml",
        "configs/experiments/grpo_local.yaml",
        "configs/profiles/full_40b.yaml",
    ):
        assert load_yaml(path)


def test_cli_rejects_unknown_stage() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "cpt", "--config", "x.yaml"])
    assert args.stage == "cpt"


def test_precision_flags_are_mutually_exclusive() -> None:
    bf16, fp16 = precision_flags({"bf16": True, "fp16": True})
    assert not (bf16 and fp16)

