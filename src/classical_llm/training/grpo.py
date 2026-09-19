from __future__ import annotations

import re
from collections import Counter
from math import log2
from typing import Any

import sacrebleu
import torch

from classical_llm.training.common import (
    base_training_args,
    ensure_fsdp_module_compat,
    finish_training,
    load_causal_model,
    load_json_dataset,
    load_tokenizer,
    lora_for_new_model,
    quantization_config,
    resolve_model,
    resolve_resume_checkpoint,
    save_run_metadata,
)
from classical_llm.utils.config import load_yaml, resolve_project_path


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def nonempty_reward(completions, **_) -> list[float]:
    return [1.0 if 8 <= len(_completion_text(item).strip()) <= 1200 else -1.0 for item in completions]


def anti_repetition_reward(completions, **_) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        text = re.sub(r"\s+", "", _completion_text(completion))
        grams = [text[index : index + 4] for index in range(max(0, len(text) - 3))]
        ratio = len(set(grams)) / max(1, len(grams))
        scores.append(max(-1.0, min(1.0, 2 * ratio - 1)))
    return scores


def task_constraint_reward(completions, task=None, **_) -> list[float]:
    tasks = task if isinstance(task, list) else [task] * len(completions)
    scores: list[float] = []
    modern_markers = re.compile(r"(非常|这个|所以|因为|我们|总的来说)")
    for completion, item_task in zip(completions, tasks, strict=False):
        text = _completion_text(completion)
        score = 0.0
        if item_task in {"modern_to_old", "creation"}:
            score += 0.5 if not modern_markers.search(text) else -0.5
        if item_task == "appreciation":
            score += 0.5 if any(mark in text for mark in ("“", "「", "原文", "句")) else -0.25
        scores.append(score)
    return scores


def reference_chrf_reward(completions, reference=None, answer_text=None, **_) -> list[float]:
    """Bounded reference-overlap signal; never use it as the sole semantic reward."""
    references = reference if reference is not None else answer_text
    if references is None:
        raise ValueError("reference_chrf reward requires a reference or answer_text column")
    if not isinstance(references, list):
        references = [references] * len(completions)
    return [
        max(
            -1.0,
            min(
                1.0,
                sacrebleu.sentence_chrf(_completion_text(item), [str(target)]).score / 50 - 1,
            ),
        )
        for item, target in zip(completions, references, strict=True)
    ]


_NUMBER = re.compile(
    r"(?:\d+(?:\.\d+)?)|[〇零一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟]+"
)
_SOURCE_NEGATIONS = re.compile(r"不|无|未|非|莫|勿|弗|毋|否")
_OUTPUT_NEGATIONS = re.compile(r"并非|没有|尚未|不能|不可|不|无|未|非|莫|勿|弗|毋|否|没")


def _multiset_f1(expected: list[str], actual: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    matched = sum(min(count, actual_counts[item]) for item, count in expected_counts.items())
    precision = matched / max(1, sum(actual_counts.values()))
    recall = matched / max(1, sum(expected_counts.values()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def critical_token_reward(completions, source_text=None, **_) -> list[float]:
    """Check numerals and explicit negation markers without rewarding verbosity."""
    if source_text is None:
        raise ValueError("critical_token reward requires a source_text column")
    sources = source_text if isinstance(source_text, list) else [source_text] * len(completions)
    scores = []
    for completion, source in zip(completions, sources, strict=True):
        text = _completion_text(completion)
        source = str(source)
        number_score = _multiset_f1(_NUMBER.findall(source), _NUMBER.findall(text))
        source_negations = ["NEG"] * len(_SOURCE_NEGATIONS.findall(source))
        output_negations = ["NEG"] * len(_OUTPUT_NEGATIONS.findall(text))
        negation_score = _multiset_f1(source_negations, output_negations)
        scores.append(number_score + negation_score - 1)
    return scores


def source_copy_penalty_reward(completions, source_text=None, **_) -> list[float]:
    """Penalize near-verbatim source copying while leaving ordinary lexical overlap alone."""
    if source_text is None:
        raise ValueError("source_copy_penalty reward requires a source_text column")
    sources = source_text if isinstance(source_text, list) else [source_text] * len(completions)
    scores = []
    for completion, source in zip(completions, sources, strict=True):
        copy_score = sacrebleu.sentence_chrf(
            re.sub(r"\s+", "", _completion_text(completion)),
            [re.sub(r"\s+", "", str(source))],
        ).score
        scores.append(-max(0.0, min(1.0, (copy_score - 70.0) / 30.0)))
    return scores


def reference_length_reward(completions, reference=None, answer_text=None, **_) -> list[float]:
    """Symmetric weak guard against omissions and runaway verbosity."""
    references = reference if reference is not None else answer_text
    if references is None:
        raise ValueError("reference_length reward requires a reference or answer_text column")
    if not isinstance(references, list):
        references = [references] * len(completions)
    scores = []
    for completion, target in zip(completions, references, strict=True):
        length = max(1, len(re.sub(r"\s+", "", _completion_text(completion))))
        target_length = max(1, len(re.sub(r"\s+", "", str(target))))
        distance = abs(log2(length / target_length))
        scores.append(max(-1.0, 1 - distance))
    return scores


def _load_reward_adapter(path: str, policy_config: dict[str, Any]):
    """Load the frozen reward adapter and its saved sequence-classification head."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSequenceClassification

    adapter = PeftConfig.from_pretrained(path)
    kwargs: dict[str, Any] = {"num_labels": 1, "trust_remote_code": False}
    quantization = quantization_config(policy_config)
    if quantization is not None:
        kwargs.update(quantization_config=quantization, device_map={"": 0})
    elif torch.cuda.is_available():
        kwargs.update(device_map={"": 0}, dtype=torch.bfloat16)
    base = AutoModelForSequenceClassification.from_pretrained(
        adapter.base_model_name_or_path, **kwargs
    )
    reward_model = PeftModel.from_pretrained(base, path, is_trainable=False)
    reward_model.eval()
    return reward_model


def run_grpo(config_path: str) -> dict[str, Any]:
    ensure_fsdp_module_compat()
    from trl import GRPOConfig, GRPOTrainer

    config = load_yaml(config_path)
    model_path = resolve_model(config)
    reward_names = list(
        config.get(
            "reward_functions", ["reward_model", "nonempty", "anti_repetition", "task_constraint"]
        )
    )
    reward_path = None
    if "reward_model" in reward_names:
        reward_path = resolve_project_path(config["reward_model_path"])
        if not reward_path.exists():
            raise FileNotFoundError(
                f"Reward Model is required for this GRPO configuration: {reward_path}"
            )
    output_dir = resolve_project_path(config["output_dir"])
    save_run_metadata(output_dir, config, model_path, "starting")
    dataset = load_json_dataset(config["dataset_path"], config.get("validation_path"))
    # GRPO needs only prompts; preference responses must not leak into rollouts.
    keep = {"prompt", "task", "source_id", "reference", "answer_text", "source_text"}
    dataset = dataset.map(lambda row: {key: row.get(key) for key in keep}, remove_columns=[
        column for column in dataset["train"].column_names if column not in keep
    ])
    tokenizer = load_tokenizer(model_path)
    model = load_causal_model(model_path, config)
    reward_tokenizer = None
    reward_model = None
    if reward_path is not None:
        reward_tokenizer = load_tokenizer(str(reward_path))
        reward_model = _load_reward_adapter(str(reward_path), config)
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    has_eval = "validation" in dataset
    base_args = base_training_args(config, has_eval)
    # Evaluation during GRPO creates extra rollouts; disable it on the 8 GB profile.
    base_args.update(eval_strategy="no", load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None)
    args = GRPOConfig(
        **base_args,
        max_completion_length=int(config.get("max_completion_length", 384)),
        num_generations=int(config.get("num_generations", 2)),
        generation_batch_size=int(config.get("generation_batch_size", config.get("num_generations", 2))),
        beta=float(config.get("beta", 0.02)),
        temperature=float(config.get("temperature", 0.8)),
        repetition_penalty=float(config.get("repetition_penalty", 1.0)),
        loss_type=str(config.get("loss_type", "dapo")),
        scale_rewards=config.get("scale_rewards", "group"),
        mask_truncated_completions=bool(config.get("mask_truncated_completions", False)),
        reward_weights=[
            float(value) for value in config.get("reward_weights", [0.65, 0.10, 0.10, 0.15])
        ],
        log_completions=True,
        num_completions_to_print=2,
    )
    registry: dict[str, Any] = {
        "reward_model": reward_model,
        "nonempty": nonempty_reward,
        "anti_repetition": anti_repetition_reward,
        "task_constraint": task_constraint_reward,
        "reference_chrf": reference_chrf_reward,
        "critical_token": critical_token_reward,
        "source_copy_penalty": source_copy_penalty_reward,
        "reference_length": reference_length_reward,
    }
    unknown = [name for name in reward_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown GRPO reward functions: {unknown}")
    reward_funcs = [registry[name] for name in reward_names]
    reward_processing_classes = [
        reward_tokenizer if name == "reward_model" else tokenizer for name in reward_names
    ]
    if len(args.reward_weights) != len(reward_funcs):
        raise ValueError("reward_weights must match reward_functions")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        reward_processing_classes=reward_processing_classes,
        peft_config=lora_for_new_model(model, config),
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    return finish_training(trainer, tokenizer, config, model_path, result)
