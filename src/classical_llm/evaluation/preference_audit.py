from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SOURCE = re.compile(r"原文[：:]\s*(.*?)\s*\n\s*译文[：:]", re.DOTALL)
_NUMBER = re.compile(r"(?:\d+(?:\.\d+)?)|[〇零一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟]+")
_NEGATIONS = ("不", "无", "未", "非", "莫", "勿", "弗", "毋", "否", "不能", "不可")


def compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def extract_source(prompt: str) -> str:
    match = _SOURCE.search(prompt)
    if not match:
        raise ValueError(f"Cannot extract source from prompt: {prompt[:120]!r}")
    return match.group(1).strip()


def _multiset_recall(expected: Iterable[str], actual: Iterable[str]) -> float:
    expected_counts = Counter(expected)
    if not expected_counts:
        return 1.0
    actual_counts = Counter(actual)
    matched = sum(min(count, actual_counts[item]) for item, count in expected_counts.items())
    return matched / sum(expected_counts.values())


def pair_features(row: dict[str, Any]) -> dict[str, Any]:
    source = extract_source(str(row["prompt"]))
    chosen = str(row["chosen"]).strip()
    rejected = str(row["rejected"]).strip()
    chosen_len = compact_length(chosen)
    rejected_len = compact_length(rejected)
    source_numbers = _NUMBER.findall(source)
    source_negations = [mark for mark in _NEGATIONS if mark in source]
    return {
        "source": source,
        "chosen_length": chosen_len,
        "rejected_length": rejected_len,
        "length_ratio": chosen_len / max(1, rejected_len),
        "chosen_number_recall": _multiset_recall(source_numbers, _NUMBER.findall(chosen)),
        "rejected_number_recall": _multiset_recall(source_numbers, _NUMBER.findall(rejected)),
        "chosen_negation_recall": _multiset_recall(
            source_negations, [mark for mark in _NEGATIONS if mark in chosen]
        ),
        "rejected_negation_recall": _multiset_recall(
            source_negations, [mark for mark in _NEGATIONS if mark in rejected]
        ),
        "has_number_constraint": bool(source_numbers),
        "has_negation_constraint": bool(source_negations),
    }


def _describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = [pair_features(row) for row in rows]
    chosen_lengths = [item["chosen_length"] for item in features]
    rejected_lengths = [item["rejected_length"] for item in features]
    with_numbers = [item for item in features if item["has_number_constraint"]]
    with_negations = [item for item in features if item["has_negation_constraint"]]
    bands = Counter(
        "low" if float(row["rejected_chrf"]) < 20 else
        "medium" if float(row["rejected_chrf"]) < 40 else "hard"
        for row in rows
    )
    return {
        "count": len(rows),
        "chosen_length": _describe([float(value) for value in chosen_lengths]),
        "rejected_length": _describe([float(value) for value in rejected_lengths]),
        "chosen_longer_rate": sum(a > b for a, b in zip(chosen_lengths, rejected_lengths, strict=True))
        / max(1, len(rows)),
        "chosen_1_5x_rate": sum(a >= 1.5 * b for a, b in zip(chosen_lengths, rejected_lengths, strict=True))
        / max(1, len(rows)),
        "rejected_1_5x_rate": sum(b >= 1.5 * a for a, b in zip(chosen_lengths, rejected_lengths, strict=True))
        / max(1, len(rows)),
        "length_balanced_rate": sum(0.8 <= item["length_ratio"] <= 1.25 for item in features)
        / max(1, len(rows)),
        "number_cases": len(with_numbers),
        "chosen_number_recall": _describe([item["chosen_number_recall"] for item in with_numbers]),
        "rejected_number_recall": _describe([item["rejected_number_recall"] for item in with_numbers]),
        "negation_cases": len(with_negations),
        "chosen_negation_recall": _describe([item["chosen_negation_recall"] for item in with_negations]),
        "rejected_negation_recall": _describe([item["rejected_negation_recall"] for item in with_negations]),
        "difficulty_bands": dict(sorted(bands.items())),
    }


def build_judge_presentations(
    splits: dict[str, list[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split, rows in splits.items():
        for row in rows:
            features = pair_features(row)
            first_chosen = int(hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), 16) % 2 == 0
            for order in (0, 1):
                chosen_is_a = first_chosen if order == 0 else not first_chosen
                output.append({
                    "presentation_id": f"{split}:{row['id']}:{order}",
                    "pair_id": str(row["id"]),
                    "split": split,
                    "order": order,
                    "source_id": row.get("source_id"),
                    "source": features["source"],
                    "option_a": row["chosen"] if chosen_is_a else row["rejected"],
                    "option_b": row["rejected"] if chosen_is_a else row["chosen"],
                    "option_a_role": "chosen" if chosen_is_a else "rejected",
                    "option_b_role": "rejected" if chosen_is_a else "chosen",
                    "features": features,
                })
    random.Random(seed).shuffle(output)
    return output


def normalize_winner(presentation: dict[str, Any], judge: dict[str, Any]) -> str:
    winner = str(judge.get("winner", "INVALID")).strip().upper()
    if winner == "A":
        return str(presentation["option_a_role"])
    if winner == "B":
        return str(presentation["option_b_role"])
    if winner == "TIE":
        return "tie"
    return "invalid"


def candidate_utility(scores: dict[str, Any]) -> float | None:
    """Map rubric scores to a fidelity-first scalar used by independent judging."""
    try:
        faithfulness = float(scores["faithfulness"])
        completeness = float(scores["completeness"])
        fluency = float(scores["fluency"])
        hallucination = float(scores["hallucination"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        0 <= faithfulness <= 4
        and 0 <= completeness <= 4
        and 0 <= fluency <= 2
        and 0 <= hallucination <= 2
    ):
        return None
    return 4 * faithfulness + 2 * completeness + fluency - 4 * hallucination


def compare_candidate_scores(
    first: dict[str, Any], second: dict[str, Any], tie_margin: float = 1.0
) -> str:
    """Compare independently produced rubric scores without exposing answer order."""
    first_utility = candidate_utility(first)
    second_utility = candidate_utility(second)
    if first_utility is None or second_utility is None:
        return "invalid"
    difference = first_utility - second_utility
    if abs(difference) <= tie_margin:
        return "tie"
    return "first" if difference > 0 else "second"


def _role_scores(presentation: dict[str, Any], judge: dict[str, Any], role: str) -> dict[str, float]:
    key = "a" if presentation["option_a_role"] == role else "b"
    raw = judge.get(key, {}) if isinstance(judge.get(key), dict) else {}
    return {
        name: float(raw.get(name, -1))
        for name in ("faithfulness", "completeness", "fluency", "hallucination")
    }


def merge_judgments(
    presentations: list[dict[str, Any]], judgment_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_presentation = {row["presentation_id"]: row for row in presentations}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgment_rows:
        presentation = by_presentation.get(str(row.get("presentation_id")))
        judge = row.get("judge")
        if presentation is None or not isinstance(judge, dict):
            continue
        by_pair[str(presentation["pair_id"])].append({
            "presentation": presentation,
            "winner_role": normalize_winner(presentation, judge),
            "chosen_scores": _role_scores(presentation, judge, "chosen"),
            "rejected_scores": _role_scores(presentation, judge, "rejected"),
            "reason": str(judge.get("reason", "")),
        })
    merged: list[dict[str, Any]] = []
    for pair_id, judgments in by_pair.items():
        if len(judgments) != 2:
            continue
        judgments.sort(key=lambda item: int(item["presentation"]["order"]))
        winners = [item["winner_role"] for item in judgments]
        chosen_faithfulness = [item["chosen_scores"]["faithfulness"] for item in judgments]
        rejected_faithfulness = [item["rejected_scores"]["faithfulness"] for item in judgments]
        chosen_hallucination = [item["chosen_scores"]["hallucination"] for item in judgments]
        rejected_hallucination = [item["rejected_scores"]["hallucination"] for item in judgments]
        features = judgments[0]["presentation"]["features"]
        high_confidence = (
            winners == ["chosen", "chosen"]
            and min(chosen_faithfulness) >= max(rejected_faithfulness) + 1
            and max(chosen_hallucination) <= min(rejected_hallucination)
            and 0.8 <= float(features["length_ratio"]) <= 1.25
        )
        merged.append({
            "pair_id": pair_id,
            "split": judgments[0]["presentation"]["split"],
            "source_id": judgments[0]["presentation"].get("source_id"),
            "winner_roles": winners,
            "position_consistent": winners[0] == winners[1],
            "chosen_faithfulness": chosen_faithfulness,
            "rejected_faithfulness": rejected_faithfulness,
            "chosen_hallucination": chosen_hallucination,
            "rejected_hallucination": rejected_hallucination,
            "high_confidence": high_confidence,
            "features": features,
            "reasons": [item["reason"] for item in judgments],
        })
    counts = Counter(
        tuple(item["winner_roles"]) if item["position_consistent"] else ("inconsistent",)
        for item in merged
    )
    report = {
        "complete_pairs": len(merged),
        "position_consistency_rate": sum(item["position_consistent"] for item in merged)
        / max(1, len(merged)),
        "double_chosen_rate": sum(item["winner_roles"] == ["chosen", "chosen"] for item in merged)
        / max(1, len(merged)),
        "double_rejected_rate": sum(item["winner_roles"] == ["rejected", "rejected"] for item in merged)
        / max(1, len(merged)),
        "high_confidence_count": sum(item["high_confidence"] for item in merged),
        "outcome_counts": {"/".join(key): value for key, value in sorted(counts.items())},
    }
    return merged, report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
