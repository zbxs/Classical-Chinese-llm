"""Summarize position bias and schema quality in a double-pass preference judge run."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from classical_llm.evaluation.preference_audit import (
    merge_judgments,
    normalize_winner,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def score_winner(judge: dict) -> str:
    """Derive a winner from the declared rubric instead of the free-form winner token."""
    scores: dict[str, float] = {}
    for key in ("a", "b"):
        value = judge.get(key)
        if not isinstance(value, dict):
            return "INVALID"
        try:
            scores[key] = (
                4 * float(value["faithfulness"])
                + 2 * float(value["completeness"])
                + float(value["fluency"])
                - 4 * float(value["hallucination"])
            )
        except (KeyError, TypeError, ValueError):
            return "INVALID"
    difference = scores["a"] - scores["b"]
    if abs(difference) <= 1:
        return "TIE"
    return "A" if difference > 0 else "B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge-v1/judge_input.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge-v1/judge_output.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/preference-diagnosis-v1/judge-v1/bias_report.json",
    )
    args = parser.parse_args()

    presentations = read_jsonl(args.input)
    judgments = read_jsonl(args.output)
    presentation_by_id = {row["presentation_id"]: row for row in presentations}
    raw_winners: Counter[str] = Counter()
    winner_by_order: dict[int, Counter[str]] = defaultdict(Counter)
    role_by_order: dict[int, Counter[str]] = defaultdict(Counter)
    pair_raw: dict[str, dict[int, str]] = defaultdict(dict)
    pair_scored_roles: dict[str, dict[int, str]] = defaultdict(dict)
    parse_errors = 0
    missing_reason = 0
    invalid_schema = 0
    spurious_c_mentions = 0

    for row in judgments:
        presentation = presentation_by_id[str(row["presentation_id"])]
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        winner = str(judge.get("winner", "INVALID")).strip().upper()
        order = int(presentation["order"])
        raw_winners[winner] += 1
        winner_by_order[order][winner] += 1
        role_by_order[order][normalize_winner(presentation, judge)] += 1
        pair_raw[str(row["pair_id"])][order] = winner
        scored = score_winner(judge)
        pair_scored_roles[str(row["pair_id"])][order] = normalize_winner(
            presentation, {"winner": scored}
        )
        parse_errors += row.get("parse_error") is not None
        missing_reason += not bool(str(judge.get("reason", "")).strip())
        spurious_c_mentions += "C" in str(judge.get("reason", "")).upper()
        if winner not in {"A", "B", "TIE", "INVALID"}:
            invalid_schema += 1
        for key in ("a", "b"):
            scores = judge.get(key)
            if not isinstance(scores, dict) or not all(
                name in scores
                for name in ("faithfulness", "completeness", "fluency", "hallucination")
            ):
                invalid_schema += 1
                break

    raw_pair_patterns = Counter(
        f"{orders.get(0, 'MISSING')}/{orders.get(1, 'MISSING')}"
        for orders in pair_raw.values()
    )
    scored_pair_patterns = Counter(
        f"{orders.get(0, 'missing')}/{orders.get(1, 'missing')}"
        for orders in pair_scored_roles.values()
    )
    score_complete = [orders for orders in pair_scored_roles.values() if len(orders) == 2]
    score_consistent = sum(orders[0] == orders[1] for orders in score_complete)
    merged, merged_report = merge_judgments(presentations, judgments)
    total = max(1, len(judgments))
    report = {
        "presentations": len(presentations),
        "judgments": len(judgments),
        "parse_error_count": parse_errors,
        "parse_error_rate": parse_errors / total,
        "missing_reason_count": missing_reason,
        "missing_reason_rate": missing_reason / total,
        "spurious_c_mention_count": spurious_c_mentions,
        "invalid_schema_count": invalid_schema,
        "raw_winner_counts": dict(raw_winners),
        "raw_winner_by_order": {
            str(order): dict(counts) for order, counts in sorted(winner_by_order.items())
        },
        "normalized_role_by_order": {
            str(order): dict(counts) for order, counts in sorted(role_by_order.items())
        },
        "raw_pair_patterns": dict(raw_pair_patterns.most_common()),
        "score_derived": {
            "formula": "4*faithfulness + 2*completeness + fluency - 4*hallucination",
            "tie_margin": 1,
            "position_consistency_rate": score_consistent / max(1, len(score_complete)),
            "outcome_patterns": dict(scored_pair_patterns.most_common()),
        },
        "merged": merged_report,
        "inconsistent_pair_examples": [
            {
                "pair_id": row["pair_id"],
                "winner_roles": row["winner_roles"],
                "reasons": row["reasons"],
            }
            for row in merged
            if not row["position_consistent"]
        ][:20],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
