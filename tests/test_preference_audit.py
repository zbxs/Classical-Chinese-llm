from classical_llm.evaluation.preference_audit import (
    build_judge_presentations,
    candidate_utility,
    compare_candidate_scores,
    extract_source,
    merge_judgments,
    pair_features,
)


def _row() -> dict:
    return {
        "id": "sample",
        "source_id": "source:1",
        "prompt": "任务：翻译。\n原文：三人未至。\n译文：",
        "chosen": "三个人尚未到。",
        "rejected": "两个人已经到了。",
        "rejected_chrf": 10.0,
    }


def test_extract_source_and_constraints() -> None:
    row = _row()
    assert extract_source(row["prompt"]) == "三人未至。"
    features = pair_features(row)
    assert features["has_number_constraint"]
    assert features["has_negation_constraint"]
    assert features["chosen_number_recall"] > features["rejected_number_recall"]
    assert features["chosen_negation_recall"] > features["rejected_negation_recall"]


def test_position_swapped_judgments_merge() -> None:
    presentations = build_judge_presentations({"train": [_row()]}, seed=7)
    assert len(presentations) == 2
    assert presentations[0]["option_a_role"] != presentations[1]["option_a_role"]
    judgments = []
    for item in presentations:
        chosen_key = "a" if item["option_a_role"] == "chosen" else "b"
        rejected_key = "b" if chosen_key == "a" else "a"
        judge = {
            "winner": chosen_key.upper(),
            chosen_key: {
                "faithfulness": 4,
                "completeness": 4,
                "fluency": 2,
                "hallucination": 0,
            },
            rejected_key: {
                "faithfulness": 1,
                "completeness": 1,
                "fluency": 2,
                "hallucination": 2,
            },
        }
        judgments.append({"presentation_id": item["presentation_id"], "judge": judge})
    merged, report = merge_judgments(presentations, judgments)
    assert report["double_chosen_rate"] == 1.0
    assert merged[0]["high_confidence"]


def test_independent_score_comparison_is_fidelity_first() -> None:
    faithful = {
        "faithfulness": 4,
        "completeness": 3,
        "fluency": 1,
        "hallucination": 0,
    }
    fluent_hallucination = {
        "faithfulness": 3,
        "completeness": 4,
        "fluency": 2,
        "hallucination": 2,
    }
    assert candidate_utility(faithful) > candidate_utility(fluent_hallucination)
    assert compare_candidate_scores(faithful, fluent_hallucination) == "first"
    assert compare_candidate_scores(faithful, dict(faithful)) == "tie"


def test_independent_score_comparison_rejects_out_of_range_values() -> None:
    invalid = {
        "faithfulness": 5,
        "completeness": 3,
        "fluency": 1,
        "hallucination": 0,
    }
    valid = {
        "faithfulness": 4,
        "completeness": 3,
        "fluency": 1,
        "hallucination": 0,
    }
    assert candidate_utility(invalid) is None
    assert compare_candidate_scores(invalid, valid) == "invalid"
