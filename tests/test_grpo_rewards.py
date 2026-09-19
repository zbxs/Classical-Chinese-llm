import pytest

from classical_llm.training.grpo import (
    critical_token_reward,
    reference_chrf_reward,
    reference_length_reward,
    source_copy_penalty_reward,
)


def test_reference_chrf_reward_prefers_exact_translation() -> None:
    scores = reference_chrf_reward(
        ["三个人尚未到。", "天气很好。"],
        reference=["三个人尚未到。", "三个人尚未到。"],
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[0] > scores[1]


def test_critical_token_reward_tracks_number_and_negation() -> None:
    scores = critical_token_reward(
        ["三个人尚未到。", "两个人已经到了。"],
        source_text=["三人未至。", "三人未至。"],
    )
    assert scores[0] > scores[1]


def test_critical_token_reward_normalizes_modern_negation() -> None:
    scores = critical_token_reward(
        ["这里没有军队。", "这里有军队。"],
        source_text=["此地无军。", "此地无军。"],
    )
    assert scores[0] > scores[1]


def test_reference_length_reward_is_symmetric() -> None:
    scores = reference_length_reward(
        ["甲乙丙丁", "甲乙", "甲乙丙丁甲乙丙丁"],
        reference=["甲乙丙丁"] * 3,
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(scores[2])


def test_source_copy_penalty_targets_verbatim_echo() -> None:
    scores = source_copy_penalty_reward(
        ["三人未至。", "三个人尚未到达。"],
        source_text=["三人未至。", "三人未至。"],
    )
    assert scores[0] == pytest.approx(-1.0)
    assert scores[1] > scores[0]
