from __future__ import annotations

from pathlib import Path

import pytest

from classical_llm.data.dedup import DuplicateDetector
from classical_llm.data.evaluation import build_domain_evaluation
from classical_llm.data.normalize import normalize_text
from classical_llm.data.preferences import build_preferences
from classical_llm.data.prepare import prepare_corpus
from classical_llm.data.quality import assess_quality
from classical_llm.data.sft import build_sft_dataset
from classical_llm.data.split import stable_split
from classical_llm.data.teacher import _is_classical_source, _passes_rules
from classical_llm.evaluation.score import make_blinded_review, summarize_blinded_review
from classical_llm.utils.io import read_jsonl, write_jsonl


def test_normalization_preserves_historical_characters() -> None:
    value = normalize_text("  學而時習之\u200b，\u3000不亦說乎？\r\n\r\n\r\n")
    assert value == "學而時習之， 不亦說乎？"
    assert "學" in value and "說" in value


def test_quality_rules_are_auditable() -> None:
    good = assess_quality("學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？", "classical")
    bad = assess_quality("點擊下載點擊下載點擊下載", "general_zh")
    assert good.accepted
    assert not bad.accepted
    assert "too_short" in bad.reasons


def test_exact_and_near_deduplication() -> None:
    detector = DuplicateDetector()
    text = "臣本布衣，躬耕於南陽，苟全性命於亂世。"
    assert detector.is_duplicate(text) == (False, None)
    assert detector.is_duplicate(text) == (True, "exact_duplicate")


def test_long_document_simhash_sampling_still_finds_near_duplicates() -> None:
    detector = DuplicateDetector()
    text = "天地玄黄，宇宙洪荒。" * 1000
    assert detector.is_duplicate(text) == (False, None)
    changed = text[:5000] + "日月盈昃，辰宿列张。" + text[5010:]
    assert detector.is_duplicate(changed)[0]


def test_group_split_is_stable_and_leakage_resistant() -> None:
    left = {"id": "one", "work": "史記·項羽本紀"}
    right = {"id": "two", "work": "史記·項羽本紀"}
    assert stable_split(left) == stable_split(right)


def test_prepare_corpus_filters_and_writes_audit(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    long_text = "臣本布衣，躬耕於南陽，苟全性命於亂世，不求聞達於諸侯。"
    write_jsonl(
        source,
        [
            {"id": "1", "work": "出師表", "category": "classical", "text": long_text},
            {"id": "2", "work": "出師表副本", "category": "classical", "text": long_text},
            {"id": "3", "category": "general_zh", "text": "太短"},
        ],
    )
    report = prepare_corpus([source], tmp_path / "clean")
    assert sum(report["accepted"].values()) == 1
    assert report["rejections"]["exact_duplicate"] == 1
    assert (tmp_path / "clean" / "audit.json").exists()


def _sft_seed_rows() -> list[dict]:
    return [
        {
            "id": "a",
            "source_id": "work-a",
            "split": "train",
            "task": "old_to_modern",
            "classical": "學而時習之，不亦說乎？",
            "modern": "学习之后经常温习，不也是很愉快吗？",
        },
        {
            "id": "b",
            "source_id": "work-b",
            "split": "train",
            "task": "modern_to_old",
            "modern": "知道就是知道，不知道就是不知道。",
            "classical": "知之為知之，不知為不知。",
        },
        {
            "id": "c",
            "source_id": "work-c",
            "split": "train",
            "task": "appreciation",
            "input": "海內存知己，天涯若比鄰。",
            "output": "诗句以空间距离反衬友情，‘天涯若比邻’体现真挚友情不受远隔阻碍。",
        },
        {
            "id": "d",
            "source_id": "work-d",
            "split": "train",
            "task": "creation",
            "input": "写四句五言诗，主题为山中秋雨。",
            "output": "秋雨入空山，寒雲度遠關。松聲清夜起，客夢伴潺湲。",
        },
    ]


def test_sft_and_preference_builders(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    write_jsonl(seed, _sft_seed_rows())
    quotas = {task: 1 for task in ("old_to_modern", "modern_to_old", "appreciation", "creation")}
    counts = build_sft_dataset([seed], tmp_path / "sft", quotas=quotas)
    assert counts["train"] == 4
    sft_rows = list(read_jsonl(tmp_path / "sft" / "train.jsonl"))
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft_rows)

    report = build_preferences(tmp_path / "sft" / "train.jsonl", tmp_path / "prefs", limit=4)
    assert sum(report["counts"].values()) == 4
    all_rows = []
    for split in ("train", "validation", "test"):
        all_rows.extend(read_jsonl(tmp_path / "prefs" / f"{split}.jsonl"))
    assert all(row["chosen"] != row["rejected"] for row in all_rows)


def test_evaluation_builder_requires_heldout_rows(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    rows = []
    for index, task in enumerate(("old_to_modern", "modern_to_old", "appreciation", "creation")):
        rows.append(
            {
                "id": str(index),
                "split": "test",
                "task": task,
                "prompt": f"question-{index}",
                "reference": f"answer-{index}",
            }
        )
    write_jsonl(path, rows)
    counts = build_domain_evaluation([path], tmp_path / "eval.jsonl", per_task=1)
    assert counts == {task: 1 for task in sorted(counts)}


def test_sft_quota_fails_instead_of_duplicating(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    write_jsonl(seed, _sft_seed_rows())
    with pytest.raises(ValueError, match="below quota"):
        build_sft_dataset([seed], tmp_path / "sft", quotas={"creation": 2})


def test_teacher_filter_rejects_general_web_text() -> None:
    assert not _is_classical_source({"category": "general_en", "text": "Ordinary web text."})
    assert _is_classical_source({"category": "classical", "text": "學而時習之，不亦說乎。溫故而知新。"})


def test_appreciation_requires_a_real_source_span() -> None:
    passage = "學而時習之，不亦說乎。"
    valid = "原文以“學而時習”說明反覆學習的快樂，又用反問語氣增強勸學意味。"
    generic = "語言優美，感情真摯，表達了深厚的思想感情。"
    assert _passes_rules("appreciation", valid, passage)[0]
    assert not _passes_rules("appreciation", generic, passage)[0]


def test_blinded_review_round_trip(tmp_path: Path) -> None:
    generations = tmp_path / "base.jsonl"
    write_jsonl(
        generations,
        [
            {
                "id": "q1",
                "task": "old_to_modern",
                "prompt": "prompt",
                "response": "response",
                "reference": "reference",
            }
        ],
    )
    review_path = tmp_path / "review.jsonl"
    make_blinded_review({"base": generations}, review_path)
    rows = list(read_jsonl(review_path))
    rows[0]["scores"] = {"faithfulness": 4, "completeness": 3, "fluency": 5}
    write_jsonl(review_path, rows)
    summary = summarize_blinded_review(
        review_path, review_path.with_suffix(".key.json"), tmp_path / "summary.json"
    )
    assert summary["models"]["base"]["overall_mean"] == 4
