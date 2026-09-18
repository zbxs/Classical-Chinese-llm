"""Validate completed evaluation and make a private, non-destructive backup.

This archive is a private backup, NOT a licence-cleared public release.
Run from the repository root using its isolated Python environment.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from classical_llm.evaluation.score import make_blinded_review
from classical_llm.utils.io import read_jsonl


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluation = root / "outputs/evaluation"
    names = ("base", "cpt", "sft", "dpo", "grpo")
    expected = list(read_jsonl(root / "data/final/evaluation/domain_400.jsonl"))
    ids = [row["id"] for row in expected]
    assert len(ids) == len(set(ids)) == 400
    report = json.loads((root / "reports/generated/results.json").read_text())
    assert len(report["evaluation_summaries"]) == 5
    assert len(report["perplexity_results"]) == 15
    audit = {"checked_at": datetime.now(UTC).isoformat(), "models": {}}
    for name in names:
        rows = list(read_jsonl(evaluation / f"{name}.jsonl"))
        assert [row["id"] for row in rows] == ids, name
        assert all(row.get("messages") and row.get("response") for row in rows), name
        tasks = Counter(row["task"] for row in rows)
        assert len(tasks) == 4 and set(tasks.values()) == {100}, name
        scored = list(read_jsonl(evaluation / f"{name}.scored.jsonl"))
        assert [row["id"] for row in scored] == ids
        assert all("chrf" in row["automatic_metrics"] for row in scored)
        audit["models"][name] = {"rows": len(rows), "tasks": dict(tasks),
                                  "distinct_responses": len({row["response"] for row in rows})}
    blind = evaluation / "blinded_review.jsonl"
    old = evaluation / "blinded_review.before-prompt-fix.jsonl"
    if blind.exists():
        existing = list(read_jsonl(blind))
        assert not any(any(v is not None for v in r.get("scores", {}).values()) for r in existing), \
            "Do not overwrite human scores"
        if not old.exists():
            shutil.copy2(blind, old)
    make_blinded_review({name: evaluation / f"{name}.jsonl" for name in names}, blind)
    review = list(read_jsonl(blind))
    assert len(review) == 2000 and all(row["prompt"] for row in review)
    audit["human_review"] = "not performed; 2000 populated question/response forms"
    (root / "reports/generated/final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    destination = root / "transfer"
    destination.mkdir(exist_ok=True)
    archive = destination / "experiment-backup-20260918.tar.gz"
    if archive.exists():
        raise FileExistsError(f"Preserve existing backup: {archive}")
    subprocess.run(["tar", "-czf", str(archive), "--exclude=__pycache__",
                    "--exclude=checkpoint-*", "--exclude=.cache", "--exclude=ssh",
                    "data", "outputs", "reports", "configs", "src", "tests", "scripts",
                    "docs", "requirements", "README.md", "pyproject.toml", ".gitignore",
                    ".gitattributes"], cwd=root, check=True)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(f"Backup ready: {archive}; sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
