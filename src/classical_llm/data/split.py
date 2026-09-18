from __future__ import annotations

import hashlib
from typing import Any


def group_key(record: dict[str, Any]) -> str:
    """Choose a leakage-resistant key before text is split into chunks."""
    for field in ("work", "source_group", "source_id", "url", "id"):
        if record.get(field):
            return f"{field}:{record[field]}"
    raise ValueError("A record needs work/source_group/source_id/url/id for stable splitting")


def stable_split(
    record: dict[str, Any], seed: int = 20260912, train: float = 0.90, validation: float = 0.05
) -> str:
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("Split ratios must leave a non-empty test partition")
    digest = hashlib.sha256(f"{seed}:{group_key(record)}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    if point < train:
        return "train"
    if point < train + validation:
        return "validation"
    return "test"

