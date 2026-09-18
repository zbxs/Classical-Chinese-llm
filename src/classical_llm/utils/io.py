from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from a UTF-8 JSONL file."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {source}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically write JSONL and return the number of rows."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return count


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
