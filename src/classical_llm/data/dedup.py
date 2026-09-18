from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from classical_llm.data.normalize import normalized_for_dedup


def simhash64(text: str, max_features: int = 512) -> int:
    """Return a sampled, deterministic 64-bit character-trigram SimHash.

    Long documents use evenly spaced shingles so near-duplicate detection stays
    bounded on production corpora while still representing the whole document.
    """
    normalized = normalized_for_dedup(text)
    if len(normalized) < 3:
        normalized = normalized.ljust(3, "_")
    feature_count = len(normalized) - 2
    if feature_count <= max_features:
        positions = range(feature_count)
    else:
        positions = (index * feature_count // max_features for index in range(max_features))
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(normalized[index : index + 3].encode(), digest_size=8).digest(),
                "little",
            )
            for index in positions
        ),
        dtype=np.uint64,
    )
    bits = np.unpackbits(hashes.view(np.uint8), bitorder="little").reshape(-1, 64)
    ones = bits.sum(axis=0)
    result = 0
    for bit, count in enumerate(ones):
        if int(count) * 2 >= len(hashes):
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(slots=True)
class DuplicateDetector:
    """Exact plus bucketed near-duplicate detection for local-sized corpora."""

    max_hamming_distance: int = 3
    _exact: set[str] = field(default_factory=set, init=False, repr=False)
    _buckets: dict[int, list[int]] = field(default_factory=dict, init=False, repr=False)

    def is_duplicate(self, text: str) -> tuple[bool, str | None]:
        normalized = normalized_for_dedup(text)
        exact = hashlib.sha256(normalized.encode()).hexdigest()
        if exact in self._exact:
            return True, "exact_duplicate"

        fingerprint = simhash64(normalized)
        # Four 16-bit bands keep comparisons local without third-party services.
        candidates: set[int] = set()
        for band in range(4):
            key = (band << 16) | ((fingerprint >> (band * 16)) & 0xFFFF)
            candidates.update(self._buckets.get(key, []))
        if any(hamming_distance(fingerprint, item) <= self.max_hamming_distance for item in candidates):
            return True, "near_duplicate"

        self._exact.add(exact)
        for band in range(4):
            key = (band << 16) | ((fingerprint >> (band * 16)) & 0xFFFF)
            self._buckets.setdefault(key, []).append(fingerprint)
        return False, None
