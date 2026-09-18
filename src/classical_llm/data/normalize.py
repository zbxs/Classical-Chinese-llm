from __future__ import annotations

import html
import re
import unicodedata

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")
_HORIZONTAL_SPACE = re.compile(r"[\t\v\f\u00a0\u3000 ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Conservatively normalize text while preserving historical characters.

    NFC is intentional: NFKC can fold compatibility ideographs and destroy
    distinctions that matter in Classical Chinese source material.
    """
    value = html.unescape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFC", value)
    value = _ZERO_WIDTH.sub("", value)
    value = _CONTROL.sub("", value)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = _BLANK_LINES.sub("\n\n", value)
    return value.strip()


def normalized_for_dedup(text: str) -> str:
    """Normalize presentation differences for duplicate detection only."""
    value = normalize_text(text).lower()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)

