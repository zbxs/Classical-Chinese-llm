from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_REPEATED_CHAR = re.compile(r"(.)\1{11,}")
_WEB_NOISE = re.compile(
    r"(点击下载|立即注册|ICP备|cookie policy|javascript is required|广告合作)", re.IGNORECASE
)
_STEM_TERMS = re.compile(
    r"(定理|证明|方程|函数|算法|数据结构|物理|化学|生物|工程|programming|algorithm|theorem)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class QualityDecision:
    accepted: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def _line_duplicate_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    return 1.0 - len(set(lines)) / len(lines)


def assess_quality(text: str, category: str = "general") -> QualityDecision:
    """Apply auditable, deliberately conservative document quality rules."""
    reasons: list[str] = []
    length = len(text)
    minimum = 20 if category == "classical" else 80
    if length < minimum:
        reasons.append("too_short")
    if length > 2_000_000:
        reasons.append("too_long")
    if "�" in text:
        reasons.append("replacement_character")
    if _REPEATED_CHAR.search(text):
        reasons.append("repeated_character_run")
    if _WEB_NOISE.search(text):
        reasons.append("web_boilerplate")
    if _line_duplicate_ratio(text) > 0.45:
        reasons.append("repeated_lines")

    visible = [char for char in text if not char.isspace()]
    cjk_ratio = len(_CJK.findall(text)) / max(1, len(visible))
    if category in {"classical", "general_zh", "stem_technical"} and cjk_ratio < 0.25:
        reasons.append("low_cjk_ratio")

    penalties = min(1.0, len(reasons) * 0.22)
    length_bonus = min(0.20, math.log10(max(length, 10)) / 20)
    score = round(max(0.0, 0.8 + length_bonus - penalties), 4)
    return QualityDecision(not reasons, score, reasons)


def looks_stem_or_technical(text: str) -> bool:
    return len(_STEM_TERMS.findall(text)) >= 2

