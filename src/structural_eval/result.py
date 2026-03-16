from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class EvalResult:
    """6개 구조 보존 지표 점수를 담는 결과 객체."""

    seg: float    # S_seg  : 세그먼트 구조 보존 (0~1)
    topic: float  # S_topic: 화제 전개 보존 (0~1)
    order: float  # S_order: 정보 배열 보존 (0~1)
    focus: float  # S_focus: 강조 구조 보존 (0~1)
    rst: float    # S_rst  : 담화 관계 구조 보존 (0~1)
    dep: float    # S_dep  : 문장 내부 구조 보존 (0~1)

    _FIELDS: ClassVar[tuple[str, ...]] = ("seg", "topic", "order", "focus", "rst", "dep")

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

    @property
    def overall(self) -> float:
        """단순 평균 종합 점수."""
        return sum(getattr(self, f) for f in self._FIELDS) / len(self._FIELDS)

    def to_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in self._FIELDS}

    def __repr__(self) -> str:
        scores = ", ".join(f"{f}={getattr(self, f):.3f}" for f in self._FIELDS)
        return f"EvalResult({scores}, overall={self.overall:.3f})"
