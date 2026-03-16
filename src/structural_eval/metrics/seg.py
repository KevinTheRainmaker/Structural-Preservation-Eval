from __future__ import annotations
from .base import BaseMetric


class SegMetric(BaseMetric):
    """S_seg: 세그먼트 구조 보존.

    원문과 번안문이 비슷한 위치에서 비슷한 단위로 분절되는지를 평가한다.

    계산 방법:
        1. 문장 임베딩 생성
        2. 인접 문장 간 cosine similarity로 경계 탐지
        3. 원문/번안문 경계 위치와 세그먼트 수 비교
        S_seg = 0.6 * S_boundary + 0.4 * S_count
    """

    def compute(self, source: str, target: str) -> float:
        raise NotImplementedError
