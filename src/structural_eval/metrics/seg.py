from __future__ import annotations
from .base import BaseMetric
from nltk.tokenize import sent_tokenize


class SegMetric(BaseMetric):
    """S_seg: 세그먼트 구조 보존. (README §5.1)"""

    def __init__(self, embedder=None, threshold_sigma: float = 0.5, delta: float = 0.2) -> None:
        """
        Args:
            embedder: sentence-transformers 호환 객체 (None이면 기본 모델 사용).
            threshold_sigma: 경계 탐지 민감도 (평균 - sigma*std 이하를 경계로).
            delta: 경계 위치 허용 오차 (원문 문장 수 대비 비율).
        """
        self._embedder = embedder  # 지연 로딩
        self.threshold_sigma = threshold_sigma
        self.delta = delta

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _sent_tokenize(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def compute(self, source: str, target: str) -> float:
        raise NotImplementedError
