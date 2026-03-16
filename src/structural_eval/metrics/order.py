from __future__ import annotations
import numpy as np
from .base import BaseMetric
from nltk.tokenize import sent_tokenize


class OrderMetric(BaseMetric):
    """S_order: 정보 배열 보존. (README §5.3)"""

    def __init__(self, embedder=None) -> None:
        """
        Args:
            embedder: sentence-transformers 호환 객체 (None이면 기본 모델 사용).
        """
        self._embedder = embedder

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _sent_tokenize(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def _embed(self, sentences: list[str]) -> np.ndarray:
        return self._get_embedder().encode(sentences, normalize_embeddings=True)

    def _match_sentences(self, src_embs: np.ndarray, tgt_embs: np.ndarray) -> np.ndarray:
        """각 원문 문장에 대해 가장 유사한 번안 문장의 인덱스 배열 반환."""
        sim_matrix = src_embs @ tgt_embs.T  # (n_src, n_tgt)
        return sim_matrix.argmax(axis=1)

    def compute(self, source: str, target: str) -> float:
        """원문과 번안문의 정보 배열 유사도를 계산한다.

        원문 문장을 번안 문장에 대응시킨 뒤 Kendall's τ 로 순서 보존도를 측정한다.

        Returns:
            S_order = (τ + 1) / 2  (0~1)
        """
        src_sents = self._sent_tokenize(source)
        tgt_sents = self._sent_tokenize(target)

        n = len(src_sents)
        if n <= 1:
            return 1.0

        src_embs = self._embed(src_sents)
        tgt_embs = self._embed(tgt_sents)

        matched = self._match_sentences(src_embs, tgt_embs)

        from scipy.stats import kendalltau
        tau, _ = kendalltau(range(n), matched)

        if np.isnan(tau):
            return 0.5

        return float((tau + 1) / 2)
