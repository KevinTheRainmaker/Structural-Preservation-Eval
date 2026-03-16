from __future__ import annotations
import numpy as np
from .base import BaseMetric
from nltk.tokenize import sent_tokenize


class TopicMetric(BaseMetric):
    """S_topic: 화제 전개 보존. (README §5.2)"""

    def __init__(self, embedder=None, threshold_sigma: float = 0.5) -> None:
        """
        Args:
            embedder: sentence-transformers 호환 객체 (None이면 기본 모델 사용).
            threshold_sigma: 경계 탐지 민감도 (평균 - sigma*std 이하를 경계로).
        """
        self._embedder = embedder
        self.threshold_sigma = threshold_sigma

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _sent_tokenize(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def _embed(self, sentences: list[str]) -> np.ndarray:
        return self._get_embedder().encode(sentences, normalize_embeddings=True)

    def _detect_boundaries(self, sentences: list[str]) -> set[int]:
        """인접 문장 간 cosine similarity 급락 지점을 경계 인덱스 집합으로 반환.

        인덱스 i 는 sentences[i] 와 sentences[i+1] 사이의 경계를 의미한다.
        """
        if len(sentences) <= 1:
            return set()

        embeddings = self._embed(sentences)
        sims = np.array([
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ])

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        if std_sim == 0.0:
            return set()

        threshold = mean_sim - self.threshold_sigma * std_sim
        return {i for i, s in enumerate(sims) if s < threshold}

    def _segment_embeddings(self, sentences: list[str]) -> np.ndarray:
        """각 세그먼트의 평균 임베딩 배열 반환. shape: (n_segments, dim)"""
        boundaries = sorted(self._detect_boundaries(sentences))

        segments: list[list[str]] = []
        prev = 0
        for b in boundaries:
            segments.append(sentences[prev:b + 1])
            prev = b + 1
        segments.append(sentences[prev:])

        result = []
        for seg in segments:
            if seg:
                embs = self._embed(seg)
                result.append(embs.mean(axis=0))
        return np.array(result)

    def _local_score(self, src_embs: np.ndarray, tgt_embs: np.ndarray) -> float:
        """비례 정렬된 세그먼트 쌍 간 cosine similarity 평균."""
        n, m = len(src_embs), len(tgt_embs)
        total = 0.0
        for i in range(n):
            j = round(i * (m - 1) / (n - 1)) if n > 1 else 0
            s, t = src_embs[i], tgt_embs[j]
            s_norm, t_norm = np.linalg.norm(s), np.linalg.norm(t)
            if s_norm > 0 and t_norm > 0:
                total += float(np.dot(s / s_norm, t / t_norm))
            else:
                total += 1.0
        return total / n

    def _flow_score(self, src_embs: np.ndarray, tgt_embs: np.ndarray) -> float:
        """전이 벡터 간 cosine similarity 평균."""
        if len(src_embs) <= 1 or len(tgt_embs) <= 1:
            return 1.0

        src_trans = src_embs[1:] - src_embs[:-1]
        tgt_trans = tgt_embs[1:] - tgt_embs[:-1]

        ns, nt = len(src_trans), len(tgt_trans)
        total = 0.0
        for i in range(ns):
            j = round(i * (nt - 1) / (ns - 1)) if ns > 1 else 0
            s, t = src_trans[i], tgt_trans[j]
            s_norm, t_norm = np.linalg.norm(s), np.linalg.norm(t)
            if s_norm > 0 and t_norm > 0:
                total += float(np.dot(s / s_norm, t / t_norm))
            else:
                total += 1.0
        return total / ns

    def compute(self, source: str, target: str) -> float:
        """원문과 번안문의 화제 전개 유사도를 계산한다.

        Returns:
            S_topic = 0.7 * S_local + 0.3 * S_flow  (0~1)
        """
        src_sents = self._sent_tokenize(source)
        tgt_sents = self._sent_tokenize(target)

        src_embs = self._segment_embeddings(src_sents)
        tgt_embs = self._segment_embeddings(tgt_sents)

        s_local = self._local_score(src_embs, tgt_embs)
        s_flow = self._flow_score(src_embs, tgt_embs)

        return 0.7 * s_local + 0.3 * s_flow
