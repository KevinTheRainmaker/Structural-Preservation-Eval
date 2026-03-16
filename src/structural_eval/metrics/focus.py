from __future__ import annotations
import numpy as np
from .base import BaseMetric
from nltk.tokenize import sent_tokenize, word_tokenize

_CUE_PHRASES = {
    "first", "second", "third", "finally", "importantly", "notably",
    "significantly", "crucially", "critically", "above all", "in conclusion",
    "in summary", "key", "main", "primary", "central", "essential",
    "fundamentally", "most importantly",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "my", "not", "no", "as", "if",
}


class FocusMetric(BaseMetric):
    """S_focus: 강조 구조 보존. (README §5.4)

    spaCy 없이 위치(P), 반복 키워드(R), cue phrase(C)로 salience를 계산한다.
    """

    def __init__(
        self,
        embedder=None,
        k: int = 3,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
    ) -> None:
        """
        Args:
            embedder: sentence-transformers 호환 객체 (None이면 기본 모델 사용).
            k: 상위 salient unit 개수.
            alpha: 위치 가중치 계수.
            beta: 반복 키워드 계수.
            gamma: cue phrase 계수.
        """
        self._embedder = embedder
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _sent_tokenize(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def _embed(self, sentences: list[str]) -> np.ndarray:
        return self._get_embedder().encode(sentences, normalize_embeddings=True)

    def _content_words(self, text: str) -> list[str]:
        words = word_tokenize(text.lower())
        return [w for w in words if w.isalpha() and w not in _STOPWORDS]

    def _position_weights(self, n: int) -> np.ndarray:
        """위치 가중치: 앞 문장일수록 높음. 범위 [0.5, 1.0]."""
        if n == 1:
            return np.array([1.0])
        return np.array([1.0 - i / (n - 1) * 0.5 for i in range(n)])

    def _repetition_weights(self, sentences: list[str]) -> np.ndarray:
        """TF 기반 반복 키워드 점수. 최대값으로 정규화."""
        all_words: list[str] = []
        for s in sentences:
            all_words.extend(self._content_words(s))

        if not all_words:
            return np.zeros(len(sentences))

        total = len(all_words)
        tf: dict[str, float] = {}
        for w in all_words:
            tf[w] = tf.get(w, 0) + 1 / total

        weights = []
        for s in sentences:
            words = self._content_words(s)
            if words:
                weights.append(sum(tf.get(w, 0.0) for w in words) / len(words))
            else:
                weights.append(0.0)

        arr = np.array(weights)
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        return arr

    def _cue_weights(self, sentences: list[str]) -> np.ndarray:
        """cue phrase 존재 여부 (0 또는 1)."""
        weights = []
        for s in sentences:
            s_lower = s.lower()
            found = any(cue in s_lower for cue in _CUE_PHRASES)
            weights.append(1.0 if found else 0.0)
        return np.array(weights)

    def _salience(self, sentences: list[str]) -> np.ndarray:
        """각 문장의 강조 점수 배열 반환."""
        n = len(sentences)
        p = self._position_weights(n)
        r = self._repetition_weights(sentences)
        c = self._cue_weights(sentences)
        return self.alpha * p + self.beta * r + self.gamma * c

    def _top_k_indices(self, salience: np.ndarray) -> list[int]:
        """상위 k개 강조 문장 인덱스 반환 (높은 순)."""
        k = min(self.k, len(salience))
        return list(np.argsort(salience)[-k:][::-1])

    def compute(self, source: str, target: str) -> float:
        """원문과 번안문의 강조 구조 유사도를 계산한다.

        Returns:
            S_focus = (1/k) * sum_i max_j cos(f_i, g_j)  (0~1)
        """
        src_sents = self._sent_tokenize(source)
        tgt_sents = self._sent_tokenize(target)

        src_sal = self._salience(src_sents)
        tgt_sal = self._salience(tgt_sents)

        src_idx = self._top_k_indices(src_sal)
        tgt_idx = self._top_k_indices(tgt_sal)

        src_top = [src_sents[i] for i in src_idx]
        tgt_top = [tgt_sents[i] for i in tgt_idx]

        src_embs = self._embed(src_top)
        tgt_embs = self._embed(tgt_top)

        sim_matrix = src_embs @ tgt_embs.T  # (k_src, k_tgt)
        return float(sim_matrix.max(axis=1).mean())
