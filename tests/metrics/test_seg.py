import pytest
import numpy as np
from structural_eval.metrics.seg import SegMetric


class FakeEmbedder:
    """테스트 전용 결정적 임베더.

    각 문장을 고정 벡터로 변환해 랜덤 요소를 제거한다.
    vectors 딕셔너리로 문장→벡터 지정; 미지정 문장은 zeros.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self._vectors = vectors or {}

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        dim = 4  # 테스트용 저차원
        result = []
        for s in sentences:
            v = np.array(self._vectors.get(s, [0.0] * dim), dtype=float)
            if normalize_embeddings and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
            result.append(v)
        return np.array(result)


def test_seg_metric_instantiable():
    m = SegMetric()
    assert m is not None


def test_seg_compute_raises_not_implemented():
    m = SegMetric()
    with pytest.raises(NotImplementedError):
        m.compute("Hello world.", "Hello world.")


def test_sent_tokenize_splits_sentences():
    """SegMetric 이 텍스트를 문장 단위로 분리하는지 확인."""
    m = SegMetric()
    text = "The cat sat. The mat was red. It was big."
    sents = m._sent_tokenize(text)
    assert len(sents) == 3
    assert sents[0] == "The cat sat."


def test_sent_tokenize_single_sentence():
    m = SegMetric()
    sents = m._sent_tokenize("Only one sentence here")
    assert len(sents) == 1
