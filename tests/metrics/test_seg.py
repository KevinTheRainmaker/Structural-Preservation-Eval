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


def make_metric(vectors=None, threshold_sigma=0.5, delta=0.2):
    return SegMetric(embedder=FakeEmbedder(vectors), threshold_sigma=threshold_sigma, delta=delta)


def test_detect_boundaries_empty_for_single_sentence():
    m = make_metric()
    # 문장 1개 → 인접 쌍 없음 → 경계 없음
    boundaries = m._detect_boundaries(["Only one."])
    assert boundaries == set()


def test_detect_boundaries_all_similar_no_boundary():
    """모든 인접 쌍 유사도가 동일하면 분산=0, threshold>=mean → 경계 없음."""
    v = {"A": [1.0, 0.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0, 0.0], "C": [1.0, 0.0, 0.0, 0.0]}
    m = make_metric(vectors=v)
    boundaries = m._detect_boundaries(["A", "B", "C"])
    assert boundaries == set()


def test_detect_boundaries_detects_drop():
    """유사도가 급락하는 쌍을 경계로 탐지한다.

    A-B 는 거의 동일 방향 (sim≈1), B-C 는 직교 (sim≈0).
    경계는 인덱스 1 (B 와 C 사이).
    """
    v = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [1.0, 0.0, 0.0, 0.0],  # A 와 동일
        "C": [0.0, 1.0, 0.0, 0.0],  # B 와 직교
    }
    m = make_metric(vectors=v, threshold_sigma=0.5)
    boundaries = m._detect_boundaries(["A", "B", "C"])
    assert 1 in boundaries  # index 1 = B↔C 경계
