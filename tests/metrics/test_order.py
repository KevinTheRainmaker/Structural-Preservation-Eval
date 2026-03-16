import pytest
import numpy as np
from structural_eval.metrics.order import OrderMetric


class FakeEmbedder:
    def __init__(self, vectors=None):
        self._vectors = vectors or {}

    def encode(self, sentences, normalize_embeddings=True):
        dim = 4
        result = []
        for s in sentences:
            v = np.array(self._vectors.get(s, [0.0] * dim), dtype=float)
            if normalize_embeddings and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
            result.append(v)
        return np.array(result)


def make_metric(vectors=None):
    return OrderMetric(embedder=FakeEmbedder(vectors))


def test_order_metric_instantiable():
    m = OrderMetric()
    assert m is not None


def test_single_source_sentence_returns_one():
    """원문 문장 1개 → 순서 비교 불가 → 1.0."""
    v = {"The cat sat.": [1.0, 0.0, 0.0, 0.0]}
    m = make_metric(vectors=v)
    score = m.compute("The cat sat.", "The cat sat.")
    assert score == 1.0


def test_match_sentences_returns_argmax():
    """각 원문 문장에 대해 가장 유사한 번안 문장 인덱스 반환."""
    # src[0] = [1,0,0,0] → 가장 유사한 tgt: tgt[0] = [1,0,0,0]
    # src[1] = [0,1,0,0] → 가장 유사한 tgt: tgt[1] = [0,1,0,0]
    src_embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    tgt_embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    m = make_metric()
    matched = m._match_sentences(src_embs, tgt_embs)
    assert list(matched) == [0, 1]


def test_match_sentences_cross_order():
    """원문 순서와 다르게 번안문이 배치된 경우."""
    # src[0] ~ tgt[1], src[1] ~ tgt[0]
    src_embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    tgt_embs = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    m = make_metric()
    matched = m._match_sentences(src_embs, tgt_embs)
    assert list(matched) == [1, 0]


def test_compute_identical_texts_score_one():
    """동일 텍스트 → 순서 완전 보존 → 1.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
        "It was red.": [0.0, 0.0, 1.0, 0.0],
    }
    m = make_metric(vectors=v)
    text = "The cat sat. The dog ran. It was red."
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-9


def test_compute_reversed_order_score_zero():
    """순서가 완전 역전 → tau = -1 → S_order = 0.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
        "It was red.": [0.0, 0.0, 1.0, 0.0],
    }
    m = make_metric(vectors=v)
    src = "The cat sat. The dog ran. It was red."
    # 번안: 순서 완전 역전 (src[0]~tgt[2], src[1]~tgt[1], src[2]~tgt[0])
    tgt = "It was red. The dog ran. The cat sat."
    score = m.compute(src, tgt)
    assert abs(score - 0.0) < 1e-9


def test_compute_partial_inversion():
    """부분 역전 → 0 < S_order < 1."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
        "It was red.": [0.0, 0.0, 1.0, 0.0],
        "The fox jumped.": [0.0, 0.0, 0.0, 1.0],
    }
    m = make_metric(vectors=v)
    src = "The cat sat. The dog ran. It was red. The fox jumped."
    # src[0]→tgt[0], src[1]→tgt[1], src[2]→tgt[3], src[3]→tgt[2] — 1 inversion
    tgt = "The cat sat. The dog ran. The fox jumped. It was red."
    score = m.compute(src, tgt)
    assert isinstance(score, float)
    assert 0.0 < score < 1.0


def test_compute_returns_float_in_range():
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
    }
    m = make_metric(vectors=v)
    score = m.compute("The cat sat. The dog ran.", "The dog ran. The cat sat.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.integration
def test_real_model_identical_returns_one():
    m = OrderMetric()
    text = (
        "The quick brown fox jumped over the lazy dog. "
        "It was a sunny afternoon. "
        "The dog did not move at all."
    )
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-6


@pytest.mark.integration
def test_real_model_score_in_range():
    m = OrderMetric()
    source = (
        "Scientists discovered a new species of bird in the Amazon. "
        "The bird has bright blue feathers. "
        "Researchers believe it evolved in isolation. "
        "They published their findings in Nature."
    )
    target = (
        "A new bird was found in the Amazon rainforest. "
        "It has blue feathers. "
        "Scientists think it evolved alone. "
        "The research was published."
    )
    score = m.compute(source, target)
    assert 0.0 <= score <= 1.0
