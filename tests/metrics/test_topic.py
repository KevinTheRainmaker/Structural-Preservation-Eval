import pytest
import numpy as np
from structural_eval.metrics.topic import TopicMetric


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


def make_metric(vectors=None, threshold_sigma=0.5):
    return TopicMetric(embedder=FakeEmbedder(vectors), threshold_sigma=threshold_sigma)


def test_topic_metric_instantiable():
    m = TopicMetric()
    assert m is not None


def test_segment_embeddings_single_segment():
    """경계 없는 텍스트는 단일 세그먼트 임베딩 반환."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The mat was red.": [1.0, 0.0, 0.0, 0.0],
    }
    m = make_metric(vectors=v)
    sents = ["The cat sat.", "The mat was red."]
    embs = m._segment_embeddings(sents)
    assert embs.shape[0] == 1


def test_segment_embeddings_two_segments():
    """경계가 탐지되면 2개 세그먼트 임베딩 반환."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [1.0, 0.0, 0.0, 0.0],
        "It was red.": [0.0, 1.0, 0.0, 0.0],
    }
    m = make_metric(vectors=v, threshold_sigma=0.5)
    sents = ["The cat sat.", "The dog ran.", "It was red."]
    embs = m._segment_embeddings(sents)
    assert embs.shape[0] == 2


def test_local_score_identical():
    """동일 세그먼트 임베딩 → S_local = 1.0."""
    m = make_metric()
    embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    score = m._local_score(embs, embs)
    assert abs(score - 1.0) < 1e-9


def test_local_score_orthogonal():
    """직교 세그먼트 임베딩 → S_local = 0.0."""
    m = make_metric()
    src = np.array([[1.0, 0.0, 0.0, 0.0]])
    tgt = np.array([[0.0, 1.0, 0.0, 0.0]])
    score = m._local_score(src, tgt)
    assert abs(score) < 1e-9


def test_flow_score_single_segment():
    """세그먼트 1개 → 전이 없음 → S_flow = 1.0."""
    m = make_metric()
    embs = np.array([[1.0, 0.0, 0.0, 0.0]])
    score = m._flow_score(embs, embs)
    assert score == 1.0


def test_flow_score_identical_transitions():
    """동일 전이 벡터 → S_flow = 1.0."""
    m = make_metric()
    embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    score = m._flow_score(embs, embs)
    assert abs(score - 1.0) < 1e-9


def test_compute_identical_texts_score_one():
    """동일 텍스트 → 1.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The mat was red.": [0.8, 0.6, 0.0, 0.0],
    }
    m = make_metric(vectors=v)
    text = "The cat sat. The mat was red."
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-9


def test_compute_returns_float_in_range():
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The mat was red.": [0.0, 1.0, 0.0, 0.0],
        "The dog ran fast.": [0.0, 0.0, 1.0, 0.0],
        "It was blue.": [0.0, 0.0, 0.0, 1.0],
    }
    m = make_metric(vectors=v)
    score = m.compute("The cat sat. The mat was red.", "The dog ran fast. It was blue.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_compute_weighted_combination():
    """S_topic = 0.7*S_local + 0.3*S_flow. 동일 텍스트 → 1.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
    }
    m = make_metric(vectors=v)
    src = "The cat sat. The dog ran."
    score = m.compute(src, src)
    assert abs(score - 1.0) < 1e-9


@pytest.mark.integration
def test_real_model_identical_returns_one():
    m = TopicMetric()
    text = (
        "The quick brown fox jumped over the lazy dog. "
        "It was a sunny afternoon. "
        "The dog did not move at all."
    )
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-6


@pytest.mark.integration
def test_real_model_score_in_range():
    m = TopicMetric()
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
