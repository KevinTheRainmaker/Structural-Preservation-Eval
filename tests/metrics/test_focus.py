import pytest
import numpy as np
from structural_eval.metrics.focus import FocusMetric


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


def make_metric(vectors=None, k=3):
    return FocusMetric(embedder=FakeEmbedder(vectors), k=k)


def test_focus_metric_instantiable():
    m = FocusMetric()
    assert m is not None


def test_position_weights_first_is_highest():
    """첫 문장의 위치 가중치가 가장 높다."""
    m = make_metric()
    weights = m._position_weights(4)
    assert weights[0] == max(weights)
    assert len(weights) == 4


def test_position_weights_single_sentence():
    m = make_metric()
    weights = m._position_weights(1)
    assert weights[0] == 1.0


def test_cue_weights_detects_cue_phrase():
    """'importantly'가 포함된 문장은 cue weight = 1.0."""
    m = make_metric()
    sents = ["The cat sat.", "Importantly, the dog ran.", "It was red."]
    weights = m._cue_weights(sents)
    assert weights[0] == 0.0
    assert weights[1] == 1.0
    assert weights[2] == 0.0


def test_cue_weights_no_cue_phrase():
    m = make_metric()
    sents = ["The cat sat.", "The dog ran."]
    weights = m._cue_weights(sents)
    assert all(w == 0.0 for w in weights)


def test_repetition_weights_repeated_words_score_higher():
    """반복되는 단어를 가진 문장의 점수가 높다."""
    m = make_metric()
    # "cat"이 두 문장에 등장 → 첫 두 문장이 높은 점수
    sents = ["The cat chased the cat.", "The dog ran.", "A bird flew away."]
    weights = m._repetition_weights(sents)
    assert weights[0] > weights[2]  # cat이 반복되므로 s0 > s2


def test_salience_returns_array_of_correct_length():
    m = make_metric()
    sents = ["The cat sat.", "The dog ran.", "It was red."]
    sal = m._salience(sents)
    assert len(sal) == 3


def test_top_k_indices_returns_k_items():
    """상위 k개 인덱스 반환."""
    m = make_metric(k=2)
    sal = np.array([0.1, 0.9, 0.5])
    indices = m._top_k_indices(sal)
    assert len(indices) == 2
    assert 1 in indices  # 최고 점수


def test_top_k_indices_caps_at_n():
    """k > n 이면 n개만 반환."""
    m = make_metric(k=5)
    sal = np.array([0.5, 0.3])
    indices = m._top_k_indices(sal)
    assert len(indices) == 2


def test_compute_identical_texts_score_one():
    """동일 텍스트 → 1.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran fast.": [0.0, 1.0, 0.0, 0.0],
        "It was a red day.": [0.0, 0.0, 1.0, 0.0],
    }
    m = make_metric(vectors=v, k=2)
    text = "The cat sat. The dog ran fast. It was a red day."
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-9


def test_compute_returns_float_in_range():
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [0.0, 1.0, 0.0, 0.0],
        "The fox jumped.": [0.0, 0.0, 1.0, 0.0],
        "The bird flew.": [0.0, 0.0, 0.0, 1.0],
    }
    m = make_metric(vectors=v, k=2)
    score = m.compute("The cat sat. The dog ran.", "The fox jumped. The bird flew.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.integration
def test_real_model_identical_returns_one():
    m = FocusMetric()
    text = (
        "Scientists discovered a new species of bird in the Amazon. "
        "The bird has bright blue feathers. "
        "Researchers believe it evolved in isolation. "
        "They published their findings in Nature."
    )
    score = m.compute(text, text)
    assert abs(score - 1.0) < 1e-6


@pytest.mark.integration
def test_real_model_score_in_range():
    m = FocusMetric()
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
