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


def test_boundary_score_identical_boundaries():
    m = make_metric(delta=0.2)
    # 원문 경계={1}, 번안 경계={1}, 원문 문장=3, 번안 문장=3 → 완전 일치
    score = m._boundary_score(src_b={1}, tgt_b={1}, n_src=3, n_tgt=3)
    assert score == 1.0


def test_boundary_score_no_source_boundaries():
    m = make_metric()
    # 원문에 경계 없음 → 정의에 따라 1.0
    score = m._boundary_score(src_b=set(), tgt_b={1}, n_src=2, n_tgt=3)
    assert score == 1.0


def test_boundary_score_no_match():
    m = make_metric(delta=0.1)
    # 원문 경계={0}, 번안 경계={5}, 거리 멀어 매칭 안됨 → 0.0
    score = m._boundary_score(src_b={0}, tgt_b={5}, n_src=10, n_tgt=10)
    assert score == 0.0


def test_count_score_same_count():
    m = make_metric()
    score = m._count_score(src_b={1, 2}, tgt_b={1, 2})
    assert score == 1.0  # 세그먼트 수 동일


def test_count_score_different_count():
    m = make_metric()
    # 원문 3세그먼트, 번안 1세그먼트
    score = m._count_score(src_b={1, 2}, tgt_b=set())
    # |3-1|/max(3,1) = 2/3
    assert abs(score - (1 - 2 / 3)) < 1e-9


def test_compute_identical_texts_score_one():
    """동일한 텍스트 → 1.0."""
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The mat was red.": [0.8, 0.6, 0.0, 0.0],
    }
    m = make_metric(vectors=v)
    text = "The cat sat. The mat was red."
    score = m.compute(text, text)
    assert score == 1.0


def test_compute_returns_float_in_range():
    v = {
        "A.": [1.0, 0.0, 0.0, 0.0],
        "B.": [0.0, 1.0, 0.0, 0.0],
        "C.": [0.0, 0.0, 1.0, 0.0],
        "D.": [0.0, 0.0, 0.0, 1.0],
    }
    m = make_metric(vectors=v)
    score = m.compute("A. B. C.", "B. C. D.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_compute_weighted_combination():
    """S_seg = 0.6*S_boundary + 0.4*S_count 가중치 확인.

    경계: 원문={1} (2세그먼트), 번안={} (1세그먼트)
    → S_boundary=0 (매칭 안됨), S_count=1-|2-1|/2=0.5
    → S_seg = 0.6*0 + 0.4*0.5 = 0.2

    Note: NLTK는 단일 대문자 약어("A.", "B.")를 문장 경계로 인식하지 않으므로
    실제 문장 형태의 텍스트를 사용한다.
    """
    # 원문: 앞 두 문장 유사, 세 번째 문장 급락 → 경계 at index 1
    v = {
        "The cat sat.": [1.0, 0.0, 0.0, 0.0],
        "The dog ran.": [1.0, 0.0, 0.0, 0.0],
        "It was red.": [0.0, 1.0, 0.0, 0.0],
        # 번안: 모두 동일 벡터 → std=0 → 경계 없음
        "The bird flew.": [1.0, 0.0, 0.0, 0.0],
        "It was blue.": [1.0, 0.0, 0.0, 0.0],
        "They were fast.": [1.0, 0.0, 0.0, 0.0],
    }
    m = make_metric(vectors=v, delta=0.1)
    src = "The cat sat. The dog ran. It was red."
    tgt = "The bird flew. It was blue. They were fast."
    score = m.compute(src, tgt)
    assert abs(score - 0.2) < 1e-6


@pytest.mark.integration
def test_real_model_identical_returns_one():
    """실제 sentence-transformers 모델로 동일 텍스트 → 1.0."""
    m = SegMetric()  # embedder=None → all-MiniLM-L6-v2 사용
    text = (
        "The quick brown fox jumped over the lazy dog. "
        "It was a sunny afternoon. "
        "The dog did not move at all."
    )
    score = m.compute(text, text)
    assert score == 1.0


@pytest.mark.integration
def test_real_model_score_in_range():
    m = SegMetric()
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
