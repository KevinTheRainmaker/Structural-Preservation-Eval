import pytest
from structural_eval.metrics.base import BaseMetric

def test_base_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric()  # 직접 인스턴스화 불가

def test_concrete_metric_must_implement_compute():
    class MyMetric(BaseMetric):
        pass
    with pytest.raises(TypeError):
        MyMetric()

def test_concrete_metric_returns_float():
    class MyMetric(BaseMetric):
        def compute(self, source: str, target: str) -> float:
            return 1.0
    m = MyMetric()
    assert m.compute("hello", "hello") == 1.0

def test_compute_result_bounded():
    class MyMetric(BaseMetric):
        def compute(self, source: str, target: str) -> float:
            return 0.5
    m = MyMetric()
    result = m.safe_compute("a", "b")
    assert 0.0 <= result <= 1.0
