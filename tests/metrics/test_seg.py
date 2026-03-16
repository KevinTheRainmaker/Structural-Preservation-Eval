import pytest
from structural_eval.metrics.seg import SegMetric


def test_seg_metric_instantiable():
    m = SegMetric()
    assert m is not None


def test_seg_compute_raises_not_implemented():
    m = SegMetric()
    with pytest.raises(NotImplementedError):
        m.compute("Hello world.", "Hello world.")
