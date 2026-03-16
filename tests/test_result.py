from structural_eval.result import EvalResult

def test_eval_result_has_six_scores():
    r = EvalResult(seg=0.9, topic=0.8, order=0.7, focus=0.6, rst=0.5, dep=0.4)
    assert r.seg == 0.9
    assert r.topic == 0.8
    assert r.order == 0.7
    assert r.focus == 0.6
    assert r.rst == 0.5
    assert r.dep == 0.4

def test_eval_result_to_dict():
    r = EvalResult(seg=1.0, topic=1.0, order=1.0, focus=1.0, rst=1.0, dep=1.0)
    d = r.to_dict()
    assert set(d.keys()) == {"seg", "topic", "order", "focus", "rst", "dep"}
    assert all(v == 1.0 for v in d.values())

def test_eval_result_overall_is_mean():
    r = EvalResult(seg=1.0, topic=0.0, order=1.0, focus=0.0, rst=1.0, dep=0.0)
    assert abs(r.overall - 0.5) < 1e-9

def test_scores_bounded_0_1():
    import pytest
    with pytest.raises(ValueError):
        EvalResult(seg=1.5, topic=0.0, order=0.0, focus=0.0, rst=0.0, dep=0.0)
