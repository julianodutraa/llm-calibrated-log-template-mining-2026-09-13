from collections import Counter

from logsentinel.drift import detect_drift, jensen_shannon_divergence


def test_identical_distributions_have_zero_divergence():
    a = Counter({"t1": 10, "t2": 5})
    b = Counter({"t1": 20, "t2": 10})  # same proportions, different volume
    assert jensen_shannon_divergence(a, b) < 1e-9


def test_disjoint_distributions_have_high_divergence():
    a = Counter({"t1": 10})
    b = Counter({"t2": 10})
    js = jensen_shannon_divergence(a, b)
    assert js > 0.9  # bounded by 1.0 with log base 2


def test_detect_drift_flags_new_template_above_threshold():
    windows = [
        Counter({"t1": 100, "t2": 20}),
        Counter({"t1": 100, "t2": 20}),
        Counter({"t1": 100, "t2": 20, "t3": 80}),  # new dominant template appears
    ]
    points = detect_drift(windows, threshold=0.05)
    assert points[0].is_drift is False
    assert points[1].is_drift is True
    assert "t3" in points[1].new_templates


def test_detect_drift_empty_new_templates_when_stable():
    windows = [Counter({"a": 5, "b": 5}), Counter({"a": 6, "b": 4})]
    points = detect_drift(windows, threshold=0.5)
    assert points[0].new_templates == []
