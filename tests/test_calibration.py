import math

from logsentinel.calibration import brier_score, expected_calibration_error, reliability_bins


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0


def test_brier_score_worst_case_is_one():
    assert brier_score([0.0, 1.0], [True, False]) == 1.0


def test_brier_score_always_half_gives_quarter_on_balanced_labels():
    probs = [0.5] * 4
    labels = [True, True, False, False]
    assert math.isclose(brier_score(probs, labels), 0.25)


def test_ece_zero_for_perfectly_calibrated_bins():
    # 10 items at confidence 0.9, exactly 9 correct -> perfectly calibrated in that bin
    probs = [0.9] * 10
    labels = [True] * 9 + [False]
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert math.isclose(ece, 0.0, abs_tol=1e-9)


def test_ece_penalizes_overconfidence():
    # confidence 0.95 but only right half the time: should be far from 0
    probs = [0.95] * 10
    labels = [True] * 5 + [False] * 5
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece > 0.4


def test_reliability_bins_cover_all_items():
    probs = [0.05, 0.15, 0.5, 0.85, 0.95]
    labels = [False, True, True, True, False]
    bins = reliability_bins(probs, labels, n_bins=5)
    assert sum(b.n for b in bins) == len(probs)
