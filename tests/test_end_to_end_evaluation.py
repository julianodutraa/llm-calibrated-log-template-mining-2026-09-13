"""A light integration test over the shipped, pre-computed evaluation
artifact (data/evaluation_report.json), checking that the headline numbers
quoted in the README are internally consistent with the underlying per-pair
data shipped alongside it. This does not re-run the full pipeline (that is
what scripts/build_dataset.py and scripts/evaluate.py are for, and what
produced this file), but it guards against the report and the README
silently drifting apart.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_report():
    with open(DATA_DIR / "evaluation_report.json") as f:
        return json.load(f)


def test_report_file_exists_and_parses():
    report = _load_report()
    assert "merge_policy_evaluation" in report
    assert "drift_detection_demo" in report


def test_per_pair_matches_summary_counts():
    report = _load_report()
    mp = report["merge_policy_evaluation"]
    per_pair = mp["per_pair"]
    assert len(per_pair) == mp["n_pairs"]
    assert sum(1 for p in per_pair if p["ground_truth_same_event"]) == mp["n_should_merge"]


def test_judge_calibration_metrics_are_in_valid_range():
    report = _load_report()
    cal = report["merge_policy_evaluation"]["judge_calibration"]
    assert 0.0 <= cal["brier_score"] <= 1.0
    assert 0.0 <= cal["expected_calibration_error_5bin"] <= 1.0


def test_drift_flagged_exactly_at_injected_windows():
    report = _load_report()
    points = report["drift_detection_demo"]["drift_points"]
    flagged = {p["window_transition"] for p in points if p["is_drift"]}
    # the synthetic corpus injects new templates starting at window 4
    assert "3->4" in flagged
    assert "4->5" in flagged
    assert "0->1" not in flagged
    assert "1->2" not in flagged
