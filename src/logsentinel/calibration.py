"""Calibration metrics for a binary probabilistic judge.

Two standard metrics are implemented, both operating on (predicted
probability of the positive class, true binary label) pairs:

Brier score
    BS = (1/N) * sum_i (p_i - y_i)^2
    The mean squared error between predicted probability and the 0/1
    outcome. Lower is better; 0 is perfect, 0.25 is what a judge who always
    outputs 0.5 achieves against a balanced dataset.

Expected Calibration Error (ECE)
    Partition [0, 1] into B equal-width confidence bins. For each bin b,
    let acc_b be the fraction of positive labels among predictions whose
    probability falls in that bin, and conf_b be the average predicted
    probability in that bin. Then

        ECE = sum_b (n_b / N) * |acc_b - conf_b|

    ECE measures whether "confidence" can be read as a real frequency: a
    judge that says 0.9 confidence should be right about 90% of the time
    it says so. Lower is better; 0 is perfect calibration.
"""

from __future__ import annotations

from dataclasses import dataclass


def brier_score(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        raise ValueError("empty input")
    n = len(probs)
    return sum((p - float(y)) ** 2 for p, y in zip(probs, labels)) / n


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    avg_confidence: float
    empirical_accuracy: float


def reliability_bins(probs: list[float], labels: list[bool], n_bins: int = 5) -> list[ReliabilityBin]:
    bins: list[ReliabilityBin] = []
    edges = [i / n_bins for i in range(n_bins + 1)]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, p in enumerate(probs) if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        avg_conf = sum(probs[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if labels[j]) / len(idx)
        bins.append(ReliabilityBin(lo, hi, len(idx), avg_conf, acc))
    return bins


def expected_calibration_error(probs: list[float], labels: list[bool], n_bins: int = 5) -> float:
    n = len(probs)
    if n == 0:
        raise ValueError("empty input")
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    return sum((b.n / n) * abs(b.empirical_accuracy - b.avg_confidence) for b in bins)
