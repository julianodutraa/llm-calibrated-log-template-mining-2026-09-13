"""Merge policies compared: naive threshold-only vs LLM-calibrated review.

Two policies decide, for each candidate cluster pair, whether to merge:

naive_threshold_policy
    Merge iff structural_similarity >= tau, for a single global tau. This
    is what an online parser effectively already does at its operating
    threshold, applied here retrospectively to a batch of already-split
    clusters for a fair comparison.

llm_calibrated_policy
    Merge iff the judge says same_event=True AND its confidence
    (interpreted as P(same_event=True), see JudgeResult.p_same_event) is
    at least tau_conf. This lets a pair with high structural similarity
    still be rejected (the SUCCESS/FAILED case) and a pair with lower
    structural similarity still be accepted (the cross-length cosmetic
    suffix case) when the judge's semantic read disagrees with raw token
    overlap.

Both policies are scored against the same ground truth with standard
binary classification metrics, where the positive class is "should merge".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyMetrics:
    tau: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) and p == p and r == r and (p + r) > 0 else float("nan")

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else float("nan")

    def as_dict(self) -> dict:
        def _safe_round(x: float) -> float | None:
            # NaN is not valid JSON; represent an undefined metric (e.g.
            # precision with zero positive predictions) as null instead of
            # letting json.dumps emit the non-standard literal `NaN`.
            return None if x != x else round(x, 4)

        return {
            "tau": self.tau,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": _safe_round(self.precision),
            "recall": _safe_round(self.recall),
            "f1": _safe_round(self.f1),
            "accuracy": _safe_round(self.accuracy),
        }


def _score(decisions: list[bool], ground_truth: list[bool], tau: float) -> PolicyMetrics:
    tp = sum(1 for d, y in zip(decisions, ground_truth) if d and y)
    fp = sum(1 for d, y in zip(decisions, ground_truth) if d and not y)
    fn = sum(1 for d, y in zip(decisions, ground_truth) if not d and y)
    tn = sum(1 for d, y in zip(decisions, ground_truth) if not d and not y)
    return PolicyMetrics(tau=tau, tp=tp, fp=fp, fn=fn, tn=tn)


def naive_threshold_policy(similarities: list[float], ground_truth: list[bool], tau: float) -> PolicyMetrics:
    decisions = [s >= tau for s in similarities]
    return _score(decisions, ground_truth, tau)


def llm_calibrated_policy(
    judge_probs: list[float], ground_truth: list[bool], tau_conf: float
) -> PolicyMetrics:
    decisions = [p >= tau_conf for p in judge_probs]
    return _score(decisions, ground_truth, tau_conf)
