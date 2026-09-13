"""Template-distribution drift detection via Jensen-Shannon divergence.

Once log lines are reduced to templates, each observation window (e.g. one
hour of pipeline activity) can be summarized as a categorical probability
distribution over template ids: P_w(template) = count(template in window w)
/ total events in window w. Comparing this distribution across consecutive
windows lets us detect a change in the *mix* of operational events, not just
a change in overall volume, which is exactly the signal that matters for
catching an emerging failure pattern (a new template appearing, or an
existing one becoming much more frequent) before it dominates the log
stream.

Jensen-Shannon divergence between two discrete distributions P and Q:

    M = (P + Q) / 2
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

where KL(P || M) = sum_i P(i) * log2(P(i) / M(i)) (0 * log(0) := 0 by
convention). JS is symmetric, always finite (unlike raw KL divergence, which
diverges when a template present in one window is entirely absent from the
other), and bounded in [0, 1] when log base 2 is used, which makes it easy
to reason about and threshold.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


def _distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _kl_divergence(p: dict[str, float], m: dict[str, float]) -> float:
    total = 0.0
    for k, pk in p.items():
        mk = m.get(k, 0.0)
        if pk > 0 and mk > 0:
            total += pk * math.log2(pk / mk)
    return total


def jensen_shannon_divergence(counter_a: Counter, counter_b: Counter) -> float:
    p = _distribution(counter_a)
    q = _distribution(counter_b)
    keys = set(p) | set(q)
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in keys}
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


@dataclass
class DriftPoint:
    window_index: int
    js_divergence: float
    is_drift: bool
    new_templates: list[str]


def detect_drift(
    window_template_counts: list[Counter], threshold: float
) -> list[DriftPoint]:
    """Compare each window to its predecessor and flag drift above threshold.

    `threshold` should be calibrated on a known-stable stretch of data (see
    the demo script, which calibrates it on windows 0-3 before any drift is
    injected) rather than picked arbitrarily; the right threshold depends on
    how many distinct templates exist and how much natural mix variation is
    expected window to window even with no real behavioral change.
    """
    points: list[DriftPoint] = []
    for i in range(1, len(window_template_counts)):
        prev, curr = window_template_counts[i - 1], window_template_counts[i]
        js = jensen_shannon_divergence(prev, curr)
        new_templates = sorted(set(curr) - set(prev))
        points.append(
            DriftPoint(
                window_index=i,
                js_divergence=round(js, 5),
                is_drift=js >= threshold,
                new_templates=new_templates,
            )
        )
    return points
