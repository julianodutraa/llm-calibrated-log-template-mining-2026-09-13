"""End-to-end evaluation: calibration of the LLM judge, naive vs LLM-informed
merge policy comparison, and drift detection on the synthetic corpus.

Writes data/evaluation_report.json with every number reported in the README,
so the README's claims are traceable to a single reproducible artifact.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logsentinel.calibration import brier_score, expected_calibration_error, reliability_bins  # noqa: E402
from logsentinel.drift import detect_drift  # noqa: E402
from logsentinel.drain import DrainParser  # noqa: E402
from logsentinel.judge import CachedLLMJudge, JudgeResult  # noqa: E402
from logsentinel.merge_policy import llm_calibrated_policy, naive_threshold_policy  # noqa: E402
from logsentinel.synthetic import generate_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def run_calibration_and_policy_comparison() -> dict:
    pairs = json.loads((DATA / "candidate_pairs.json").read_text())
    judge = CachedLLMJudge(DATA / "llm_judgments_cache.json")

    ground_truth = [p["ground_truth_same_event"] for p in pairs]
    similarities = [p["structural_similarity"] for p in pairs]

    judge_results: list[JudgeResult] = [
        judge.judge_pair(p["template_a"], p["template_b"], pair_id=p["pair_id"]) for p in pairs
    ]
    judge_probs = [r.p_same_event() for r in judge_results]

    bs = brier_score(judge_probs, ground_truth)
    ece = expected_calibration_error(judge_probs, ground_truth, n_bins=5)
    bins = reliability_bins(judge_probs, ground_truth, n_bins=5)

    # Sweep thresholds for both policies and report the best F1 each can
    # reach, plus the metrics at a fixed, typical operating point, so the
    # comparison is not an artifact of hand-picking one policy's best point
    # and the other's worst.
    sim_taus = [round(0.05 * i, 2) for i in range(1, 20)]
    naive_sweep = [naive_threshold_policy(similarities, ground_truth, tau) for tau in sim_taus]
    best_naive = max(naive_sweep, key=lambda m: (m.f1 if m.f1 == m.f1 else -1))

    conf_taus = [round(0.05 * i, 2) for i in range(1, 20)]
    llm_sweep = [llm_calibrated_policy(judge_probs, ground_truth, tau) for tau in conf_taus]
    best_llm = max(llm_sweep, key=lambda m: (m.f1 if m.f1 == m.f1 else -1))

    naive_at_090 = naive_threshold_policy(similarities, ground_truth, 0.90)
    llm_at_050 = llm_calibrated_policy(judge_probs, ground_truth, 0.50)

    return {
        "n_pairs": len(pairs),
        "n_should_merge": sum(ground_truth),
        "n_should_not_merge": len(ground_truth) - sum(ground_truth),
        "judge_calibration": {
            "brier_score": round(bs, 4),
            "expected_calibration_error_5bin": round(ece, 4),
            "reliability_bins": [
                {
                    "range": [b.lo, b.hi],
                    "n": b.n,
                    "avg_confidence": round(b.avg_confidence, 4),
                    "empirical_accuracy": round(b.empirical_accuracy, 4),
                }
                for b in bins
            ],
        },
        "policy_comparison": {
            "naive_threshold_operating_point_tau_0.90": naive_at_090.as_dict(),
            "naive_threshold_best_f1_over_sweep": best_naive.as_dict(),
            "llm_calibrated_operating_point_tau_0.50": llm_at_050.as_dict(),
            "llm_calibrated_best_f1_over_sweep": best_llm.as_dict(),
        },
        "per_pair": [
            {
                "pair_id": p["pair_id"],
                "template_a": p["template_a"],
                "template_b": p["template_b"],
                "structural_similarity": p["structural_similarity"],
                "ground_truth_same_event": p["ground_truth_same_event"],
                "judge_same_event": r.same_event,
                "judge_confidence": r.confidence,
                "judge_p_same_event": round(r.p_same_event(), 4),
                "judge_rationale": r.rationale,
            }
            for p, r in zip(pairs, judge_results)
        ],
    }


def run_drift_demo() -> dict:
    events = generate_corpus()
    n_windows = max(e.window for e in events) + 1
    parser = DrainParser(similarity_threshold=0.90)
    window_counts: list[Counter] = [Counter() for _ in range(n_windows)]
    for e in events:
        cluster = parser.add_log_message(e.message)
        window_counts[e.window][cluster.template()] += 1

    # Calibrate the drift threshold on the two known-stable transitions
    # (window 0->1 and 1->2, before any new template is injected), then
    # apply the same threshold to the remaining transitions rather than
    # picking a threshold that is fit to the drift windows themselves.
    stable_points = detect_drift(window_counts[:3], threshold=1.0)  # threshold=1.0: nothing flags yet
    baseline_js = [p.js_divergence for p in stable_points]
    calibrated_threshold = round(max(baseline_js) * 1.5, 5) if baseline_js else 0.02

    all_points = detect_drift(window_counts, threshold=calibrated_threshold)

    return {
        "n_windows": n_windows,
        "calibrated_threshold": calibrated_threshold,
        "baseline_js_divergences_windows_0_to_2": baseline_js,
        "drift_points": [
            {
                "window_transition": f"{p.window_index - 1}->{p.window_index}",
                "js_divergence": p.js_divergence,
                "is_drift": p.is_drift,
                "new_templates": p.new_templates,
            }
            for p in all_points
        ],
    }


def main() -> None:
    report = {
        "merge_policy_evaluation": run_calibration_and_policy_comparison(),
        "drift_detection_demo": run_drift_demo(),
    }
    out_path = DATA / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    mp = report["merge_policy_evaluation"]
    print("Brier score:", mp["judge_calibration"]["brier_score"])
    print("ECE (5 bin):", mp["judge_calibration"]["expected_calibration_error_5bin"])
    print("naive @0.90:", mp["policy_comparison"]["naive_threshold_operating_point_tau_0.90"])
    print("naive best F1:", mp["policy_comparison"]["naive_threshold_best_f1_over_sweep"])
    print("llm @0.50:", mp["policy_comparison"]["llm_calibrated_operating_point_tau_0.50"])
    print("llm best F1:", mp["policy_comparison"]["llm_calibrated_best_f1_over_sweep"])
    dd = report["drift_detection_demo"]
    print("drift threshold:", dd["calibrated_threshold"])
    for p in dd["drift_points"]:
        print(" ", p["window_transition"], p["js_divergence"], "DRIFT" if p["is_drift"] else "", p["new_templates"])


if __name__ == "__main__":
    main()
