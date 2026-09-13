from logsentinel.merge_policy import llm_calibrated_policy, naive_threshold_policy


def test_naive_threshold_policy_basic_counts():
    sims = [0.9, 0.8, 0.5, 0.95]
    gt = [True, True, False, True]
    m = naive_threshold_policy(sims, gt, tau=0.85)
    # >=0.85: indices 0 (0.9, True->TP), 3 (0.95, True->TP); index1 0.8 below tau (True,FN); index2 0.5 below tau (False,TN)
    assert m.tp == 2
    assert m.fn == 1
    assert m.tn == 1
    assert m.fp == 0
    assert m.recall == 2 / 3


def test_llm_calibrated_policy_basic_counts():
    probs = [0.9, 0.4, 0.6, 0.2]
    gt = [True, False, True, False]
    m = llm_calibrated_policy(probs, gt, tau_conf=0.5)
    assert m.tp == 2
    assert m.fp == 0
    assert m.fn == 0
    assert m.tn == 2
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_metrics_handle_no_positive_predictions_gracefully():
    probs = [0.1, 0.2]
    gt = [True, False]
    m = llm_calibrated_policy(probs, gt, tau_conf=0.9)
    assert m.tp == 0
    assert m.fp == 0
    # precision is nan (0/0); recall is well-defined 0.0
    assert m.recall == 0.0
    assert m.precision != m.precision  # nan check
