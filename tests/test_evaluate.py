"""
Unit tests for src.evaluate.

Focused on the pure cost/threshold logic and metric computation, which
can be tested with small hand-crafted arrays -- not on fitting real
models, which belongs in a manual/reporting run (see evaluate.main()).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    find_best_threshold_by_cost,
    evaluate_at_threshold,
    get_oof_probabilities,
)


def test_high_false_negative_cost_favors_a_low_threshold():
    """If missing a default is made VERY expensive relative to a false
    positive, the cost-minimizing threshold should be low (flag more
    borrowers as risky, accept more false positives to catch more
    defaults)."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    # probabilities roughly separating the classes, but not perfectly
    y_probs = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.3, 0.9, 10)])

    low_fn_cost_threshold, _, _ = find_best_threshold_by_cost(y_true, y_probs, fn_cost=1.0, fp_cost=1.0)
    high_fn_cost_threshold, _, _ = find_best_threshold_by_cost(y_true, y_probs, fn_cost=50.0, fp_cost=1.0)

    assert high_fn_cost_threshold <= low_fn_cost_threshold


def test_high_false_positive_cost_favors_a_high_threshold():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    y_probs = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.3, 0.9, 10)])

    balanced_threshold, _, _ = find_best_threshold_by_cost(y_true, y_probs, fn_cost=1.0, fp_cost=1.0)
    high_fp_cost_threshold, _, _ = find_best_threshold_by_cost(y_true, y_probs, fn_cost=1.0, fp_cost=50.0)

    assert high_fp_cost_threshold >= balanced_threshold


def test_evaluate_at_threshold_matches_hand_computed_values():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.6, 0.4, 0.9])  # at threshold 0.5: preds = [0, 1, 0, 1]

    metrics = evaluate_at_threshold(y_true, y_probs, threshold=0.5)

    # TP=1 (idx3), FP=1 (idx1), FN=1 (idx2), TN=1 (idx0)
    assert metrics["precision"] == 0.5  # TP / (TP + FP) = 1/2
    assert metrics["recall"] == 0.5     # TP / (TP + FN) = 1/2
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_get_oof_probabilities_shape_and_range():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int))

    model = LogisticRegression(max_iter=200)
    oof_probs = get_oof_probabilities(model, X, y)

    assert len(oof_probs) == n
    assert (oof_probs >= 0).all() and (oof_probs <= 1).all()


def test_threshold_sweep_never_picks_outside_valid_range():
    rng = np.random.default_rng(1)
    y_true = (rng.uniform(size=500) < 0.1).astype(int)
    y_probs = rng.uniform(size=500)

    threshold, thresholds, costs = find_best_threshold_by_cost(y_true, y_probs)
    assert 0.0 < threshold < 1.0
    assert len(thresholds) == len(costs)
