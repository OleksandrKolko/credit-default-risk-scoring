"""
Final evaluation of the trained model on the held-out test set.

This module touches the test set exactly once, for reporting only. Every
decision that shaped the final model -- model family, imbalance strategy,
hyperparameters -- was made using only cross-validation on the training
set (see `train_models.py`). The numbers here are therefore an honest,
unbiased estimate of how the model would perform on genuinely new
borrowers, not a number that was implicitly optimized against.

Threshold selection without touching the test set
----------------------------------------------------
The default classification threshold of 0.5 is arbitrary -- it doesn't
reflect that, in credit risk, missing an actual default is typically far
costlier to a lender than declining a borrower who would have repaid (the
lender loses a large share of the loan principal in the first case, vs.
just the foregone profit margin on one loan in the second).

To pick a better threshold WITHOUT peeking at the test set (threshold
selection is still a modeling decision -- doing it on the test set would
be a subtle form of leakage), this module:

1. Gets out-of-fold (OOF) predicted probabilities for the TRAINING set
   only, via the same 5-fold stratified CV used everywhere else in this
   project (`get_oof_probabilities`).
2. Sweeps thresholds and picks the one minimizing an illustrative total
   cost: `FALSE_NEGATIVE_COST * missed_defaults + FALSE_POSITIVE_COST *
   wrongly_declined_borrowers` (`find_best_threshold_by_cost`).
3. Applies that ONE fixed threshold to the test set for final reporting.

The specific 5:1 cost ratio used here is illustrative, motivated by the
common credit-risk heuristic that loss-given-default is several times
larger than a single loan's foregone profit margin -- it demonstrates the
METHODOLOGY of a business-driven threshold, not a claim that 5:1 is the
correct ratio for any specific lender (that would come from the
institution's actual loss and margin data).
"""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    FEATURES_TRAIN_PATH, FEATURES_TEST_PATH, TARGET_COL, RANDOM_STATE, N_SPLITS,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, FINAL_MODEL_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FALSE_NEGATIVE_COST = 5.0  # illustrative cost of missing an actual default
FALSE_POSITIVE_COST = 1.0  # illustrative cost of declining a borrower who would not have defaulted


def load_final_model():
    return joblib.load(FINAL_MODEL_PATH)


def load_train_test():
    train_df = pd.read_csv(FEATURES_TRAIN_PATH)
    test_df = pd.read_csv(FEATURES_TEST_PATH)
    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    return X_train, y_train, X_test, y_test


def get_oof_probabilities(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold predicted probabilities for X/y using the SAME 5-fold
    stratified CV setup as the rest of the project. `model` is cloned and
    refit inside each fold by `cross_val_predict` -- passing an already-
    fitted model in is safe and does not leak its full-data fit into the
    OOF predictions."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def find_best_threshold_by_cost(y_true, y_probs,
                                 fn_cost: float = FALSE_NEGATIVE_COST,
                                 fp_cost: float = FALSE_POSITIVE_COST):
    """Sweep thresholds 0.01-0.99 and return the one minimizing
    `fn_cost * false_negatives + fp_cost * false_positives`."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        preds = (y_probs >= t).astype(int)
        fn = int(((preds == 0) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        costs[i] = fn_cost * fn + fp_cost * fp

    best_idx = int(costs.argmin())
    return float(thresholds[best_idx]), thresholds, costs


def evaluate_at_threshold(y_true, y_probs, threshold: float) -> dict:
    preds = (np.asarray(y_probs) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "pr_auc": float(average_precision_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def plot_roc_and_pr_curves(y_true, y_probs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    axes[0].plot(fpr, tpr, color="#4C72B0", label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (test set)")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    baseline = float(np.mean(y_true))
    axes[1].plot(recall, precision, color="#C44E52", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(baseline, linestyle="--", color="grey", label=f"Baseline (default rate = {baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (test set)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved ROC/PR curves to %s", save_path)


def plot_confusion_matrix(y_true, y_probs, threshold, save_path):
    preds = (np.asarray(y_probs) >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No default", "Default"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (test set, threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)


def plot_threshold_cost_curve(thresholds, costs, best_threshold, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, costs, color="#55A868")
    ax.axvline(best_threshold, linestyle="--", color="#C44E52",
               label=f"Chosen threshold = {best_threshold:.2f}")
    ax.axvline(0.5, linestyle=":", color="grey", label="Default threshold = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Illustrative total cost (FN:FP = {FALSE_NEGATIVE_COST:.0f}:{FALSE_POSITIVE_COST:.0f})")
    ax.set_title("Threshold selection via out-of-fold predictions (training set only)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved threshold cost curve to %s", save_path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_final_model()
    X_train, y_train, X_test, y_test = load_train_test()

    # --- Step 1: choose threshold using TRAIN-only OOF predictions ---
    logger.info("Computing out-of-fold predictions on the training set for threshold selection...")
    oof_probs = get_oof_probabilities(model, X_train, y_train)
    best_threshold, thresholds, costs = find_best_threshold_by_cost(y_train, oof_probs)
    logger.info("Chosen threshold (cost-minimizing, FN:FP = %.0f:%.0f) = %.2f",
                FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, best_threshold)

    plot_threshold_cost_curve(thresholds, costs, best_threshold, FIGURES_DIR / "threshold_cost_curve.png")

    # --- Step 2: the ONE honest look at the test set ---
    logger.info("Evaluating on the held-out test set (first and only time it's used)...")
    y_test_probs = model.predict_proba(X_test)[:, 1]

    metrics_default = evaluate_at_threshold(y_test, y_test_probs, 0.5)
    metrics_chosen = evaluate_at_threshold(y_test, y_test_probs, best_threshold)

    print("\n=== Final test-set evaluation (evaluated ONCE, 30,000 held-out rows) ===")
    print(f"Threshold-independent -- ROC-AUC: {metrics_chosen['roc_auc']:.4f}  PR-AUC: {metrics_chosen['pr_auc']:.4f}")
    print(f"\nAt default threshold (0.50): Precision={metrics_default['precision']:.3f}  "
          f"Recall={metrics_default['recall']:.3f}  F1={metrics_default['f1']:.3f}")
    print(f"At chosen threshold ({best_threshold:.2f}): Precision={metrics_chosen['precision']:.3f}  "
          f"Recall={metrics_chosen['recall']:.3f}  F1={metrics_chosen['f1']:.3f}")

    plot_roc_and_pr_curves(y_test, y_test_probs, FIGURES_DIR / "roc_pr_curves.png")
    plot_confusion_matrix(y_test, y_test_probs, best_threshold, FIGURES_DIR / "confusion_matrix.png")

    final_report = {
        "chosen_threshold": best_threshold,
        "cost_assumption": {"false_negative_cost": FALSE_NEGATIVE_COST, "false_positive_cost": FALSE_POSITIVE_COST},
        "metrics_at_chosen_threshold": metrics_chosen,
        "metrics_at_default_threshold_0.5": metrics_default,
    }
    report_path = REPORTS_DIR / "final_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Saved final evaluation report to %s", report_path)


if __name__ == "__main__":
    main()
