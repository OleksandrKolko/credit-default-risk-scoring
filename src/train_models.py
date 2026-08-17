"""
Model training and comparison for credit risk scoring.

Compares:
- 4 model families: Logistic Regression (baseline), Random Forest,
  XGBoost, LightGBM.
- 2 class-imbalance strategies: class weighting vs. SMOTE oversampling.

...giving 8 total configurations, each evaluated via stratified 5-fold
cross-validation on the TRAINING set only. The held-out test set is not
touched here at all -- it's reserved for a single, final, honest
evaluation of the chosen model in `evaluate.py`, after hyperparameter
tuning.

Why these metrics
------------------
The target is heavily imbalanced (6.7% default rate), so plain accuracy
is misleading: a model that always predicts "no default" would already
score ~93% accuracy while being useless for risk decisions.

- **ROC-AUC** summarizes ranking quality (how well the model orders
  borrowers by risk) across all classification thresholds.
- **PR-AUC** (average precision) is more sensitive to performance on the
  minority (default) class specifically -- which is what a lender
  actually cares about -- and is used here as the PRIMARY criterion for
  picking the best configuration, with ROC-AUC reported alongside as
  corroboration.
- **Precision / Recall / F1** are reported at the default 0.5
  probability threshold to make the precision/recall trade-off
  concrete; the final decision threshold can be tuned separately once a
  model is chosen (see `evaluate.py`).

Why class weights AND SMOTE
-----------------------------
Both are legitimate ways to counter imbalance, with different
trade-offs: class weighting changes the loss function without touching
the data, while SMOTE synthesizes new minority-class examples. Which
works better is an empirical question for this specific dataset --
that's what this comparison answers, rather than assuming one is
"correct" in general.

Leakage note on SMOTE
----------------------
SMOTE must only ever see the TRAINING fold, never the validation fold
-- otherwise synthetic points derived from validation-fold neighbors
would leak information into evaluation. This is handled by using
`imblearn.pipeline.Pipeline` (not plain sklearn Pipeline), which
guarantees resampling happens only inside each fold's training portion
when used with `cross_validate`.
"""

import json
import logging
import time

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    FEATURES_TRAIN_PATH, TARGET_COL, RANDOM_STATE, N_SPLITS, REPORTS_DIR, MODELS_DIR,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
IMBALANCE_STRATEGIES = ["class_weight", "smote"]

NEEDS_SCALING = {
    "logistic_regression": True,
    "random_forest": False,
    "xgboost": False,
    "lightgbm": False,
}

# Comparison-stage hyperparameters are deliberately modest defaults, not
# tuned -- the winning (model, imbalance_strategy) combination gets
# properly tuned with Optuna in the next step. The point here is a fair
# like-for-like comparison, not the best possible score for any one model.
N_ESTIMATORS = 100

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def load_training_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_estimator(model_name: str, class_weight_mode=None, scale_pos_weight=1.0):
    """Construct a fresh estimator instance. `class_weight_mode` is either
    None or "balanced"; XGBoost has no `class_weight` param so it uses
    `scale_pos_weight` instead to achieve the equivalent effect."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight_mode
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight=class_weight_mode,
        )
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=(scale_pos_weight if class_weight_mode == "balanced" else 1.0),
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            class_weight=class_weight_mode,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def build_pipeline(model_name: str, imbalance_strategy: str, scale_pos_weight: float):
    steps = []
    if NEEDS_SCALING[model_name]:
        steps.append(("scaler", StandardScaler()))

    if imbalance_strategy == "class_weight":
        estimator = build_estimator(model_name, class_weight_mode="balanced",
                                     scale_pos_weight=scale_pos_weight)
        steps.append(("clf", estimator))
        return SkPipeline(steps)

    if imbalance_strategy == "smote":
        estimator = build_estimator(model_name, class_weight_mode=None)
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)

    raise ValueError(f"Unknown imbalance_strategy: {imbalance_strategy}")


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    rows = []
    for model_name in MODEL_NAMES:
        for imbalance_strategy in IMBALANCE_STRATEGIES:
            pipeline = build_pipeline(model_name, imbalance_strategy, scale_pos_weight)

            start = time.time()
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
            elapsed = time.time() - start

            row = {
                "model": model_name,
                "imbalance_strategy": imbalance_strategy,
                "roc_auc_mean": cv_results["test_roc_auc"].mean(),
                "roc_auc_std": cv_results["test_roc_auc"].std(),
                "pr_auc_mean": cv_results["test_pr_auc"].mean(),
                "pr_auc_std": cv_results["test_pr_auc"].std(),
                "precision_mean": cv_results["test_precision"].mean(),
                "recall_mean": cv_results["test_recall"].mean(),
                "f1_mean": cv_results["test_f1"].mean(),
                "fit_time_seconds": round(elapsed, 1),
            }
            rows.append(row)
            logger.info(
                "%-20s | %-13s -> ROC-AUC=%.4f  PR-AUC=%.4f  Precision=%.3f  Recall=%.3f  F1=%.3f  (%.1fs)",
                model_name, imbalance_strategy,
                row["roc_auc_mean"], row["pr_auc_mean"],
                row["precision_mean"], row["recall_mean"], row["f1_mean"], elapsed,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna) of the winning configuration from
# compare_models(). Currently only a LightGBM search space is implemented,
# since that was the winning model family in our comparison -- if the
# winner changes (e.g. after adding new features), add a
# build_*_estimator_from_trial function for that model family and branch
# on model_name below.
# ---------------------------------------------------------------------------

def build_lightgbm_estimator_from_trial(trial: "optuna.Trial", class_weight_mode):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    return lgb.LGBMClassifier(
        **params,
        subsample_freq=1,  # required for `subsample` (bagging) to actually take effect in LightGBM
        class_weight=class_weight_mode,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


ESTIMATOR_BUILDERS_FROM_TRIAL = {
    "lightgbm": build_lightgbm_estimator_from_trial,
}


def tune_best_model(X: pd.DataFrame, y: pd.Series, model_name: str, imbalance_strategy: str,
                     n_trials: int = 30) -> "optuna.Study":
    """Runs an Optuna study (TPE sampler) to tune hyperparameters of the
    winning (model, imbalance_strategy) combination from `compare_models`,
    using the SAME 5-fold stratified CV and PR-AUC objective (average
    precision) as the comparison stage, for a consistent, comparable
    criterion end-to-end."""
    if model_name not in ESTIMATOR_BUILDERS_FROM_TRIAL:
        raise NotImplementedError(
            f"No Optuna search space implemented for '{model_name}'. "
            "Add a build_*_estimator_from_trial function and register it "
            "in ESTIMATOR_BUILDERS_FROM_TRIAL."
        )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    build_estimator_from_trial = ESTIMATOR_BUILDERS_FROM_TRIAL[model_name]

    def objective(trial):
        class_weight_mode = "balanced" if imbalance_strategy == "class_weight" else None
        estimator = build_estimator_from_trial(trial, class_weight_mode)

        if imbalance_strategy == "smote":
            pipeline = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", estimator)])
        else:
            pipeline = SkPipeline([("clf", estimator)])

        scores = cross_validate(pipeline, X, y, cv=cv, scoring={"pr_auc": "average_precision"}, n_jobs=1)
        return scores["test_pr_auc"].mean()

    sampler = TPESampler(seed=RANDOM_STATE)
    storage_path = REPORTS_DIR / "optuna_study.db"
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name=f"{model_name}_{imbalance_strategy}_tuning",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    logger.info("Starting Optuna tuning: target %s total trials for %s + %s "
                "(%s already completed in existing study, if any)",
                n_trials, model_name, imbalance_strategy, len(study.trials))
    start = time.time()
    trials_needed = max(0, n_trials - len(study.trials))
    if trials_needed > 0:
        study.optimize(objective, n_trials=trials_needed, show_progress_bar=False)
    else:
        logger.info("Study already has %s trials (>= target %s) -- skipping further optimization.",
                     len(study.trials), n_trials)
    elapsed = time.time() - start

    logger.info("Optuna tuning finished in %.1fs. Best PR-AUC: %.4f (baseline default params: see reports/model_comparison.csv)",
                elapsed, study.best_value)
    logger.info("Best params: %s", study.best_params)

    return study


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()
    logger.info("Training data: %s rows, %s features, default rate %.2f%%",
                X.shape[0], X.shape[1], y.mean() * 100)

    # --- Stage 1: compare 4 models x 2 imbalance strategies ---
    results_df = compare_models(X, y)

    results_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved comparison results to %s", results_path)

    print("\n=== Model comparison (5-fold stratified CV on training set) ===")
    print(results_df.sort_values("pr_auc_mean", ascending=False).to_string(index=False))

    best_row = results_df.loc[results_df["pr_auc_mean"].idxmax()]
    best_model_name = best_row["model"]
    best_imbalance_strategy = best_row["imbalance_strategy"]
    logger.info(
        "Best config by PR-AUC: %s + %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
        best_model_name, best_imbalance_strategy,
        best_row["pr_auc_mean"], best_row["roc_auc_mean"],
    )

    best_config = {
        "model": best_model_name,
        "imbalance_strategy": best_imbalance_strategy,
        "pr_auc_mean": float(best_row["pr_auc_mean"]),
        "roc_auc_mean": float(best_row["roc_auc_mean"]),
    }
    best_config_path = REPORTS_DIR / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Saved best config to %s", best_config_path)

    # --- Stage 2: Optuna tuning of the winning configuration ---
    if best_model_name not in ESTIMATOR_BUILDERS_FROM_TRIAL:
        logger.warning(
            "Skipping Optuna tuning: no search space implemented for '%s'. "
            "Using default hyperparameters for the final model instead.",
            best_model_name,
        )
        best_params = {}
    else:
        study = tune_best_model(X, y, best_model_name, best_imbalance_strategy, n_trials=30)
        best_params = study.best_params

        print("\n=== Optuna tuning result ===")
        print(f"Best CV PR-AUC (tuned):   {study.best_value:.4f}")
        print(f"Best CV PR-AUC (default): {best_row['pr_auc_mean']:.4f}")
        print(f"Best params: {json.dumps(best_params, indent=2)}")

        best_hyperparams_path = REPORTS_DIR / "best_hyperparams.json"
        with open(best_hyperparams_path, "w") as f:
            json.dump({
                "model": best_model_name,
                "imbalance_strategy": best_imbalance_strategy,
                "best_cv_pr_auc": study.best_value,
                "default_cv_pr_auc": float(best_row["pr_auc_mean"]),
                "params": best_params,
            }, f, indent=2)
        logger.info("Saved tuned hyperparameters to %s", best_hyperparams_path)

    # --- Stage 3: fit the FINAL pipeline on the full training set ---
    # This is the only place the model is fit on all 120k training rows at
    # once (CV above only ever fits on ~96k-row folds). The held-out test
    # set is still untouched -- evaluation happens once, in evaluate.py.
    class_weight_mode = "balanced" if best_imbalance_strategy == "class_weight" else None
    if best_model_name in ESTIMATOR_BUILDERS_FROM_TRIAL and best_params:
        final_estimator = lgb.LGBMClassifier(
            **best_params, subsample_freq=1, class_weight=class_weight_mode,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    else:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        final_estimator = build_estimator(best_model_name, class_weight_mode, scale_pos_weight)

    final_steps = []
    if NEEDS_SCALING[best_model_name]:
        final_steps.append(("scaler", StandardScaler()))
    if best_imbalance_strategy == "smote":
        final_steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        final_pipeline = ImbPipeline(final_steps + [("clf", final_estimator)])
    else:
        final_pipeline = SkPipeline(final_steps + [("clf", final_estimator)])

    logger.info("Fitting final pipeline (%s + %s) on the full training set...",
                best_model_name, best_imbalance_strategy)
    final_pipeline.fit(X, y)

    import joblib
    final_model_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(final_pipeline, final_model_path)
    logger.info("Saved final fitted pipeline to %s", final_model_path)


if __name__ == "__main__":
    main()
