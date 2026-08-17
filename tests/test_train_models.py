"""
Unit tests for src.train_models.

These focus on pipeline CONSTRUCTION correctness and the SMOTE leakage
guarantee, not on full model performance -- fitting 4 real models with
5-fold CV is what `main()` does, and belongs in a manual/reporting run,
not a fast unit test suite.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from src.train_models import build_pipeline, build_estimator, MODEL_NAMES, NEEDS_SCALING


def _make_tiny_imbalanced_dataset(n=200, n_features=5, random_state=0):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    # ~10% positive class, deliberately imbalanced like the real dataset
    y = pd.Series((rng.uniform(size=n) < 0.1).astype(int))
    return X, y


def test_class_weight_strategy_returns_plain_sklearn_pipeline():
    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(model_name, "class_weight", scale_pos_weight=13.0)
        assert isinstance(pipeline, SkPipeline)
        assert "smote" not in pipeline.named_steps


def test_smote_strategy_returns_imblearn_pipeline_with_smote_step():
    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(model_name, "smote", scale_pos_weight=13.0)
        assert isinstance(pipeline, ImbPipeline)
        assert "smote" in pipeline.named_steps


def test_scaling_only_applied_where_needed():
    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(model_name, "class_weight", scale_pos_weight=13.0)
        has_scaler = "scaler" in pipeline.named_steps
        assert has_scaler == NEEDS_SCALING[model_name]


def test_xgboost_uses_scale_pos_weight_not_class_weight():
    est = build_estimator("xgboost", class_weight_mode="balanced", scale_pos_weight=13.0)
    assert est.get_params()["scale_pos_weight"] == 13.0

    est_none = build_estimator("xgboost", class_weight_mode=None)
    assert est_none.get_params()["scale_pos_weight"] == 1.0


def test_smote_pipeline_fits_and_predicts_on_tiny_dataset():
    """End-to-end smoke test on a tiny synthetic dataset (fast) rather
    than the real 120k-row data."""
    X, y = _make_tiny_imbalanced_dataset()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_val = X.iloc[150:]

    pipeline = build_pipeline("logistic_regression", "smote", scale_pos_weight=9.0)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    # predict() must return one prediction per validation row -- i.e.
    # SMOTE-resampled training data must not change how many predictions
    # come out for a held-out set of a different size.
    assert len(preds) == len(X_val)


def test_class_weight_pipeline_fits_and_predicts_on_tiny_dataset():
    X, y = _make_tiny_imbalanced_dataset()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_val = X.iloc[150:]

    pipeline = build_pipeline("random_forest", "class_weight", scale_pos_weight=9.0)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    assert len(preds) == len(X_val)
