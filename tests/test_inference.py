"""
Unit tests for src.inference.

`CreditRiskPredictor` loads real fitted artifacts from disk (the
DataCleaner, the final model) and builds a SHAP explainer -- these tests
exercise it end-to-end on a tiny scale (a handful of predictions), since
that's the realistic integration point, rather than mocking everything.
`select_representative_clients` is pure logic and tested in isolation.
"""

import numpy as np
import pandas as pd
import pytest

from src.inference import select_representative_clients, CreditRiskPredictor, FEATURE_DISPLAY_NAMES
from src.config import RAW_FEATURE_COLS, DATA_CLEANER_PATH, FINAL_MODEL_PATH


class _FakeModel:
    """Minimal stand-in exposing predict_proba, so select_representative_clients
    can be tested without loading the real (large) model."""
    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def test_select_representative_clients_picks_correct_indices():
    probs = [0.1, 0.9, 0.5, 0.3, 0.05]
    model = _FakeModel(probs)
    X = pd.DataFrame({"f1": range(5)})
    y = pd.Series([0, 1, 0, 1, 0])  # index 3 is the only actual defaulter besides index 1

    clients = select_representative_clients(model, X, y, threshold=0.5)

    assert clients["highest_risk_client"] == 1     # prob 0.9
    assert clients["lowest_risk_client"] == 4       # prob 0.05
    assert clients["borderline_client"] == 2        # prob 0.5, exactly at threshold
    # missed_default: lowest prob AMONG actual defaulters (indices 1, 3) -> index 3 (prob 0.3 < 0.9)
    assert clients["missed_default_client"] == 3


def test_select_representative_clients_handles_no_defaulters_in_sample():
    probs = [0.1, 0.2, 0.3]
    model = _FakeModel(probs)
    X = pd.DataFrame({"f1": range(3)})
    y = pd.Series([0, 0, 0])  # no defaulters at all

    clients = select_representative_clients(model, X, y, threshold=0.5)
    # should fall back gracefully rather than crash
    assert clients["missed_default_client"] == clients["lowest_risk_client"]


def test_feature_display_names_cover_all_raw_features():
    for col in RAW_FEATURE_COLS:
        assert col in FEATURE_DISPLAY_NAMES, f"{col} is missing a human-readable display name"


@pytest.mark.skipif(
    not (DATA_CLEANER_PATH.exists() and FINAL_MODEL_PATH.exists()),
    reason="requires fitted artifacts from running src.data_preprocessing and src.train_models first",
)
def test_predictor_predict_returns_expected_shape():
    predictor = CreditRiskPredictor()
    demo_client = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }
    result = predictor.predict(demo_client, top_n=5)

    assert 0.0 <= result["default_probability"] <= 1.0
    assert len(result["top_factors"]) == 5
    for factor in result["top_factors"]:
        assert "feature" in factor and "impact_percentage_points" in factor


@pytest.mark.skipif(
    not (DATA_CLEANER_PATH.exists() and FINAL_MODEL_PATH.exists()),
    reason="requires fitted artifacts from running src.data_preprocessing and src.train_models first",
)
def test_predictor_raises_on_missing_fields():
    predictor = CreditRiskPredictor()
    incomplete_client = {"age": 45}  # missing most required fields

    with pytest.raises(ValueError):
        predictor.predict(incomplete_client)
