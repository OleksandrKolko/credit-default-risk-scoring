"""
Unit tests for src.data_preprocessing.DataCleaner.

These use small synthetic DataFrames (not the real dataset) so the
tests run fast and each one isolates a single cleaning rule.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataCleaner


def _make_raw_df(n=20):
    """A minimal DataFrame with all columns DataCleaner expects."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "SeriousDlqin2yrs": rng.integers(0, 2, n),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, n),
        "age": rng.integers(25, 70, n),
        "NumberOfTime30-59DaysPastDueNotWorse": np.zeros(n, dtype=int),
        "DebtRatio": rng.uniform(0, 1, n),
        "MonthlyIncome": rng.uniform(2000, 10000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 15, n),
        "NumberOfTimes90DaysLate": np.zeros(n, dtype=int),
        "NumberRealEstateLoansOrLines": rng.integers(0, 3, n),
        "NumberOfTime60-89DaysPastDueNotWorse": np.zeros(n, dtype=int),
        "NumberOfDependents": rng.integers(0, 4, n).astype(float),
    })
    return df


def test_age_zero_is_imputed():
    df = _make_raw_df()
    df.loc[0, "age"] = 0
    cleaner = DataCleaner().fit(df)
    out = cleaner.transform(df)

    assert out.loc[0, "age"] != 0
    assert out.loc[0, "age"] == cleaner.age_median_
    assert (out["age"] > 0).all()


def test_past_due_sentinel_code_is_flagged_and_replaced():
    df = _make_raw_df()
    df.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"] = 98
    df.loc[0, "NumberOfTime60-89DaysPastDueNotWorse"] = 98
    df.loc[0, "NumberOfTimes90DaysLate"] = 98

    cleaner = DataCleaner().fit(df)
    out = cleaner.transform(df)

    assert out.loc[0, "had_past_due_sentinel_code"] == 1
    assert out.loc[1, "had_past_due_sentinel_code"] == 0
    # the sentinel value itself must not survive into the modeling features
    assert out["NumberOfTime30-59DaysPastDueNotWorse"].max() < 96
    assert out["NumberOfTimes90DaysLate"].max() < 96


def test_monthly_income_missing_is_flagged_and_imputed():
    df = _make_raw_df()
    df.loc[0, "MonthlyIncome"] = np.nan

    cleaner = DataCleaner().fit(df)
    out = cleaner.transform(df)

    assert out.loc[0, "monthly_income_missing"] == 1
    assert out.loc[1, "monthly_income_missing"] == 0
    assert not out["MonthlyIncome"].isna().any()


def test_dependents_missing_is_imputed_without_error():
    df = _make_raw_df()
    df.loc[0, "NumberOfDependents"] = np.nan

    cleaner = DataCleaner().fit(df)
    out = cleaner.transform(df)

    assert not out["NumberOfDependents"].isna().any()


def test_winsorize_caps_extreme_values():
    df = _make_raw_df()
    df.loc[0, "DebtRatio"] = 100_000  # implausible outlier
    df.loc[1, "RevolvingUtilizationOfUnsecuredLines"] = 50_000

    cleaner = DataCleaner().fit(df)
    out = cleaner.transform(df)

    assert out.loc[0, "DebtRatio"] < 100_000
    assert out.loc[0, "DebtRatio"] == cleaner.winsorize_caps_["DebtRatio"]
    assert out.loc[1, "RevolvingUtilizationOfUnsecuredLines"] == pytest.approx(
        cleaner.winsorize_caps_["RevolvingUtilizationOfUnsecuredLines"]
    )


def test_fit_only_on_train_does_not_change_when_transforming_test():
    """Statistics learned on train should be applied as-is to a
    differently-distributed test set (no re-fitting on test)."""
    train_df = _make_raw_df(n=50)
    test_df = _make_raw_df(n=10)
    test_df.loc[0, "age"] = 0  # only appears in test

    cleaner = DataCleaner().fit(train_df)
    train_median_age = cleaner.age_median_

    out_test = cleaner.transform(test_df)
    assert out_test.loc[0, "age"] == train_median_age
