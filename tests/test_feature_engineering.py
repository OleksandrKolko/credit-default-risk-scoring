"""
Unit tests for src.feature_engineering.FeatureEngineer.
"""

import pandas as pd

from src.feature_engineering import FeatureEngineer


def _make_clean_df():
    """A minimal already-cleaned DataFrame (no NaNs, no sentinel codes) --
    mirrors the output shape of DataCleaner.transform()."""
    return pd.DataFrame({
        "SeriousDlqin2yrs": [0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.9, 0.1],
        "age": [25, 50, 70],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 0],
        "DebtRatio": [0.3, 1.5, 0.2],
        "MonthlyIncome": [3000.0, 5000.0, 4000.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 0, 8],
        "NumberOfTimes90DaysLate": [0, 1, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0],
        "NumberOfDependents": [0.0, 3.0, 1.0],
        "had_past_due_sentinel_code": [0, 0, 0],
        "monthly_income_missing": [0, 0, 0],
    })


def test_total_past_due_and_flag():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    assert out.loc[0, "total_past_due_count"] == 0
    assert out.loc[0, "has_any_past_due"] == 0
    assert out.loc[1, "total_past_due_count"] == 3  # 2 + 0 + 1
    assert out.loc[1, "has_any_past_due"] == 1


def test_income_per_dependent_no_division_by_zero():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    # row 0 has 0 dependents -> should not raise / produce inf
    assert out.loc[0, "income_per_dependent"] == 3000.0 / 1
    assert not out["income_per_dependent"].isin([float("inf"), float("-inf")]).any()


def test_utilization_x_credit_lines_handles_zero_lines():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    # row 1 has 0 open credit lines -> product should just be 0, not NaN/inf
    assert out.loc[1, "utilization_x_credit_lines"] == 0.0


def test_real_estate_loan_share_no_division_by_zero():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    # row 1 has 0 open credit lines -> +1 guard should avoid div-by-zero
    assert out.loc[1, "real_estate_loan_share"] == 0.0 / 1


def test_age_group_buckets():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    assert out.loc[0, "age_group"] == 0  # age 25 -> <30
    assert out.loc[1, "age_group"] == 2  # age 50 -> 45-60
    assert out.loc[2, "age_group"] == 3  # age 70 -> 60+


def test_transform_preserves_original_columns():
    df = _make_clean_df()
    out = FeatureEngineer().fit_transform(df)

    for col in df.columns:
        assert col in out.columns
