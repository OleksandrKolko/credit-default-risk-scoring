"""
Feature engineering for the credit risk scoring project.

Takes the CLEANED data (output of `src.data_preprocessing.DataCleaner`)
and derives new features that make specific financial-risk signals more
directly accessible to the models, on top of the existing columns.

Unlike `DataCleaner`, this transformer is (mostly) stateless -- every
derived feature below is a fixed arithmetic combination of existing
columns, not a statistic learned from the training distribution. It's
still implemented as a scikit-learn Transformer (fit/transform) purely
for API consistency, so it composes into the same pipeline as
`DataCleaner` in `train_models.py` and `inference.py`.

Scaling note: features here are intentionally left UNSCALED. Only
Logistic Regression needs standardized inputs; scaling is applied as
part of that model's own pipeline in `train_models.py` (fit inside each
CV fold via StandardScaler), rather than baked into this shared,
model-agnostic feature set that Random Forest / XGBoost / LightGBM also
consume unscaled.
"""

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CLEAN_TRAIN_PATH,
    CLEAN_TEST_PATH,
    FEATURES_TRAIN_PATH,
    FEATURES_TEST_PATH,
    DATA_PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Fixed, domain-chosen age buckets (not learned from data).
AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = [0, 1, 2, 3]  # ordinal: <30, 30-45, 45-60, 60+


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds the following derived features on top of the cleaned columns:

    - `total_past_due_count`: sum of the three past-due count columns.
      A simpler, more stable aggregate delinquency signal, complementing
      (not replacing) the individual counts. By construction this is
      correlated with its own components -- an accepted trade-off given
      L2 regularization for Logistic Regression and the natural
      robustness of tree models to collinearity.
    - `has_any_past_due`: binary flag, 1 if `total_past_due_count > 0`.
    - `income_per_dependent`: `MonthlyIncome / (NumberOfDependents + 1)`
      -- financial pressure relative to household size (+1 avoids
      division by zero for borrowers with 0 dependents).
    - `estimated_monthly_debt_payment`: `DebtRatio * MonthlyIncome` --
      reconstructs an absolute debt-payment magnitude rather than a pure
      ratio. Caveat: for rows where `monthly_income_missing == 1`,
      `DebtRatio` may not represent a genuine ratio to begin with (see
      EDA notebook) -- this feature is noisier for that subset, and the
      `monthly_income_missing` flag lets models learn to downweight it
      there if useful.
    - `utilization_x_credit_lines`: `RevolvingUtilizationOfUnsecuredLines
      * NumberOfOpenCreditLinesAndLoans` -- captures compounding risk of
      high utilization spread across many open credit lines.
    - `real_estate_loan_share`: proportion of open credit lines that are
      real estate loans -- mortgage debt is typically lower-risk than
      revolving/unsecured debt, so this may act as a mildly protective
      signal.
    - `age_group`: ordinal bucket of age (<30 / 30-45 / 45-60 / 60+).
      Trees can already split on raw `age` directly, so this mainly
      benefits Logistic Regression by exposing a non-linear age effect
      to a linear model without a manual polynomial term.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless: every feature below is a fixed formula, not a
        # statistic learned from the data. fit() exists only so this
        # class is a drop-in, Pipeline-compatible Transformer alongside
        # DataCleaner (which IS stateful).
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["total_past_due_count"] = X[PAST_DUE_COLS].sum(axis=1)
        X["has_any_past_due"] = (X["total_past_due_count"] > 0).astype(int)

        X["income_per_dependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1)

        X["estimated_monthly_debt_payment"] = X["DebtRatio"] * X["MonthlyIncome"]

        X["utilization_x_credit_lines"] = (
            X["RevolvingUtilizationOfUnsecuredLines"] * X["NumberOfOpenCreditLinesAndLoans"]
        )

        X["real_estate_loan_share"] = X["NumberRealEstateLoansOrLines"] / (
            X["NumberOfOpenCreditLinesAndLoans"] + 1
        )

        X["age_group"] = pd.cut(
            X["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
        ).astype(int)

        return X


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(CLEAN_TRAIN_PATH)
    test_df = pd.read_csv(CLEAN_TEST_PATH)
    logger.info("Loaded cleaned train (%s, %s) and test (%s, %s)",
                *train_df.shape, *test_df.shape)

    engineer = FeatureEngineer()
    engineer.fit(train_df)

    train_features = engineer.transform(train_df)
    test_features = engineer.transform(test_df)

    train_features.to_csv(FEATURES_TRAIN_PATH, index=False)
    test_features.to_csv(FEATURES_TEST_PATH, index=False)

    new_cols = [c for c in train_features.columns if c not in train_df.columns]
    logger.info("Saved engineered train data to %s (%s rows, %s columns)",
                FEATURES_TRAIN_PATH, *train_features.shape)
    logger.info("Saved engineered test data to %s (%s rows, %s columns)",
                FEATURES_TEST_PATH, *test_features.shape)
    logger.info("New columns added: %s", new_cols)


if __name__ == "__main__":
    main()
