"""
Data preprocessing for the credit risk scoring project.

Handles: loading the raw Kaggle "Give Me Some Credit" CSV, a stratified
train/test split, and cleaning the known data-quality issues (missing
values, sentinel/error codes, extreme outliers) discovered during EDA
(see notebooks/01_eda.ipynb).

Design note on leakage
----------------------
`DataCleaner` is a scikit-learn compatible Transformer (fit/transform).
It is fit ONCE on the training split only; the same fitted statistics
(medians, outlier caps) are then applied to the test split and, later,
to new clients at inference time. This is what prevents the test set
from leaking into preprocessing decisions.

For simplicity, the cleaner is fit once on the full training split
rather than refit inside every individual cross-validation fold during
model comparison. With >100k training rows the fitted medians and
percentile caps are stable, so per-fold refitting would not meaningfully
change results here -- but the class is written as a proper Transformer
specifically so it CAN be composed into a single sklearn Pipeline with
the model if that stricter guarantee is ever needed (e.g. for a
production retraining job).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_TRAIN_PATH,
    CLEAN_TRAIN_PATH,
    CLEAN_TEST_PATH,
    DATA_CLEANER_PATH,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Columns known to contain the 96 / 98 sentinel/error code.
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_CODES = [96, 98]

# Ratio-type features with an extreme long right tail, handled via
# percentile-based winsorization rather than a hand-picked domain cutoff.
WINSORIZE_COLS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
WINSORIZE_UPPER_PERCENTILE = 0.975


def load_raw_data(path=RAW_TRAIN_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV. The first column is an unnamed row index."""
    df = pd.read_csv(path, index_col=0)
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def split_train_test(df: pd.DataFrame):
    """Stratified train/test split on the target, done on RAW data before
    any cleaning statistics are computed -- this is what keeps the test
    set genuinely held out."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    logger.info(
        "Split: train=%s rows (default rate %.2f%%), test=%s rows (default rate %.2f%%)",
        len(train_df), train_df[TARGET_COL].mean() * 100,
        len(test_df), test_df[TARGET_COL].mean() * 100,
    )
    return train_df.copy(), test_df.copy()


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans the known data-quality issues in the raw dataset:

    1. `age == 0` (a single impossible value in this dataset) is treated
       as missing and imputed with the training median age.

    2. `96` / `98` in the three "past due" count columns are a
       sentinel/error code, not real counts -- the 269 rows carrying
       this code have a ~55% default rate vs. ~6.7% overall, so the
       code itself is highly informative and must not just be dropped.
       It's captured in a binary flag (`had_past_due_sentinel_code`),
       then the raw 96/98 values are replaced with each column's
       training median (computed excluding the sentinel rows) so they
       don't distort model training as an implausible magnitude.

    3. `MonthlyIncome` is missing for ~20% of rows. This missingness is
       not random: ~94% of these rows also have `DebtRatio > 1`,
       suggesting `DebtRatio` may hold a different quantity (e.g. an
       absolute debt figure) when income wasn't captured. A binary flag
       (`monthly_income_missing`) preserves this signal; the value
       itself is imputed with the training median income.

    4. `NumberOfDependents` is missing for ~2.6% of rows; imputed with
       the training median (0). No comparably strong evidence of
       informative missingness was found for this column, so no flag
       is added here (kept proportionate to the signal found).

    5. `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` are ratio
       features with an extreme long right tail (observed up to
       ~50,708 and ~329,664 respectively, where both should typically
       stay well under ~2). Rather than pick an arbitrary domain
       cutoff, both are winsorized (capped) at the 97.5th percentile
       learned on the training data.

    All statistics (medians, percentile caps) are learned in `fit` and
    must be fit on the training split only.
    """

    def __init__(self):
        self.age_median_ = None
        self.past_due_medians_ = {}
        self.monthly_income_median_ = None
        self.dependents_median_ = None
        self.winsorize_caps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # age: median computed excluding the impossible age == 0 rows
        valid_age = X.loc[X["age"] > 0, "age"]
        self.age_median_ = valid_age.median()

        # past-due columns: median computed excluding sentinel codes,
        # so the imputed value reflects genuine borrower behaviour
        for col in PAST_DUE_COLS:
            valid_vals = X.loc[~X[col].isin(SENTINEL_CODES), col]
            self.past_due_medians_[col] = valid_vals.median()

        self.monthly_income_median_ = X["MonthlyIncome"].median()
        self.dependents_median_ = X["NumberOfDependents"].median()

        for col in WINSORIZE_COLS:
            self.winsorize_caps_[col] = X[col].quantile(WINSORIZE_UPPER_PERCENTILE)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. age
        X.loc[X["age"] == 0, "age"] = np.nan
        X["age"] = X["age"].fillna(self.age_median_)

        # 2. past-due sentinel codes -> one combined flag (all three
        # columns carry 96/98 together for the same rows), then impute
        sentinel_mask = pd.Series(False, index=X.index)
        for col in PAST_DUE_COLS:
            sentinel_mask = sentinel_mask | X[col].isin(SENTINEL_CODES)
        X["had_past_due_sentinel_code"] = sentinel_mask.astype(int)

        for col in PAST_DUE_COLS:
            X.loc[X[col].isin(SENTINEL_CODES), col] = np.nan
            X[col] = X[col].fillna(self.past_due_medians_[col])

        # 3. MonthlyIncome
        X["monthly_income_missing"] = X["MonthlyIncome"].isna().astype(int)
        X["MonthlyIncome"] = X["MonthlyIncome"].fillna(self.monthly_income_median_)

        # 4. NumberOfDependents
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(self.dependents_median_)

        # 5. winsorize heavy-tailed ratio features
        for col in WINSORIZE_COLS:
            cap = self.winsorize_caps_[col]
            X[col] = X[col].clip(upper=cap)

        return X


# When this module is run directly via `python -m src.data_preprocessing`,
# Python sets this module's __name__ to "__main__", and pickle/joblib would
# then record DataCleaner's location as "__main__" -- which breaks loading
# it back from any OTHER entry point (train_models.py, inference.py, tests,
# ...), since none of those are the "__main__" module. Pinning __module__
# explicitly keeps the saved artifact portable regardless of how this file
# was executed when it was created.
DataCleaner.__module__ = "src.data_preprocessing"


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    train_df, test_df = split_train_test(df)

    cleaner = DataCleaner()
    cleaner.fit(train_df)

    train_clean = cleaner.transform(train_df)
    test_clean = cleaner.transform(test_df)

    train_clean.to_csv(CLEAN_TRAIN_PATH, index=False)
    test_clean.to_csv(CLEAN_TEST_PATH, index=False)
    logger.info("Saved cleaned train data to %s (%s rows)", CLEAN_TRAIN_PATH, len(train_clean))
    logger.info("Saved cleaned test data to %s (%s rows)", CLEAN_TEST_PATH, len(test_clean))

    import joblib
    joblib.dump(cleaner, DATA_CLEANER_PATH)
    logger.info("Saved fitted DataCleaner to %s", DATA_CLEANER_PATH)


if __name__ == "__main__":
    # Running this file directly makes Python set this module's __name__ to
    # "__main__", so classes defined here (DataCleaner) would get pickled
    # with module="__main__" -- unloadable from any other entry point
    # (train_models.py, inference.py, tests, ...). Re-importing this file
    # under its real package path first makes Python register it normally
    # under "src.data_preprocessing" in sys.modules, so calling THAT
    # module's main() pickles DataCleaner with a stable, portable module
    # reference instead.
    from src.data_preprocessing import main as _canonical_main
    _canonical_main()
