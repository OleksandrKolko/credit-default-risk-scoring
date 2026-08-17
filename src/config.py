"""
Central configuration for the credit risk scoring project.

Keeping paths, column names and constants in one place avoids magic
strings scattered across preprocessing / feature engineering / training
/ inference modules, and makes it obvious what would need to change if
the raw data source ever changes.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_TRAIN_PATH = DATA_RAW_DIR / "cs-training.csv"
CLEAN_TRAIN_PATH = DATA_PROCESSED_DIR / "train_clean.csv"
CLEAN_TEST_PATH = DATA_PROCESSED_DIR / "test_clean.csv"
FEATURES_TRAIN_PATH = DATA_PROCESSED_DIR / "train_features.csv"
FEATURES_TEST_PATH = DATA_PROCESSED_DIR / "test_features.csv"

DATA_CLEANER_PATH = MODELS_DIR / "data_cleaner.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"

# --- Data schema ---------------------------------------------------------
# "Give Me Some Credit" (Kaggle) raw column names.
# The first column in the raw CSV is an unnamed row index -> dropped on load.
TARGET_COL = "SeriousDlqin2yrs"

RAW_FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# --- Reproducibility ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2  # held-out test split, stratified on TARGET_COL

# --- Cross-validation ------------------------------------------------------
N_SPLITS = 5  # for StratifiedKFold during model comparison / Optuna
