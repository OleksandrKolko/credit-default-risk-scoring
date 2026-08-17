"""
Inference for a single new client: cleaning, feature engineering,
prediction, and a SHAP-based explanation of that specific prediction --
reusing the exact same fitted `DataCleaner` and `FeatureEngineer` used
during training, so a new client is transformed identically to how the
training data was.

This module also produces the client-level SHAP analysis requested for
the project: waterfall plots for a few representative clients (highest
risk, lowest risk, a borderline case, and a case the model got wrong),
plus a global feature-importance summary -- see `main()`.

Why probability-space SHAP values
------------------------------------
`shap.TreeExplainer`'s default output for a LightGBM binary classifier is
in log-odds (margin) space, which is exact and fast but not something a
credit analyst or a loan applicant can intuitively read ("this feature
contributed -0.3 log-odds" means nothing to a non-technical audience).
Instead, this module uses `feature_perturbation="interventional"` with
`model_output="probability"` and a small background sample, so SHAP
values are directly in percentage points of default probability (e.g.
"high revolving utilization added 6 percentage points of risk") --
slightly more expensive to compute than the default, but fast enough in
practice (~3ms/row) and far easier to explain to a non-technical
audience, which is the whole point of this step.
"""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    DATA_CLEANER_PATH, FINAL_MODEL_PATH, FEATURES_TRAIN_PATH, FEATURES_TEST_PATH,
    TARGET_COL, RAW_FEATURE_COLS, FIGURES_DIR, REPORTS_DIR,
)
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Human-readable labels for the SHAP plots and top-factor output -- a
# credit analyst reading "DebtRatio" has to guess what it means; "Debt
# payments relative to income" doesn't.
FEATURE_DISPLAY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization (% of limit used)",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days late",
    "DebtRatio": "Debt payments relative to income",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit lines/loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Number of real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days late",
    "NumberOfDependents": "Number of dependents",
    "had_past_due_sentinel_code": "Had unusual/flagged delinquency record",
    "monthly_income_missing": "Income was not reported",
    "total_past_due_count": "Total past-due incidents (all categories)",
    "has_any_past_due": "Has any past-due history",
    "income_per_dependent": "Income per household dependent",
    "estimated_monthly_debt_payment": "Estimated absolute monthly debt payment",
    "utilization_x_credit_lines": "Utilization spread across credit lines",
    "real_estate_loan_share": "Share of credit that is real-estate loans",
    "age_group": "Age group",
}

BACKGROUND_SAMPLE_SIZE = 100
RANDOM_STATE_FOR_BACKGROUND = 42


class CreditRiskPredictor:
    """
    Loads the fitted DataCleaner and the final tuned model once, then
    exposes `predict()`, which takes a new client's RAW features (the
    same columns as the original Kaggle CSV) and returns a default
    probability plus a ranked, human-readable SHAP explanation of THAT
    specific prediction.
    """

    def __init__(self):
        self.cleaner = joblib.load(DATA_CLEANER_PATH)
        self.feature_engineer = FeatureEngineer()  # stateless -- no fitted state to load
        self.model = joblib.load(FINAL_MODEL_PATH)

        background_df = pd.read_csv(FEATURES_TRAIN_PATH).drop(columns=[TARGET_COL])
        background = shap.sample(background_df, BACKGROUND_SAMPLE_SIZE,
                                  random_state=RANDOM_STATE_FOR_BACKGROUND)
        self.explainer = shap.TreeExplainer(
            self.model, data=background,
            feature_perturbation="interventional", model_output="probability",
        )

    def _prepare_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.cleaner.transform(raw_df)
        return self.feature_engineer.transform(cleaned)

    def explain_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """SHAP values (probability space) for an already-cleaned,
        already-engineered feature DataFrame. Useful for global analysis
        over many rows at once (see main())."""
        return self.explainer.shap_values(features_df)

    def predict(self, raw_client: dict, top_n: int = 8) -> dict:
        """
        raw_client: dict with the ORIGINAL Kaggle column names (see
        RAW_FEATURE_COLS in src/config.py), e.g.
        {"RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 45, ...}
        Do NOT include the target column.

        Returns the predicted default probability plus the top_n
        features that pushed that specific prediction up or down the
        most, in percentage points.
        """
        missing = [c for c in RAW_FEATURE_COLS if c not in raw_client]
        if missing:
            raise ValueError(f"Missing required client fields: {missing}")

        raw_df = pd.DataFrame([raw_client])[RAW_FEATURE_COLS]
        features_df = self._prepare_features(raw_df)

        probability = float(self.model.predict_proba(features_df)[0, 1])
        shap_values = self.explainer.shap_values(features_df)[0]
        base_value = float(self.explainer.expected_value)

        contributions = sorted(
            zip(features_df.columns, features_df.iloc[0].values, shap_values),
            key=lambda row: abs(row[2]), reverse=True,
        )

        top_factors = [
            {
                "feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "raw_feature_name": feat,
                "value": float(val),
                "impact_percentage_points": round(float(shap_val) * 100, 2),
            }
            for feat, val, shap_val in contributions[:top_n]
        ]

        return {
            "default_probability": round(probability, 4),
            "base_rate_percentage_points": round(base_value * 100, 2),
            "top_factors": top_factors,
        }


def plot_client_waterfall(predictor: CreditRiskPredictor, features_row: pd.Series,
                           title: str, save_path):
    shap_values = predictor.explainer.shap_values(features_row.to_frame().T)[0]
    explanation = shap.Explanation(
        values=shap_values,
        base_values=predictor.explainer.expected_value,
        data=features_row.values,
        feature_names=[FEATURE_DISPLAY_NAMES.get(c, c) for c in features_row.index],
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(title, fontsize=11, loc="left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved waterfall plot to %s", save_path)


def plot_global_summary(predictor: CreditRiskPredictor, X_sample: pd.DataFrame, save_path,
                         n_sample=500):
    sample = X_sample.sample(n=min(n_sample, len(X_sample)), random_state=RANDOM_STATE_FOR_BACKGROUND)
    shap_values = predictor.explain_batch(sample)

    display_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in sample.columns]
    plt.figure()
    shap.summary_plot(
        shap_values, sample, feature_names=display_names, show=False, max_display=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP summary plot to %s (n=%s)", save_path, len(sample))


def select_representative_clients(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Picks a few clients whose stories are worth telling:
    - highest predicted risk
    - lowest predicted risk
    - borderline (closest to the chosen decision threshold)
    - a notable miss: an actual defaulter the model was most confident
      was safe (the costliest kind of error for a lender)
    """
    probs = model.predict_proba(X_test)[:, 1]

    highest_risk_idx = int(np.argmax(probs))
    lowest_risk_idx = int(np.argmin(probs))
    borderline_idx = int(np.argmin(np.abs(probs - threshold)))

    actual_defaulters_mask = (y_test.values == 1)
    if actual_defaulters_mask.any():
        defaulter_probs = np.where(actual_defaulters_mask, probs, np.inf)
        missed_default_idx = int(np.argmin(defaulter_probs))
    else:
        missed_default_idx = lowest_risk_idx

    return {
        "highest_risk_client": highest_risk_idx,
        "lowest_risk_client": lowest_risk_idx,
        "borderline_client": borderline_idx,
        "missed_default_client": missed_default_idx,
    }


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictor = CreditRiskPredictor()

    test_df = pd.read_csv(FEATURES_TEST_PATH)
    y_test = test_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])

    try:
        with open(REPORTS_DIR / "final_evaluation.json") as f:
            threshold = json.load(f)["chosen_threshold"]
    except FileNotFoundError:
        threshold = 0.5
        logger.warning("reports/final_evaluation.json not found -- using default threshold 0.5")

    # --- Global feature importance, on a sample of the test set ---
    plot_global_summary(predictor, X_test, FIGURES_DIR / "shap_summary.png")

    # --- Individual client waterfall plots ---
    clients = select_representative_clients(predictor.model, X_test, y_test, threshold)
    probs = predictor.model.predict_proba(X_test)[:, 1]

    summaries = {}
    for name, idx in clients.items():
        row = X_test.iloc[idx]
        actual = int(y_test.iloc[idx])
        prob = float(probs[idx])
        title = f"{name.replace('_', ' ').title()} (P(default)={prob:.1%}, actual={'defaulted' if actual else 'no default'})"

        plot_client_waterfall(predictor, row, title, FIGURES_DIR / f"shap_waterfall_{name}.png")

        shap_values = predictor.explainer.shap_values(row.to_frame().T)[0]
        top3 = sorted(zip(row.index, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
        summaries[name] = {
            "predicted_probability": round(prob, 4),
            "actual_label": actual,
            "top_factors": [
                {"feature": FEATURE_DISPLAY_NAMES.get(f, f), "impact_percentage_points": round(v * 100, 2)}
                for f, v in top3
            ],
        }
        logger.info("%s -> P(default)=%.1f%%, actual=%s, top factors: %s",
                    name, prob * 100, actual, [t["feature"] for t in summaries[name]["top_factors"]])

    with open(REPORTS_DIR / "shap_client_examples.json", "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Saved client SHAP summaries to %s", REPORTS_DIR / "shap_client_examples.json")

    # --- Demo: a hypothetical new client through the full predict() API ---
    demo_client = {
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "age": 29,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.6,
        "MonthlyIncome": 2800,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }
    result = predictor.predict(demo_client)
    print("\n=== Demo: new client prediction ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
