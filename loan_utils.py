"""
loan_utils.py - Core Feature Engineering, Schema Definition, and Model Pipeline
Part of the AI Credit Risk & Financial Inclusion Platform
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Model registry
MODEL_REGISTRY = {
    "XGBoost": "default_risk_pipeline.joblib",
    "Random Forest": "rf_risk_pipeline.joblib",
    "Logistic Regression": "lr_risk_pipeline.joblib",
}

MODEL_DESCRIPTIONS = {
    "XGBoost": {
        "description": "Gradient-boosted ensemble of decision trees. Highest accuracy, handles non-linearity and interactions.",
        "roc_auc": 0.781,
        "f1": 0.712,
        "latency": "~14 ms",
        "interpretability": "Medium (SHAP required)",
        "icon": "⚡",
        "color": "#6366F1"
    },
    "Random Forest": {
        "description": "Bagged ensemble of independent decision trees. Robust to overfitting, excellent OOB estimation.",
        "roc_auc": 0.763,
        "f1": 0.694,
        "latency": "~22 ms",
        "interpretability": "Medium (feature importances)",
        "icon": "🌲",
        "color": "#10B981"
    },
    "Logistic Regression": {
        "description": "Linear probabilistic classifier. Maximum interpretability, fast inference, and regulatory transparency.",
        "roc_auc": 0.731,
        "f1": 0.668,
        "latency": "~3 ms",
        "interpretability": "High (linear coefficients)",
        "icon": "📐",
        "color": "#F59E0B"
    },
}

# -------------------------------------------------------------
# Financial Inclusion thresholds.
# Single source of truth shared by the inclusion flag in
# predict_credit_risk() and by symbolic rule R7 in neuro_symbolic.py,
# so the two can never drift apart again.
# -------------------------------------------------------------
INCLUSION_FREE_CASH_MIN = 500.0   # EUR of discretionary monthly free cash
INCLUSION_P2I_MAX = 0.20          # payment-to-income ceiling (20%)

# -------------------------------------------------------------
# 1. Feature Specifications
# -------------------------------------------------------------
NUMERIC_FEATURES = [
    "Age",
    "AppliedAmount",
    "Amount",
    "Interest",
    "LoanDuration",
    "MonthlyPayment",
    "IncomeTotal",
    "ExistingLiabilities",
    "LiabilitiesTotal",
    "DebtToIncome",
    "FreeCash",
    "AmountOfPreviousLoansBeforeLoan",
    "NoOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan",
    # Engineered features
    "PaymentToIncome",
    "LiabilityToIncome",
    "LoanToAnnualIncome",
    "FundingGap",
    "PreviousRepaymentRatio"
]

CATEGORICAL_FEATURES = [
    "Country",
    "NewCreditCustomer",
    "Education",
    "EmploymentStatus",
    "HomeOwnershipType"
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_OPTIONS = {
    "Country": ["EE", "FI", "ES", "SK"],
    "NewCreditCustomer": ["No", "Yes"],
    "Education": ["Primary", "Basic", "Vocational", "Secondary", "Higher"],
    "EmploymentStatus": ["Fully employed", "Part-time", "Self-employed", "Entrepreneur", "Retiree", "Unemployed"],
    "HomeOwnershipType": ["Owner", "Tenant", "Mortgage", "Living with parents", "Other"]
}

# -------------------------------------------------------------
# 2. Feature Engineering
# -------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes derived application-time risk ratios without future leakage."""
    data = df.copy()

    # Safe divisions avoiding zero-division
    income = np.maximum(data.get("IncomeTotal", 1.0), 1.0)
    data["PaymentToIncome"] = data.get("MonthlyPayment", 0.0) / income
    data["LiabilityToIncome"] = data.get("LiabilitiesTotal", 0.0) / income
    data["LoanToAnnualIncome"] = data.get("Amount", 0.0) / (income * 12.0)
    data["FundingGap"] = data.get("AppliedAmount", 0.0) - data.get("Amount", 0.0)

    prev_amount = np.maximum(data.get("AmountOfPreviousLoansBeforeLoan", 0.0), 1.0)
    prev_repay = data.get("PreviousRepaymentsBeforeLoan", 0.0)
    has_prev = data.get("NoOfPreviousLoansBeforeLoan", 0) > 0
    data["PreviousRepaymentRatio"] = np.where(has_prev, np.clip(prev_repay / prev_amount, 0.0, 2.0), 0.0)

    # Ensure all required features are present
    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = CATEGORICAL_OPTIONS[col][0]

    return data[ALL_MODEL_FEATURES]

# -------------------------------------------------------------
# 3. Model Pipeline Builder & Surrogate Fallback
# -------------------------------------------------------------
def create_preprocessing_pipeline() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def _build_synthetic_training_data(n_samples: int = 3000):
    """Generates synthetic Bondora-calibrated training data. Returns (X_train, y_train, raw_synth)."""
    np.random.seed(42)
    age = np.random.normal(38, 11, n_samples).clip(18, 72)
    income = np.random.lognormal(7.3, 0.6, n_samples).clip(300, 8000)
    applied_amount = np.random.lognormal(7.6, 0.8, n_samples).clip(500, 10000)
    amount = applied_amount * np.random.uniform(0.85, 1.0, n_samples)
    duration = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.35, 0.2, 0.15])
    interest = np.random.normal(24.0, 10.0, n_samples).clip(6.0, 60.0)
    monthly_payment = (amount * (1 + interest / 100.0)) / duration
    existing_liab = np.random.poisson(1.8, n_samples).clip(0, 8)
    liab_total = (existing_liab * np.random.uniform(50, 300, n_samples)).clip(0, income * 0.8)
    debt_to_income = liab_total / income
    free_cash = np.maximum(income - monthly_payment - liab_total, -200)
    has_prev = np.random.binomial(1, 0.45, n_samples)
    no_prev = (has_prev * np.random.randint(1, 5, n_samples)).astype(int)
    amt_prev = has_prev * np.random.uniform(1000, 8000, n_samples)
    repay_prev = has_prev * (amt_prev * np.random.uniform(0.4, 1.1, n_samples))
    countries = np.random.choice(["EE", "FI", "ES", "SK"], n_samples, p=[0.55, 0.25, 0.18, 0.02])
    new_cust = np.random.choice(["No", "Yes"], n_samples, p=[0.45, 0.55])
    education = np.random.choice(["Higher", "Secondary", "Vocational", "Basic", "Primary"], n_samples, p=[0.25, 0.45, 0.20, 0.07, 0.03])
    emp_status = np.random.choice(["Fully employed", "Part-time", "Self-employed", "Entrepreneur", "Retiree", "Unemployed"], n_samples, p=[0.68, 0.08, 0.10, 0.05, 0.06, 0.03])
    home = np.random.choice(["Owner", "Tenant", "Mortgage", "Living with parents", "Other"], n_samples, p=[0.38, 0.32, 0.18, 0.08, 0.04])

    raw_synth = pd.DataFrame({
        "Age": age, "AppliedAmount": applied_amount, "Amount": amount,
        "Interest": interest, "LoanDuration": duration, "MonthlyPayment": monthly_payment,
        "IncomeTotal": income, "ExistingLiabilities": existing_liab, "LiabilitiesTotal": liab_total,
        "DebtToIncome": debt_to_income, "FreeCash": free_cash,
        "AmountOfPreviousLoansBeforeLoan": amt_prev, "NoOfPreviousLoansBeforeLoan": no_prev,
        "PreviousRepaymentsBeforeLoan": repay_prev,
        "Country": countries, "NewCreditCustomer": new_cust, "Education": education,
        "EmploymentStatus": emp_status, "HomeOwnershipType": home
    })
    X_train = engineer_features(raw_synth)
    latent_score = (
        0.04 * (interest - 22.0) + 0.025 * (duration - 36.0)
        + 1.8 * (X_train["PaymentToIncome"] - 0.15)
        + 1.5 * (X_train["LiabilityToIncome"] - 0.20)
        - 0.0003 * (income - 1500)
        - 0.8 * X_train["PreviousRepaymentRatio"]
        + 0.45 * (countries == "ES") + 0.25 * (countries == "FI")
        - 0.30 * (education == "Higher") + 0.35 * (new_cust == "Yes")
        + np.random.normal(0, 0.6, n_samples)
    )
    p_default = 1.0 / (1.0 + np.exp(-latent_score))
    y_train = (np.random.rand(n_samples) < p_default).astype(int)
    return X_train, y_train


def get_or_create_model_pipeline(artifact_path: str = "default_risk_pipeline.joblib") -> Pipeline:
    """Loads saved XGBoost pipeline from disk if available, else builds and caches it."""
    return _load_or_build_pipeline(
        artifact_path=artifact_path,
        model_name="XGBoost"
    )


def _load_or_build_pipeline(artifact_path: str, model_name: str) -> Pipeline:
    """Generic loader: tries disk paths, falls back to building and caching."""
    search_paths = [
        artifact_path,
        os.path.join(os.path.dirname(__file__), artifact_path),
    ]
    for p in search_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                print(f"[loan_utils] Loaded {model_name} from {p}")
                return model
            except Exception as e:
                print(f"[loan_utils] Warning: could not load {p}: {e}")

    print(f"[loan_utils] Building {model_name} pipeline from scratch...")
    X_train, y_train = _build_synthetic_training_data()
    preprocessor = create_preprocessing_pipeline()

    if model_name == "XGBoost":
        classifier = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, eval_metric="logloss"
        )
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:  # Logistic Regression
        classifier = LogisticRegression(
            C=0.5, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42
        )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)

    save_path = os.path.join(os.path.dirname(__file__), artifact_path)
    try:
        joblib.dump(pipeline, save_path)
        print(f"[loan_utils] Saved {model_name} pipeline to {save_path}")
    except Exception as e:
        print(f"[loan_utils] Could not save pipeline: {e}")

    return pipeline


def get_all_model_pipelines() -> Dict[str, Pipeline]:
    """Returns a dict of {model_name: pipeline} for all 3 models."""
    pipelines = {}
    for name, fname in MODEL_REGISTRY.items():
        pipelines[name] = _load_or_build_pipeline(artifact_path=fname, model_name=name)
    return pipelines

# -------------------------------------------------------------
# 4. Scoring & Risk Rating Assessment
# -------------------------------------------------------------
def predict_credit_risk(pipeline: Pipeline, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runs single applicant inference and produces PD, Risk Category, and Credit Grade."""
    # Derive cash-flow fields once if the caller omitted them or passed None,
    # so grading, rules and the inclusion flag all see the same concrete values.
    if applicant_data.get("DebtToIncome") is None:
        applicant_data["DebtToIncome"] = (
            applicant_data.get("LiabilitiesTotal", 0.0) / max(1.0, applicant_data.get("IncomeTotal", 1.0))
        )
    if applicant_data.get("FreeCash") is None:
        applicant_data["FreeCash"] = (
            applicant_data.get("IncomeTotal", 0.0)
            - applicant_data.get("MonthlyPayment", 0.0)
            - applicant_data.get("LiabilitiesTotal", 0.0)
        )

    input_df = pd.DataFrame([applicant_data])
    feat_df = engineer_features(input_df)

    prob_default = float(pipeline.predict_proba(feat_df)[0, 1])

    # Grade and category
    if prob_default < 0.15:
        grade = "A"
        category = "Very Low Risk"
        recommendation = "Approved (Prime Terms)"
        color = "#10B981"  # Emerald
    elif prob_default < 0.28:
        grade = "B"
        category = "Low Risk"
        recommendation = "Approved (Standard Terms)"
        color = "#34D399"
    elif prob_default < 0.45:
        grade = "C"
        category = "Moderate Risk"
        recommendation = "Approved with Risk Premium"
        color = "#F59E0B"  # Amber
    elif prob_default < 0.60:
        grade = "D"
        category = "Elevated Risk"
        recommendation = "Conditional / Manual Review"
        color = "#F97316"  # Orange
    elif prob_default < 0.75:
        grade = "E"
        category = "High Risk"
        recommendation = "Declined (High Risk Threshold)"
        color = "#EF4444"  # Red
    else:
        grade = "F"
        category = "Severe Risk"
        recommendation = "Declined (Critical Risk)"
        color = "#991B1B"

    # Financial Inclusion Flag — thresholds must match symbolic rule R7
    # (shared constants INCLUSION_FREE_CASH_MIN / INCLUSION_P2I_MAX).
    free_cash = float(applicant_data.get("FreeCash") or 0.0)
    p_to_i = applicant_data.get("MonthlyPayment", 0.0) / max(1.0, applicant_data.get("IncomeTotal", 1.0))
    is_inclusion_candidate = (
        applicant_data.get("NewCreditCustomer") == "Yes"
        and prob_default < 0.55
        and free_cash > INCLUSION_FREE_CASH_MIN
        and p_to_i < INCLUSION_P2I_MAX
    )

    return {
        "probability_of_default": prob_default,
        "credit_grade": grade,
        "risk_category": category,
        "recommendation": recommendation,
        "accent_color": color,
        "is_inclusion_candidate": is_inclusion_candidate,
        "engineered_features": feat_df.iloc[0].to_dict()
    }
