"""
xai_engine.py - SHAP and LIME Explainability Engine
Provides local and global model interpretability with additivity checks and feature attributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import shap
from sklearn.linear_model import Ridge

# Tolerance for verifying that base_val + sum_shap matches the model's own output.
# XGBoost SHAP values live in raw margin (log-odds) space, where exact additivity holds
# to float32 precision (~1e-5). RandomForest TreeExplainer operates in probability space,
# where the same relationship holds to ~1e-4 relative error. A single universal tolerance
# balances both model families.
ADDITIVITY_TOL = 1e-4


class XAIEngine:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
        self.classifier = getattr(pipeline, "named_steps", {}).get("classifier", None)

    @staticmethod
    def _classifier_module(cls) -> str:
        """Return the top-level module name of a classifier class (e.g. 'xgboost')."""
        return getattr(cls, "__module__", "").split(".")[0] if cls else ""

    def explain_shap(self, input_features_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates SHAP tree attributions and verifies mathematical additivity."""
        try:
            # Transform input features using pipeline preprocessor
            if self.preprocessor is not None:
                X_trans = self.preprocessor.transform(input_features_df)
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                X_trans = input_features_df.values
                feature_names = input_features_df.columns

            # Initialize TreeExplainer
            explainer = shap.TreeExplainer(self.classifier)
            shap_values = explainer.shap_values(X_trans)

            # Handle binary classification shapes
            if isinstance(shap_values, list) and len(shap_values) == 2:
                vals = shap_values[1][0]
                base_val = float(explainer.expected_value[1])
            elif hasattr(shap_values, "values"):
                vals = shap_values.values[0]
                base_val = float(explainer.expected_value)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                vals = shap_values[0]
                base_val = float(explainer.expected_value)
            else:
                vals = np.array(shap_values).flatten()
                base_val = 0.0

            sum_shap = float(np.sum(vals))
            raw_sum = base_val + sum_shap

            # Reference output from the model itself, in the same space as the SHAP values.
            pred_prob = float(np.clip(self.pipeline.predict_proba(input_features_df)[0, 1], 1e-12, 1 - 1e-12))
            if self._classifier_module(self.classifier).startswith("xgboost"):
                # XGBoost SHAP values are in log-odds margin space; compare against logit(p)
                reference_output = float(np.log(pred_prob / (1.0 - pred_prob)))
                # implied probability from the verified margin
                prob_implied = 1.0 / (1.0 + np.exp(-raw_sum))
            else:
                # Tree ensemble (e.g. RandomForest) SHAP values live in probability space;
                # the raw sum already approximates the predicted probability.
                reference_output = pred_prob
                prob_implied = np.clip(raw_sum, 0.0, 1.0)

            # Verify additivity: exact match between (base + sum) and the model's own output.
            additivity_verified = bool(abs(raw_sum - reference_output) <= ADDITIVITY_TOL)
            detail = None if additivity_verified else (
                f"|f(x) - (base + sum)| = {abs(raw_sum - reference_output):.2e} exceeds tolerance {ADDITIVITY_TOL}"
            )

            # Format feature names for clean UI display
            clean_names = []
            for name in feature_names:
                clean = name.replace("num__", "").replace("cat__", "").replace("_", " ")
                clean_names.append(clean)

            # Build contributions list
            items = []
            for name, val in zip(clean_names, vals):
                items.append({
                    "feature": name,
                    "shap_value": float(val),
                    "abs_val": abs(float(val)),
                    "direction": "Increases Risk" if val > 0 else "Lowers Risk"
                })

            # Sort by absolute impact
            items.sort(key=lambda x: x["abs_val"], reverse=True)
            top_drivers = items[:12]

            return {
                "success": True,
                "base_value": base_val,
                "sum_shap": sum_shap,
                "implied_probability": prob_implied,
                "additivity_verified": additivity_verified,
                "detail": detail,
                "top_drivers": top_drivers
            }

        except Exception as e:
            # Fallback heuristic feature importance if shap tree fails — be honest:
            # do NOT report success=True or additivity_verified=True with fake values.
            warning_msg = f"[xai_engine] SHAP TreeExplainer warning: {e}. Using permutation surrogate."
            print(warning_msg)
            row = input_features_df.iloc[0]
            base_prob = float(self.pipeline.predict_proba(input_features_df)[0, 1])

            drivers = [
                {"feature": "Interest Rate", "shap_value": float(0.04 * (row.get("Interest", 20.0) - 20.0))},
                {"feature": "Payment to Income", "shap_value": float(1.2 * (row.get("PaymentToIncome", 0.15) - 0.12))},
                {"feature": "Loan Duration", "shap_value": float(0.02 * (row.get("LoanDuration", 36) - 36))},
                {"feature": "Discretionary Free Cash", "shap_value": float(-0.0003 * (row.get("FreeCash", 500) - 400))},
                {"feature": "Total Liabilities", "shap_value": float(0.0004 * (row.get("LiabilitiesTotal", 200) - 150))},
                {"feature": "Previous Repayment History", "shap_value": float(-0.6 * row.get("PreviousRepaymentRatio", 0.0))}
            ]
            for d in drivers:
                d["abs_val"] = abs(d["shap_value"])
                d["direction"] = "Increases Risk" if d["shap_value"] > 0 else "Lowers Risk"

            drivers.sort(key=lambda x: x["abs_val"], reverse=True)

            return {
                "success": False,
                "base_value": 0.0,
                "sum_shap": float(sum(d["shap_value"] for d in drivers)),
                "implied_probability": base_prob,
                "additivity_verified": False,
                "detail": f"Exact TreeExplainer attribution unavailable ({e}); showing heuristic directional sensitivities only.",
                "top_drivers": drivers
            }

    def _heuristic_shap_fallback(self, input_features_df: pd.DataFrame) -> Dict[str, Any]:
        """Deprecated: use the outer try/except fallback instead."""
        row = input_features_df.iloc[0]
        base_prob = float(self.pipeline.predict_proba(input_features_df)[0, 1])

        drivers = [
            {"feature": "Interest Rate", "shap_value": float(0.04 * (row.get("Interest", 20.0) - 20.0))},
            {"feature": "Payment to Income", "shap_value": float(1.2 * (row.get("PaymentToIncome", 0.15) - 0.12))},
            {"feature": "Loan Duration", "shap_value": float(0.02 * (row.get("LoanDuration", 36) - 36))},
            {"feature": "Discretionary Free Cash", "shap_value": float(-0.0003 * (row.get("FreeCash", 500) - 400))},
            {"feature": "Total Liabilities", "shap_value": float(0.0004 * (row.get("LiabilitiesTotal", 200) - 150))},
            {"feature": "Previous Repayment History", "shap_value": float(-0.6 * row.get("PreviousRepaymentRatio", 0.0))}
        ]
        for d in drivers:
            d["abs_val"] = abs(d["shap_value"])
            d["direction"] = "Increases Risk" if d["shap_value"] > 0 else "Lowers Risk"

        drivers.sort(key=lambda x: x["abs_val"], reverse=True)
        # This method is no longer called directly; the outer except block handles the
        # fallback path with proper honesty about success=False / additivity_verified=False.
        return {
            "success": False,
            "base_value": 0.0,
            "sum_shap": float(sum(d["shap_value"] for d in drivers)),
            "implied_probability": base_prob,
            "additivity_verified": False,
            "top_drivers": drivers
        }

    def explain_lime(self, original_applicant_dict: Dict[str, Any], n_samples: int = 150) -> Dict[str, Any]:
        """
        Runs LIME local surrogate on the unencoded original applicant feature domain,
        preventing unrealistic one-hot combinations as highlighted in notebook review.
        """
        np.random.seed(42)
        base_applicant = original_applicant_dict.copy()
        from loan_utils import engineer_features
        base_df = engineer_features(pd.DataFrame([base_applicant]))
        pred_base = float(self.pipeline.predict_proba(base_df)[0, 1])

        # Generate local perturbations in original applicant continuous space
        numeric_perturb_cols = ["Interest", "MonthlyPayment", "IncomeTotal", "LiabilitiesTotal", "FreeCash", "LoanDuration"]
        perturbed_rows = []

        for _ in range(n_samples):
            pert = base_applicant.copy()
            for col in numeric_perturb_cols:
                curr_val = float(pert.get(col, 100.0))
                std_noise = max(1.0, curr_val * 0.15)
                pert[col] = max(0.0, curr_val + np.random.normal(0, std_noise))
            perturbed_rows.append(pert)

        pert_df = pd.DataFrame(perturbed_rows)
        # Import feature engineering from loan_utils
        from loan_utils import engineer_features
        feat_pert_df = engineer_features(pert_df)
        y_pert = self.pipeline.predict_proba(feat_pert_df)[:, 1]

        # Fit interpretable local Ridge surrogate
        X_local = pert_df[numeric_perturb_cols].values
        scaler = (X_local - X_local.mean(axis=0)) / (X_local.std(axis=0) + 1e-6)

        surrogate = Ridge(alpha=1.0)
        surrogate.fit(scaler, y_pert)
        local_r2 = float(surrogate.score(scaler, y_pert))

        explanations = []
        for col_name, coef in zip(numeric_perturb_cols, surrogate.coef_):
            val = float(base_applicant.get(col_name, 0.0))
            explanations.append({
                "feature": col_name,
                "current_value": f"{val:.1f}",
                "local_weight": float(coef),
                "direction": "Increases Risk" if coef > 0 else "Lowers Risk",
                "abs_weight": abs(float(coef))
            })

        explanations.sort(key=lambda x: x["abs_weight"], reverse=True)

        return {
            "success": True,
            "local_fidelity_r2": max(0.0, min(1.0, local_r2)),
            "base_prediction": pred_base,
            "local_intercept": float(surrogate.intercept_),
            "local_explanations": explanations
        }
