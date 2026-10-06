"""
main.py - Aegis Production FastAPI Microservice
Enterprise REST API for Credit Risk Scoring, Triple-Layer XAI, and Regulatory Audits.
Fully documented with OpenAPI (Swagger UI) at /docs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import uvicorn
import os

# Internal modules
from loan_utils import (
    get_or_create_model_pipeline,
    predict_credit_risk,
    engineer_features,
    CATEGORICAL_OPTIONS
)
from neuro_symbolic import NeuroSymbolicEngine
from xai_engine import XAIEngine
from fairness_audit import get_fairness_audit_data

# -------------------------------------------------------------
# 1. API Initialization & Middleware
# -------------------------------------------------------------
app = FastAPI(
    title="Aegis AI Credit Risk & Financial Inclusion Engine",
    description="Production-grade asynchronous REST API providing calibrated default risk predictions, "
                "SHAP attributions, LIME local continuous surrogates, First-Order Logic neuro-symbolic audits, "
                "and algorithmic fairness analytics on the Bondora P2P seasoned loan population.",
    version="2.0.0",
    contact={
        "name": "AIML Case Study Evaluation",
        "department": "AIML Department, Symbiosis Institute of Technology, Pune"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline and engines at startup
pipeline = get_or_create_model_pipeline()
symbolic_engine = NeuroSymbolicEngine()
xai_engine = XAIEngine(pipeline)

# -------------------------------------------------------------
# 2. Pydantic Schemas for Strict Type Safety
# -------------------------------------------------------------
class ApplicantRequest(BaseModel):
    Age: int = Field(35, ge=18, le=85, description="Applicant chronological age")
    AppliedAmount: float = Field(3500.0, ge=100.0, le=50000.0, description="Requested loan amount in EUR")
    Amount: float = Field(3500.0, ge=100.0, le=50000.0, description="Approved funded principal in EUR")
    Interest: float = Field(17.5, ge=1.0, le=80.0, description="Nominal annual interest rate in percent")
    LoanDuration: int = Field(36, description="Loan duration in months (e.g. 12, 24, 36, 48, 60)")
    MonthlyPayment: float = Field(125.0, ge=5.0, le=5000.0, description="Contractual monthly repayment in EUR")
    IncomeTotal: float = Field(2600.0, ge=100.0, le=100000.0, description="Verified total monthly income in EUR")
    ExistingLiabilities: int = Field(1, ge=0, le=30, description="Number of existing active debt obligations")
    LiabilitiesTotal: float = Field(200.0, ge=0.0, le=50000.0, description="Total monthly debt servicing payments in EUR")
    DebtToIncome: Optional[float] = Field(None, description="Optional DTI; auto-derived if omitted")
    FreeCash: Optional[float] = Field(None, description="Discretionary net cash after all living and debt obligations")
    Country: str = Field("EE", description="Jurisdiction code: EE (Estonia), FI (Finland), ES (Spain), SK (Slovakia)")
    NewCreditCustomer: str = Field("No", description="Thin-file indicator: 'Yes' or 'No'")
    Education: str = Field("Higher", description="Primary, Basic, Vocational, Secondary, Higher")
    EmploymentStatus: str = Field("Fully employed", description="Fully employed, Part-time, Self-employed, Entrepreneur, Retiree, Unemployed")
    HomeOwnershipType: str = Field("Owner", description="Owner, Tenant, Mortgage, Living with parents, Other")
    NoOfPreviousLoansBeforeLoan: int = Field(1, ge=0, description="Prior Bondora settled facilities count")
    AmountOfPreviousLoansBeforeLoan: float = Field(2500.0, ge=0.0, description="Cumulative prior borrowed principal in EUR")
    PreviousRepaymentsBeforeLoan: float = Field(2500.0, ge=0.0, description="Total prior principal repayments satisfied in EUR")

    @model_validator(mode="after")
    def derive_cashflow_fields(self) -> "ApplicantRequest":
        """Derives DebtToIncome and FreeCash exactly once, at the schema boundary.

        Every endpoint (/predict, /explain/shap, /explain/lime, /audit/rules)
        therefore receives the same concrete values instead of None, which
        previously caused float(None) crashes and silent median imputation.
        """
        if self.DebtToIncome is None:
            self.DebtToIncome = self.LiabilitiesTotal / max(1.0, self.IncomeTotal)
        if self.FreeCash is None:
            self.FreeCash = self.IncomeTotal - self.MonthlyPayment - self.LiabilitiesTotal
        return self

class PredictionResponse(BaseModel):
    probability_of_default: float
    credit_grade: str
    risk_category: str
    recommendation: str
    accent_color: str
    is_inclusion_candidate: bool
    engineered_features: Dict[str, Any]

class SHAPExplanationResponse(BaseModel):
    success: bool
    base_value: float
    sum_shap: float
    implied_probability: float
    additivity_verified: bool
    top_drivers: List[Dict[str, Any]]
    detail: Optional[str] = Field(None, description="Diagnostic message when exact attribution was unavailable")

class LIMEExplanationResponse(BaseModel):
    success: bool
    local_fidelity_r2: float
    base_prediction: float
    local_intercept: float
    local_explanations: List[Dict[str, Any]]

class NeuroSymbolicResponse(BaseModel):
    symbolic_score: float
    symbolic_risk_probability: float = Field(
        ...,
        description="Normalized symbolic risk index in [0,1] (scorecard points / 100). "
                    "This is NOT a calibrated probability of default; consensus with the ML "
                    "model is decided on shared decision bands, not on this raw number."
    )
    ml_decision_band: str = Field(..., description="Underwriting decision band derived from the ML PD")
    symbolic_decision_band: str = Field(..., description="Underwriting decision band derived from the symbolic scorecard index")
    triggered_rules: List[Dict[str, Any]]
    negative_rules_count: int
    positive_rules_count: int
    consensus_status: str
    consensus_desc: str
    status_color: str
    deduction_summary: str

# -------------------------------------------------------------
# 3. Microservice Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "service": "Aegis Credit Risk Microservice",
        "status": "HEALTHY",
        "model_loaded": pipeline is not None,
        "api_docs": "/docs"
    }

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Scoring Engine"])
async def predict(applicant: ApplicantRequest):
    """Computes real-time Probability of Default (PD), regulatory credit grade, and underwriting recommendation."""
    data_dict = applicant.model_dump()
    res = predict_credit_risk(pipeline, data_dict)
    return res

@app.post("/api/v1/explain/shap", response_model=SHAPExplanationResponse, tags=["Explainable AI"])
async def explain_shap(applicant: ApplicantRequest):
    """Generates exact SHAP tree attributions and verifies mathematical additivity."""
    data_dict = applicant.model_dump()
    feat_df = engineer_features(pd.DataFrame([data_dict]))
    shap_res = xai_engine.explain_shap(feat_df)
    return shap_res

@app.post("/api/v1/explain/lime", response_model=LIMEExplanationResponse, tags=["Explainable AI"])
async def explain_lime(applicant: ApplicantRequest):
    """Fits local continuous surrogate (LIME) within unencoded original applicant feature domain."""
    data_dict = applicant.model_dump()
    lime_res = xai_engine.explain_lime(data_dict)
    return lime_res

@app.post("/api/v1/audit/rules", response_model=NeuroSymbolicResponse, tags=["Neuro-Symbolic Governance"])
async def audit_symbolic(applicant: ApplicantRequest):
    """Executes First-Order Logic expert rulebook and evaluates consensus with empirical ML."""
    data_dict = applicant.model_dump()
    feat_df = engineer_features(pd.DataFrame([data_dict]))
    ml_prob = float(pipeline.predict_proba(feat_df)[0, 1])
    audit_res = symbolic_engine.evaluate_applicant(data_dict, feat_df.iloc[0].to_dict(), ml_prob)
    return audit_res

@app.get("/api/v1/fairness", tags=["Responsible AI"])
async def get_fairness_metrics():
    """Returns demographic parity, disparate impact ratios, and proxy variable audits."""
    return get_fairness_audit_data()

@app.get("/api/v1/benchmarks", tags=["Model Card"])
async def get_model_card_benchmarks():
    """Returns 7-model cross-validation and test benchmark leaderboard."""
    card_path = os.path.join(os.path.dirname(__file__), "model_card.json")
    if os.path.exists(card_path):
        import json
        with open(card_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "model_card.json not found"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
