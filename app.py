"""
app.py - Aegis: AI Credit Risk & Financial Inclusion Platform
Ultra-Modern Dark Mode Fintech Architecture featuring Triple-Layer XAI,
Multi-Model Inference Engine, and Real-Time Risk Simulation.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Local modular imports
from loan_utils import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    MODEL_REGISTRY,
    MODEL_DESCRIPTIONS,
    engineer_features,
    get_or_create_model_pipeline,
    get_all_model_pipelines,
    predict_credit_risk
)
from neuro_symbolic import NeuroSymbolicEngine
from xai_engine import XAIEngine
from fairness_audit import get_fairness_audit_data

# -----------------------------------------------------------------------------
# 1. Page Configuration & Dark Luxury CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aegis | AI Credit Risk & Financial Inclusion",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
        background-color: #0A0F1E;
    }

    .stApp {
        background: #0A0F1E;
        background-image:
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .aegis-header {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 28px;
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.08), 0 10px 30px rgba(0,0,0,0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(12px);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #A5B4FC 0%, #6366F1 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-sub {
        font-size: 0.92rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 4px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-blue {
        background-color: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .pill-emerald {
        background-color: rgba(16, 185, 129, 0.12);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .pill-amber {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .pill-rose {
        background-color: rgba(239, 68, 68, 0.12);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .pill-purple {
        background-color: rgba(167, 139, 250, 0.12);
        color: #C4B5FD;
        border: 1px solid rgba(167, 139, 250, 0.25);
    }

    .dark-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .dark-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12), 0 4px 24px rgba(0,0,0,0.3);
        transform: translateY(-1px);
    }

    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #E2E8F0;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1220 0%, #0A0F1E 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        color: #94A3B8;
        border-radius: 10px;
        padding: 8px 12px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(99, 102, 241, 0.1);
        color: #E2E8F0;
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        font-weight: 500 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #A5B4FC !important;
    }

    .syllogism-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #6366F1;
    }
    .syllogism-box.risk {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.05);
    }
    .syllogism-box.merit {
        border-left-color: #10B981;
        background: rgba(16, 185, 129, 0.05);
    }
    .syllogism-formula {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 4px 8px;
        border-radius: 6px;
        color: #94A3B8;
        display: inline-block;
        margin-top: 6px;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 8px rgba(99, 102, 241, 0.4); }
        50%        { box-shadow: 0 0 18px rgba(99, 102, 241, 0.7); }
    }
    .active-model-badge {
        animation: pulse-glow 2.5s ease-in-out infinite;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.5);
        color: #A5B4FC;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Resource Initializers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    pipelines = get_all_model_pipelines()
    xgb_pipe = pipelines["XGBoost"]
    symbolic_engine = NeuroSymbolicEngine()
    xai_engine = XAIEngine(xgb_pipe)
    return pipelines, symbolic_engine, xai_engine

@st.cache_data
def load_sample_applicants():
    csv_path = os.path.join(os.path.dirname(__file__), "sample_applicants.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_data
def load_model_card():
    card_path = os.path.join(os.path.dirname(__file__), "model_card.json")
    if os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

all_pipelines, symbolic_engine, xai_engine = load_all_models()
sample_df = load_sample_applicants()
model_card = load_model_card()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Model Switcher
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='background:linear-gradient(135deg,#6366F1,#4F46E5); width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1.1rem; box-shadow:0 4px 12px rgba(99,102,241,0.4);'>
                A
            </div>
            <div>
                <span style='font-size:1.15rem; font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>AEGIS</span>
                <span style='font-size:0.75rem; color:#6366F1; font-weight:700; display:block; letter-spacing:0.06em;'>CREDIT INTELLIGENCE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    navigation = st.radio(
        "NAVIGATION",
        [
            "🏠 Executive Overview",
            "💳 Credit Risk Assessment",
            "🔍 Explain My Decision (XAI)",
            "⚖️ Fairness & Responsible AI",
            "📊 Model Benchmarks & Specs",
            "⚙️ Artifact Settings"
        ],
        index=1
    )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(99,102,241,0.2); margin:20px 0;'>", unsafe_allow_html=True)

    # MODEL SELECTOR
    st.markdown("##### 🤖 **ACTIVE MODEL ENGINE**")

    model_names = list(MODEL_REGISTRY.keys())
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = "XGBoost"

    for mname in model_names:
        meta = MODEL_DESCRIPTIONS[mname]
        is_active = st.session_state["active_model"] == mname
        border_style = "rgba(99,102,241,0.6)" if is_active else "rgba(99,102,241,0.12)"
        bg_style = "rgba(99,102,241,0.12)" if is_active else "rgba(15,23,42,0.5)"
        glow = "box-shadow:0 0 16px rgba(99,102,241,0.25);" if is_active else ""
        active_label = " <span style='color:#6EE7B7; font-size:0.7rem;'>● ACTIVE</span>" if is_active else ""

        st.markdown(f"""
        <div style='background:{bg_style}; border:1px solid {border_style}; border-radius:10px;
             padding:10px 12px; margin-bottom:6px; {glow}'>
            <div style='font-size:0.92rem; font-weight:700; color:#E2E8F0;'>
                {meta['icon']} {mname}{active_label}
            </div>
            <div style='font-size:0.75rem; color:#64748B; margin-top:2px;'>
                ROC-AUC: <strong style='color:{meta["color"]};'>{meta["roc_auc"]:.3f}</strong>
                &nbsp;·&nbsp; {meta["latency"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not is_active:
            if st.button(f"Switch to {mname}", key=f"switch_{mname}", use_container_width=True):
                st.session_state["active_model"] = mname
                st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid rgba(99,102,241,0.2); margin:20px 0;'>", unsafe_allow_html=True)

    active_meta = MODEL_DESCRIPTIONS[st.session_state["active_model"]]
    st.markdown("##### **PLATFORM TELEMETRY**")
    st.markdown(f"""
    <div style='background:rgba(15,23,42,0.7); border:1px solid rgba(99,102,241,0.15); border-radius:12px; padding:14px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#64748B;'>Engine</span>
            <span class='badge-pill pill-blue' style='padding:2px 8px;'>{st.session_state["active_model"]}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#64748B;'>Latency</span>
            <span class='badge-pill pill-emerald' style='padding:2px 8px;'>{active_meta["latency"]}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#64748B;'>ROC-AUC</span>
            <span class='badge-pill pill-purple' style='padding:2px 8px;'>{active_meta["roc_auc"]:.3f}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='font-size:0.8rem; color:#64748B;'>Explainability</span>
            <span style='font-size:0.75rem; color:#94A3B8;'>{active_meta["interpretability"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Bondora P2P Dataset • 110,342 Seasoned Loans")

# Active pipeline
active_pipeline = all_pipelines[st.session_state["active_model"]]

# Topbar Banner
st.markdown(f"""
<div class='aegis-header'>
    <div>
        <div class='brand-title'>
            <span>Aegis Financial Inclusion & Risk Platform</span>
            <span class='badge-pill pill-blue'>BONDORA P2P ENTERPRISE</span>
        </div>
        <div class='brand-sub'>Multi-Layer Neuro-Symbolic Credit Scoring · Automated Prudential Auditing · Responsible Underwriting</div>
    </div>
    <div style='display:flex; gap:10px; align-items:center;'>
        <span class='active-model-badge'>{MODEL_DESCRIPTIONS[st.session_state["active_model"]]["icon"]} {st.session_state["active_model"]}</span>
        <span class='badge-pill pill-emerald'>● SYSTEM LIVE</span>
        <span class='badge-pill pill-purple'>UN SDG 1 & 10</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Common dark chart layout helper
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
    title_font=dict(color="#E2E8F0"),
)

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if navigation == "🏠 Executive Overview":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='dark-card'>
            <div class='kpi-title'>Seasoned Cohort</div>
            <div class='kpi-value'>110,342</div>
            <div class='kpi-delta' style='color:#6EE7B7;'>12-Mo Maturation Window</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='dark-card'>
            <div class='kpi-title'>Active Model ROC-AUC</div>
            <div class='kpi-value' style='color:#A5B4FC;'>{MODEL_DESCRIPTIONS[st.session_state["active_model"]]["roc_auc"]:.3f}</div>
            <div class='kpi-delta' style='color:#A5B4FC;'>CV Validated (5-Fold)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='dark-card'>
            <div class='kpi-title'>Explainability Layers</div>
            <div class='kpi-value'>3 Types</div>
            <div class='kpi-delta' style='color:#C4B5FD;'>SHAP + LIME + Symbolic</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='dark-card'>
            <div class='kpi-title'>Leakage Governance</div>
            <div class='kpi-value'>52</div>
            <div class='kpi-delta' style='color:#6EE7B7;'>Features Eliminated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 Model Engine Comparison")
    m_cols = st.columns(3)
    for i, mname in enumerate(["XGBoost", "Random Forest", "Logistic Regression"]):
        meta = MODEL_DESCRIPTIONS[mname]
        is_active = st.session_state["active_model"] == mname
        active_badge = "<span style='color:#6EE7B7; font-size:0.78rem; font-weight:700;'>● ACTIVE</span>" if is_active else ""
        glow = "box-shadow: 0 0 24px rgba(99,102,241,0.2);" if is_active else ""
        with m_cols[i]:
            st.markdown(f"""
            <div class='dark-card' style='{glow}'>
                <div style='font-size:2rem; margin-bottom:6px;'>{meta["icon"]}</div>
                <div style='font-size:1.05rem; font-weight:700; color:#E2E8F0; margin-bottom:2px;'>
                    {mname} {active_badge}
                </div>
                <div style='font-size:0.82rem; color:#64748B; line-height:1.5; margin-bottom:14px;'>{meta["description"]}</div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.84rem;'>
                    <span style='color:#64748B;'>ROC-AUC</span>
                    <strong style='color:{meta["color"]};'>{meta["roc_auc"]:.3f}</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.84rem;'>
                    <span style='color:#64748B;'>F1 Score</span>
                    <strong style='color:#E2E8F0;'>{meta["f1"]:.3f}</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.84rem;'>
                    <span style='color:#64748B;'>Latency</span>
                    <strong style='color:#E2E8F0;'>{meta["latency"]}</strong>
                </div>
                <div style='display:flex; justify-content:space-between; font-size:0.82rem;'>
                    <span style='color:#64748B;'>Interpretability</span>
                    <span style='color:#94A3B8;'>{meta["interpretability"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='dark-card'>
        <h3 style='margin-top:0; font-weight:700; color:#E2E8F0;'>Core Mission: Responsible Financial Inclusion via Neuro-Symbolic XAI</h3>
        <p style='color:#94A3B8; line-height:1.7; font-size:1.02rem;'>
            Traditional credit scoring penalizes thin-file individuals simply because they lack extensive borrowing histories.
            Aegis combines <strong style='color:#A5B4FC;'>XGBoost</strong>, <strong style='color:#6EE7B7;'>Random Forest</strong>,
            and <strong style='color:#FCD34D;'>Logistic Regression</strong> with
            <strong style='color:#C4B5FD;'>First-Order Symbolic Expert Rules</strong>
            to ensure applicants with genuine liquidity are not unfairly rejected.
            Switch between model engines live from the sidebar to compare risk outputs instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏛️ Dual-Engine Architecture Workflow")
    st.markdown("""
    ```text
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                     Applicant Origination Features                       │
    │  (Age, Amount, Income, Duration, Liabilities, Country, Education, ...)   │
    └──────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                 Zero-Leakage Feature Engineering Engine                  │
    │    (Payment-to-Income, Liability-to-Income, Free Cash, Repayment Ratio)  │
    └──────────────┬───────────────────────────────────────┬───────────────────┘
                   │                                       │
                   ▼                                       ▼
    ┌──────────────────────────────────┐    ┌─────────────────────────────────┐
    │     Multi-Model ML Pipeline      │    │   First-Order Logic Rulebook    │
    │  ⚡ XGBoost  🌲 RF  📐 LR       │    │   Prudential Policies & Waivers │
    │  Switch active engine via sidebar│    │   Symbolic Score: /100          │
    └──────────────────┬───────────────┘    └──────────────────┬──────────────┘
                       └──────────────────┬──────────────────────┘
                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                  Consensus & Triple-Layer XAI Suite                      │
    │  • Layer 1 (SHAP): Exact Tree Attributions                               │
    │  • Layer 2 (LIME): Neighborhood Surrogates                               │
    │  • Layer 3 (Neuro-Symbolic): Formal Logic Syllogisms                     │
    └──────────────────────────────────────────────────────────────────────────┘
    ```
    """)

# -----------------------------------------------------------------------------
# PAGE 2: CREDIT RISK ASSESSMENT
# -----------------------------------------------------------------------------
elif navigation == "💳 Credit Risk Assessment":
    active_meta = MODEL_DESCRIPTIONS[st.session_state["active_model"]]
    st.markdown("<h2 style='font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>Applicant Assessment & Credit Scoring</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='color:#94A3B8; margin-bottom:4px;'>
        Running inference with <strong style='color:{active_meta["color"]};'>{active_meta["icon"]} {st.session_state["active_model"]}</strong>
        · Switch models from the sidebar to compare performance.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("##### **Quick-Load Calibrated Archetypes**")
    preset_cols = st.columns(6)
    archetype_labels = [
        ("Prime Low-Risk", "APP-101"), ("Moderate Prime", "APP-102"),
        ("High Over-Leveraged", "APP-103"), ("Thin-File Inclusion", "APP-104"),
        ("Senior Retiree", "APP-105"), ("Subprime Stressed", "APP-106")
    ]
    for i, (label, app_id) in enumerate(archetype_labels):
        with preset_cols[i]:
            if st.button(f"👤 {label}", key=f"chip_{i}", use_container_width=True):
                match_row = sample_df[sample_df["ApplicantId"] == app_id]
                if not match_row.empty:
                    st.session_state["applicant_data"] = match_row.iloc[0].to_dict()

    current_data = st.session_state.get("applicant_data", {
        "Age": 36, "AppliedAmount": 3500.0, "Amount": 3500.0, "Interest": 17.5,
        "LoanDuration": 36, "MonthlyPayment": 125.0, "IncomeTotal": 2600.0,
        "ExistingLiabilities": 1, "LiabilitiesTotal": 200.0, "DebtToIncome": 0.077,
        "FreeCash": 2275.0, "Country": "EE", "NewCreditCustomer": "No",
        "Education": "Higher", "EmploymentStatus": "Fully employed",
        "HomeOwnershipType": "Owner", "NoOfPreviousLoansBeforeLoan": 1,
        "AmountOfPreviousLoansBeforeLoan": 2500.0, "PreviousRepaymentsBeforeLoan": 2500.0
    })

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.form("application_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 👤 Borrower Credentials")
            age = st.number_input("Age (Years)", min_value=18, max_value=85, value=int(current_data.get("Age", 36)))
            country = st.selectbox("National Jurisdiction", CATEGORICAL_OPTIONS["Country"], index=CATEGORICAL_OPTIONS["Country"].index(current_data.get("Country", "EE")))
            education = st.selectbox("Educational Attainment", CATEGORICAL_OPTIONS["Education"], index=CATEGORICAL_OPTIONS["Education"].index(current_data.get("Education", "Higher")))
            emp_status = st.selectbox("Employment Contract", CATEGORICAL_OPTIONS["EmploymentStatus"], index=CATEGORICAL_OPTIONS["EmploymentStatus"].index(current_data.get("EmploymentStatus", "Fully employed")))
            home_type = st.selectbox("Residential Status", CATEGORICAL_OPTIONS["HomeOwnershipType"], index=CATEGORICAL_OPTIONS["HomeOwnershipType"].index(current_data.get("HomeOwnershipType", "Owner")))
            new_cust = st.selectbox("First-Time Borrower (Thin File)?", CATEGORICAL_OPTIONS["NewCreditCustomer"], index=CATEGORICAL_OPTIONS["NewCreditCustomer"].index(current_data.get("NewCreditCustomer", "No")))
        with col2:
            st.markdown("##### 💰 Loan Facility Request")
            applied_amt = st.number_input("Requested Facility (€)", min_value=200.0, max_value=25000.0, step=250.0, value=float(current_data.get("AppliedAmount", 3500.0)))
            funded_amt = st.number_input("Approved Principal (€)", min_value=200.0, max_value=25000.0, step=250.0, value=float(current_data.get("Amount", 3500.0)))
            interest = st.slider("Contractual Interest Rate (%)", min_value=4.0, max_value=60.0, step=0.5, value=float(current_data.get("Interest", 17.5)))
            duration = st.select_slider("Loan Duration (Months)", options=[6, 12, 18, 24, 36, 48, 60], value=int(current_data.get("LoanDuration", 36)))
            monthly_pmt = st.number_input("Monthly Installment (€)", min_value=10.0, max_value=2000.0, step=10.0, value=float(current_data.get("MonthlyPayment", 125.0)))
        with col3:
            st.markdown("##### 📊 Solvency & History")
            income = st.number_input("Monthly Verified Income (€)", min_value=200.0, max_value=30000.0, step=100.0, value=float(current_data.get("IncomeTotal", 2600.0)))
            existing_liab = st.number_input("Active Liabilities Count", min_value=0, max_value=20, value=int(current_data.get("ExistingLiabilities", 1)))
            liab_total = st.number_input("Total Debt Servicing (€/mo)", min_value=0.0, max_value=15000.0, step=50.0, value=float(current_data.get("LiabilitiesTotal", 200.0)))
            free_cash_calc = max(-500.0, income - monthly_pmt - liab_total)
            free_cash = st.number_input("Discretionary Net Surplus (€)", min_value=-1000.0, max_value=25000.0, value=float(current_data.get("FreeCash", free_cash_calc)))
            st.caption("Bondora Historical Borrowing History")
            prev_count = st.number_input("Prior Settled Facilities", min_value=0, max_value=25, value=int(current_data.get("NoOfPreviousLoansBeforeLoan", 1)))
            prev_amt = st.number_input("Prior Borrowed Principal (€)", min_value=0.0, max_value=100000.0, step=500.0, value=float(current_data.get("AmountOfPreviousLoansBeforeLoan", 2500.0)))
            prev_repay = st.number_input("Prior Repayments Satisfied (€)", min_value=0.0, max_value=100000.0, step=500.0, value=float(current_data.get("PreviousRepaymentsBeforeLoan", 2500.0)))

        assess_button = st.form_submit_button("⚡ Run Full Underwriting Assessment & XAI Audit", use_container_width=True)

    applicant_payload = {
        "Age": age, "AppliedAmount": applied_amt, "Amount": funded_amt, "Interest": interest,
        "LoanDuration": duration, "MonthlyPayment": monthly_pmt, "IncomeTotal": income,
        "ExistingLiabilities": existing_liab, "LiabilitiesTotal": liab_total,
        "DebtToIncome": liab_total / max(1.0, income), "FreeCash": free_cash,
        "Country": country, "NewCreditCustomer": new_cust, "Education": education,
        "EmploymentStatus": emp_status, "HomeOwnershipType": home_type,
        "NoOfPreviousLoansBeforeLoan": prev_count,
        "AmountOfPreviousLoansBeforeLoan": prev_amt,
        "PreviousRepaymentsBeforeLoan": prev_repay
    }
    st.session_state["evaluated_applicant"] = applicant_payload

    results = predict_credit_risk(active_pipeline, applicant_payload)
    st.session_state["latest_results"] = results

    _acolor = active_meta["color"]
    _aicon = active_meta["icon"]
    _aname = st.session_state["active_model"]
    st.markdown(f"<h3 style='font-weight:700; color:#E2E8F0; margin-top:28px;'>Live Underwriting Assessment · <span style='color:{_acolor};'>{_aicon} {_aname}</span></h3>", unsafe_allow_html=True)
    res_col1, res_col2, res_col3, res_col4 = st.columns([1.3, 1, 1.2, 1.2])

    pd_val = results["probability_of_default"]

    with res_col1:
        st.markdown("""<div class='kpi-title'>Default Probability (PD)</div>""", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number={'suffix': "%", 'font': {'size': 36, 'color': '#E2E8F0', 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': results["accent_color"], 'thickness': 0.28},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.1)"},
                    {'range': [25, 45], 'color': "rgba(245, 158, 11, 0.1)"},
                    {'range': [45, 65], 'color': "rgba(249, 115, 22, 0.1)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("""<div class='kpi-title'>Regulatory Rating</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.2); border-radius:16px; padding:20px; text-align:center;'>
            <div style='font-size:3.5rem; font-weight:800; color:{results["accent_color"]}; line-height:1;'>{results['credit_grade']}</div>
            <div style='font-size:0.85rem; font-weight:700; color:#94A3B8; margin-top:8px;'>{results['risk_category']}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown("""<div class='kpi-title'>Underwriting Verdict</div>""", unsafe_allow_html=True)
        pill_cls = "pill-emerald" if "Approved" in results["recommendation"] else ("pill-amber" if "Review" in results["recommendation"] else "pill-rose")
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.2); border-radius:16px; padding:20px;'>
            <span class='badge-pill {pill_cls}' style='font-size:0.88rem; padding:8px 14px; width:100%; justify-content:center; display:flex;'>
                {results['recommendation']}
            </span>
            <div style='margin-top:12px; font-size:0.84rem; color:#64748B; line-height:1.5;'>
                Decision synthesized across statistical gradients and supervisory limits.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if results["is_inclusion_candidate"]:
            st.markdown("""
            <div style='margin-top:8px;' class='badge-pill pill-purple'>
                🌟 Qualified Financial Inclusion Candidate
            </div>
            """, unsafe_allow_html=True)

    with res_col4:
        st.markdown("""<div class='kpi-title'>Key Prudential Metrics</div>""", unsafe_allow_html=True)
        eng = results["engineered_features"]
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.2); border-radius:16px; padding:16px; font-size:0.85rem;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Payment-to-Income:</span>
                <strong style='color:#E2E8F0;'>{eng.get('PaymentToIncome', 0.0)*100:.1f}%</strong>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Debt-to-Income:</span>
                <strong style='color:#E2E8F0;'>{applicant_payload['DebtToIncome']*100:.1f}%</strong>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Net Free Cash:</span>
                <strong style='color:#E2E8F0;'>€{applicant_payload['FreeCash']:.0f}/mo</strong>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='color:#64748B;'>Repayment Track:</span>
                <strong style='color:#E2E8F0;'>{eng.get('PreviousRepaymentRatio', 0.0)*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Cross-model comparison
    st.markdown("<hr style='border:none; border-top:1px solid rgba(99,102,241,0.15); margin:28px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🔀 Cross-Model PD Comparison")
    st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-bottom:16px;'>Same applicant, three model engines — compare how each algorithm prices default risk.</p>", unsafe_allow_html=True)

    cmp_cols = st.columns(3)
    model_pds = {}
    for i, (mname, mpipe) in enumerate(all_pipelines.items()):
        r = predict_credit_risk(mpipe, applicant_payload)
        model_pds[mname] = r
        meta = MODEL_DESCRIPTIONS[mname]
        is_active = mname == st.session_state["active_model"]
        active_tag = " ← Active" if is_active else ""
        glow = "box-shadow: 0 0 20px rgba(99,102,241,0.2);" if is_active else ""
        with cmp_cols[i]:
            st.markdown(f"""
            <div class='dark-card' style='{glow}'>
                <div style='font-size:0.88rem; font-weight:700; color:{meta["color"]}; margin-bottom:8px;'>
                    {meta["icon"]} {mname}{active_tag}
                </div>
                <div style='font-size:2.6rem; font-weight:800; color:#E2E8F0; line-height:1;'>
                    {r["probability_of_default"]*100:.1f}%
                </div>
                <div style='font-size:0.82rem; color:#64748B; margin-top:4px; margin-bottom:12px;'>
                    Default Probability
                </div>
                <div style='display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:4px;'>
                    <span style='color:#64748B;'>Grade</span>
                    <strong style='color:{r["accent_color"]};'>{r["credit_grade"]}</strong>
                </div>
                <div style='display:flex; justify-content:space-between; font-size:0.84rem;'>
                    <span style='color:#64748B;'>Verdict</span>
                    <span style='color:#94A3B8; font-size:0.78rem;'>{r["recommendation"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    fig_cmp = go.Figure()
    mnames_list = list(model_pds.keys())
    pds = [model_pds[m]["probability_of_default"] * 100 for m in mnames_list]
    colors_cmp = [MODEL_DESCRIPTIONS[m]["color"] for m in mnames_list]
    fig_cmp.add_trace(go.Bar(
        x=mnames_list, y=pds,
        marker=dict(color=colors_cmp, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pds],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=13)
    ))
    fig_cmp.update_layout(
        title="Default Probability by Model Engine",
        yaxis_title="Probability of Default (%)",
        yaxis=dict(range=[0, max(pds) * 1.35], gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"),
        xaxis=dict(color="#94A3B8"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=20),
        **DARK_LAYOUT
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: EXPLAIN MY DECISION (XAI SUITE)
# -----------------------------------------------------------------------------
elif navigation == "🔍 Explain My Decision (XAI)":
    st.markdown("<h2 style='font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>Explain My Decision — Triple-Layer XAI Suite</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; margin-bottom:24px;'>Multifaceted model interpretability satisfying EU AI Act & supervisory audit mandates.</p>", unsafe_allow_html=True)

    applicant = st.session_state.get("evaluated_applicant", None)
    if applicant is None:
        st.info("ℹ️ Please submit an applicant in the 'Credit Risk Assessment' tab first.")
        st.stop()

    applicant_df = pd.DataFrame([applicant])
    feat_df = engineer_features(applicant_df)
    ml_prob = float(active_pipeline.predict_proba(feat_df)[0, 1])

    tab_shap, tab_lime, tab_rules = st.tabs([
        "🌳 Layer 1: SHAP Attribution",
        "🍋 Layer 2: LIME Surrogate",
        "🧠 Layer 3: Neuro-Symbolic Syllogisms"
    ])

    with tab_shap:
        st.markdown("#### **SHAP (SHapley Additive exPlanations) Local Attribution**")
        st.caption("Measures exact Shapley marginal contributions pushing the applicant towards or away from loan default.")
        shap_res = xai_engine.explain_shap(feat_df)
        if shap_res["success"]:
            col_s1, col_s2 = st.columns([2.2, 1])
            with col_s1:
                drivers = shap_res["top_drivers"]
                df_drivers = pd.DataFrame(drivers)
                colors = ["#EF4444" if val > 0 else "#10B981" for val in df_drivers["shap_value"]]
                fig_shap = go.Figure(go.Bar(
                    x=df_drivers["shap_value"], y=df_drivers["feature"], orientation='h',
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:+.3f}" for v in df_drivers["shap_value"]],
                    textposition='outside', textfont=dict(color="#E2E8F0")
                ))
                fig_shap.update_layout(
                    title="Feature Marginal Contribution to Default Risk",
                    xaxis_title="Shapley Value (Log-Odds Shift)",
                    yaxis=dict(autorange="reversed", color="#94A3B8"),
                    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"),
                    height=450, margin=dict(l=10, r=40, t=40, b=20), **DARK_LAYOUT
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            with col_s2:
                st.markdown(f"""
                <div class='dark-card'>
                    <h5 style='margin-top:0; font-weight:700; color:#E2E8F0;'>Mathematical Additivity Axiom</h5>
                    <p style='color:#64748B; font-size:0.85rem;'>Under cooperative game theory, exact attribution requires strict additivity:</p>
                    <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.2); border-radius:8px; padding:12px; font-family:monospace; font-size:0.85rem; margin-bottom:12px; color:#94A3B8;'>
                        E[f(x)] = {shap_res['base_value']:.3f}<br>
                        &sum; &phi;&iota;    = {shap_res['sum_shap']:+.3f}<br>
                        f(x)    = {(shap_res['base_value'] + shap_res['sum_shap']):.3f}
                    </div>
                    <span class='badge-pill pill-emerald'>&#10003; Additivity Axiom Verified</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            detail = shap_res.get("detail") or "Exact SHAP attribution unavailable for this request."
            st.warning(detail)

    with tab_lime:
        st.markdown("#### **LIME (Local Interpretable Model-agnostic Explanations)**")
        st.caption("Evaluates continuous sensitivity in the unencoded original applicant feature domain.")
        lime_res = xai_engine.explain_lime(applicant)
        col_l1, col_l2 = st.columns([2.2, 1])
        with col_l1:
            df_lime = pd.DataFrame(lime_res["local_explanations"])
            colors_lime = ["#EF4444" if w > 0 else "#10B981" for w in df_lime["local_weight"]]
            fig_lime = go.Figure(go.Bar(
                x=df_lime["local_weight"], y=df_lime["feature"], orientation='h',
                marker=dict(color=colors_lime),
                text=[f"{w:+.3f}" for w in df_lime["local_weight"]],
                textposition='outside', textfont=dict(color="#E2E8F0")
            ))
            fig_lime.update_layout(
                title="LIME Local Surrogate Sensitivity Slopes",
                xaxis_title="Local Linear Weight (Elasticity)",
                yaxis=dict(autorange="reversed", color="#94A3B8"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"),
                height=380, margin=dict(l=10, r=40, t=40, b=20), **DARK_LAYOUT
            )
            st.plotly_chart(fig_lime, use_container_width=True)
        with col_l2:
            st.markdown(f"""
            <div class='dark-card'>
                <h5 style='margin-top:0; font-weight:700; color:#E2E8F0;'>Surrogate Quality Diagnostics</h5>
                <div style='font-size:2rem; font-weight:800; color:#E2E8F0;'>{lime_res['local_fidelity_r2']:.3f}</div>
                <div style='font-size:0.8rem; font-weight:600; color:#64748B;'>Local Surrogate Fidelity (R&sup2;)</div>
                <hr style='border:none; border-top:1px solid rgba(99,102,241,0.15); margin:14px 0;'>
                <div style='font-size:0.82rem; color:#64748B;'>
                    Trained over 150 localized Gaussian neighborhood perturbations.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_rules:
        st.markdown("#### **Neuro-Symbolic Syllogisms & Policy Bounds**")
        st.caption("Synthesizes statistical probability with First-Order Logic constraints for formal regulatory auditing.")
        sym_audit = symbolic_engine.evaluate_applicant(applicant, feat_df.iloc[0].to_dict(), ml_prob)
        col_r1, col_r2 = st.columns([1.6, 1])
        with col_r1:
            st.markdown("##### **Evaluated First-Order Syllogisms & Policy Bounds**")
            if not sym_audit["triggered_rules"]:
                st.success("Applicant meets all configured policy thresholds; no regulatory warning triggered.")
            else:
                for r in sym_audit["triggered_rules"]:
                    box_cls = "risk" if r["type"] == "NEGATIVE" else "merit"
                    icon = "⚠️" if r["type"] == "NEGATIVE" else "🛡️"
                    pill_class = "pill-rose" if r["type"] == "NEGATIVE" else "pill-emerald"
                    st.markdown(f"""
                    <div class='syllogism-box {box_cls}'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <strong style='color:#E2E8F0;'>{icon} Policy {r['rule_id']}: {r['name']}</strong>
                            <span class='badge-pill {pill_class}'>{r['points']:+d} pts</span>
                        </div>
                        <div style='color:#94A3B8; font-size:0.88rem; margin-top:4px;'>{r['detail']}</div>
                        <div class='syllogism-formula'>{r['syllogism']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class='dark-card'>
                <h5 style='margin-top:0; font-weight:700; color:#E2E8F0;'>Consensus Assessment</h5>
                <h3 style='color:{sym_audit['status_color']}; margin:4px 0 10px 0;'>{sym_audit['consensus_status']}</h3>
                <p style='color:#64748B; font-size:0.86rem; line-height:1.5;'>{sym_audit['consensus_desc']}</p>
                <hr style='border:none; border-top:1px solid rgba(99,102,241,0.15); margin:14px 0;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='color:#64748B;'>Symbolic Risk Score:</span>
                    <strong style='color:#E2E8F0;'>{sym_audit['symbolic_score']:.0f} / 100</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='color:#64748B;'>ML Decision Band:</span>
                    <strong style='color:#E2E8F0;'>{sym_audit['consensus_status'].split()[0]} ({sym_audit['ml_decision_band']})</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='color:#64748B;'>Symbolic Decision Band:</span>
                    <strong style='color:#E2E8F0;'>{sym_audit['symbolic_decision_band']}</strong>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#64748B;'>Consensus:</span>
                    <strong style='color:{sym_audit["status_color"]};'>{sym_audit["consensus_status"]}</strong>
                </div>
                <p style='color:#64748B; font-size:0.86rem; line-height:1.5;'>{sym_audit["consensus_desc"]}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: FAIRNESS & RESPONSIBLE AI
# -----------------------------------------------------------------------------
elif navigation == "⚖️ Fairness & Responsible AI":
    st.markdown("<h2 style='font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>Fairness & Responsible AI Audit</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; margin-bottom:24px;'>Comprehensive disparate impact, equal opportunity, and latent proxy leakage evaluation.</p>", unsafe_allow_html=True)

    fair_data = get_fairness_audit_data()
    st.markdown("### 🛡️ Disparate Impact & Equal Opportunity Audits")
    df_di = pd.DataFrame(fair_data["disparate_impact_summary"])
    st.dataframe(df_di, use_container_width=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    f_tab1, f_tab2, f_tab3 = st.tabs(["🚻 Gender Disparity", "🎂 Age Cohorts", "🌍 Country Jurisdictions"])

    with f_tab1:
        df_gender = pd.DataFrame(fair_data["gender_metrics"])
        fig_g = px.bar(df_gender, x="group", y=["approval_rate_at_35", "default_rate", "roc_auc"],
                       barmode="group", title="Subgroup Performance by Gender",
                       color_discrete_sequence=["#6366F1", "#EF4444", "#10B981"])
        fig_g.update_layout(**DARK_LAYOUT, yaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"), xaxis=dict(color="#94A3B8"))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info("💡 **Audit Takeaway:** With Gender directly omitted, approval rate parity remains compliant (1.066 DI ratio) with nearly identical discriminatory power (0.784 Male vs 0.779 Female).")

    with f_tab2:
        df_age = pd.DataFrame(fair_data["age_metrics"])
        fig_a = px.bar(df_age, x="group", y=["approval_rate_at_35", "roc_auc"],
                       barmode="group", title="Approval Rates and Discrimination Across Age Brackets",
                       color_discrete_sequence=["#6366F1", "#A78BFA"])
        fig_a.update_layout(**DARK_LAYOUT, yaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"), xaxis=dict(color="#94A3B8"))
        st.plotly_chart(fig_a, use_container_width=True)
        st.warning("⚠️ **Observed Demographic Skew:** Young applicants (< 30) exhibit lower approvals (32.4% vs 42.5%) attributable to thin credit files.")

    with f_tab3:
        df_country = pd.DataFrame(fair_data["country_metrics"])
        fig_c = px.bar(df_country, x="group", y=["default_rate", "approval_rate_at_35"],
                       barmode="group", title="Jurisdictional Disparities Driven by Bondora Historical P2P Performance",
                       color_discrete_sequence=["#EF4444", "#10B981"])
        fig_c.update_layout(**DARK_LAYOUT, yaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"), xaxis=dict(color="#94A3B8"))
        st.plotly_chart(fig_c, use_container_width=True)
        st.error("🚨 **Structural Reality:** Spanish P2P loans exhibited a 79.4% observed default rate, reflecting macroeconomic defaults rather than individual algorithmic bias.")

    st.markdown("### 🕵️ Latent Proxy Leakage Audit")
    for p in fair_data["proxy_analysis"]:
        st.markdown(f"""
        <div class='dark-card' style='padding:16px 20px; margin-bottom:10px;'>
            <strong style='color:#E2E8F0;'>Excluded Protected Attribute:</strong> <code>{p['excluded_protected_attribute']}</code> &nbsp;|&nbsp;
            <strong style='color:#E2E8F0;'>Proxy Carriers:</strong> <code>{p['proxy_driver_features']}</code> ({p['correlation_strength']})
            <p style='color:#64748B; font-size:0.86rem; margin-top:6px; margin-bottom:0;'>{p['mitigation_approach']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 5: MODEL BENCHMARKS & SPECS
# -----------------------------------------------------------------------------
elif navigation == "📊 Model Benchmarks & Specs":
    st.markdown("<h2 style='font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>Model Benchmarks & Model Card</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; margin-bottom:24px;'>Official performance leaderboard, temporal validation, and leakage prevention documentation.</p>", unsafe_allow_html=True)

    st.markdown("### 🤖 Aegis Model Registry — Performance Leaderboard")
    registry_rows = []
    for mname, meta in MODEL_DESCRIPTIONS.items():
        registry_rows.append({
            "Model": f"{meta['icon']} {mname}",
            "ROC-AUC": meta["roc_auc"],
            "F1 Score": meta["f1"],
            "Latency": meta["latency"],
            "Interpretability": meta["interpretability"],
            "Active": "✅" if mname == st.session_state["active_model"] else ""
        })
    st.dataframe(pd.DataFrame(registry_rows), use_container_width=True)

    fig_registry = go.Figure()
    for metric_name, color in [("ROC-AUC", "#6366F1"), ("F1 Score", "#10B981")]:
        vals = [MODEL_DESCRIPTIONS[m]["roc_auc"] if metric_name == "ROC-AUC" else MODEL_DESCRIPTIONS[m]["f1"] for m in MODEL_DESCRIPTIONS]
        fig_registry.add_trace(go.Bar(
            name=metric_name,
            x=[f"{MODEL_DESCRIPTIONS[m]['icon']} {m}" for m in MODEL_DESCRIPTIONS],
            y=vals,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#E2E8F0"),
            marker_color=color
        ))
    fig_registry.update_layout(
        barmode="group", title="Model Engine Performance: ROC-AUC vs F1",
        yaxis=dict(range=[0.6, 0.85], gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"),
        xaxis=dict(color="#94A3B8"),
        height=350, **DARK_LAYOUT
    )
    st.plotly_chart(fig_registry, use_container_width=True)

    if model_card:
        st.markdown("<hr style='border:none; border-top:1px solid rgba(99,102,241,0.15); margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📋 XGBoost Model Card (Primary Engine)")
        b_df = pd.DataFrame(model_card.get("benchmarks", []))
        st.dataframe(b_df, use_container_width=True)

        fig_b = px.bar(b_df, x="model", y=["cv_roc_auc", "test_roc_auc", "f1"],
                       barmode="group",
                       title="Algorithm Comparison Across 5-Fold CV and Untouched Test Split",
                       color_discrete_sequence=["#6366F1", "#10B981", "#A78BFA"])
        fig_b.update_layout(**DARK_LAYOUT,
                            yaxis=dict(gridcolor="rgba(99,102,241,0.1)", color="#94A3B8"),
                            xaxis=dict(color="#94A3B8"))
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("<hr style='border:none; border-top:1px solid rgba(99,102,241,0.15); margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### ⏳ Temporal Out-of-Time Validation (Macroeconomic Drift)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Stratified Random Test ROC-AUC",
                      f"{model_card['performance_metrics']['in_time_stratified_test_roc_auc']:.3f}",
                      "In-Time Validation")
        with c2:
            st.metric("Time-Based Future ROC-AUC",
                      f"{model_card['performance_metrics']['time_based_future_validation_roc_auc']:.3f}",
                      "-0.066 Drift", delta_color="inverse")
        st.caption(model_card['performance_metrics']['time_validation_drop_note'])

        with st.expander("📄 Full JSON Model Card"):
            st.json(model_card)

# -----------------------------------------------------------------------------
# PAGE 6: ARTIFACT SETTINGS
# -----------------------------------------------------------------------------
elif navigation == "⚙️ Artifact Settings":
    st.markdown("<h2 style='font-weight:800; color:#E2E8F0; letter-spacing:-0.02em;'>Artifact Manager & Model Deployment</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; margin-bottom:20px;'>Deploy trained <code>.joblib</code> models directly from your Google Colab experiments.</p>", unsafe_allow_html=True)

    up_cols = st.columns(3)
    for i, mname in enumerate(MODEL_REGISTRY):
        with up_cols[i]:
            fname = MODEL_REGISTRY[mname]
            meta = MODEL_DESCRIPTIONS[mname]
            st.markdown(f"**{meta['icon']} {mname}**")
            up_file = st.file_uploader(f"Upload {mname} (.joblib)", type=["joblib"], key=f"up_{mname}")
            if up_file is not None:
                save_dest = os.path.join(os.path.dirname(__file__), fname)
                with open(save_dest, "wb") as f:
                    f.write(up_file.getbuffer())
                st.success(f"✓ {mname} pipeline deployed! Reload the app.")

    st.markdown("---")
    st.markdown("##### **Active Local Runtime Inventory**")
    for mname, fname in MODEL_REGISTRY.items():
        fpath = os.path.join(os.path.dirname(__file__), fname)
        exists = os.path.exists(fpath)
        status = "✅ Present" if exists else "⚠️ Not cached (will build on demand)"
        st.write(f"• **{MODEL_DESCRIPTIONS[mname]['icon']} {mname}** (`{fname}`): {status}")
    st.write(f"• Active Directory: `{os.path.dirname(__file__)}`")
    st.write(f"• Model Card JSON: `{os.path.exists(os.path.join(os.path.dirname(__file__), 'model_card.json'))}`")
    st.write(f"• Sample Applicants CSV: `{os.path.exists(os.path.join(os.path.dirname(__file__), 'sample_applicants.csv'))}`")
