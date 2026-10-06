"""
neuro_symbolic.py - First-Order Logic & Symbolic Knowledge Base Engine
Implements the Neuro-Symbolic XAI layer combining Black-Box Machine Learning (XGBoost)
with First-Order Symbolic Expert Rules for auditability and explainability.

Important: all numeric thresholds in this rulebook are project-configured policy
parameters (illustrative consumer-lending underwriting heuristics). They are NOT
statutory limits and NOT Basel III requirements — Basel III (BCBS) regulates bank
capital and liquidity ratios, not consumer DTI, interest-rate or free-cash caps.
Calibrate them per jurisdiction before any production use.
"""

from typing import Dict, Any, List

from loan_utils import INCLUSION_FREE_CASH_MIN, INCLUSION_P2I_MAX


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion: None/missing/unparsable values fall back to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Decision bands mirror the ML recommendation ladder in loan_utils.predict_credit_risk
# (Approved < 0.45 | Manual Review 0.45-0.60 | Declined >= 0.60). The symbolic scorecard
# index (0-100 points) and the ML probability of default live on different scales, so
# consensus is decided only after both engines are mapped onto these shared bands —
# never by comparing a raw point total against a calibrated probability.
DECISION_BANDS = ((0.60, "Declined"), (0.45, "Manual Review"), (0.0, "Approved"))
_BAND_RANK = {"Approved": 0, "Manual Review": 1, "Declined": 2}


def decision_band(risk_value_01: float) -> str:
    """Maps a 0-1 risk value (ML PD or normalized symbolic index) to a shared decision band."""
    for cutoff, label in DECISION_BANDS:
        if risk_value_01 >= cutoff:
            return label
    return "Approved"

class SymbolicRule:
    def __init__(self, rule_id: str, name: str, condition_desc: str, impact: str, weight: float):
        self.rule_id = rule_id
        self.name = name
        self.condition_desc = condition_desc
        self.impact = impact  # 'NEGATIVE' (increases risk) or 'POSITIVE' (mitigates risk)
        self.weight = weight

class NeuroSymbolicEngine:
    def __init__(self):
        self.rules = [
            SymbolicRule("R1", "Excessive Debt Burden", "Debt-to-Income ratio > 40% or Total Liabilities > 35% of monthly income", "NEGATIVE", +25),
            SymbolicRule("R2", "High Interest Stress", "Annual interest rate exceeds 28.0%, compounding borrowing cost", "NEGATIVE", +20),
            SymbolicRule("R3", "Inadequate Liquidity Buffer", "Discretionary Free Cash after living costs is below 250 EUR", "NEGATIVE", +25),
            SymbolicRule("R4", "Long-Horizon Commitment", "Loan duration >= 48 months with principal exceeding 5,000 EUR", "NEGATIVE", +15),
            SymbolicRule("R5", "Payment-to-Income Strain", "Monthly loan repayment consumes more than 30% of total verified income", "NEGATIVE", +20),
            SymbolicRule("R6", "Proven Repayment History", "Borrower previously repaid >= 90% of past loans on time", "POSITIVE", -25),
            SymbolicRule("R7", "Financial Inclusion Positive Buffer", "New credit applicant maintains solid Free Cash (> 500 EUR) and low payment-to-income (< 20%)", "POSITIVE", -20),
            SymbolicRule("R8", "High Academic Attainment Protection", "Applicant possesses Higher Education degree and full-time employment", "POSITIVE", -15),
            SymbolicRule("R9", "Structural Jurisdiction Risk", "Origination in higher historical default region (Spain/ES) with high interest (> 25%)", "NEGATIVE", +15)
        ]

    def evaluate_applicant(self, applicant: Dict[str, Any], engineered: Dict[str, Any], ml_prob: float) -> Dict[str, Any]:
        """Evaluates symbolic expert rules against applicant features and checks consensus with ML."""
        income = _to_float(applicant.get("IncomeTotal"), 1.0)
        payment = _to_float(applicant.get("MonthlyPayment"), 0.0)
        liab_total = _to_float(applicant.get("LiabilitiesTotal"), 0.0)
        free_cash = _to_float(applicant.get("FreeCash"), income - payment - liab_total)
        interest = _to_float(applicant.get("Interest"), 0.0)
        duration = int(_to_float(applicant.get("LoanDuration"), 36))
        amount = _to_float(applicant.get("Amount"), 0.0)
        dti = _to_float(applicant.get("DebtToIncome"), liab_total / max(1.0, income))
        p_to_i = _to_float(engineered.get("PaymentToIncome"), payment / max(1.0, income))
        l_to_i = _to_float(engineered.get("LiabilityToIncome"), liab_total / max(1.0, income))
        prev_ratio = _to_float(engineered.get("PreviousRepaymentRatio"), 0.0)
        prev_count = int(_to_float(applicant.get("NoOfPreviousLoansBeforeLoan"), 0))
        new_cust = applicant.get("NewCreditCustomer", "No")
        education = applicant.get("Education", "Secondary")
        emp_status = applicant.get("EmploymentStatus", "Fully employed")
        country = applicant.get("Country", "EE")

        triggered_rules: List[Dict[str, Any]] = []
        symbolic_score = 50.0  # Base neutral baseline

        # Rule 1
        if dti > 0.40 or l_to_i > 0.35:
            triggered_rules.append({
                "rule_id": "R1",
                "name": "Excessive Debt Burden",
                "type": "NEGATIVE",
                "points": +25,
                "detail": f"DTI is {dti*100:.1f}% (threshold 40%) or Liability/Income is {l_to_i*100:.1f}% (threshold 35%).",
                "syllogism": "∀x (DebtBurden(x) > 40% → ElevatedDefaultRisk(x))"
            })
            symbolic_score += 25

        # Rule 2
        if interest > 28.0:
            triggered_rules.append({
                "rule_id": "R2",
                "name": "High Interest Cost Surcharge",
                "type": "NEGATIVE",
                "points": +20,
                "detail": f"Interest rate of {interest:.1f}% exceeds the policy threshold of 28.0%.",
                "syllogism": "∀x (InterestRate(x) > 28% → HighServiceBurden(x))"
            })
            symbolic_score += 20

        # Rule 3
        if free_cash < 250.0:
            triggered_rules.append({
                "rule_id": "R3",
                "name": "Inadequate Liquidity Buffer",
                "type": "NEGATIVE",
                "points": +25,
                "detail": f"Remaining discretionary cash {free_cash:.1f} EUR is below safety cushion of 250 EUR.",
                "syllogism": "∀x (FreeCash(x) < 250€ → VulnerableToShocks(x))"
            })
            symbolic_score += 25

        # Rule 4
        if duration >= 48 and amount > 5000:
            triggered_rules.append({
                "rule_id": "R4",
                "name": "Long-Horizon Principal Exposure",
                "type": "NEGATIVE",
                "points": +15,
                "detail": f"Loan duration {duration} months with high principal {amount:.0f} EUR increases macroeconomic sensitivity.",
                "syllogism": "∀x (Duration(x) ≥ 48m ∧ Amount(x) > 5000€ → LongTermExposure(x))"
            })
            symbolic_score += 15

        # Rule 5
        if p_to_i > 0.30:
            triggered_rules.append({
                "rule_id": "R5",
                "name": "Payment-to-Income Strain",
                "type": "NEGATIVE",
                "points": +20,
                "detail": f"Monthly installment absorbs {p_to_i*100:.1f}% of income (policy threshold is 30%).",
                "syllogism": "∀x (PaymentToIncome(x) > 30% → HighDebtServicingRatio(x))"
            })
            symbolic_score += 20

        # Rule 6
        if prev_count > 0 and prev_ratio >= 0.90:
            triggered_rules.append({
                "rule_id": "R6",
                "name": "Proven Repayment Reliability",
                "type": "POSITIVE",
                "points": -25,
                "detail": f"Borrower successfully repaid {prev_ratio*100:.1f}% across {prev_count} past credit facility/facilities.",
                "syllogism": "∀x (PreviousRepayments(x) ≥ 90% → ProvenCreditworthiness(x))"
            })
            symbolic_score -= 25

        # Rule 7 (Financial Inclusion Rule) — thresholds are the shared constants
        # INCLUSION_FREE_CASH_MIN / INCLUSION_P2I_MAX used by the inclusion flag in
        # loan_utils.predict_credit_risk(). Note: R7 grants a point credit and (via that
        # flag) an inclusion tag; it does not override the ML underwriting decision.
        if new_cust == "Yes" and free_cash > INCLUSION_FREE_CASH_MIN and p_to_i < INCLUSION_P2I_MAX:
            triggered_rules.append({
                "rule_id": "R7",
                "name": "Financial Inclusion Safe Harbors",
                "type": "POSITIVE",
                "points": -20,
                "detail": f"Unbanked/Thin-file applicant exhibits healthy surplus ({free_cash:.0f} EUR > {INCLUSION_FREE_CASH_MIN:.0f} EUR) and modest installment ({p_to_i*100:.1f}% of income < {INCLUSION_P2I_MAX*100:.0f}%).",
                "syllogism": f"∀x (ThinFile(x) ∧ FreeCash(x) > {INCLUSION_FREE_CASH_MIN:.0f}€ ∧ P2I(x) < {INCLUSION_P2I_MAX*100:.0f}% → ResponsibleInclusion(x))"
            })
            symbolic_score -= 20

        # Rule 8
        if education == "Higher" and emp_status in ["Fully employed", "Entrepreneur"]:
            triggered_rules.append({
                "rule_id": "R8",
                "name": "Human Capital & Employment Stability",
                "type": "POSITIVE",
                "points": -15,
                "detail": f"Higher educational degree paired with {emp_status.lower()} stability provides structural resilience.",
                "syllogism": "∀x (Education(x) = Higher ∧ Employed(x) → CareerResilience(x))"
            })
            symbolic_score -= 15

        # Rule 9
        if country == "ES" and interest > 25.0:
            triggered_rules.append({
                "rule_id": "R9",
                "name": "Macro-Jurisdiction Warning",
                "type": "NEGATIVE",
                "points": +15,
                "detail": "Historical Bondora P2P performance in Spanish cohort demonstrated elevated systemic default rates.",
                "syllogism": "∀x (Jurisdiction(x) = Spain ∧ HighYield(x) → MacroeconomicVulnerability(x))"
            })
            symbolic_score += 15

        symbolic_score = max(0.0, min(100.0, symbolic_score))
        # Normalized 0-1 index for reporting ONLY. This is a scorecard point total,
        # not a calibrated probability of default, so it is never compared numerically
        # against ml_prob below — both engines are first mapped to decision bands.
        symbolic_risk_index = symbolic_score / 100.0

        # Assess Consensus between Gradient Boosting ML and First-Order Logic
        # on the shared decision-band scale (the two raw scales are not comparable).
        ml_band = decision_band(ml_prob)
        symbolic_band = decision_band(symbolic_risk_index)
        ml_rank = _BAND_RANK[ml_band]
        symbolic_rank = _BAND_RANK[symbolic_band]

        if ml_band == symbolic_band:
            consensus_status = "Strong Consensus"
            consensus_desc = (
                f"The ML model and the symbolic rulebook both map this applicant to the "
                f"'{ml_band}' decision band — the statistical and rule-based layers agree on the outcome."
            )
            status_color = "#10B981"
        elif ml_rank > symbolic_rank:
            consensus_status = "ML Conservative Divergence"
            consensus_desc = (
                f"The ML model maps the applicant to the stricter '{ml_band}' band while the rulebook "
                f"reaches '{symbolic_band}' — the statistical model detects non-linear risk interactions "
                f"beyond the individual rule triggers."
            )
            status_color = "#F59E0B"
        else:
            consensus_status = "Symbolic Policy Constraint Triggered"
            consensus_desc = (
                f"The rulebook maps the applicant to the stricter '{symbolic_band}' band while the ML "
                f"model reaches '{ml_band}' — configured policy rules flag limits even when the "
                f"statistical score is lower."
            )
            status_color = "#EF4444"

        # Generate human-readable formal reasoning deduction
        deduction_lines = []
        positive_rules = [r for r in triggered_rules if r["type"] == "POSITIVE"]
        negative_rules = [r for r in triggered_rules if r["type"] == "NEGATIVE"]

        if negative_rules:
            deduction_lines.append(f"Risk Multipliers Identified ({len(negative_rules)}):")
            for r in negative_rules:
                deduction_lines.append(f"  • {r['name']}: {r['detail']}")
        else:
            deduction_lines.append("Risk Multipliers: None triggered (Compliant with configured policy bounds).")

        if positive_rules:
            deduction_lines.append(f"Risk Mitigants & Financial Inclusion Credits ({len(positive_rules)}):")
            for r in positive_rules:
                deduction_lines.append(f"  • {r['name']}: {r['detail']}")

        deduction_lines.append(f"Conclusion: Symbolic Scorecard index {symbolic_score:.0f}/100 supports {consensus_status.lower()}.")

        return {
            "symbolic_score": symbolic_score,
            "symbolic_risk_probability": symbolic_risk_index,
            "ml_decision_band": ml_band,
            "symbolic_decision_band": symbolic_band,
            "triggered_rules": triggered_rules,
            "negative_rules_count": len(negative_rules),
            "positive_rules_count": len(positive_rules),
            "consensus_status": consensus_status,
            "consensus_desc": consensus_desc,
            "status_color": status_color,
            "deduction_summary": "\n".join(deduction_lines)
        }
