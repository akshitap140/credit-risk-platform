# VIVA VOCE & PROJECT DEFENSE MASTERGUIDE
## Course: Applications and Use Cases of Machine Learning (TE7483)
**Symbiosis Institute of Technology (SIT), Pune — AIML Department**  
**Faculty Evaluators:** Dr. Aniket Shahade, Dr. Shrikishna Kolhar, Dr. Sumanto Dutta  
**Goal:** Full Marks (3/3 for Viva Defense + 2/2 for Presentation)

---

## 🎯 1. 2-Minute Presentation Pitch (Structured & Time-Managed)

> *"Good morning respected faculty members. Our project is **Aegis: an AI Credit Risk & Financial Inclusion Platform** developed on 179,235 loans from the Bondora Peer-to-Peer lending ecosystem.*  
> 
> *The central problem in consumer lending today is a dual challenge: traditional scorecards exclude thin-file, credit-invisible borrowers (violating UN SDGs 1 and 10), while modern machine learning models act as un-auditable black boxes that violate the EU AI Act and Fair Lending regulations.*  
> 
> *To solve this, we formulated a zero-leakage, time-seasoned pipeline over 110,342 loans. We benchmarked seven algorithms, selecting an **XGBoost Classifier that achieves a 0.781 test ROC-AUC**.*  
> 
> *Our key innovation is a **Triple-Layer Explainable AI (XAI) Suite**:*  
> *1. **SHAP** for mathematically additive game-theoretic attributions.*  
> *2. **LIME** in the un-encoded continuous feature space to prevent impossible dummy collisions.*  
> *3. A **Neuro-Symbolic Expert Rulebook** that translates black-box gradient boosting into formal First-Order Logic syllogisms.*  
> 
> *Finally, unlike typical student prototypes that remain confined to a single script, we engineered a production-grade decoupled architecture: an **asynchronous FastAPI REST microservice** with interactive OpenAPI documentation, paired with a modern white fintech client. We are excited to present our live demonstration."*

---

## 🧠 2. Expected Viva Questions & Perfect Technical Answers

### Q1: "Why did you use the Bondora dataset instead of standard datasets like Kaggle German Credit?"
* **Answer:**  
  *"German Credit has only 1,000 synthetic rows and 20 features, which leads to immediate overfitting and unrealistic modern credit dynamics. Bondora is an authentic, multi-national European P2P lending dataset with 179,235 real loans and 112 features. It includes authentic loan seasoning, macroeconomic default cycles across 4 countries, and real-world cashflow attributes, allowing us to evaluate true algorithmic generalizability and fairness."*

---

### Q2: "What is right-censoring, and why did you use a 12-month seasoning window?"
* **Answer:**  
  *"In credit risk, a loan originated last week cannot be classified as 'safe' or 'good' simply because it has not yet missed a payment; its true risk has not matured. This is called right-censoring. If included, recent good loans artificially inflate model performance.  
  By implementing a 12-month seasoning window, we restricted our modeling population to the 110,342 loans that have had sufficient tenure to experience their true default outcome or full repayment, ensuring rigorous, unbiased ground-truth labels."*

---

### Q3: "How did you prevent data leakage across 112 columns?"
* **Answer:**  
  *"We performed a strict four-category column audit:*  
  *1. Dropped all 52 post-origination columns that occur after loan disbursement—such as `PrincipalBalance`, `CurrentDebtDaysPrimary`, `RecoveryStage`, and `WriteOffs`.*  
  *2. Dropped platform-derived leakage like Bondora's own `Rating` and `ExpectedLoss` so our model learns from raw borrower fundamentals rather than piggybacking on their internal model.*  
  *3. Dropped identifiers like `LoanId` and `UserName` to prevent memorization.*  
  *4. All imputation and scaling parameters were strictly fitted on the 80% training split inside scikit-learn Pipeline objects to prevent test-fold contamination."*

---

### Q4: "Why did you choose XGBoost over Deep Learning or Random Forest?"
* **Answer:**  
  *"In our 5-fold cross-validation benchmark across seven algorithms:*  
  *• XGBoost achieved the top CV ROC-AUC (0.782) and holdout test ROC-AUC (0.781).*  
  *• While Deep Learning struggles with tabular heterogeneous features and requires vast unregularized tuning, XGBoost utilizes second-order gradient approximations with built-in L1/L2 leaf penalties.*  
  *• Crucially, XGBoost supports exact TreeSHAP computation with $O(TLD^2)$ complexity, allowing us to generate real-time local game-theoretic attributions in under 15 milliseconds."*

---

### Q5: "Why report a drop from 0.781 to 0.715 ROC-AUC on the time-based future test? Doesn't that mean your model failed?"
* **Answer:**  
  *"No, respected faculty, that drop represents scientific honesty and is a strength of our evaluation. In random stratified splits, training and testing come from the same time period, masking macroeconomic regime shifts. When evaluated on future time periods, factors like interest rate hikes and inflation cause distribution drift. Reporting this 0.066 drop demonstrates that we tested temporal robustness rather than making over-optimistic production claims."*

---

### Q6: "Why do you need Neuro-Symbolic AI if you already have SHAP and LIME?"
* **Answer:**  
  *"SHAP and LIME provide post-hoc local correlations, but they cannot enforce legal constraints or statutory policy rules. For example, Our underwriting policies cap Debt-to-Income around 40% and require liquidity buffers as configurable policy parameters (not statutory Basel III limits); our rulebook encodes such thresholds as project-configured parameters for demonstration purposes.  
  Our Neuro-Symbolic layer integrates First-Order Logic syllogisms with the empirical gradient boosted output. If the ML model outputs a low risk score but the borrower violates statutory debt caps, our symbolic consensus engine flags the divergence, guaranteeing regulatory auditability."*

---

### Q7: "How does your system address Financial Inclusion (UN SDG 1 & 10)?"
* **Answer:**  
  *"Standard bureau algorithms penalize first-time credit applicants (`NewCreditCustomer == Yes`) simply due to a lack of past credit footprint. Aegis incorporates a 'Financial Inclusion Safe Harbor Flag' in our symbolic rulebook (shared thresholds with rule R7: FreeCash > €500 and P2I < 20% for thin-file borrowers). The flag grants a −20 point policy credit and tags the applicant as a Financial Inclusion candidate; the ML underwriting decision itself is not overridden."*

---

### Q8: "Why did you build both a FastAPI backend and a Web Client?"
* **Answer:**  
  *"Standard college submissions use a monolithic Streamlit script where model loading, computation, and rendering are coupled in one thread. In enterprise banking, this is an anti-pattern.  
  We decoupled the architecture into an asynchronous FastAPI microservice with type-safe Pydantic schemas and interactive OpenAPI/Swagger documentation (`/docs`), communicating via REST with our client. This enables horizontal scalability, cross-platform mobile integration, and sub-15ms inference latency."*

---

## 📋 3. Live Demonstration Cheat-Sheet for Full Marks

1. **Step 1: Open the Swagger UI at `http://localhost:8000/docs`**  
   *Say: "Here is our production FastAPI backend. All endpoints—predict, SHAP, LIME, rules, and fairness—are fully documented with Pydantic request models."*
2. **Step 2: Open the Web Platform at `http://localhost:8501`**  
   *Say: "Here is our client dashboard styled with a high-contrast white fintech aesthetic."*
3. **Step 3: Click 'Thin-File Inclusion' profile (APP-104)**  
   *Show how the default probability is evaluated, and point out the purple badge: '🌟 Qualified Financial Inclusion Candidate'.*
4. **Step 4: Click the 'Explain My Decision (XAI)' tab**  
   *Show Layer 1 (SHAP Additivity verified), Layer 2 (LIME original-space perturbation), and Layer 3 (First-Order Logic Syllogisms).*
5. **Step 5: Click 'Model Benchmarks & Specs' tab**  
   *Show the 7-model leaderboard and point to the honest reporting of in-time vs out-of-time temporal drift.*
