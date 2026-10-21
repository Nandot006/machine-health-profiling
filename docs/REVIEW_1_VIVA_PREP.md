# 🎓 ML Capstone Project: Review 1 Viva & Defense Master Guide
**Course:** 23CSE301 - Machine Learning Capstone | **Review 1 Weightage:** 25 Marks (Presentation: 1m, Viva: 5m)  
**Project Title:** Intelligent Predictive Maintenance and Machine Health Analysis Using Machine Learning  
**Tracks Evaluated:** Full Regression Track (Concrete Strength) + Classification Track Part A (AI4I Maintenance)

---

## Table of Contents
1. [Core Capstone Architecture & Workflow](#1-core-capstone-architecture--workflow)
2. [Regression Track Defense (Concrete Compressive Strength)](#2-regression-track-defense-concrete-compressive-strength)
   - [All 10 Algorithms: Theory, Formulas & Hyperparameters](#the-10-regression-algorithms)
   - [Domain Feature Engineering (Abram's Law)](#regression-feature-engineering)
   - [Evaluation Metrics: R², RMSE, MAE](#regression-metrics)
3. [Classification Track Part A Defense (AI4I Predictive Maintenance)](#3-classification-track-part-a-defense-ai4i-predictive-maintenance)
   - [All 5 Part-A Algorithms: Theory & Inner Workings](#the-5-part-a-classification-algorithms)
   - [The Class Imbalance Dilemma (96.61% vs 3.39%)](#the-class-imbalance-dilemma)
   - [Domain Feature Engineering (Thermodynamics & Mechanics)](#classification-feature-engineering)
   - [Evaluation Metrics: Accuracy vs Recall vs F1 vs ROC-AUC](#classification-metrics)
4. [High-Probability Viva Questions & Model Answers ("Examiner Traps")](#4-high-probability-viva-questions--model-answers)
5. [Presentation Strategy & Team Split Guide](#5-presentation-strategy--team-split-guide)

---

## 1. Core Capstone Architecture & Workflow

### The End-to-End ML Pipeline
Our implementation follows the strict industry standard pipeline:
$$\text{Data Ingestion} \longrightarrow \text{Audit \& EDA} \longrightarrow \text{Leak-Proof Preprocessing} \longrightarrow \text{Domain Feature Engineering} \longrightarrow \text{Model Training} \longrightarrow \text{Hyperparameter Tuning} \longrightarrow \text{Diagnostic Visualisation}$$

### Rigorous Data Leakage Avoidance Protocol
A primary inspection point in the rubric is **preventing data leakage**:
1. **Scalers & Encoders**: `StandardScaler` is fitted **exclusively on the training split** (`X_train`), and only `.transform()` is invoked on `X_test`.
   - *Why?* If `StandardScaler` is fitted on the full dataset, the test set's mean $\mu$ and standard deviation $\sigma$ leak into the training representations, yielding overly optimistic evaluation metrics.
2. **Stratified Splitting**: For classification, `train_test_split(..., stratify=y)` ensures both training and testing partitions retain the exact $3.39\%$ failure ratio.
3. **Leakage Columns Purged**: In the predictive maintenance dataset, failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are dropped from features.

---

## 2. Regression Track Defense (Concrete Compressive Strength)

### Dataset Overview
- **Samples**: 1,030 mix formulations; 9 attributes.
- **Target**: `Compressive_Strength` (MPa) — Range: $2.33$ to $82.60\text{ MPa}$, Mean: $35.82\text{ MPa}$.
- **Duplicates**: 25 duplicate rows were audited and pruned to yield 1,005 unique formulations.

### Regression Feature Engineering
We engineered physical civil-engineering interaction variables based on **Abram's Law** (1918):
1. **Water-to-Cement Ratio ($w/c$)**:
   $$\text{Water\_to\_Cement\_Ratio} = \frac{\text{Water}}{\text{Cement}}$$
   *Theoretical Justification*: Abram's law states compressive strength $\sigma = \frac{A}{B^{w/c}}$. Strength decreases exponentially as excess mix water increases capillary porosity.
2. **Total Cementitious Binder**:
   $$\text{Total\_Binder} = \text{Cement} + \text{Blast\_Furnace\_Slag} + \text{Fly\_Ash}$$
   *Theoretical Justification*: Secondary pozzolanic reactions from slag and fly ash react with lime ($\text{Ca(OH)}_2$) to synthesize supplemental Calcium Silicate Hydrate (C-S-H) gel.
3. **Water-to-Binder Ratio ($w/b$)**:
   $$\text{Water\_to\_Binder\_Ratio} = \frac{\text{Water}}{\text{Total\_Binder}}$$

---

### The 10 Regression Algorithms

| # | Algorithm | Core Mechanism / Loss Function | Key Hyperparameters & Notes |
|---|---|---|---|
| **1** | **Linear Regression** | Ordinary Least Squares (OLS): $\min_{\mathbf{w}} \sum (y_i - \mathbf{w}^T \mathbf{x}_i)^2$ | Baseline model; coefficients reveal directional feature impacts. |
| **2** | **Ridge Regression** | OLS + $L_2$ Regularization: $\min_{\mathbf{w}} \|y - X\mathbf{w}\|^2 + \alpha \|\mathbf{w}\|_2^2$ | $\alpha=10.0$; handles collinearity by shrinking weights smoothly towards zero. |
| **3** | **Lasso Regression** | OLS + $L_1$ Regularization: $\min_{\mathbf{w}} \|y - X\mathbf{w}\|^2 + \alpha \|\mathbf{w}\|_1$ | $\alpha=0.05$; induces feature sparsity by driving irrelevant weights strictly to zero. |
| **4** | **ElasticNet** | OLS + Combined $L_1$ and $L_2$: $\alpha \left( \rho \|\mathbf{w}\|_1 + \frac{1-\rho}{2} \|\mathbf{w}\|_2^2 \right)$ | $\alpha=0.05, \text{l1\_ratio}=0.5$; balances group selection with sparsity. |
| **5** | **Polynomial Regression** | Non-linear feature mapping: $\mathbf{x} \mapsto [\mathbf{x}, \mathbf{x}_i \mathbf{x}_j, \mathbf{x}^2]$ then OLS | Degree 2 captures interaction terms ($R^2 \approx 0.82$); Degree 3 causes extreme overfitting ($>400$ dimensions). |
| **6** | **Decision Tree Regressor** | Recursive binary partitioning minimizing MSE: $\sum (y - \hat{y}_{\text{region}})^2$ | `max_depth=6`; non-linear, non-parametric, shows feature importances. |
| **7** | **Random Forest Regressor** | Bagging (Bootstrap Aggregating) of decorrelated decision trees | `n_estimators=300`, `max_depth=15`; reduces variance without increasing bias. |
| **8** | **Gradient Boosting Regressor** | Sequential boosting: each new tree fits negative gradient of residual loss | `learning_rate=0.05`, `n_estimators=350`, `max_depth=4`; champion model ($R^2 > 0.93$). |
| **9** | **Support Vector Regressor (SVR)** | $\epsilon$-insensitive loss function with kernel trick: $L_\epsilon(y, f(x)) = \max(0, \|y - f(x)\| - \epsilon)$ | RBF kernel, $C=50.0, \epsilon=0.2$; requires standardized feature scaling. |
| **10** | **K-Nearest Neighbors Regressor** | Local non-parametric regression: $\hat{y} = \frac{\sum w_i y_i}{\sum w_i}$ | $k=5$, distance weighting ($w_i = \frac{1}{d_i}$); highly sensitive to feature scaling. |

---

### Regression Metrics
1. **$R^2$ (Coefficient of Determination)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
   - *Interpretation*: The proportion of variance in compressive strength explained by the mixture components. Values closer to $1.0$ indicate high explanatory fidelity.
2. **RMSE (Root Mean Squared Error)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   - *Interpretation*: Penalizes larger prediction errors quadratically; expressed in target units ($\text{MPa}$).
3. **MAE (Mean Absolute Error)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   - *Interpretation*: Average absolute magnitude of error in $\text{MPa}$; robust to extreme outliers.
4. **5-Fold Cross-Validation $R^2$**:
   - Evaluated across 5 disjoint validation folds to verify that test metrics do not represent a fortunate split.

---

## 3. Classification Track Part A Defense (AI4I Predictive Maintenance)

### Dataset Overview & The Class Imbalance Dilemma
- **Samples**: 10,000 industrial milling machine sensor records.
- **Target**: `Machine failure` (0 = Healthy, 1 = Breakdown).
- **Class Breakdown**:
  - Class 0: **9,661 samples (96.61%)**
  - Class 1: **339 samples (3.39%)**
- **The Imbalance Trap**: If an algorithm predicts "0" for every machine, its accuracy is **96.61%**, but **0% of machine breakdowns are prevented**.
- **Defense Strategy**: We use `stratify=y`, apply `class_weight='balanced'`, and evaluate primarily on **Recall, Precision, Weighted F1-Score, and ROC-AUC**.

---

### Classification Feature Engineering (Physics & Mechanics)
1. **Temperature Difference ($\Delta T$)**:
   $$\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$$
   *Physics Justification*: Directly governs Heat Dissipation Failure (HDF). When $\Delta T < 8.6\text{ K}$ during slow spindle operation, convective heat transfer collapses.
2. **Mechanical Spindle Power ($P_{\text{kW}}$)**:
   $$P_{\text{kW}} = \frac{\text{Torque [Nm]} \times \text{Rotational speed [rpm]} \times \frac{2\pi}{60}}{1000}$$
   *Physics Justification*: Power Failure (PWF) occurs when cutting power drops below $3.5\text{ kW}$ (tool stalling) or surges past $9.0\text{ kW}$ (motor overload).
3. **Cumulative Mechanical Strain**:
   $$\text{Cumulative\_Strain} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
   *Physics Justification*: Overstrain Failure (OSF) occurs when high contact resistance meets accumulated cutting-edge micro-fractures.

---

### The 5 Part-A Classification Algorithms

| # | Algorithm | Decision Boundary & Mechanics | Key Insights for Viva |
|---|---|---|---|
| **1** | **Logistic Regression** | Linear log-odds hyperplane: $P(y=1\|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$ | Baseline classifier; `class_weight='balanced'`; Odds ratios $\exp(w_j)$ quantify relative risk. |
| **2** | **K-Nearest Neighbors (KNN)** | Majority class vote among $k$ nearest Euclidean/Manhattan neighbors | Sensitive to local minority density; requires scaled features to prevent RPM from dominating. |
| **3** | **Gaussian Naive Bayes** | Bayes' Theorem: $P(y\|\mathbf{x}) \propto P(y) \prod P(x_i\|y)$ assuming conditional independence | Fast; assumes normal distribution; mildly violates independence due to torque-speed coupling ($P=\tau \cdot \omega$). |
| **4** | **Decision Tree Classifier** | Recursive greedy splits maximizing Information Gain / Gini Impurity reduction | `max_depth=5`; excellent for orthogonal threshold failures; full visual tree rendered. |
| **5** | **Support Vector Machine (SVC)** | Maximum-margin hyperplane with non-linear RBF Kernel | Scales features; maps data to high-dimensional Hilbert space to find optimal separating boundary. |

---

### Classification Metrics Explained
1. **Confusion Matrix Terminology**:
   - **True Positive (TP)**: Failing machine correctly flagged (saved factory downtime).
   - **False Negative (FN)**: Failing machine predicted healthy (**catastrophic failure occurs**).
   - **False Positive (FP)**: Healthy machine flagged as failing (false maintenance inspection).
   - **True Negative (TN)**: Healthy machine correctly identified.
2. **Recall (Sensitivity)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   - In predictive maintenance, **Recall is paramount** because an undetected failure (FN) halts an entire manufacturing assembly line.
3. **Precision**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   - Measures reliability of maintenance alerts.
4. **F1-Score**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - The harmonic mean penalizing models that sacrifice recall for precision or vice versa.
5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   - Measures class separability across all possible classification decision thresholds (from $0.0$ to $1.0$). Unaffected by class imbalance.

---

## 4. High-Probability Viva Questions & Model Answers

### Q1: "Why did you drop TWF, HDF, PWF, OSF, and RNF from the classification features?"
> **Model Answer:** "These columns represent specific post-failure diagnoses (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure). Including them in the feature matrix $X$ would constitute 100% data leakage because an operational predictive maintenance system does not know the cause of a breakdown before the breakdown occurs. If included, the model trivially learns $\text{Failure} = \text{TWF} \lor \text{HDF} \dots$ without learning any physical sensor relationships."

### Q2: "Why did you fit the StandardScaler only on X_train?"
> **Model Answer:** "Fitting a scaler computes the empirical mean $\mu$ and standard deviation $\sigma$. If fitted across the entire dataset prior to splitting, the statistics of the test set contaminate the training distribution. This violates the fundamental machine learning assumption that the test set represents unseen future data."

### Q3: "Why is accuracy an inadequate metric for the AI4I dataset?"
> **Model Answer:** "The dataset exhibits extreme class imbalance: 9,661 healthy samples (96.61%) vs 339 failures (3.39%). A naive classifier that predicts healthy for every single input achieves 96.61% accuracy while completely failing its industrial purpose. Hence, we prioritize Recall, F1-Score, and ROC-AUC."

### Q4: "What is the difference between $L_1$ (Lasso) and $L_2$ (Ridge) regularization?"
> **Model Answer:** "Ridge ($L_2$) adds the squared sum of weights ($\alpha \sum w_i^2$) to the loss function. It shrinks collinear weights proportionally towards zero but never sets them exactly to zero. Lasso ($L_1$) adds the absolute sum of weights ($\alpha \sum |w_i|$), creating sharp diamond-shaped constraint boundaries that drive less informative weights to exact zero, effectively performing embedded feature selection."

### Q5: "What is Abram's Law and how did you incorporate it?"
> **Model Answer:** "Abram's Law (1918) is a foundational principle of concrete materials science stating that compressive strength is inversely proportional to the water-to-cement ratio ($w/c$). We engineered the explicit ratio $\text{Water} / \text{Cement}$ and $\text{Water} / \text{Total\_Binder}$. In tree models and linear models, this engineered ratio provides a direct linearizable representation of paste porosity."

### Q6: "Why did Degree 3 Polynomial Regression perform worse than Degree 2?"
> **Model Answer:** "With 12 input features, generating degree 3 polynomial combinations produces over 450 interaction terms $\binom{n+d}{d}$. This triggers the curse of dimensionality, where the model begins fitting random sampling noise in the training split, leading to high variance and severe test set error degradation."

---

## 5. Presentation Strategy & Team Split Guide (3-Member Team)

According to Section 7.4 of the guidelines, **work must be divided by track, not by step** (e.g. not "one does EDA, one does models"):

| Team Member | Assigned Track / Scope | Presentation Responsibilities |
|---|---|---|
| **Member 1** | **Regression Track Lead (Parts A, B)** | Presents Concrete dataset audit, EDA distributions, correlation heatmap, deduplication, and domain feature engineering (Abram's law). |
| **Member 2** | **Regression Track Lead (Part C)** | Presents the 10 regression algorithms, model ranking table, 5-fold cross-validation, GridSearchCV hyperparameter tuning, and diagnostic plots (residual & predicted vs actual). |
| **Member 3 (Geethika - @Geethika0609)** | **Classification Track Lead (Parts A, B, D)** | Presents AI4I dataset audit, handling class imbalance (96.6% vs 3.4%), leak prevention (dropping TWF/HDF), 5 Part-A classification models, confusion matrices, ROC curves, and the Decision Tree diagram. |

---
*Good luck with Review 1! Keep this guide open and practice the model answers.*
