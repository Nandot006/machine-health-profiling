# ⚙️ Classification Track: Intelligent Predictive Maintenance & Machine Health Analysis

### 23CSE301 Machine Learning Capstone | Review 1 Deliverable
- **Track Lead:** Geethika ([@Geethika0609](https://github.com/Geethika0609))
- **Email:** geethikar06@gmail.com
- **Assigned Dataset:** AI4I 2020 Predictive Maintenance Dataset (`datasets/ai4i2020.csv`)
- **Primary Notebook:** [`notebooks/02_Classification_AI4I.ipynb`](../notebooks/02_Classification_AI4I.ipynb)
- **Target Variable:** `Machine failure` (Binary: 0 = Healthy, 1 = Failure)

---

## 📌 1. Track Overview & Evaluation Scope

In industrial milling manufacturing, unanticipated machine tool failures induce catastrophic downtime and high financial losses. This track builds an intelligent classification pipeline to predict machine failures before they occur using continuous physical telemetry.

### Review 1 Rubric Breakdown (25 Marks)
1. **Section A: Exploratory Data Analysis & Sensor Audit (4 Marks)**
   - Dataset profile: 10,000 samples, 14 original attributes.
   - Severe class imbalance: **96.61% Healthy (0)** vs. **3.39% Failure (1)**.
   - Sensor distributions: Ambient Air Temperature, Process Temperature, Spindle Rotational Speed (rpm), Torque (Nm), and Tool Wear (min).
   - Correlation analysis & physical failure dynamics.

2. **Section B: Preprocessing & Domain Feature Engineering (3 Marks)**
   - **Data Leakage Safeguard**: Critical removal of post-failure symptom flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and arbitrary IDs (`UDI`, `Product ID`) before modeling.
   - **Stratified Train-Test Split (80:20)**: Preserves the exact 3.39% failure incidence in both splits (`random_state=42`).
   - **Leak-Proof Scaling**: `StandardScaler` fitted strictly on training data to prevent statistical leakage into test evaluation.
   - **Domain Feature Engineering**:
     - **$\Delta T$ (Temperature Difference)**: $T_{\text{process}} - T_{\text{air}}$ (monitors convective cooling efficiency and Heat Dissipation Failure).
     - **Mechanical Power ($P_{\text{kW}}$)**: $\frac{\tau \cdot \omega}{1000}$ (captures excessive power demand).
     - **Cumulative Mechanical Strain**: $\text{Tool Wear} \times \text{Torque}$ (isolates tool wear fatigue).

3. **Section D: Implementation & Benchmark of 5 Part-A Classifiers (3 Marks)**
   - Logistic Regression (interpretable odds ratios)
   - K-Nearest Neighbors (KNN with distance metric comparison)
   - Gaussian Naive Bayes (conditional independence benchmark)
   - Decision Tree Classifier (pruned with Gini impurity tree visualization)
   - Support Vector Classifier (SVC with RBF kernel and probability calibration)

---

## 📊 2. Comparative Performance Benchmark

All models were evaluated on the held-out test split (2,000 samples). Because of the severe class imbalance (3.39% failure rate), **ROC-AUC Score** and **Failure Recall (Class 1)** are prioritized over raw accuracy.

| Rank | Algorithm | Accuracy | Precision (Failure) | Recall (Failure) | F1 (Weighted) | F1 (Failure Class) | ROC-AUC Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Support Vector Classifier (SVC)** | 93.90% | 0.3430 | **86.76%** | 0.9514 | 0.4917 | **0.9689** |
| 🥈 | **Decision Tree Classifier** | 94.55% | 0.3758 | **91.18%** | 0.9561 | **0.5322** | **0.9415** |
| 🥉 | **Logistic Regression** | 85.80% | 0.1786 | **88.24%** | 0.8998 | 0.2970 | **0.9394** |
| 4 | **Naive Bayes (Gaussian)** | 95.85% | 0.4118 | 51.47% | 0.9607 | 0.4575 | 0.9349 |
| 5 | **K-Nearest Neighbors (KNN)** | **97.55%** | **0.8276** | 35.29% | **0.9707** | 0.4948 | 0.8834 |

---

## 🔍 3. Key Technical Takeaways for Viva Defense

1. **Why SVC & Decision Trees Dominated**:
   - Machine failures in AI4I are governed by physical boundary rules (e.g., $P_{\text{kW}} > 9.0$ or $\Delta T < 8.6\text{ K}$).
   - Decision trees naturally capture these orthogonal partition thresholds.
   - The RBF kernel in SVC maps non-linear sensor interactions into higher-dimensional separation margins, yielding the top ROC-AUC of **0.9689**.

2. **Why High Accuracy Alone is Deceptive**:
   - A naive model that predicts zero failures achieves 96.6% accuracy but catches 0% of breakdowns.
   - KNN achieves 97.55% accuracy but only catches 35.29% of failures (misses 44 breakdowns!).
   - In industrial maintenance, missed failures cause expensive machine breakdowns; hence **Recall** and **ROC-AUC** are the defining operational metrics.

3. **Domain Features Confirmed Decisive**:
   - In the Decision Tree visualization, the primary splits occur on `Cumulative_Strain` and `Power_kW`, verifying that the engineered domain features are the primary determinants of machine degradation.
