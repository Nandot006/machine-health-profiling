# 🚀 Intelligent Predictive Maintenance and Machine Health Analysis Using Machine Learning
### 23CSE301 Machine Learning Capstone Project | Academic Year 2026–2027

---

## 📌 Project Overview
This repository hosts the end-to-end Machine Learning Capstone project developed for **23CSE301: Machine Learning Capstone**. The project explores real-world industrial intelligence across three core ML tracks: **Regression**, **Classification**, and **Clustering**.

For **Review 1**, the repository implements:
1. **Full Regression Track**: NASA Turbofan Engine Degradation (CMAPSS) prediction of Remaining Useful Life (RUL) across 10 supervised learning algorithms.
2. **Classification Track (Part A)**: Industrial Predictive Maintenance & Machine Health Analysis across 5 foundational classification algorithms.

---

## 👥 Team Responsibilities & Track Ownership

As mandated by Section 7.4 of the Capstone Guidelines, track ownership is partitioned by problem track:

| Member | Track Ownership | Core Responsibilities |
|---|---|---|
| **Member 1** | **Regression Track (EDA & Preprocessing)** | Dataset audit, distributions, correlation analysis, outlier assessment, and domain feature engineering for engine health degradation modeling. |
| **Member 2** | **Regression Track (Algorithms & Tuning)** | Implementation of all 10 regressors, 5-fold cross-validation, GridSearchCV tuning, residual & feature importance plots. |
| **Member 3** | **Classification Track (Part A Lead)** | AI4I sensor EDA, class imbalance handling, leak-proof preprocessing, training 5 Part-A classifiers, confusion matrices, and ROC curves. |

---

## 📁 Repository Directory Structure

The repository follows the official Capstone directory specification:

```text
ML_Capstone/
│
├── datasets/                               <- Processed datasets for regression & classification
│   ├── turbofan_data.csv                   <- NASA Turbofan Engine Degradation dataset (CMAPSS)
│   └── ai4i2020.csv                        <- AI4I 2020 Predictive Maintenance dataset (10,000 samples)
│
├── notebooks/                              <- Executed Jupyter Notebooks with all outputs visible
│   ├── 01_Regression_Turbofan.ipynb        <- Full Regression pipeline (10 algorithms, tuning, CV)
│   └── 02_Classification_AI4I.ipynb       <- Classification Part A (5 algorithms, ROC-AUC, tree plot)
│
├── results/                                <- Exported metric summaries
│   ├── regression_results.csv              <- Comparative metrics (R², RMSE, MAE) for 10 regressors
│   ├── classification_results.csv          <- Preliminary comparative metrics for 5 classifiers
│   └── classification_results.xlsx         <- Excel format of classification metrics
│
├── docs/                                   <- Viva and defense documentation
│   └── REVIEW_1_VIVA_PREP.md               <- Comprehensive Review 1 Viva Q&A master guide
│
├── requirements.txt                        <- Python environment dependencies
└── README.md                               <- Master capstone documentation
```

---

## 📊 Review 1 Results Summary

### 1. Regression Track: NASA Turbofan Engine Degradation (10 Models)
- **Problem**: Predicting Remaining Useful Life (RUL) in engine cycles from sensor telemetry and operating conditions.
- **Engineered Features**: sensor trends, operating condition indicators, cycle-based degradation proxies, and health index features.
- **Validation**: 80:20 split, zero data leakage (scalers fitted only on train split), 5-Fold Cross-Validation on champions.

| Rank | Model | Test R² | Test RMSE (cycles) | Test MAE (cycles) | Train R² | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **1** | **Gradient Boosting Regressor** | **0.9340** | **4.4365** | **2.8811** | 0.9821 | **Champion Regressor** (learning_rate=0.1, n_est=150) |
| **2** | **Random Forest Regressor** | **0.9201** | **4.8835** | **3.4748** | 0.9845 | **Runner-up Ensemble** (n_estimators=150, max_depth=12) |
| 3 | Decision Tree Regressor | 0.8564 | 6.5443 | 5.0120 | 0.9127 | max_depth=6 |
| 4 | Support Vector Regressor (SVR) | 0.8516 | 6.6543 | 4.6613 | 0.9030 | RBF kernel, C=50.0, eps=0.2 |
| 5 | Polynomial Regression (Deg 2) | 0.7873 | 7.9650 | 5.9294 | 0.8336 | Degree 2 interactions |
| 6 | K-Nearest Neighbors (KNN) | 0.7754 | 8.1861 | 5.9519 | 0.9972 | k=5, distance-weighted |
| 7 | Ridge Regression (L2) | 0.5906 | 11.0513 | 8.7971 | 0.6161 | alpha=10.0 (smooth shrinkage) |
| 8 | ElasticNet Regression | 0.5901 | 11.0587 | 8.8007 | 0.6151 | alpha=0.05, l1_ratio=0.5 |
| 9 | Lasso Regression (L1) | 0.5898 | 11.0621 | 8.8054 | 0.6163 | alpha=0.05 (sparse selection) |
| 10 | Linear Regression (Baseline) | 0.5868 | 11.1027 | 8.8623 | 0.6172 | Ordinary Least Squares |

> **5-Fold Cross-Validation**: Gradient Boosting achieved Mean CV $R^2 = 0.9193 \pm 0.0182$, and Random Forest achieved $0.9074 \pm 0.0195$, verifying high generalization robustness for the turbofan degradation task.

---

### 2. Classification Track Part A: AI4I Predictive Maintenance (5 Models)
- **Problem**: Predicting impending machine breakdowns (`Machine failure`) from multi-sensor telemetry.
- **Class Imbalance**: 9,661 Normal (96.61%) vs. 339 Failures (3.39%).
- **Leakage Prevention**: Excluded post-failure cause flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and IDs (`UDI`, `Product ID`).
- **Engineered Features**: $\Delta T$ (Thermal gradient), Mechanical Power ($P_{\text{kW}}$), Cumulative Strain ($\text{Wear} \times \text{Torque}$).

| Rank | Model | Accuracy | Recall (Failure) | Precision | F1 (Weighted) | ROC-AUC Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Support Vector Classifier (SVC)** | 0.9390 | **0.8676** | 0.3430 | **0.9514** | **0.9689** |
| **2** | **Decision Tree Classifier** | 0.9455 | **0.9118** | 0.3758 | **0.9561** | **0.9415** |
| 3 | Logistic Regression (Balanced) | 0.8580 | 0.8824 | 0.1786 | 0.8998 | 0.9394 |
| 4 | Gaussian Naive Bayes | 0.9585 | 0.5147 | 0.4118 | 0.9607 | 0.9349 |
| 5 | K-Nearest Neighbors (KNN) | **0.9755** | 0.3529 | **0.8276** | 0.9707 | 0.8834 |

---

## 🛠️ Environment Setup & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Virtual environment recommended (`venv` or `conda`)

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/ML_Capstone.git
cd ML_Capstone
```

### Step 2: Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### Step 3: Run the Notebooks
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```
Navigate to `notebooks/` and execute:
- `01_Regression_Turbofan.ipynb`
- `02_Classification_AI4I.ipynb`

Both notebooks run fully top-to-bottom without manual intervention.

---

## 📖 Evaluation & Viva Defense Preparation
Comprehensive documentation covering theoretical questions, loss formulas, data leakage arguments, and examiner traps can be viewed in:
👉 [`docs/REVIEW_1_VIVA_PREP.md`](docs/REVIEW_1_VIVA_PREP.md)

---
*Developed for 23CSE301 Machine Learning Capstone, Academic Year 2026–2027.*
