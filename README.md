# 💳 Credit Card Fraud Detection using Ensemble Machine Learning Models

This project demonstrates the use of various **machine learning models** to detect fraudulent credit card transactions. The models include **Random Forest**, **LightGBM**, **Gradient Boosting**, **XGBoost**, **Logistic Regression**, and a **hybrid ensemble model** optimized using `scipy.optimize`.

---

## 📌 Project Objectives

- Handle class imbalance in fraud detection tasks through aggressive undersampling.
- Train and evaluate multiple classification models on credit card data.
- Optimize and combine predictions from multiple models (RF, LGBM, Logistic Regression) into a **hybrid ensemble**.
- Evaluate all models using accuracy, precision, recall, and F1-score.

---

## 🧠 Models Implemented

| Model                  | Notes                                  |
|------------------------|----------------------------------------|
| Random Forest          | Simple with few estimators and low depth |
| LightGBM               | Tuned with regularization & subsampling |
| Gradient Boosting      | Basic gradient boosting with low depth |
| XGBoost                | Tuned with scale_pos_weight, depth     |
| Logistic Regression    | Used as baseline (called "PUMA" style) |
| **Hybrid (RF + LGBM + PUMA)** | Weighted ensemble with optimized weights |

---

## 📊 Dataset

The project uses the `creditcard.csv` dataset containing transaction records and fraud labels.

- **Class 1**: Fraudulent transaction  
- **Class 0**: Legitimate transaction  

Features are anonymized (e.g., V1, V2, ..., V28) along with `Time`, `Amount`, and the `Class` label.

---

## ✨ Features

### ✅ Data Processing
- NaN handling
- Undersampling (Class 1 retained, Class 0 downsampled)
- Feature selection (limited to 3–5 features to simulate low-resource scenarios)
- Train-test split using stratification

### ⚙️ Models & Evaluation
- Training and prediction using 5 ML models
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Manual hybrid ensemble model using weighted averaging of probability outputs
- Optimization of ensemble weights using `scipy.optimize.minimize`

---

## 🚀 How to Run

### 1. Clone this Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-ensemble.git
cd credit-card-fraud-ensemble
