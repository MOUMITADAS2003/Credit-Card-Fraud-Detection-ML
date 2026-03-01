# Credit-Card-Fraud-Detection-ML
Credit Card Fraud Detection using Machine Learning with imbalance handling and performance-driven model selection.

# 💳 Credit Card Fraud Detection System

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Since fraud detection is a highly imbalanced classification problem, special care was taken to handle class imbalance and optimize recall performance.

The objective of this project is to build a reliable model that minimizes false negatives (i.e., missing fraudulent transactions), which is critical in financial systems.

---

## 📊 Dataset Information

- Source: Kaggle - Credit Card Fraud Detection Dataset  
- Total Transactions: 284,807  
- Fraud Cases: 492 (~0.17%)  
- Features:
  - V1–V28 (PCA transformed features)
  - Time
  - Amount
  - Class (Target Variable)
    - 0 → Normal Transaction
    - 1 → Fraud Transaction  

⚠️ The dataset is highly imbalanced, making accuracy an unreliable metric.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Joblib

---

## 🔎 Project Workflow

### 1️⃣ Data Exploration (EDA)
- Loaded and inspected dataset
- Checked missing values
- Analyzed class imbalance
- Visualized fraud vs non-fraud distribution
- Generated correlation heatmap

### 2️⃣ Data Preprocessing
- Train-Test Split (Stratified)
- Feature Scaling using StandardScaler
- Applied SMOTE on training data only (to prevent data leakage)

### 3️⃣ Model Training
Three models were trained and compared:
- Logistic Regression
- Random Forest
- XGBoost

### 4️⃣ Model Evaluation
Models were evaluated using:
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Since fraud detection prioritizes catching fraudulent transactions, **Recall for Class 1 (Fraud)** was used as the primary metric.

### 5️⃣ Final Model Selection
Logistic Regression achieved the highest recall (~91.8%) for fraud detection and was selected as the final model.

---

## 📈 Why Recall is Important in Fraud Detection?

In fraud detection systems:

- False Negative → Fraud not detected ❌ (Very dangerous)
- False Positive → Normal transaction flagged as fraud (Less critical)

Therefore, maximizing Recall ensures more fraudulent transactions are detected.

---

## 💾 Model Saving

The final trained model and scaler were saved using Joblib:

models/
│── best_model.pkl  
│── scaler.pkl  

---

## 📁 Project Structure

credit-card-fraud-detection/
│
├data
│   └── creditcard.csv
│
├notebooks
│   └── fraud_detection.ipynb
│
├models
│   ├── best_model.pkl
│   └── scaler.pkl
│
├requirements.txt
└README.md

---

## 🚀 How to Run This Project

1. Clone the repository  
2. Install dependencies:

pip install -r requirements.txt

3. Open the notebook:

notebooks/fraud_detection.ipynb

4. Run all cells

---

## 🎯 Key Learnings

- Handling highly imbalanced datasets
- Importance of Recall in risk-sensitive problems
- Avoiding data leakage during SMOTE
- Comparing multiple ML models systematically
- Saving trained models for deployment

---

## 👩‍💻 Author

Moumita Das  
Machine Learning Enthusiast  
