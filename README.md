# 📉 Customer Churn Prediction using Machine Learning:
An AI internship project that predicts whether a telecom customer is likely to
churn (cancel their subscription), using supervised machine learning models
trained on customer demographic, account, and service data.

## 📌 Project Overview:
Customer churn is when a subscriber stops using a company's service. For
subscription-based businesses like telecom providers, retaining an existing
customer is far cheaper than acquiring a new one — so predicting churn early
lets a business take action (offers, outreach, etc.) before a customer
actually leaves.

This project builds and evaluates several classification models to predict
churn, with a focus on handling class imbalance, tuning the best model, and
diagnosing/fixing overfitting.

## 📊 Dataset:
- **Name:** Telco Customer Churn Dataset
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Records:** 7,043 customers
- **Features:** 20 (after dropping `customerID`) — demographics, account
  information, subscribed services, and billing details
- **Target:** `Churn` (Yes / No) — 26.5% churn rate (imbalanced)

## 🛠️ Tools & Libraries:
- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing, model training, evaluation, GridSearchCV
- XGBoost — gradient boosting classifier
- imbalanced-learn — SMOTE and random undersampling
- Google Colab — development environment

## 🔍 What the Notebook Covers:
1. **Data cleaning & preprocessing** — removed unique identifiers, fixed
   blank values, label-encoded categorical columns
2. **Exploratory data analysis** — distribution plots, box plots,
   correlation heatmap, count plots
3. **Class imbalance handling** — SMOTE oversampling vs. random
   undersampling, compared head-to-head
4. **Model training & comparison** — Decision Tree, Random Forest, and
   XGBoost compared via 5-fold and Stratified K-Fold cross-validation
5. **Model evaluation** — confusion matrix, precision, recall, F1-score on
   held-out test data
6. **Hyperparameter tuning** — GridSearchCV used to tune the best model
   (Random Forest)
7. **Overfitting diagnosis & fix** — compared training vs. test accuracy,
   identified a 22.2% gap, and corrected it through regularization
   (reduced to 8.0%)

## 📈 Results (Final Model: Regularized Random Forest):
| Metric | Score |
|---|---|
| Test Accuracy | 77.5% |
| Churn Recall | 74% |
| Churn Precision | 56% |
| Churn F1-Score | 64% |
| Train/Test Overfitting Gap | 8.0% (down from 22.2%) |

**Key churn predictors:** contract type, tenure, and monthly charges were
the strongest signals for predicting churn.

## 📁 Repository Contents:
| File | Description |
|---|---|
| `Customer_Churn_Prediction_using_ML.ipynb` | Full notebook — data cleaning, EDA, modeling, tuning, overfitting fix |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset used for training and evaluation |
| `Customer_Churn_Prediction_Report.docx` | Detailed written project report |
| `Customer_Churn_Prediction.pptx` | Presentation summary (10 slides) |
| `README.md` | This file |

## 🚀 How to Run:
1. Open `Customer_Churn_Prediction_using_ML.ipynb` in
   [Google Colab](https://colab.research.google.com).
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` when prompted by the file
   upload cell (or mount Google Drive and update the file path).
3. Run all cells from top to bottom (**Runtime → Run all**).

## 🔮 Future Scope:
- Try additional models (LightGBM, CatBoost) for further accuracy gains
- Deploy the model as a web app for real-time churn predictions
- Add SHAP-based explainability to interpret individual predictions

## 👤 Author

**Deepali Das**
AI Internship Program — Naviotech Solution Pvt. Ltd.
