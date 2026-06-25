Fraud Detection — ML Pipeline

End-to-end machine learning pipeline to detect fraudulent transactions using Python.

DataSet 

File: fraud.csv
Target: is_fraud (0 = Legit, 1 = Fraud)
Class balance: ~90% Legit / ~10% Fraud

Tools & Libraries


Data: pandas, numpy
Visualization: matplotlib, seaborn
Models: scikit-learn, xgboost
Imbalance handling: imbalanced-learn (SMOTE)
Web app: streamlit

Pipeline Steps


Exploratory Data Analysis
Handle missing values — median for continuous, mode for categorical
Feature engineering — amount_per_distance, low_history_flag
Scale features with StandardScaler (fit on train only)
Apply SMOTE on training set to fix class imbalance
Train & compare 4 models via 5-Fold Cross-Validation (ROC-AUC)
Evaluate best model — Confusion Matrix, ROC Curve, Feature Importance

Models Compared


Logistic Regression
Random Forest
Gradient Boosting
XGBoost
