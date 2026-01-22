# =========================================
# IDENTITAS
# =========================================
# Nama  : Cindy Alya Putri
# NIM   : 22533644
# Kelas : TI 7D
# Dataset : Kaggle - Alzheimer's Disease Dataset
# =========================================

# =========================================
# IMPORT LIBRARY
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier

# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv("alzheimers_disease_data.csv")

print(df.head())
print(df.info())

# =========================================
# PREPROCESSING
# =========================================
# Drop kolom yang tidak relevan
df = df.drop(columns=["PatientID", "DoctorInCharge"])

# Cek missing value
print(df.isnull().sum())

# =========================================
# FEATURE & TARGET
# =========================================
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# =========================================
# SPLIT DATA
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# FEATURE SCALING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# TRAINING MODEL (XGBOOST)
# =========================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================================
# PREDIKSI DATA UJI
# =========================================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# =========================================
# EVALUASI MODEL
# =========================================
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================================
# SIMPAN MODEL & HASIL EVALUASI
# =========================================
joblib.dump(model, "xgboost_alzheimer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

evaluation_results = {
    "accuracy": accuracy,
    "roc_auc": roc_auc,
    "confusion_matrix": cm,
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}

joblib.dump(evaluation_results, "evaluation_results.pkl")

# =========================================
# PREDIKSI DATA BARU
# =========================================
new_data = pd.DataFrame([{
    "Age": 75,
    "Gender": 1,
    "Ethnicity": 0,
    "EducationLevel": 2,
    "BMI": 23.5,
    "Smoking": 0,
    "AlcoholConsumption": 8.5,
    "PhysicalActivity": 5.2,
    "DietQuality": 1.3,
    "SleepQuality": 7.8,
    "FamilyHistoryAlzheimers": 1,
    "CardiovascularDisease": 0,
    "Diabetes": 1,
    "Depression": 0,
    "HeadInjury": 0,
    "Hypertension": 1,
    "SystolicBP": 145,
    "DiastolicBP": 85,
    "CholesterolTotal": 240,
    "CholesterolLDL": 160,
    "CholesterolHDL": 45,
    "CholesterolTriglycerides": 180,
    "MMSE": 18,
    "FunctionalAssessment": 5.5,
    "MemoryComplaints": 1,
    "BehavioralProblems": 1,
    "ADL": 3.2,
    "Confusion": 1,
    "Disorientation": 1,
    "PersonalityChanges": 1,
    "DifficultyCompletingTasks": 1,
    "Forgetfulness": 1
}])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)[0]
prediction_proba = model.predict_proba(new_data_scaled)[0][1]

print("Prediksi Diagnosis:", "Alzheimer" if prediction == 1 else "Non-Alzheimer")
print("Probabilitas Alzheimer:", round(prediction_proba, 3))
