import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("xgboost_alzheimer_model.pkl")
scaler = joblib.load("scaler.pkl")
eval_results = joblib.load("evaluation_results.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sistem Deteksi Alzheimer",
    layout="centered"
)

st.title("🧠 Sistem Deteksi Alzheimer Berbasis Machine Learning")
st.write(
    "Aplikasi ini menggunakan algoritma **XGBoost Classifier** "
    "untuk membantu mendeteksi risiko penyakit Alzheimer pada lansia."
)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📈 Evaluasi Model", "🧪 Prediksi Pasien Baru"])

# ==========================================================
# TAB 1 : EVALUASI MODEL
# ==========================================================
with tab1:
    st.subheader("📊 Performa Model")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{eval_results['accuracy']:.3f}")
    col2.metric("ROC-AUC", f"{eval_results['roc_auc']:.3f}")

    st.markdown("### Confusion Matrix")
    cm = eval_results["confusion_matrix"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.markdown("### Classification Report")
    report_df = pd.DataFrame(eval_results["classification_report"]).transpose()
    st.dataframe(report_df)

# ==========================================================
# TAB 2 : PREDIKSI DATA BARU
# ==========================================================
with tab2:
    st.subheader("🧪 Prediksi Diagnosis Alzheimer")

    with st.form("prediction_form"):
        age = st.number_input("Usia", 40, 100, 70)
        gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        ethnicity = st.selectbox("Etnis", [0, 1, 2, 3])
        education = st.selectbox("Tingkat Pendidikan", [0, 1, 2, 3])
        bmi = st.number_input("BMI", 10.0, 40.0, 23.0)
        smoking = st.selectbox("Merokok", [0, 1])
        alcohol = st.number_input("Konsumsi Alkohol", 0.0, 30.0, 5.0)
        activity = st.number_input("Aktivitas Fisik", 0.0, 10.0, 5.0)
        diet = st.number_input("Kualitas Diet", 0.0, 5.0, 2.5)
        sleep = st.number_input("Kualitas Tidur", 0.0, 10.0, 7.0)
        family = st.selectbox("Riwayat Alzheimer Keluarga", [0, 1])
        cardio = st.selectbox("Penyakit Kardiovaskular", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        depression = st.selectbox("Depresi", [0, 1])
        head = st.selectbox("Riwayat Cedera Kepala", [0, 1])
        hyper = st.selectbox("Hipertensi", [0, 1])
        sys = st.number_input("Tekanan Darah Sistolik", 80, 200, 130)
        dia = st.number_input("Tekanan Darah Diastolik", 50, 120, 80)
        chol = st.number_input("Kolesterol Total", 100, 350, 220)
        ldl = st.number_input("LDL", 50, 250, 140)
        hdl = st.number_input("HDL", 20, 100, 50)
        trig = st.number_input("Trigliserida", 50, 400, 150)
        mmse = st.number_input("Skor MMSE", 0, 30, 20)
        func = st.number_input("Functional Assessment", 0.0, 10.0, 5.0)
        memory = st.selectbox("Keluhan Memori", [0, 1])
        behavior = st.selectbox("Masalah Perilaku", [0, 1])
        adl = st.number_input("ADL", 0.0, 10.0, 4.0)
        confusion = st.selectbox("Kebingungan", [0, 1])
        disorientation = st.selectbox("Disorientasi", [0, 1])
        personality = st.selectbox("Perubahan Kepribadian", [0, 1])
        difficulty = st.selectbox("Sulit Menyelesaikan Tugas", [0, 1])
        forgetful = st.selectbox("Pelupa", [0, 1])

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_data = pd.DataFrame([[  
            age, gender, ethnicity, education, bmi, smoking, alcohol,
            activity, diet, sleep, family, cardio, diabetes, depression,
            head, hyper, sys, dia, chol, ldl, hdl, trig, mmse, func,
            memory, behavior, adl, confusion, disorientation,
            personality, difficulty, forgetful
        ]], columns=[
            "Age","Gender","Ethnicity","EducationLevel","BMI","Smoking",
            "AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality",
            "FamilyHistoryAlzheimers","CardiovascularDisease","Diabetes",
            "Depression","HeadInjury","Hypertension","SystolicBP","DiastolicBP",
            "CholesterolTotal","CholesterolLDL","CholesterolHDL",
            "CholesterolTriglycerides","MMSE","FunctionalAssessment",
            "MemoryComplaints","BehavioralProblems","ADL","Confusion",
            "Disorientation","PersonalityChanges",
            "DifficultyCompletingTasks","Forgetfulness"
        ])

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("### 📊 Hasil Prediksi")
        st.write("**Diagnosis:**", "🟥 Alzheimer" if pred == 1 else "🟩 Non-Alzheimer")
        st.write("**Probabilitas Alzheimer:**", f"{prob:.2%}")
