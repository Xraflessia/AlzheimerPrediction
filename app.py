import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL & EVALUASI
# =========================
model = joblib.load("xgboost_alzheimer_model.pkl")
scaler = joblib.load("scaler.pkl")
eval_results = joblib.load("evaluation_results.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sistem Cerdas Deteksi Alzheimer",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🧠 Sistem Cerdas Deteksi Alzheimer")
st.markdown("""
**Nama**  : Cindy Alya Putri  
**NIM**   : 22533644  
**Kelas** : TI 7D  

Aplikasi ini merupakan **tugas mata kuliah Sistem Cerdas**  
untuk mendeteksi **risiko penyakit Alzheimer pada lansia**
menggunakan **algoritma XGBoost Classifier**.
""")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Beranda",
    "📊 Dataset",
    "🤖 Metode XGBoost",
    "📈 Evaluasi Model",
    "🧪 Prediksi Pasien Baru"
])

# ==========================================================
# TAB 1 : BERANDA
# ==========================================================
with tab1:
    st.subheader("📌 Latar Belakang")
    st.write("""
    Penyakit Alzheimer merupakan salah satu gangguan neurodegeneratif
    yang banyak dialami oleh lansia dan dapat menyebabkan penurunan fungsi kognitif.
    Oleh karena itu, diperlukan sistem cerdas yang mampu membantu
    mendeteksi risiko Alzheimer secara dini berdasarkan data klinis.
    """)

    st.subheader("🎯 Tujuan Sistem")
    st.write("""
    Tujuan dari sistem ini adalah membangun model klasifikasi
    untuk mengelompokkan pasien lansia ke dalam dua kelas:
    **Alzheimer** dan **Non-Alzheimer**
    menggunakan pendekatan machine learning.
    """)

# ==========================================================
# TAB 2 : DATASET
# ==========================================================
with tab2:
    st.subheader("📊 Deskripsi Dataset")
    st.write("""
    Dataset yang digunakan berasal dari Kaggle:

    **Alzheimer’s Disease Dataset**  
    Sumber: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

    Dataset ini berisi data klinis, gaya hidup, dan kondisi kognitif pasien lansia.
    """)

    st.subheader("🔢 Fitur dan Target")
    st.markdown("""
    - **Fitur**: Usia, BMI, tekanan darah, kolesterol, MMSE, kondisi perilaku, dll  
    - **Target**: `Diagnosis`
        - `0` → Non-Alzheimer  
        - `1` → Alzheimer
    """)

    st.info("Dataset digunakan untuk **klasifikasi biner (supervised learning)**.")

# ==========================================================
# TAB 3 : METODE XGBOOST
# ==========================================================
with tab3:
    st.subheader("🤖 Algoritma XGBoost")

    st.write("""
    XGBoost (Extreme Gradient Boosting) adalah algoritma
    ensemble learning yang membangun banyak decision tree
    secara bertahap untuk meningkatkan akurasi prediksi.
    """)

    st.subheader("🧠 Cara Kerja XGBoost dalam Sistem Ini")
    st.markdown("""
    1. Data pasien lansia dimasukkan ke dalam sistem  
    2. Setiap decision tree mempelajari pola dari data klinis  
    3. Kesalahan prediksi diperbaiki oleh tree berikutnya  
    4. Hasil akhir berupa:
       - Prediksi kelas (Alzheimer / Non-Alzheimer)
       - Probabilitas Alzheimer
    """)

    st.subheader("✅ Alasan Pemilihan XGBoost")
    st.markdown("""
    - Cocok untuk data tabular & numerik  
    - Performa tinggi pada klasifikasi biner  
    - Banyak digunakan pada penelitian kesehatan  
    """)

# ==========================================================
# TAB 4 : EVALUASI MODEL
# ==========================================================
with tab4:
    st.subheader("📈 Hasil Evaluasi Model")

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

    st.info("""
    Evaluasi dilakukan menggunakan data uji (test data)
    untuk mengukur performa model dalam mengklasifikasikan pasien.
    """)

# ==========================================================
# TAB 5 : PREDIKSI DATA BARU
# ==========================================================
with tab5:
    st.subheader("🧪 Prediksi Diagnosis Alzheimer")

    with st.form("prediction_form"):
        age = st.number_input("Usia", 40, 100, 70)
        gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        education = st.selectbox("Tingkat Pendidikan", [0, 1, 2, 3])
        bmi = st.number_input("BMI", 10.0, 40.0, 23.0)
        mmse = st.number_input("Skor MMSE", 0, 30, 20)
        hypertension = st.selectbox("Hipertensi", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        forgetful = st.selectbox("Pelupa", [0, 1])

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_data = pd.DataFrame([[
            age, gender, 0, education, bmi, 0, 0, 0, 0, 0, 0, 0,
            diabetes, 0, 0, hypertension, 0, 0, 0, 0, 0, 0,
            mmse, 0, 0, 0, 0, 0, 0, 0, 0, forgetful
        ]], columns=model.feature_names_in_)

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("### 📊 Hasil Prediksi")
        st.write("**Diagnosis:**", "🟥 Alzheimer" if pred == 1 else "🟩 Non-Alzheimer")
        st.write("**Probabilitas Alzheimer:**", f"{prob:.2%}")
