import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL
# =========================
model = joblib.load("xgboost_alzheimer_model.pkl")
scaler = joblib.load("scaler.pkl")
eval_results = joblib.load("evaluation_results.pkl")

st.set_page_config(
    page_title="Sistem Cerdas Deteksi Alzheimer",
    layout="wide"
)

st.title("🧠 Sistem Cerdas Deteksi Alzheimer")
st.markdown("""
**Nama**  : Cindy Alya Putri  
**NIM**   : 22533644  
**Kelas** : TI 7D  
""")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔁 Alur Prediksi Sistem",
    "📊 Dataset",
    "📈 Evaluasi Model",
    "🧪 Prediksi"
])

# ==========================================================
# TAB 1 : ALUR PREDIKSI SISTEM
# ==========================================================
with tab1:
    st.subheader("🔁 Alur Prediksi Sistem Cerdas")

    st.markdown("""
    Alur prediksi pada sistem deteksi Alzheimer ditunjukkan sebagai berikut:
    """)

    st.code("""
Input Data Pasien
        ↓
Preprocessing Data
- Penyesuaian fitur
- Normalisasi (StandardScaler)
        ↓
Model XGBoost
- Ensemble Decision Tree
- Gradient Boosting
        ↓
Output Prediksi
- Diagnosis (Alzheimer / Non-Alzheimer)
- Probabilitas Risiko
        ↓
Tampilan Hasil di Streamlit
    """)

    st.success("""
    Setiap kali pengguna menekan tombol **Prediksi**, sistem akan
    menjalankan seluruh alur di atas secara otomatis.
    """)

# ==========================================================
# TAB 2 : DATASET
# ==========================================================
with tab2:
    st.subheader("📊 Dataset yang Digunakan")
    st.write("""
    Dataset berasal dari Kaggle dan berisi data klinis pasien lansia
    yang digunakan sebagai dasar pembelajaran model XGBoost.
    """)

    st.markdown("""
    **Target Klasifikasi:**
    - 0 → Non-Alzheimer
    - 1 → Alzheimer
    """)

# ==========================================================
# TAB 3 : EVALUASI MODEL
# ==========================================================
with tab3:
    st.subheader("📈 Evaluasi Model")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{eval_results['accuracy']:.3f}")
    col2.metric("ROC-AUC", f"{eval_results['roc_auc']:.3f}")

    cm = eval_results["confusion_matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ==========================================================
# TAB 4 : PREDIKSI
# ==========================================================
with tab4:
    st.subheader("🧪 Prediksi Pasien Baru")

    with st.form("prediction_form"):
        age = st.number_input("Usia", 40, 100, 70)
        bmi = st.number_input("BMI", 10.0, 40.0, 23.0)
        mmse = st.number_input("Skor MMSE", 0, 30, 20)
        forgetful = st.selectbox("Pelupa", [0, 1])
        submit = st.form_submit_button("Prediksi")

    if submit:
        # =========================
        # STEP 1: INPUT
        # =========================
        st.info("📥 Step 1: Data pasien diterima")

        input_data = pd.DataFrame([[  
            age, 0, 0, 0, bmi, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            mmse, 0, 0, 0, 0, 0, 0, 0, 0, forgetful
        ]], columns=model.feature_names_in_)

        # =========================
        # STEP 2: PREPROCESSING
        # =========================
        st.info("⚙️ Step 2: Normalisasi data (StandardScaler)")
        input_scaled = scaler.transform(input_data)

        # =========================
        # STEP 3: PREDIKSI
        # =========================
        st.info("🤖 Step 3: Prediksi oleh model XGBoost")
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # =========================
        # STEP 4: OUTPUT
        # =========================
        st.success("📊 Step 4: Hasil prediksi ditampilkan")
        st.write("**Diagnosis:**", "🟥 Alzheimer" if pred == 1 else "🟩 Non-Alzheimer")
        st.write("**Probabilitas Alzheimer:**", f"{prob:.2%}")
