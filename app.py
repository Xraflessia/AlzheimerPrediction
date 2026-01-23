import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Sistem Cerdas Deteksi Alzheimer",
    layout="wide"
)

# =====================================================
# LOAD MODEL & HASIL EVALUASI
# =====================================================
model = joblib.load("xgboost_alzheimer_model.pkl")
scaler = joblib.load("scaler.pkl")
eval_results = joblib.load("evaluation_results.pkl")

FEATURE_COLUMNS = [
    "Age","Gender","Ethnicity","EducationLevel","BMI","Smoking",
    "AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality",
    "FamilyHistoryAlzheimers","CardiovascularDisease","Diabetes",
    "Depression","HeadInjury","Hypertension","SystolicBP","DiastolicBP",
    "CholesterolTotal","CholesterolLDL","CholesterolHDL",
    "CholesterolTriglycerides","MMSE","FunctionalAssessment",
    "MemoryComplaints","BehavioralProblems","ADL","Confusion",
    "Disorientation","PersonalityChanges",
    "DifficultyCompletingTasks","Forgetfulness"
]

# =====================================================
# HEADER
# =====================================================
st.title("🧠 Sistem Cerdas Deteksi Alzheimer")
st.markdown("""
**Nama** : Cindy Alya Putri  
**NIM** : 22533644  
**Kelas** : TI 7D  

Sistem ini menggunakan **algoritma XGBoost Classifier**  
untuk mendeteksi **risiko penyakit Alzheimer pada lansia**
berdasarkan data klinis.
""")

# =====================================================
# TAB
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Penjelasan Sistem",
    "📈 Evaluasi Model",
    "🔁 Alur Deteksi",
    "🧪 Prediksi Pasien Baru"
])

# =====================================================
# TAB 1 – PENJELASAN SISTEM
# =====================================================
with tab1:
    st.subheader("📘 Penjelasan Kegiatan pada Sistem")

    st.markdown("""
    **1. Load Dataset**  
    Dataset berasal dari Kaggle dan berisi data klinis pasien lansia.

    **2. Preprocessing**  
    - Menghapus kolom identitas (PatientID, DoctorInCharge)  
    - Normalisasi data menggunakan StandardScaler  

    **3. Training Model**  
    Model XGBoost dilatih untuk mempelajari pola hubungan
    antara fitur klinis dan diagnosis Alzheimer.

    **4. Evaluasi Model**  
    Evaluasi dilakukan menggunakan:
    - Accuracy
    - Confusion Matrix
    - Classification Report
    - ROC-AUC

    **5. Prediksi Data Baru**  
    Data pasien baru diproses dan diklasifikasikan
    menjadi Alzheimer atau Non-Alzheimer.
    """)

# =====================================================
# TAB 2 – EVALUASI MODEL
# =====================================================
with tab2:
    st.subheader("📈 Evaluasi Performa Model")

    # ===============================
    # ACCURACY
    # ===============================
    st.markdown("### ✅ Accuracy")
    acc = eval_results["accuracy"]
    st.metric("Accuracy", f"{acc:.3f}")

    st.markdown(f"""
    **Interpretasi Accuracy:**

    Nilai accuracy sebesar **{acc:.2%}** menunjukkan bahwa dari seluruh data uji,
    sekitar **{int(acc*100)}% diagnosis pasien berhasil diprediksi dengan benar** oleh model.
    Hal ini menandakan bahwa model XGBoost memiliki tingkat ketepatan yang tinggi
    dalam membedakan pasien Alzheimer dan Non-Alzheimer.
    """)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    st.markdown("### 🔍 Confusion Matrix")

    cm = eval_results["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=110)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 9},
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=9, pad=6)

    st.pyplot(fig, use_container_width=False)

    st.markdown(f"""
    **Interpretasi Confusion Matrix:**

    - **{cm[0][0]} pasien Non-Alzheimer** berhasil diprediksi dengan benar sebagai Non-Alzheimer.
    - **{cm[1][1]} pasien Alzheimer** berhasil diprediksi dengan benar sebagai Alzheimer.
    - **{cm[0][1]} pasien Non-Alzheimer** salah diprediksi sebagai Alzheimer (*false positive*).
    - **{cm[1][0]} pasien Alzheimer** salah diprediksi sebagai Non-Alzheimer (*false negative*).

    Jumlah *false negative* yang relatif kecil menunjukkan bahwa
    **sebagian besar pasien Alzheimer berhasil terdeteksi oleh sistem**,
    yang sangat penting dalam konteks medis agar risiko keterlambatan diagnosis dapat diminimalkan.
    """)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.markdown("### 📋 Classification Report")

    report_df = pd.DataFrame(eval_results["classification_report"]).transpose()
    st.dataframe(report_df)

    prec = report_df.loc["1", "precision"]
    rec = report_df.loc["1", "recall"]
    f1 = report_df.loc["1", "f1-score"]

    st.markdown(f"""
    **Interpretasi Classification Report (Kelas Alzheimer):**

    - **Precision = {prec:.2f}**  
      Artinya, sebagian besar pasien yang diprediksi Alzheimer memang benar mengalami Alzheimer.

    - **Recall = {rec:.2f}**  
      Artinya, model mampu mendeteksi sekitar **{rec:.0%} pasien Alzheimer** dari seluruh kasus Alzheimer yang ada.

    - **F1-Score = {f1:.2f}**  
      Menunjukkan keseimbangan yang baik antara precision dan recall,
      sehingga model tidak hanya akurat tetapi juga sensitif terhadap kasus Alzheimer.
    """)

    # ===============================
    # ROC-AUC
    # ===============================
    st.markdown("### 📈 ROC-AUC")

    roc = eval_results["roc_auc"]
    st.metric("ROC-AUC", f"{roc:.3f}")

    st.markdown(f"""
    **Interpretasi ROC-AUC:**

    Nilai ROC-AUC sebesar **{roc:.2f}** menunjukkan bahwa model memiliki
    **kemampuan diskriminasi yang sangat baik** dalam membedakan pasien Alzheimer
    dan Non-Alzheimer. Nilai ini jauh di atas 0.5 (tebakan acak),
    sehingga model dinilai andal untuk digunakan sebagai sistem pendukung keputusan medis.
    """)


# =====================================================
# TAB 3 – ALUR DETEKSI
# =====================================================
with tab3:
    st.subheader("🔁 Alur Deteksi Sistem Cerdas")

    st.code("""
Input Data Pasien
        ↓
Preprocessing Data
(Normalisasi dengan StandardScaler)
        ↓
Model XGBoost
(Ensemble Decision Tree)
        ↓
Output
(Diagnosis & Probabilitas Alzheimer)
    """)

    st.info("""
    Setiap kali pengguna melakukan prediksi,
    sistem akan melalui seluruh tahapan di atas secara otomatis.
    """)

# =====================================================
# TAB 4 – PREDIKSI PASIEN BARU
# =====================================================
with tab4:
    st.subheader("🧪 Prediksi Pasien Baru")

    with st.form("form_prediksi"):
        age = st.number_input("Usia", 40, 100, 75)
        gender = st.selectbox("Jenis Kelamin (0=Perempuan, 1=Laki-laki)", [0, 1])
        bmi = st.number_input("BMI", 10.0, 40.0, 23.5)
        mmse = st.number_input("Skor MMSE", 0, 30, 18)
        diabetes = st.selectbox("Diabetes", [0, 1])
        hypertension = st.selectbox("Hipertensi", [0, 1])
        forgetful = st.selectbox("Pelupa", [0, 1])

        submit = st.form_submit_button("Prediksi")

    if submit:
        st.markdown("### 📥 Data Mentah")
        input_data = pd.DataFrame([[  
            age, gender, 0, 2, bmi, 0,
            8.5, 5.2, 1.3, 7.8,
            1, 0, diabetes, 0, 0, hypertension,
            145, 85, 240, 160, 45, 180,
            mmse, 5.5, 1, 1, 3.2, 1,
            1, 1, 1, forgetful
        ]], columns=FEATURE_COLUMNS)

        st.dataframe(input_data)

        st.markdown("### ⚙️ Data Setelah Preprocessing")
        scaled = scaler.transform(input_data)
        st.dataframe(pd.DataFrame(scaled, columns=FEATURE_COLUMNS))

        st.markdown("### 🤖 Hasil Deteksi")
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        st.success("Deteksi Selesai")
        st.write("**Diagnosis:**", "🟥 Alzheimer" if pred == 1 else "🟩 Non-Alzheimer")
        st.write("**Probabilitas Alzheimer:**", f"{prob:.2%}")
        st.markdown("### 📝 Rekomendasi Sistem")
        if pred == 1:
    st.warning("""
    **Hasil menunjukkan risiko Alzheimer.**

    🔹 Disarankan untuk:
    - Melakukan konsultasi dengan **dokter spesialis saraf**
    - Menjalani **pemeriksaan lanjutan** (tes kognitif lanjutan, MRI/CT Scan)
    - Melibatkan **keluarga atau caregiver** dalam pemantauan aktivitas harian
    - Melakukan **monitoring rutin** terhadap kondisi pasien

    ⚠️ *Hasil ini bersifat pendukung keputusan dan bukan diagnosis medis final.*
    """)
else:
    st.success("""
    **Hasil menunjukkan risiko Alzheimer rendah (Non-Alzheimer).**

    🔹 Disarankan untuk:
    - Menjaga **pola hidup sehat** (olahraga, nutrisi, tidur cukup)
    - Melatih fungsi kognitif (membaca, permainan otak)
    - Mengontrol faktor risiko seperti **hipertensi dan diabetes**
    - Melakukan **pemeriksaan berkala** sebagai langkah pencegahan
    """)

