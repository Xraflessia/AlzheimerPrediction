#  Sistem Deteksi Alzheimer Berbasis Machine Learning

Aplikasi ini merupakan sistem **klasifikasi biner** untuk membantu **deteksi risiko penyakit Alzheimer pada lansia** menggunakan **algoritma XGBoost Classifier**.
Sistem dikembangkan menggunakan **Python**, **scikit-learn**, **XGBoost**, dan **Streamlit** sebagai antarmuka pengguna.

---

## 📌 Fitur Utama

* ✅ Klasifikasi **Alzheimer / Non-Alzheimer**
* ✅ Input data pasien secara interaktif
* ✅ Menampilkan **probabilitas risiko Alzheimer**
* ✅ Menampilkan **evaluasi performa model** (Accuracy, ROC-AUC, Confusion Matrix, Classification Report)
* ✅ Antarmuka web sederhana berbasis **Streamlit**

---

## 🗂️ Struktur Folder

```
project/
│── app.py                          # Aplikasi Streamlit
│── xgboost_alzheimer_model.pkl    # Model terlatih
│── scaler.pkl                      # Scaler normalisasi data
│── evaluation_results.pkl          # Hasil evaluasi model
│── alzheimers_disease_data.csv     # Dataset (opsional)
│── README.md                       # Dokumentasi proyek
```

---

## 📊 Dataset

Dataset berisi data klinis dan gaya hidup pasien lansia, dengan fitur antara lain:

* Demografi (usia, jenis kelamin, pendidikan)
* Faktor kesehatan (BMI, tekanan darah, kolesterol, diabetes)
* Faktor kognitif (MMSE, functional assessment)
* Gejala perilaku (confusion, forgetfulness, personality changes)

**Target:**
`Diagnosis`

* `0` → Non-Alzheimer
* `1` → Alzheimer

---

## ⚙️ Metodologi

1. **Preprocessing Data**

   * Menghapus kolom tidak relevan
   * Normalisasi fitur menggunakan `StandardScaler`

2. **Model**

   * Algoritma: **XGBoost Classifier**
   * Task: **Klasifikasi Biner**

3. **Evaluasi Model**

   * Accuracy
   * Confusion Matrix
   * Classification Report
   * ROC-AUC Curve

4. **Deployment**

   * Model disimpan dalam format `.pkl`
   * Diintegrasikan ke aplikasi **Streamlit**

---

## 📈 Hasil Evaluasi Model

* **Accuracy**: ± 94–95%
* **ROC-AUC**: ± 0.94
* Model menunjukkan performa yang baik dalam membedakan pasien Alzheimer dan Non-Alzheimer.

> Evaluasi dilakukan secara **offline**, kemudian hasilnya ditampilkan pada aplikasi Streamlit sebagai informasi performa model.

---

## 🧪 Cara Menjalankan Aplikasi

### 1️⃣ Clone Repository

```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Jika belum ada `requirements.txt`, install manual:

```bash
pip install streamlit pandas scikit-learn xgboost joblib matplotlib seaborn
```

### 3️⃣ Jalankan Streamlit

```bash
streamlit run app.py
```

Aplikasi akan berjalan di browser pada:

```
http://localhost:8501
```

---

## 🧠 Tampilan Aplikasi

Aplikasi terdiri dari dua tab utama:

1. **Evaluasi Model**
   Menampilkan performa model klasifikasi.
2. **Prediksi Pasien Baru**
   Input data pasien → hasil diagnosis + probabilitas.

---

## ⚠️ Catatan Penting

* Aplikasi ini **bukan alat diagnosis medis**, melainkan **alat bantu pendukung keputusan**.
* Hasil prediksi harus dikonsultasikan dengan tenaga medis profesional.

---

## 👨‍🎓 Tujuan Pengembangan

Proyek ini dikembangkan untuk keperluan:

* Tugas akhir / skripsi
* Pembelajaran machine learning
* Demonstrasi sistem klasifikasi kesehatan berbasis web

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT License**
Silakan digunakan dan dikembangkan untuk keperluan akademik.

---

## 🙌 Penutup

Semoga aplikasi ini dapat membantu dalam pengembangan sistem deteksi dini Alzheimer dan menjadi referensi pembelajaran machine learning berbasis kesehatan.

---

Kalau kamu mau, aku bisa:

* ✅ bikinkan **`requirements.txt`**
* ✅ bikinkan **MIT LICENSE**
* ✅ bantu **deskripsi repo GitHub (About section)**
* ✅ bantu **deployment ke Streamlit Cloud + badge**

Tinggal bilang mau lanjut yang mana 👌
