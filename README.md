# Analisis Sentimen Aplikasi Grab Menggunakan LSTM, SVM, dan Random Forest

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk membangun sistem analisis sentimen berbasis teks dalam Bahasa Indonesia terhadap ulasan aplikasi Grab yang diambil dari Google Playstore. Sistem ini mengklasifikasikan sentimen pengguna menjadi **positif**, **netral**, dan **negatif** menggunakan tiga pendekatan utama:

- **Deep Learning** dengan model **LSTM** dan Word Embedding
- **Machine Learning** klasik menggunakan **SVM** dan **Random Forest** berbasis TF-IDF

Tujuan utama dari proyek ini adalah membandingkan efektivitas dan akurasi dari ketiga pendekatan tersebut dengan berbagai skema pelatihan.

---

## 📚 Dataset

- Berasal dari hasil **scraping** ulasan pengguna aplikasi Grab di Playstore.
- Total jumlah data: **20.000 data ulasan**.
- Label sentimen: **Positif**, **Netral**, dan **Negatif**.
- File dataset:
  - `grab_reviews.csv` — hasil scraping ulasan
  - `slangwords.json` — daftar kata gaul/slang untuk preprocessing

---

## 🧪 Skema Pelatihan & Metode

Proyek ini melibatkan **tiga skema percobaan pelatihan berbeda**, dengan variasi:
- Algoritma: LSTM, SVM, Random Forest
- Fitur: Word Embedding, TF-IDF
- Rasio pembagian data: 80/20 dan 70/30

### 🔹 LSTM + Word Embedding
- Pendekatan deep learning berbasis urutan kata.
- Preprocessing menggunakan tokenizer dan padding.
- Akurasi pelatihan terakhir: **99.46%**
- Akurasi pengujian: **95.20%**

### 🔹 SVM + TF-IDF
- Menggunakan ekstraksi fitur TF-IDF.
- Cocok untuk klasifikasi teks pendek.
- Akurasi pelatihan: **97.90%**
- Akurasi pengujian: **93.93%**

### 🔹 Random Forest + TF-IDF
- Model ensemble berbasis decision tree.
- Lebih toleran terhadap outlier dan noise.
- Akurasi pelatihan: **96.19%**
- Akurasi pengujian: **92.52%**

---

## 📊 Ringkasan Evaluasi Model

| Model                          | Akurasi Uji | Rasio Data | Jenis Fitur     |
|-------------------------------|-------------|-------------|------------------|
| LSTM + Word Embedding         | 95.20%      | 80/20       | Word Embedding   |
| SVM + TF-IDF                  | 93.93%      | 80/20       | TF-IDF           |
| Random Forest + TF-IDF        | 92.52%      | 70/30       | TF-IDF           |

---

## 🚀 Inference & Prediksi

- Proyek ini menyediakan file `.ipynb` dan `.py` untuk melakukan **inference** atau pengujian model terhadap data baru.
- Output berupa **kelas kategorikal**: `positif`, `netral`, atau `negatif`.
- Notebook `inference.ipynb` dapat digunakan untuk melihat prediksi dari ketiga model sekaligus terhadap input teks baru.

---
