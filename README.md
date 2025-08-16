# Analisis Sentimen Ulasan Aplikasi Grab di Google Play Store

## 📝 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi **Grab** yang datanya diambil (scraping) dari Google Play Store. Tujuannya adalah untuk mengklasifikasikan ulasan ke dalam dua kategori sentimen: **positif** atau **negatif**.

Model yang telah dilatih kemudian diimplementasikan ke dalam sebuah aplikasi web sederhana menggunakan Streamlit, di mana pengguna dapat memasukkan teks ulasan dan mendapatkan prediksi sentimennya secara langsung.

🔗 **Aplikasi Streamlit yang telah dideploy:**  
👉 [Analisis Sentimen Ulasan Grab](https://pembelajaranmesin-2022150182.streamlit.app/)

---

## 📂 Struktur Folder Proyek

.
├── Datasets
│ ├── grab_reviews.csv
│ └── slangwords.json
├── Models
│ ├── best_ensemble_model.pkl
│ ├── feature_selector.pkl
│ └── tfidf_vectorizer.pkl
├── Notebooks
│ ├── analisis_sentimen_grab.ipynb
│ ├── inference.ipynb
│ └── scraping.ipynb
├── app.py
├── requirements.txt
└── README.md


-   **`/Datasets`**: Berisi data mentah ulasan dan kamus kata-kata slang.  
-   **`/Models`**: Menyimpan model machine learning, vectorizer, dan feature selector yang telah dilatih.  
-   **`/Notebooks`**: Kumpulan Jupyter Notebook untuk setiap tahap, mulai dari scraping, analisis, hingga uji coba model.  
-   **`app.py`**: File utama untuk menjalankan aplikasi web Streamlit.  
-   **`requirements.txt`**: Daftar library Python yang dibutuhkan untuk menjalankan proyek.  

---

## ⚙️ Alur Pengerjaan Proyek

1.  **Scraping Data**: Data ulasan aplikasi Grab dikumpulkan dari Google Play Store menggunakan notebook `scraping.ipynb`.  
2.  **Text Preprocessing**: Teks ulasan dibersihkan melalui beberapa tahapan seperti *case folding*, menghapus karakter yang tidak perlu, dan normalisasi kata-kata slang menggunakan `slangwords.json`.  
3.  **Ekstraksi Fitur**: Teks yang sudah bersih diubah menjadi representasi numerik menggunakan metode **TF-IDF (Term Frequency-Inverse Document Frequency)**.  
4.  **Pemodelan**: Berbagai model machine learning dievaluasi, dan model **Ensemble** terpilih sebagai model terbaik untuk klasifikasi sentimen.  
5.  **Implementasi**: Model terbaik disimpan dan diimplementasikan dalam sebuah aplikasi web menggunakan **Streamlit** untuk prediksi secara *real-time*.  

---

## 🚀 Cara Menjalankan Aplikasi

1.  **Clone Repository**  
    Pastikan Anda sudah meng-clone repository ini ke mesin lokal Anda.

2.  **Buat Virtual Environment (Opsional tapi Direkomendasikan)**  
    ```bash
    python -m venv venv
    ```  
    Aktifkan environment:  
    -   Windows: `.\venv\Scripts\activate`  
    -   macOS/Linux: `source venv/bin/activate`  

3.  **Instalasi Dependensi**  
    ```bash
    pip install -r requirements.txt
    ```  

4.  **Jalankan Aplikasi Streamlit**  
    ```bash
    streamlit run app.py
    ```  

5.  **Buka di Browser**  
    Aplikasi akan secara otomatis terbuka di browser Anda pada alamat `http://localhost:8501`. Sekarang Anda dapat mencoba memasukkan ulasan untuk diprediksi sentimennya.

---

## 🧑‍💻 Dibuat Oleh

-   **Kunti Najma Jalia**  
-   Proyek ini dibuat untuk memenuhi tugas Ujian Akhir Semester (UAS) mata kuliah Pembelajaran Mesin.
