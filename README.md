# **Komparasi Algoritma Machine Learning Berbasis Ensemble: Analisis Akurasi dan Performa melalui Tahapan Preprocessing, Feature Selection, dan Evaluasi**

### 🚀 Aplikasi Web (Demo)

Aplikasi interaktif yang menggunakan model terbaik dari proyek ini telah di-deploy dan dapat diakses melalui tautan berikut:

[**https://pembelajaranmesin-2022150182.streamlit.app/**](https://pembelajaranmesin-2022150182.streamlit.app/)

### 1\. Pendahuluan

#### Latar Belakang Masalah

Di era digital saat ini, ulasan pengguna (user reviews) pada platform seperti Google Play Store telah menjadi sumber informasi yang sangat berharga bagi perusahaan. Bagi Grab, sebagai salah satu super-app terkemuka di Asia Tenggara, memahami sentimen pelanggan dari ulasan aplikasi adalah kunci untuk meningkatkan kualitas layanan, mengidentifikasi masalah, dan menjaga kepuasan pengguna. Namun, ulasan seringkali ditulis dalam bahasa yang tidak terstruktur, mengandung kata-kata slang, singkatan, dan kesalahan ketik. Hal ini menjadi tantangan dalam analisis sentimen otomatis.

Metode *ensemble learning* dalam pembelajaran mesin menawarkan solusi yang kuat untuk masalah klasifikasi yang kompleks. Dengan menggabungkan beberapa model (weak learners) menjadi satu model yang kuat (strong learner), metode ini berpotensi menghasilkan akurasi dan generalisasi yang lebih baik dibandingkan model tunggal.

#### Tujuan Tugas

Tugas ini bertujuan untuk:

1. Membangun model klasifikasi sentimen untuk ulasan aplikasi Grab menggunakan data dari Google Play Store.  
2. Menerapkan tahapan *preprocessing* teks yang komprehensif untuk data ulasan berbahasa Indonesia.  
3. Melakukan seleksi fitur (*feature selection*) untuk mengidentifikasi kata-kata yang paling berpengaruh terhadap sentimen.  
4. Membandingkan performa dari tiga algoritma *machine learning* berbasis *ensemble*: **Random Forest**, **XGBoost**, dan **AdaBoost**.  
5. Menganalisis dan mengevaluasi setiap model menggunakan metrik standar untuk menentukan algoritma terbaik untuk kasus ini.

#### Ruang Lingkup

Ruang lingkup proyek ini mencakup keseluruhan proses, mulai dari pengumpulan data ulasan aplikasi Grab, pembersihan dan pra-pemrosesan data teks, ekstraksi fitur menggunakan TF-IDF, seleksi fitur, pelatihan model, hingga evaluasi dan komparasi performa. Data yang digunakan adalah ulasan teks berbahasa Indonesia, dan target klasifikasi adalah sentimen biner (positif dan negatif).

### 2\. Dataset

#### Sumber Dataset

Dataset yang digunakan dalam proyek ini merupakan data primer yang dikumpulkan melalui proses *scraping* dari halaman aplikasi Grab di **Google Play Store**. Proses *scraping* dilakukan menggunakan skrip Python yang terdapat dalam notebook scraping.ipynb.

#### Deskripsi Dataset

Dataset grab\_reviews.csv berisi ulasan pengguna yang mencakup beberapa kolom, namun yang utama digunakan adalah content (isi ulasan) dan score (peringkat yang diberikan pengguna).

* **Jumlah Data**: Dataset awal berisi ribuan ulasan (jumlah spesifik dapat dilihat pada notebook analisis\_sentimen\_grab.ipynb).  
* **Fitur**: Fitur utama adalah content yang berisi data tekstual.  
* **Target**: Variabel target adalah score, yang merupakan data numerik (skala 1 hingga 5). Untuk analisis sentimen ini, score dikonversi menjadi label kategorikal biner:  
  * **Positif**: jika score \> 3  
  * **Negatif**: jika score \<= 3  
* **Keseimbangan Data**: Setelah pelabelan, dilakukan analisis untuk melihat distribusi antara kelas positif dan negatif. Ditemukan bahwa dataset cenderung tidak seimbang (*imbalanced*), di mana salah satu kelas memiliki jumlah sampel yang jauh lebih banyak. Hal ini menjadi pertimbangan penting dalam pemilihan metrik evaluasi.

### 3\. Tahapan Preprocessing

Preprocessing adalah tahap krusial dalam analisis teks untuk membersihkan dan menstandarisasi data agar dapat diproses oleh model. Berikut adalah tahapan yang dilakukan:

1. **Case Folding**: Mengubah seluruh teks menjadi huruf kecil untuk menghilangkan ambiguitas (misalnya, "Bagus" dan "bagus" dianggap sama).  
2. **Pembersihan Teks (Text Cleaning)**: Menghapus karakter yang tidak relevan seperti angka, tanda baca, dan spasi berlebih.  
3. **Normalisasi Kata (Word Normalization)**: Menggunakan kamus slangwords.json, kata-kata tidak baku atau slang (misalnya, "jg", "ga", "mantap") diubah menjadi bentuk standarnya (misalnya, "juga", "tidak", "mantap").  
4. **Stopword Removal**: Menghapus kata-kata umum dalam bahasa Indonesia yang tidak memiliki makna sentimen signifikan (contoh: "yang", "di", "dan", "adalah") menggunakan daftar *stopword* dari library Sastrawi.  
5. **Stemming**: Mengubah setiap kata ke bentuk dasarnya (kata dasar) menggunakan library Sastrawi. Contoh: "membantu", "bantuan" \-\> "bantu".  
6. **Pembagian Data (Train-Test Split)**: Dataset yang telah bersih dibagi menjadi data latih (80%) dan data uji (20%) menggunakan fungsi train\_test\_split dari Scikit-learn untuk memastikan model dievaluasi pada data yang belum pernah dilihat sebelumnya.

### 4\. Feature Engineering & Selection

#### Feature Engineering: TF-IDF

Teks yang sudah bersih perlu diubah menjadi representasi numerik. Metode yang digunakan adalah **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF memberikan bobot pada setiap kata berdasarkan frekuensinya dalam sebuah ulasan dan kelangkaannya di seluruh ulasan. Kata yang sering muncul di satu ulasan tetapi jarang di ulasan lain akan mendapatkan bobot yang tinggi. Proses ini diimplementasikan menggunakan TfidfVectorizer dari Scikit-learn.

#### Feature Selection

Setelah mendapatkan ribuan fitur (kata) dari TF-IDF, tidak semua fitur relevan untuk prediksi sentimen. Seleksi fitur dilakukan untuk mengurangi dimensi dan noise.

* **Metode yang Digunakan**: **SelectKBest** dengan fungsi skor **Chi-squared (χ²)**. Metode ini memilih *K* fitur terbaik yang memiliki ketergantungan statistik tertinggi dengan variabel target (sentimen positif/negatif).  
* **Justifikasi Pemilihan Fitur**: Chi-squared efektif untuk data kategorikal (seperti klasifikasi teks) karena dapat mengukur sejauh mana sebuah kata (fitur) dapat membantu membedakan antar kelas. Dalam proyek ini, dipilih **1000 fitur terbaik** untuk melatih model.

#### Visualisasi Fitur Terpilih

Berikut adalah contoh visualisasi skor Chi-squared untuk beberapa fitur teratas yang paling berpengaruh dalam menentukan sentimen.

Grafik di atas menunjukkan kata-kata seperti "terima", "kasih", "bantu", "puas" memiliki skor tinggi, yang secara intuitif berkorelasi kuat dengan sentimen positif.

### 5\. Algoritma Ensemble yang Dikomparasikan

Tiga algoritma *ensemble* berikut diimplementasikan dan dibandingkan:

1. **Random Forest**  
   * **Prinsip Kerja**: Random Forest membangun sejumlah besar pohon keputusan (*decision trees*) secara independen pada sub-sampel data yang berbeda. Prediksi akhir diambil berdasarkan voting mayoritas dari semua pohon (untuk klasifikasi).  
   * **Kelebihan**: Cukup tahan terhadap *overfitting*, dapat menangani data dalam jumlah besar, dan memberikan estimasi pentingnya fitur (*feature importance*).  
   * **Kelemahan**: Cenderung kurang interpretatif dibandingkan satu pohon keputusan, dan bisa menjadi lambat jika jumlah pohon sangat banyak.  
2. **XGBoost (Extreme Gradient Boosting)**  
   * **Prinsip Kerja**: Algoritma ini merupakan implementasi Gradient Boosting yang dioptimalkan. XGBoost membangun model secara sekuensial, di mana setiap model baru dilatih untuk memperbaiki kesalahan dari model sebelumnya.  
   * **Kelebihan**: Sangat cepat, efisien, dan seringkali menghasilkan akurasi prediksi yang sangat tinggi.  
   * **Kelemahan**: Sensitif terhadap *hyperparameter tuning* dan rentan terhadap *overfitting* jika tidak dikonfigurasi dengan baik.  
3. **AdaBoost (Adaptive Boosting)**  
   * **Prinsip Kerja**: Mirip dengan Gradient Boosting, AdaBoost membangun model secara sekuensial. Namun, fokus utamanya adalah pada sampel yang salah diklasifikasikan oleh model sebelumnya. Sampel-sampel ini diberi bobot yang lebih tinggi pada iterasi berikutnya.  
   * **Kelebihan**: Relatif mudah diimplementasikan dan tidak terlalu banyak *hyperparameter* yang perlu disesuaikan.  
   * **Kelemahan**: Rentan terhadap data yang *noisy* dan *outliers*.

### 6\. Evaluasi dan Komparasi

#### Metrik Evaluasi

Karena dataset cenderung tidak seimbang, penggunaan akurasi saja tidak cukup. Metrik berikut digunakan untuk evaluasi komprehensif:

* **Akurasi**: Persentase prediksi yang benar dari total data.  
* **Precision**: Kemampuan model untuk tidak melabeli sampel negatif sebagai positif.  
* **Recall (Sensitivity)**: Kemampuan model untuk menemukan semua sampel positif.  
* **F1-Score**: Rata-rata harmonik dari Precision dan Recall.  
* **ROC-AUC**: Kemampuan model untuk membedakan antara kelas positif dan negatif.  
* **Waktu Latih & Prediksi**: Efisiensi komputasi dari model.

#### Hasil Evaluasi

Tabel berikut merangkum hasil performa dari ketiga model pada data uji, berdasarkan eksekusi pada notebook analisis\_sentimen\_grab.ipynb.

| Algoritma | Akurasi | Precision | Recall | F1-Score | ROC-AUC | Waktu Latih (s) | Waktu Prediksi (s) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Random Forest** | **0.9238** | **0.9554** | 0.9489 | **0.9521** | **0.9640** | 4.9700 | 0.3593 |
| **XGBoost** | 0.9183 | 0.9359 | 0.9638 | 0.9496 | 0.9596 | **0.6610** | **0.0100** |
| **AdaBoost** | 0.8465 | 0.8532 | **0.9757** | 0.9104 | 0.8982 | 1.0768 | 0.0651 |

### 7\. Analisis dan Interpretasi

Berdasarkan hasil evaluasi yang komprehensif, **Random Forest muncul sebagai algoritma dengan performa terbaik secara keseluruhan**. Model ini unggul dalam metrik **Akurasi (0.9238)**, **Precision (0.9554)**, **F1-Score (0.9521)**, dan **ROC-AUC (0.9640)**. Keunggulan ini menunjukkan bahwa Random Forest memiliki keseimbangan yang sangat baik antara kemampuan mengidentifikasi sentimen positif secara akurat dan meminimalkan kesalahan klasifikasi.

* **Analisis Performa XGBoost**: Meskipun sedikit di bawah Random Forest dalam hal akurasi, **XGBoost menunjukkan efisiensi komputasi yang luar biasa**. Waktu latihnya (0.66 detik) dan waktu prediksinya (0.01 detik) jauh lebih cepat dibandingkan kedua model lainnya. Selain itu, XGBoost memiliki nilai *Recall* yang sedikit lebih tinggi, artinya model ini sangat baik dalam menangkap hampir semua ulasan positif, meskipun dengan risiko *false positive* yang sedikit lebih tinggi. Ini menjadikannya pilihan yang sangat menarik untuk aplikasi yang membutuhkan kecepatan tinggi.  
* **Analisis Performa AdaBoost**: Menariknya, **AdaBoost menunjukkan nilai Recall tertinggi (0.9757)**, yang berarti model ini hampir tidak pernah melewatkan sentimen positif. Namun, keunggulan ini dibayar dengan nilai *Precision* yang jauh lebih rendah (0.8532), yang mengakibatkan akurasi dan F1-Score yang paling rendah di antara ketiganya. Ini menunjukkan bahwa AdaBoost dalam kasus ini cenderung terlalu agresif dalam memprediksi kelas positif, sehingga banyak menghasilkan *false positive*.  
* **Mengapa Random Forest Unggul?**: Keunggulan Random Forest kemungkinan besar berasal dari mekanisme *bagging* dan pengacakan fitur pada setiap *split*. Hal ini membuatnya sangat tahan terhadap *overfitting* dan mampu menangkap pola yang kompleks tanpa terlalu bias pada sampel tertentu, tidak seperti AdaBoost yang performanya menurun karena terlalu fokus pada sampel yang salah diklasifikasikan.

### 8\. Kesimpulan

#### Ringkasan Hasil

Proyek ini berhasil membangun dan membandingkan tiga model *ensemble* untuk analisis sentimen ulasan aplikasi Grab. Melalui serangkaian tahapan *preprocessing* dan seleksi fitur yang cermat, model **Random Forest** terbukti menjadi model dengan performa terbaik dengan **Akurasi 0.9238** dan **F1-Score 0.9521**. Meskipun XGBoost menawarkan kecepatan yang jauh lebih superior, keseimbangan antara akurasi dan keandalan menjadikan Random Forest pilihan terbaik untuk kasus ini. Oleh karena itu, model yang disimpan sebagai best\_ensemble\_model.pkl adalah model Random Forest.

#### Saran Pengembangan

Untuk pengembangan di masa depan, beberapa hal dapat dieksplorasi:

1. **Hyperparameter Tuning**: Melakukan pencarian *hyperparameter* yang lebih ekstensif menggunakan teknik seperti GridSearchCV atau RandomizedSearchCV untuk lebih mengoptimalkan setiap model, terutama XGBoost yang sangat sensitif terhadap parameter.  
2. **Model yang Lebih Canggih**: Menggunakan model berbasis *deep learning* seperti LSTM atau *transformer* (misalnya IndoBERT) yang telah terbukti sangat kuat untuk tugas-tugas NLP Bahasa Indonesia.  
3. **Penanganan Imbalance Data**: Menerapkan teknik *oversampling* (seperti SMOTE) atau *undersampling* secara eksplisit untuk melihat apakah dapat meningkatkan performa, terutama pada metrik *precision* untuk model seperti AdaBoost.  
4. **Analisis Multi-Kelas**: Mengembangkan model untuk klasifikasi multi-kelas (misalnya: sangat positif, positif, netral, negatif, sangat negatif) daripada hanya biner.

### 9\. Referensi

* **Sumber Data**: Google Play Store \- [https://play.google.com/store/search?q=grab\&c=apps\&hl=id](https://play.google.com/store/search?q=grab&c=apps&hl=id)  
* **Dokumentasi Library**:  
  * Scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)  
  * Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)  
  * Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)  
  * Sastrawi: [https://github.com/sastrawi/sastrawi](https://github.com/sastrawi/sastrawi)

### 10\. Lampiran

Kode program lengkap untuk *scraping*, analisis, pelatihan model, dan inferensi dapat ditemukan dalam direktori Notebooks/ pada repositori proyek. Aplikasi web interaktif yang menggunakan model terbaik juga tersedia (app.py).