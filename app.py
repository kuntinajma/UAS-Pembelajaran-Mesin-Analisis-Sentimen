import streamlit as st
import pickle
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings

warnings.filterwarnings("ignore")

# --- Bagian 1: Mengunduh Data NLTK ---
# Fungsi ini memastikan data yang diperlukan NLTK (untuk tokenisasi dan stopwords)
# sudah terunduh di lingkungan server Streamlit Cloud.
# @st.cache_resource akan menyimpan data ini sehingga tidak diunduh berulang kali.
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

# Panggil fungsi untuk mengunduh data saat aplikasi pertama kali dimuat
download_nltk_data()


# --- Bagian 2: Memuat Model dan Artefak Lain ---
# @st.cache_resource digunakan agar model dan file lainnya hanya dimuat sekali,
# sehingga aplikasi berjalan lebih cepat setelah pemuatan pertama.
@st.cache_resource
def load_artifacts():
    """Memuat model, vectorizer, feature selector, dan kamus slang dari file."""
    with open("Models/best_ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("Models/feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open('Datasets/slangwords.json', 'r') as file:
        slang_dict = json.load(file)
    
    # Membuat stemmer dan daftar stopwords juga di-cache di sini
    stemmer = StemmerFactory().create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    
    return model, vectorizer, selector, slang_dict, stemmer, stop_words

# Memuat semua artefak yang diperlukan
model, vectorizer, selector, slang_dict, stemmer, stop_words = load_artifacts()


# --- Bagian 3: Fungsi Preprocessing Teks ---
# Fungsi ini sekarang lebih ringkas dan efisien.
def preprocess_text(text, slang_dict, stemmer, stop_words):
    """Membersihkan dan memproses satu teks input dari pengguna."""
    text = text.lower() # 1. Case folding
    words = word_tokenize(text) # 2. Tokenisasi
    
    # 3. Normalisasi, Filtering, dan Hapus Stopwords dalam satu langkah
    processed_words = []
    for word in words:
        # Hanya proses kata yang alfanumerik
        if word.isalnum():
            # Normalisasi kata slang
            normalized_word = slang_dict.get(word, word)
            # Cek apakah kata bukan stopword setelah dinormalisasi
            if normalized_word not in stop_words:
                processed_words.append(normalized_word)
                
    text = ' '.join(processed_words) # Gabungkan kembali kata-kata yang sudah bersih
    text = stemmer.stem(text) # 4. Stemming
    return text

# --- Bagian 4: Antarmuka Pengguna (UI) Streamlit ---
st.set_page_config(page_title="Analisis Sentimen Ulasan", layout="wide")
st.title("📱 Aplikasi Analisis Sentimen Ulasan")
st.write("Masukkan teks ulasan (dalam Bahasa Indonesia) di bawah ini untuk memprediksi sentimennya (Positif atau Negatif).")

# Input teks dari pengguna
user_input = st.text_area("Teks Ulasan:", "Aplikasinya sangat membantu dan mudah digunakan!", height=150)

# Tombol untuk memicu prediksi
if st.button("Analisis Sentimen"):
    if user_input:
        # Tampilkan spinner saat sedang memproses
        with st.spinner('Sedang menganalisis... 🧐'):
            # 1. Preprocessing teks input
            cleaned_text = preprocess_text(user_input, slang_dict, stemmer, stop_words)

            # 2. Transformasi teks menggunakan TF-IDF dan Feature Selector
            text_tfidf = vectorizer.transform([cleaned_text])
            text_selected = selector.transform(text_tfidf)

            # 3. Lakukan prediksi
            prediction = model.predict(text_selected)
            prediction_proba = model.predict_proba(text_selected)

            # 4. Tampilkan hasil
            st.subheader("Hasil Analisis:")
            
            label_map = {0: "NEGATIF", 1: "POSITIF"}
            result = label_map[prediction[0]]
            
            if result == "POSITIF":
                st.success(f"**Sentimen: {result}** 👍")
            else:
                st.error(f"**Sentimen: {result}** 👎")

            # Tampilkan skor probabilitas
            st.write("Skor Kepercayaan:")
            proba_df = {
                'Sentimen': ['Negatif', 'Positif'],
                'Probabilitas': [f"{prediction_proba[0][0]*100:.2f}%", f"{prediction_proba[0][1]*100:.2f}%"]
            }
            st.table(proba_df)
    else:
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")

st.sidebar.header("Tentang Proyek")
st.sidebar.info(
    "Aplikasi ini menggunakan model machine learning *ensemble* yang telah dilatih "
    "untuk mengklasifikasikan sentimen ulasan aplikasi ke dalam kategori positif atau negatif."
)