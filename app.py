import streamlit as st
import pickle
import re
import json
import numpy as np
import nltk # Ditambahkan
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings
import nltk
nltk.download('punkt')
nltk.download('stopwords')

warnings.filterwarnings("ignore")

# --- Improvisasi: Fungsi untuk mengunduh data NLTK ---
# Ini akan memastikan data yang diperlukan selalu tersedia.
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

# Panggil fungsi download saat aplikasi dimulai
download_nltk_data()


# --- Fungsi Caching untuk Memuat Model dan Artefak ---
# Caching memastikan model hanya dimuat sekali saat aplikasi pertama kali dijalankan.
@st.cache_resource
def load_artifacts():
    """Memuat model, vectorizer, dan feature selector dari file."""
    # --- PERBAIKAN: Mengubah path file menjadi relatif terhadap direktori utama ---
    with open("Models/best_ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("Models/feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open('Datasets/slangwords.json', 'r') as file:
        slang_dict = json.load(file)
    return model, vectorizer, selector, slang_dict

# --- Fungsi Tunggal untuk Preprocessing Teks ---
def preprocess_text(text, slang_dict):
    """Membersihkan dan memproses satu teks input."""
    stop_words = set(stopwords.words('indonesian'))
    stemmer = StemmerFactory().create_stemmer()

    text = text.lower() # Case folding
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words] # Normalisasi slang
    text = ' '.join(normalized_words)
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words] # Hapus stopwords
    text = ' '.join(filtered_words)
    text = stemmer.stem(text) # Stemming
    return text

# --- Memuat Artefak ---
model, vectorizer, selector, slang_dict = load_artifacts()

# --- Antarmuka Pengguna (UI) Streamlit ---
st.set_page_config(page_title="Analisis Sentimen Ulasan", layout="wide")
st.title("Aplikasi Analisis Sentimen Ulasan")
st.write("Masukkan teks ulasan (dalam Bahasa Indonesia) di bawah ini untuk memprediksi sentimennya (Positif atau Negatif).")

# Input teks dari pengguna
user_input = st.text_area("Teks Ulasan:", "Aplikasinya sangat membantu dan mudah digunakan!", height=150)

# Tombol untuk memicu prediksi
if st.button("Analisis Sentimen"):
    if user_input:
        # Tampilkan spinner saat sedang memproses
        with st.spinner('Sedang menganalisis...'):
            # 1. Preprocessing teks input
            cleaned_text = preprocess_text(user_input, slang_dict)

            # 2. Transformasi teks menggunakan TF-IDF dan Feature Selector
            text_tfidf = vectorizer.transform([cleaned_text])
            text_selected = selector.transform(text_tfidf)

            # 3. Lakukan prediksi
            prediction = model.predict(text_selected)
            prediction_proba = model.predict_proba(text_selected)

            # 4. Tampilkan hasil
            st.subheader("Hasil Analisis:")
            
            # Konversi label numerik ke string
            label_map = {0: "NEGATIF", 1: "POSITIF"}
            result = label_map[prediction[0]]
            
            if result == "POSITIF":
                st.success(f"**Sentimen: {result}**")
            else:
                st.error(f"**Sentimen: {result}**")

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