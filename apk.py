import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import subprocess
subprocess.run(['pip', 'install', 'imbalanced-learn'])
from imblearn.over_sampling import RandomOverSampler
from math import sqrt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import re
import time
import seaborn as sns
import os
os.system('pip install nltk')
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="Analisis Sentimen Rumah Makan",
    page_icon="https://raw.githubusercontent.com/dinia28/skripsi/main/rumah.jpg",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.write(
    """<h1 style="font-size: 40px;">Analisis Sentimen Rumah Makan</h1>""",
    unsafe_allow_html=True,
)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write(
                """<h2 style = "text-align: center;"><img src="https://raw.githubusercontent.com/dinia28/skripsi/main/home.png" width="130" height="130"><br></h2>""",
                unsafe_allow_html=True,
            ),
            [
                "Home",
                "Data",
                "Preprocessing",
                "TF-IDF",
                "Information Gain",
                "Model WKNN",

            ],
            icons=[
                "house",
                "person",
                "gear",
                "bar-chart",
                "arrow-down-square",
                "file-earmark-font",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#87CEEB"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "white",
                },
                "nav-link-selected": {"background-color": "#005980"},
            },
        )

    if selected == "Home":
        st.write(
            """<h3 style = "text-align: center;">
        <img src="https://raw.githubusercontent.com/dinia28/skripsi/main/bebek.jpeg" width="500" height="300">
        </h3>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Deskripsi Aplikasi""")
        st.write(
            """
        ANALISIS SENTIMEN RUMAH MAKAN MELALUI ULASAN GOOGLE MAPS MENGGUNAKAN METODE WEIGHT K-NEAREST NEIGHBOR DENGAN SELEKSI FITUR INFORMATION GAIN
        """
        )

    elif selected == "Data":

        st.subheader("""Deskripsi Data""")
        st.write(
            """
        Data yang digunakan dalam aplikasi ini yaitu data dari hasil scrapping ulasan pada google maps
        """
        )
        
        st.subheader("Dataset")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/dinia28/skripsi/main/bebek.xlsx"
        )
        st.dataframe(df, width=600)
        
        st.subheader("Label")
        # Menampilkan frekuensi dari masing-masing label
        label_counts = df['Label'].value_counts()
        st.write(label_counts)
        
    elif selected == "Preprocessing":
        # Cleansing
        st.subheader("Preprocessing")
    
        import re
        import pandas as pd
        
        # Mendefinisikan fungsi cleaning
        def cleaning(text):
            try:
                text = re.sub(r'\$\w*', '', str(text))
                text = re.sub(r'^rt[\s]+', '', str(text))
                text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', str(text))
                text = re.sub(r'&quot;', " ", str(text))
                text = re.sub(r"\d+", " ", str(text))
                text = re.sub(r"\b[a-zA-Z]\b", "", str(text))
                text = re.sub(r"[^\w\s]", " ", str(text))
                text = re.sub(r'(.)\1+', r'\1\1', str(text))
                text = re.sub(r"\s+", " ", str(text))
                text = re.sub(r'#', '', str(text))
                text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text))
                text = re.sub(r'\s\s+', ' ', str(text))
                text = re.sub(r'^RT[\s]+', '', str(text))
                text = re.sub(r'^b[\s]+', '', str(text))
                text = re.sub(r'^link[\s]+', '', str(text))
                return text
            except Exception as e:
                st.write(f"Error cleaning text: {e}")
                return text
        
        # Mengambil data dari file Excel
        df = pd.read_excel("https://raw.githubusercontent.com/dinia28/skripsi/main/bebek.xlsx")
        # Cek kolom dan isi untuk memastikan kolom 'Ulasan' ada
        st.write("Data contoh sebelum cleaning:", df['Ulasan'].head())
        
        # Mengisi nilai NaN dengan string kosong untuk kolom 'Ulasan'
        df['Ulasan'] = df['Ulasan'].fillna("")
        
        # Menerapkan fungsi cleaning
        df['Cleaning'] = df['Ulasan'].apply(cleaning)
        st.write("Hasil Cleansing:")
        st.dataframe(df[['Ulasan', 'Cleaning']])
        
        # Menambahkan proses case folding
        df['CaseFolding'] = df['Cleaning'].str.lower()
        st.write("Hasil Case Folding:")
        st.dataframe(df[['Ulasan', 'Cleaning', 'CaseFolding']])
        
        # Membaca file slang words
        slangword_normalisasi = pd.read_csv("combined_slang_words.csv")
        
        # Membuat kamus slang words untuk normalisasi
        kata_normalisasi_dict = {row[0]: row[1] for _, row in slangword_normalisasi.iterrows()}
        
        # Fungsi untuk normalisasi kata slang
        def normalisasi_kata(document):
            return ' '.join([kata_normalisasi_dict.get(term, term) for term in document.split()])
        
        # Menerapkan fungsi normalisasi slang words
        df['CaseFolding'] = df['CaseFolding'].fillna('').astype(str)
        df['slangword'] = df['CaseFolding'].apply(normalisasi_kata)
        
        # Tampilkan hasil akhir setelah normalisasi slang words
        st.write("Hasil Normalisasi Slang Words:")
        st.dataframe(df[['Ulasan', 'Cleaning', 'CaseFolding', 'slangword']])

        # Tokenizing
        def tokenizer(text):
            if isinstance(text, str):
                return text.split()  # Tokenisasi sederhana dengan split
                return []
        
        # Menerapkan tokenizing pada kolom 'slangword'
        df['Tokenizing'] = df['slangword'].apply(tokenizer)
        
        # Tampilkan hasil akhir setelah tokenizing
        st.write("Hasil Tokenizing:")
        st.dataframe(df[['Ulasan', 'Cleaning', 'CaseFolding', 'slangword', 'Tokenizing']])
        
        # Stopword removal
        sw = pd.read_csv("combined_stop_words.csv", header=None)[0].tolist()
        
        # Gabungkan stopword default dengan stopword tambahan
        corpus = sw
        
        # Fungsi stopword removal
        def stopword_removal(words):
            return [word for word in words if word not in corpus]
        
        # Menerapkan stopword removal pada kolom 'Tokenizing'
        df['Stopword_Removal'] = df['Tokenizing'].apply(stopword_removal)
        
        # Menampilkan hasil di Streamlit
        st.write("Data setelah stopword removal:")
        st.dataframe(df[['Ulasan', 'Cleaning', 'CaseFolding', 'slangword', 'Tokenizing', 'Stopword_Removal']])

        # Inisialisasi Porter Stemmer
        stemmer = PorterStemmer()
        
        # Fungsi stemming
        def stemText(words):
            return [stemmer.stem(word) for word in words]
        
        # Menerapkan stemming pada kolom 'Stopword_Removal'
        df['Stemming'] = df['Stopword_Removal'].apply(stemText)
        
        # Menggabungkan hasil stemming menjadi satu kalimat
        df['Full_Text_Stemmed'] = df['Stemming'].apply(lambda x: ' '.join(x))
        
        # Menampilkan hasil akhir di Streamlit
        st.write("Data setelah Stemming:")
        st.dataframe(df[['Ulasan', 'Cleaning', 'CaseFolding', 'slangword', 'Tokenizing', 'Stopword_Removal', 'Stemming', 'Full_Text_Stemmed']])

    elif selected == "TF-IDF":
        # Load the dataset from 'hasil_preprocessing.xlsx'
        df = pd.read_excel("hasil_preprocessing.xlsx")
        # Assume 'Full_Text_Stemmed' is the column with the processed text for TF-IDF
        # Create a new DataFrame for TF-IDF
        df_tfidf = df[['Full_Text_Stemmed', 'Label']]  
        # Initialize the TfidfVectorizer
        vectorizer = TfidfVectorizer()
        # Transform the 'Full_Text_Stemmed' column into a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(df_tfidf['Full_Text_Stemmed'].values.astype('U')) 
        # Get feature names (the words corresponding to the TF-IDF values)
        feature_names = vectorizer.get_feature_names_out()
        # Convert the TF-IDF matrix into a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names) 
        # Add the 'Label' column to the DataFrame
        tfidf_df['Label'] = df_tfidf['Label']
        # Display the TF-IDF result
        st.subheader("TF-IDF Results")
        st.dataframe(tfidf_df)
        # Optionally, you can save the vectorizer for future use (for example, to use in predictions)
        # import joblib
        # joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    elif selected == "Information Gain":
        import requests
        from io import BytesIO
        st.subheader("Information Gain")
        st.write("Proses Information Gain")  # Debugging tambahan
        url = "https://raw.githubusercontent.com/dinia28/skripsi/main/hasil_ig.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            data = BytesIO(response.content)
            df = pd.read_excel(data)
            st.dataframe(df, width=600)
        else:
            st.error("Gagal mengambil file. Periksa URL atau koneksi internet.")
    
    elif selected == "Model WKNN":
        def load_data(file_path):
            return pd.read_excel("hasil_tfidf.xlsx")
            
        # Load model dan selector fitur
        def load_selector_and_model(selector_file, model_file):
            selector = joblib.load(selector_file)
            model = joblib.load(model_file)
            return selector, model
        
        # Aplikasi Streamlit
        st.title("Aplikasi WKNN dengan Seleksi Fitur (Model Tersimpan)")
        
        # Pilih persentase seleksi fitur
        percentage_options = [65, 70, 75, 80, 85, 90, 95]
        selected_percentage = st.selectbox("Pilih Persentase Seleksi Fitur:", percentage_options)
        # Path file model dan selektor fitur
        selector_file = f"/mnt/data/feature_selector_{selected_percentage}percent.pkl"
        model_file = f"/mnt/data/best_knn_model_{selected_percentage}percent.pkl"
        
            # Load selector dan model
        selector, model = load_selector_and_model(selector_file, model_file)
        
            # Memisahkan fitur dan label
        if 'Label' in tfidf_df.columns:
            X = tfidf_df.drop(columns=['Label'])
            y = tfidf_df['Label']
            st.write("Kolom Label ditemukan, data akan digunakan untuk evaluasi.")
        else:
            X = tfidf_df
            y = None
            st.write("Kolom Label tidak ditemukan, data akan digunakan untuk prediksi.")
        
            # Seleksi fitur
            X_selected = selector.transform(X)
            st.write(f"Data setelah Seleksi Fitur ({selected_percentage}%):")
            st.dataframe(pd.DataFrame(X_selected))
        
            if y is not None:
                # Evaluasi model
                st.write("Evaluasi Model:")
                y_pred = model.predict(X_selected)
                st.write("Classification Report:")
                st.text(classification_report(y, y_pred))
                cm = confusion_matrix(y, y_pred)
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                st.pyplot(fig)
            else:
                # Prediksi baru
                st.write("Hasil Prediksi:")
                predictions = model.predict(X_selected)
                tfidf_df['Prediction'] = predictions
                st.dataframe(tfidf_df)
        
        except FileNotFoundError:
            st.error(f"File {tfidf_file} tidak ditemukan. Pastikan file berada di lokasi yang sesuai.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

st.markdown("---")  # Menambahkan garis pemisah
st.write("Syamsyiya Tuddiniyah-200441100016 (Sistem Informasi)")
