import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
    page_icon="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png",
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
                """<h2 style = "text-align: center;"><img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png" width="130" height="130"><br></h2>""",
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
        <img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/udara.jpeg" width="500" height="300">
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
        # Fungsi utama untuk training model
        def model_training(X, y, n_neighbors_options, weights_options, metric_options):
            # Membagi data latih dan uji
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Variabel untuk menyimpan performa terbaik
            best_accuracy = 0
            best_model = None
            best_param_set = {}
            elapsed_time = 0
        
            # Coba semua kombinasi hyperparameter
            for n_neighbors in n_neighbors_options:
                for weights in weights_options:
                    for metric in metric_options:
                        start_time = time.time()
                        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
                        knn_model.fit(X_train, y_train)
                        accuracy = knn_model.score(X_test, y_test)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = knn_model
                            best_param_set = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
        
                        # Menampilkan hasil sementara di Streamlit
                        st.write(f"Params: n_neighbors={n_neighbors}, weights={weights}, metric={metric} | "
                                 f"Accuracy: {accuracy:.4f} | Time: {elapsed_time:.2f} seconds")
        
            # Menampilkan laporan klasifikasi dan confusion matrix menggunakan model terbaik
            y_pred = best_model.predict(X_test)
            class_report = classification_report(y_test, y_pred)
            st.write("Classification Report:\n", class_report)
        
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            st.pyplot(fig)
        
            # Simpan model terbaik
            joblib.dump(best_model, 'best_knn_model.pkl')
        
            return best_accuracy, best_model, best_param_set, elapsed_time
        
        # Setup Streamlit UI
        st.title("Model Training with k-Nearest Neighbors (KNN)")
        initial_percentage = st.number_input("Initial Percentage", min_value=1, max_value=100, value=95)
        max_percentage = st.number_input("Max Percentage", min_value=1, max_value=100, value=60)
        step_percentage = st.number_input("Step Percentage", min_value=1, max_value=100, value=-5)
        n_neighbors_options = [3, 5, 7, 9]
        weights_options = ['distance']
        metric_options = ['euclidean', 'manhattan']
        
        # Dummy data untuk contoh
        X, y = ...  # Pastikan data X, y sudah siap untuk digunakan
        
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        percentages = []
        accuracies = []
        elapsed_times = []
        best_models = []
        best_params = []
        
        progress_bar = st.progress(0)
        for i, percentage in enumerate(range(initial_percentage, max_percentage + 1, step_percentage)):
            X_selected, feature_rankings = feature_selection(X_resampled, y_resampled, percentage)
            accuracy, best_model, best_param_set, elapsed_time = model_training(
                X_selected, y_resampled, n_neighbors_options, weights_options, metric_options)
        
            percentages.append(percentage)
            accuracies.append(accuracy)
            elapsed_times.append(elapsed_time)
            best_models.append(best_model)
            best_params.append(best_param_set)
        
            model_filename = f'best_knn_model_{percentage}percent.pkl'
            joblib.dump(best_model, model_filename)
            st.write(f"Model saved as: {model_filename}")
            st.write(f"Best Params for {percentage}% features: {best_param_set}")
            st.write(f"Best Accuracy on Test Data: {accuracy:.4f}")
            st.write(f"Total Elapsed Time for Best Model: {elapsed_time:.2f} seconds")
        
            progress_bar.progress((i + 1) / len(range(initial_percentage, max_percentage + 1, step_percentage)))
        
        # Simpan hasil pelatihan ke file Excel
        results_df = pd.DataFrame({
            'Percentage': percentages,
            'Accuracy': accuracies,
            'Elapsed Time (s)': elapsed_times,
            'Best Parameters': best_params
        })
        
        results_df.to_excel('training_results.xlsx', index=False)
        st.write("Training results saved as 'training_results.xlsx'")
        
        # Tampilkan tombol unduh
        with open('training_results.xlsx', 'rb') as file:
            st.download_button(
                label="Download Training Results",
                data=file,
                file_name="training_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
st.markdown("---")  # Menambahkan garis pemisah
st.write("Syamsyiya Tuddiniyah-200441100016 (Sistem Informasi)")
