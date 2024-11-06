import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


st.set_page_config(
    page_title="Prediksi Kualitas Udara DKI Jakarta",
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
    """<h1 style="font-size: 40px;">Prediksi Kualitas Udara di DKI Jakarta</h1>""",
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
                "Missing Value & Normalisasi",
                "Hasil MAPE",
                "Next Day",

            ],
            icons=[
                "house",
                "file-earmark-font",
                "bar-chart",
                "gear",
                "arrow-down-square",
                "person",
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
         Aplikasi Prediksi kualitas Udara di DKI Jakarta merupakan aplikasi yang digunakan untuk meramalkan 6 konsentrasi polutan udara di DKI Jakarta yang meliputi PM10, PM25, SO2, CO, NO2, dan O3 serta menentukan kategori untuk hari berikutnya..
        """
        )

    elif selected == "Data":

        st.subheader("""Deskripsi Data""")
        st.write(
            """
        Data yang digunakan dalam aplikasi ini yaitu data ISPU DKI Jakarta periode 1 Desember 2022 sampai 30 November 2023. Data yang ditampilkan adalah data ispu yang diperoleh per harinya. 
        """
        )

        st.subheader("""Sumber Dataset""")
        st.write(
            """
        Sumber data didapatkan dari website "Satu Data DKI Jakarta". Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://satudata.jakarta.go.id/search?q=data%20ispu&halaman=all&kategori=all&topik=all&organisasi=all&status=all&sort=desc&page_no=1&show_filter=true&lang=id">Klik disini</a>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Dataset""")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        st.dataframe(df, width=600)
        
        st.subheader("Penghapusan kolom")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        data.tanggal = pd.to_datetime(data.tanggal)
        
        # Menampilkan dataframe setelah penghapusan kolom
        st.dataframe(data, width=600)
        
    elif selected == "Missing Value & Normalisasi":
        # MEAN IMPUTATION
        st.subheader("""Mean Imputation""")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Menampilkan jumlah missing value per kolom
        missing_values = data.isnull().sum()
        st.write("Jumlah Missing Value per Kolom:")
        st.dataframe(missing_values[missing_values > 0].reset_index(name='missing_values'))
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        # Menyimpan data ke format XLSX
        data.to_excel('kualitas_udara.xlsx', index=False)
        
        # Menampilkan data yang telah diproses
        st.dataframe(data, width=600)

        # PLOTING DATA
        st.subheader("""Ploting Data""")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Resample data harian dan menghitung rata-rata
        data_resample = data.set_index('tanggal').resample('D').mean().reset_index()
        
        
        # Menentukan ukuran figure untuk subplot
        plt.figure(figsize=(12, 18))  # Ukuran figure untuk 6 subplot
        
        # Plot PM10
        plt.subplot(6, 1, 1)  # 6 baris, 1 kolom, subplot ke-1
        plt.plot(data_resample['tanggal'], data_resample['pm_sepuluh'], color='red')
        plt.title('Konsentrasi PM10')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot PM2.5
        plt.subplot(6, 1, 2)  # subplot ke-2
        plt.plot(data_resample['tanggal'], data_resample['pm_duakomalima'], color='yellow')
        plt.title('Konsentrasi PM2.5')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Karbon Monoksida
        plt.subplot(6, 1, 3)  # subplot ke-3
        plt.plot(data_resample['tanggal'], data_resample['karbon_monoksida'], color='green')
        plt.title('Konsentrasi Karbon Monoksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Ozon
        plt.subplot(6, 1, 4)  # subplot ke-4
        plt.plot(data_resample['tanggal'], data_resample['ozon'], color='magenta')
        plt.title('Konsentrasi Ozon')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Nitrogen Dioksida
        plt.subplot(6, 1, 5)  # subplot ke-5
        plt.plot(data_resample['tanggal'], data_resample['nitrogen_dioksida'], color='black')
        plt.title('Konsentrasi Nitrogen Dioksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Sulfur Dioksida
        plt.subplot(6, 1, 6)  # subplot ke-6
        plt.plot(data_resample['tanggal'], data_resample['sulfur_dioksida'], color='blue')
        plt.title('Konsentrasi Sulfur Dioksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Menyesuaikan layout untuk menghindari tumpang tindih
        plt.tight_layout()
        
        # Menampilkan plot
        plt.show()
        # Setelah plotting
        st.pyplot()

        # Standardisasi DATA
        # Normalisasi Data
        st.subheader("Normalisasi Data")
        scaler = MinMaxScaler()
        
        # Daftar kolom polutan
        pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
        
        # Melakukan normalisasi untuk setiap kolom polutan
        for col in pollutants:
            with st.expander(f"{col.upper()} - Normalisasi Data"):
                # Reshape data menjadi 2D array
                values = data[col].values.reshape(-1, 1)
                
                # Fit dan transform data
                normalized_values = scaler.fit_transform(values)
                
                # Masukkan hasil normalisasi ke dalam DataFrame
                data[f'{col}_normalized'] = normalized_values
                
                # Tampilkan hasil normalisasi
                st.write(data[[col, f'{col}_normalized']])
        
        # Tampilkan semua kolom polutan dan kolom normalisasi mereka
        st.write(data[[f'{col}' for col in pollutants] + [f'{col}_normalized' for col in pollutants]])

    elif selected == "Hasil MAPE":
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom PM10 ke tipe data integer
        data['pm_sepuluh'] = data['pm_sepuluh'].astype(int)
        # Menyimpan data ke format XLSX
        data.to_excel('kualitas_udara_.xlsx', index=False)
        
        # Bagian MODELLING
        st.subheader("Modelling PM10")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk PM10
        def fuzzy_knn_predict(data, k=3, test_size=0.3):
            # Normalisasi data PM10
            imports = data['pm_sepuluh'].values.reshape(-1, 1)
            data['pm_sepuluh_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['pm_sepuluh_normalized'].values[:-1].reshape(-1, 1)
            y = data['pm_sepuluh_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
            # Plotting nilai aktual dan prediksi
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(results['Actual'], label='Nilai Aktual', color='blue', linewidth=2)
            ax.plot(results['Predicted'], label='Nilai Prediksi', color='orange', linewidth=2)
            ax.set_title('Perbandingan Nilai Aktual dan Prediksi')
            ax.set_xlabel('Indeks')
            ax.set_ylabel('Nilai PM10')
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            return mape
        # Menyimpan MAPE untuk setiap rasio
        mapes = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict(data, k=3, test_size=test_size)
            mapes.append(mape)
        
        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes) + 10)
        plt.grid(axis='y')
        st.pyplot(plt)
    
        # Modeling PM2.5
        st.subheader("Modelling PM2.5")

        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
            
        # Fungsi Fuzzy KNN untuk PM2.5
        def fuzzy_knn_predict_pm25(data, k=3, test_size=0.3):
            # Normalisasi data PM2.5
            imports = data['pm_duakomalima'].values.reshape(-1, 1)
            data['pm_duakomalima_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['pm_duakomalima_normalized'].values[:-1].reshape(-1, 1)
            y = data['pm_duakomalima_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio PM2.5
        mapes_pm25 = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_pm25(data, k=3, test_size=test_size)
            mapes_pm25.append(mape)
        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes_pm25, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data PM2.5')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes_pm25) + 10)
        plt.grid(axis='y')
        
        # Menggunakan Streamlit untuk menampilkan plot
        st.pyplot(plt)
        
        # Modeling Sulfur Dioksida
        st.subheader("Modelling Sulfur Dioksida")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk Sulfur Dioksida
        def fuzzy_knn_predict_sulfur(data, k=3, test_size=0.3):
            # Normalisasi data Sulfur Dioksida
            imports = data['sulfur_dioksida'].values.reshape(-1, 1)
            data['sulfur_dioksida_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['sulfur_dioksida_normalized'].values[:-1].reshape(-1, 1)
            y = data['sulfur_dioksida_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio Sulfur Dioksida
        mapes_sulfur = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_sulfur(data, k=3, test_size=test_size)
            mapes_sulfur.append(mape)

        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes_sulfur, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes_sulfur) + 10)
        plt.grid(axis='y')
        st.pyplot(plt)
        
        # Modeling Karbon Monoksida
        st.subheader("Modelling Karbon Monoksida")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN
        def fuzzy_knn_predict_co(data, k=3, sigma=1.0, test_size=0.3):
            # Normalisasi data
            imports = data['karbon_monoksida'].values.reshape(-1, 1)
            data['karbon_monoksida_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['karbon_monoksida_normalized'].values[:-1].reshape(-1, 1)
            y = data['karbon_monoksida_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk Pembagian Data {int((1-test_size)*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)

            return mape
    
        # Menyimpan MAPE untuk setiap rasio Karbon Monoksida
        mapes_co = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_co(data, k=6, sigma=1.0, test_size=test_size)
            mapes_co.append(mape)
        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes_co, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes_co) + 10)
        plt.grid(axis='y')
        st.pyplot(plt)
        
        # Modeling Ozon
        st.subheader("Modelling Ozon")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk Ozon
        def fuzzy_knn_predict_ozon(data, k=3, test_size=0.3):
            # Normalisasi data Ozon
            imports = data['ozon'].values.reshape(-1, 1)
            data['ozon_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['ozon_normalized'].values[:-1].reshape(-1, 1)
            y = data['ozon_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio Ozon
        mapes_ozon = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_ozon(data, k=3, test_size=test_size)
            mapes_ozon.append(mape)
        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes_ozon, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes_ozon) + 10)
        plt.grid(axis='y')
        st.pyplot(plt)

        # Modeling Nitrogen Dioksida
        st.subheader("Modelling Nitrogen Dioksida")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy dengan sigma
        def calculate_membership_inverse(distances, sigma=1.0):
            memberships = np.exp(- (distances ** 2) / (2 * sigma ** 2))
            return memberships
        
        # Fungsi Fuzzy KNN untuk Nitrogen Dioksida
        def fuzzy_knn_predict_nitrogen(data, k=6, test_size=0.3, sigma=1.0):
            # Normalisasi data Nitrogen Dioksida
            imports = data['nitrogen_dioksida'].values.reshape(-1, 1)
            data['nitrogen_dioksida_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['nitrogen_dioksida_normalized'].values[:-1].reshape(-1, 1)
            y = data['nitrogen_dioksida_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances, sigma)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio Nitrogen Dioksida
        mapes_nitrogen = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_nitrogen(data, k=6, test_size=test_size, sigma=1.0)
            mapes_nitrogen.append(mape)
        
        # Plotting MAPE
        plt.figure(figsize=(8, 5))
        plt.bar(['70%-30%', '80%-20%', '90%-10%'], mapes_nitrogen, color=['blue', 'orange', 'green'])
        plt.title('MAPE untuk Berbagai Pembagian Data Nitrogen Dioksida')
        plt.xlabel('Rasio Pembagian Data')
        plt.ylabel('MAPE (%)')
        plt.ylim(0, max(mapes_nitrogen) + 10)
        plt.grid(axis='y')
        st.pyplot(plt)
         
        
    elif selected == "Next Day":   
        st.subheader("PM10")       
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Menampilkan jumlah missing value per kolom
        missing_values = data.isnull().sum()
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna
        user_input = st.number_input("Masukkan konsentrasi PM 10:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi10"):
            prediction = fuzzy_knn_predict(data, "pm_sepuluh", user_input, k=3)
            st.write(f"Prediksi konsentrasi PM 10 esok hari: {prediction:.2f}")

        st.subheader("PM25")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk PM2.5
        user_input = st.number_input("Masukkan konsentrasi PM 2.5:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi 25"):
            prediction = fuzzy_knn_predict(data, "pm_duakomalima", user_input, k=3)
            st.write(f"Prediksi konsentrasi PM 2.5 esok hari: {prediction:.2f}")
        
        st.subheader("SO2")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Sulfur Dioksida
        user_input = st.number_input("Masukkan konsentrasi Sulfur Dioksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi SO2"):
            prediction = fuzzy_knn_predict(data, "sulfur_dioksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Sulfur Dioksida esok hari: {prediction:.2f}")

        st.subheader("CO")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Karbon Monoksida
        user_input = st.number_input("Masukkan konsentrasi Karbon Monoksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi CO"):
            prediction = fuzzy_knn_predict(data, "karbon_monoksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Karbon Monoksida esok hari: {prediction:.2f}")
        
        st.subheader("O3")  
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Ozon
        user_input = st.number_input("Masukkan konsentrasi Ozon:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi O3"):
            prediction = fuzzy_knn_predict(data, "ozon", user_input, k=3)
            st.write(f"Prediksi konsentrasi Ozon esok hari: {prediction:.2f}")

        st.subheader("NO2")
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            # Normalisasi data
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            # Bagi data menjadi train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            # Normalisasi input dari pengguna
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(1)
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                # Hitung membership
                memberships = calculate_membership_inverse(neighbor_distances)
        
                # Hitung prediksi sebagai weighted average
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk Nitrogen Dioksida
        user_input = st.number_input("Masukkan konsentrasi Nitrogen Dioksida:", min_value=0.0)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi NO2"):
            prediction = fuzzy_knn_predict(data, "nitrogen_dioksida", user_input, k=3)
            st.write(f"Prediksi konsentrasi Nitrogen Dioksida esok hari: {prediction:.2f}")

        # After all predictions, gather maximum values
        max_values = {
            "Polutan": ["PM10", "PM2.5", "SO2", "CO", "O3", "NO2"],
            "Nilai Maksimum": [
                data['pm_sepuluh'].max(),
                data['pm_duakomalima'].max(),
                data['sulfur_dioksida'].max(),
                data['karbon_monoksida'].max(),
                data['ozon'].max(),
                data['nitrogen_dioksida'].max()
            ]
        }
        
        # Convert to DataFrame
        max_values_data = pd.DataFrame(max_values)
        
        # Display the table
        st.subheader("Nilai Maksimum Per Polutan")
        st.table(max_values_data)

        st.subheader("Prediksi Kualitas Udara")
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk prediksi
        def fuzzy_knn_predict(data, pollutant, user_input, k=3):
            imports = data[pollutant].values.reshape(-1, 1)
            data[f'{pollutant}_normalized'], scaler = normalize_data(imports)
        
            X = data[f'{pollutant}_normalized'].values[:-1].reshape(-1, 1)
            y = data[f'{pollutant}_normalized'].values[1:]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
        
            user_input_scaled = scaler.transform(np.array([[user_input]]))
        
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=k, return_distance=True)
        
            y_pred = np.zeros(1)
        
            for i in range(1):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
        
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
        
            return y_pred_original[0][0]
        
        # Muat dan bersihkan data dari file
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Imputasi mean untuk kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Input dari pengguna untuk semua polutan
        user_inputs = {}
        pollutants = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"]
        
        for pollutant in pollutants:
            user_inputs[pollutant] = st.number_input(f"Masukkan konsentrasi {pollutant.replace('_', ' ').title()}:", min_value=0.0, key=pollutant)
        
        # Prediksi berdasarkan input pengguna
        if st.button("Prediksi Semua Polutan"):
            predictions = {}
            for pollutant in pollutants:
                prediction = fuzzy_knn_predict(data, pollutant, user_inputs[pollutant], k=3)
                predictions[pollutant] = prediction
            
            # Tampilkan semua prediksi dalam format tabel
            predictions_data = pd.DataFrame(predictions, index=[0])
            st.write(predictions_data)
            # Menentukan polutan tertinggi dan kategorinya
            max_pollutant = max(predictions, key=predictions.get)
            max_value = predictions[max_pollutant]
            
            # Menentukan kategori berdasarkan nilai
            if max_value <= 50:
                category = "Baik"
            elif max_value <= 100:
                category = "Sedang"
            elif max_value <= 150:
                category = "Tidak Sehat"
            else:
                category = "Berbahaya"
            
            # Buat DataFrame untuk polutan tertinggi
            highest_pollutant_data = pd.DataFrame({
                "Polutan_Tertinggi": [max_pollutant],
                "Nilai_Tertinggi": [max_value],
                "Kategori": [category]
            })
            
            # Tampilkan tabel polutan tertinggi
            st.write(highest_pollutant_data)

            # Membuat grafik untuk input dan hasil prediksi
            fig = go.Figure()
        
            # Menambahkan trace untuk input
            fig.add_trace(go.Bar(
                x=pollutants,
                y=list(user_inputs.values()),
                name='Input',
                marker_color='blue'
            ))
        
            # Menambahkan trace untuk prediksi
            fig.add_trace(go.Bar(
                x=pollutants,
                y=list(predictions.values()),
                name='Prediksi',
                marker_color='orange'
            ))
        
            # Menambahkan layout
            fig.update_layout(
                title='Input dan Hasil Prediksi Kualitas Udara',
                xaxis_title='Polutan',
                yaxis_title='Konsentrasi',
                barmode='group'
            )
        
            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig)
        
            # Buat grafik keriting
            plt.figure(figsize=(10, 6))
            plt.plot(predictions.keys(), predictions.values(), marker='x', label='Hasil Prediksi', color='blue')
            
            # Tambahkan input pengguna ke grafik
            plt.plot(predictions.keys(), user_inputs.values(), marker='x', label='Input Pengguna', color='red')
            
            plt.title('Grafik Prediksi Kualitas Udara')
            plt.xlabel('Polutan')
            plt.ylabel('Konsentrasi (µg/m³)')
            plt.legend()
            plt.grid()
            st.pyplot(plt)

    # Menampilkan penanda
    st.markdown("---")  # Menambahkan garis pemisah
    st.write("Shinta Alya Imani Putri-200411100005 (Teknik Informatika)")
