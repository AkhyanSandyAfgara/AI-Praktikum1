import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Iris Predictor", page_icon="🌿", layout="wide")

# --- FUNGSI LOAD & TRAIN ---
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    X = df.drop("species", axis=1)
    y = df["species"]
    
    # Training Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return df, model, accuracy

# Memuat data dan model
df, model, accuracy = load_and_train()

# --- USER INTERFACE (SIDEBAR) ---
st.sidebar.header("📥 Input Parameter Bunga")
st.sidebar.write("Geser slider di bawah untuk menyesuaikan nilai (dalam cm):")

def user_input_features():
    sepal_l = st.sidebar.slider("Panjang Kelopak (Sepal Length)", 4.0, 8.0, 5.8)
    sepal_w = st.sidebar.slider("Lebar Kelopak (Sepal Width)", 2.0, 4.5, 3.1)
    petal_l = st.sidebar.slider("Panjang Mahkota (Petal Length)", 1.0, 7.0, 3.8)
    petal_w = st.sidebar.slider("Lebar Mahkota (Petal Width)", 0.1, 2.5, 1.2)
    
    data = {
        'Sepal Length': sepal_l,
        'Sepal Width': sepal_w,
        'Petal Length': petal_l,
        'Petal Width': petal_w
    }
    # Perhatikan urutan array harus sama dengan urutan saat training
    return np.array([[sepal_l, sepal_w, petal_l, petal_w]]), data

input_data, input_df_display = user_input_features()

# --- MAIN PAGE (TAMPILAN UTAMA) ---
st.title("🌸 Aplikasi Klasifikasi Spesies Iris")
st.markdown(f"""
Selamat datang! Aplikasi ini menggunakan algoritma **Machine Learning (Random Forest)** untuk memprediksi jenis bunga Iris (*Setosa, Versicolor, atau Virginica*).
* **Akurasi Model Saat Ini:** `{accuracy*100:.2f}%`
""")

st.divider() # Garis pemisah biar rapi

# Menggunakan kolom untuk layout yang lebih rapi
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📋 Parameter Input")
    st.info("Nilai ini diambil dari sidebar di sebelah kiri.")
    # Menampilkan input pengguna dalam bentuk tabel yang rapi
    st.table(pd.DataFrame(input_df_display, index=["Nilai (cm)"]).T)

with col2:
    st.subheader("🎯 Hasil Prediksi")
    
    # Tombol Prediksi
    if st.button("Lakukan Prediksi", type="primary", use_container_width=True):
        # Animasi loading singkat biar terasa seperti aplikasi beneran
        with st.spinner('Menganalisis data...'):
            time.sleep(0.8) # Jeda sedikit untuk efek visual
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)
        
        st.success(f"Berdasarkan data, bunga ini diprediksi sebagai: **Iris {prediction.capitalize()}**")
        
        # Menampilkan Probabilitas
        st.write("**Tingkat Keyakinan Model (Probabilitas):**")
        prob_df = pd.DataFrame(prediction_proba, columns=[c.capitalize() for c in model.classes_])
        # Menggunakan st.dataframe dengan format persentase
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)
    else:
        st.warning("👈 Atur parameter di sidebar, lalu klik tombol prediksi di atas.")

st.divider()

# --- BAGIAN DATASET ---
# Menggunakan expander agar dataset tidak menuhi layar kecuali diklik
with st.expander("Tampilkan Dataset Asli (Referensi)"):
    st.write(df)
