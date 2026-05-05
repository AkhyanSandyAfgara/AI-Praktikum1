import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="KNN Classification", layout="centered")

st.title("🤖 Aplikasi KNN Classification")

# CEK FILE DI FOLDER (biar tidak bingung)
st.write("📁 Isi folder saat ini:")
st.write(os.listdir())

# PILIH SUMBER DATA
option = st.radio("Pilih Sumber Data", ["Upload CSV", "Gunakan Dataset Iris (Excel)"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Silakan upload dataset terlebih dahulu!")
        st.stop()

else:
    # LANGSUNG BACA FILE EXCEL IRIS
    try:
        df = pd.read_excel("Iris.xlsx")  # sesuaikan nama file kamu
    except:
        st.error("File Iris.xlsx tidak ditemukan! Pastikan nama & lokasi benar.")
        st.stop()

# TAMPILKAN DATASET
st.write("### Dataset")
st.dataframe(df)

# PILIH TARGET
target = st.selectbox("Pilih Kolom Target", df.columns)

if target:
    X = df.drop(columns=[target])
    y = df[target]

    # Encode jika target kategorikal
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Ambil kolom numerik saja
    X = X.select_dtypes(include=np.number)

    if X.shape[1] > 0:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Input nilai K
        k = st.slider("Nilai K (Jumlah Tetangga)", 1, 15, 3)

        if st.button("Train Model"):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)

            acc = model.score(X_test, y_test)

            st.success(f"Akurasi Model: {acc*100:.2f}%")

            st.write("### Prediksi Data Baru")

            input_data = []
            for col in X.columns:
                val = st.number_input(
                    f"Masukkan {col}",
                    float(X[col].min()),
                    float(X[col].max())
                )
                input_data.append(val)

            if st.button("Prediksi"):
                prediction = model.predict([input_data])[0]

                st.success(f"Hasil Prediksi: {prediction}")
    else:
        st.error("Dataset tidak memiliki kolom numerik!")
