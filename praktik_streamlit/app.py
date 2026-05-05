import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="KNN Classification", layout="centered")
st.title("🤖 KNN Classification (Dataset Kaggle)")

# DOWNLOAD DATASET (sekali saja, lalu cache)
@st.cache_data
def load_dataset():
    path = kagglehub.dataset_download(
        "santiago123678/evolution-of-humans-datasets-for-clasification"
    )
    return path

path = load_dataset()

st.write("📁 Lokasi dataset:", path)
st.write("📄 Isi folder:")
files = os.listdir(path)
st.write(files)

# PILIH FILE CSV DI DALAM DATASET
csv_files = [f for f in files if f.endswith(".csv")]

if not csv_files:
    st.error("Tidak ada file CSV di dataset!")
    st.stop()

selected_file = st.selectbox("Pilih file CSV", csv_files)

df = pd.read_csv(os.path.join(path, selected_file))

st.write("### Dataset")
st.dataframe(df)

# RAPIIKAN NAMA KOLOM
df.columns = df.columns.str.strip()

# PILIH TARGET
target = st.selectbox("Pilih Kolom Target (Label)", df.columns)

if target:
    X = df.drop(columns=[target])
    y = df[target]

    # Encode target jika kategori
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Ambil fitur numerik saja
    X = X.select_dtypes(include=np.number)

    if X.shape[1] > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        k = st.slider("Nilai K", 1, 15, 3)

        if st.button("Train Model"):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)

            acc = model.score(X_test, y_test)
            st.success(f"Akurasi: {acc*100:.2f}%")

            st.write("### Prediksi Data Baru")

            input_data = []
            for col in X.columns:
                val = st.number_input(
                    f"{col}",
                    float(X[col].min()),
                    float(X[col].max())
                )
                input_data.append(val)

            if st.button("Prediksi"):
                pred = model.predict([input_data])[0]
                st.success(f"Hasil Prediksi: {pred}")
    else:
        st.error("Tidak ada kolom numerik untuk training!")
