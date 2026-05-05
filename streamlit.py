import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset (pakai Iris.csv lokal)
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")  # pastikan file ada di folder yang sama
    return df

# Train model
@st.cache_data
def train_model(df):
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Main app
def main():
    st.set_page_config(
        page_title="Klasifikasi Iris dengan Streamlit",
        layout="centered"
    )

    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.write("Masukkan nilai fitur bunga iris:")

    df = load_data()
    model, accuracy = train_model(df)

    # Input fitur
    sepal_length = st.slider(
        "Sepal Length (cm)",
        float(df.sepal_length.min()),
        float(df.sepal_length.max()),
        float(df.sepal_length.mean())
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        float(df.sepal_width.min()),
        float(df.sepal_width.max()),
        float(df.sepal_width.mean())
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        float(df.petal_length.min()),
        float(df.petal_length.max()),
        float(df.petal_length.mean())
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        float(df.petal_width.min()),
        float(df.petal_width.max()),
        float(df.petal_width.mean())
    )

    # Prediksi
    if st.button("Prediksi"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        st.success(f"🌼 Prediksi jenis iris: **{prediction}**")
        st.info(f"Akurasi model: {accuracy*100:.2f}%")

    # Tampilkan dataset
    if st.checkbox("Tampilkan Dataset Iris"):
        st.dataframe(df)

# Run app
if __name__ == "__main__":
    main()
