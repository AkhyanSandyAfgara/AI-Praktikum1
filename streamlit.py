import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="K-Means Clustering", layout="centered")

st.title("📊 Aplikasi K-Means Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset")
    st.dataframe(df)

    # Pilih kolom numerik
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    selected_columns = st.multiselect(
        "Pilih kolom untuk clustering",
        numeric_columns,
        default=numeric_columns[:2]
    )

    if len(selected_columns) >= 2:
        X = df[selected_columns]

        # Input jumlah cluster
        k = st.slider("Jumlah Cluster (K)", 2, 10, 3)

        if st.button("Proses Clustering"):
            model = KMeans(n_clusters=k, random_state=42)
            df["Cluster"] = model.fit_predict(X)

            st.success("Clustering berhasil!")

            st.write("### Hasil Clustering")
            st.dataframe(df)

            # Visualisasi (2D)
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                ax.scatter(
                    df[selected_columns[0]],
                    df[selected_columns[1]],
                    c=df["Cluster"]
                )
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title("Visualisasi Cluster")

                st.pyplot(fig)

    else:
        st.warning("Pilih minimal 2 kolom numerik!")