# app.py
import streamlit as st
import pandas as pd
import joblib
import re
import unicodedata
import numpy as np


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s,\.!?\-\'\"]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# Load Kaggle-trained pipeline
# ----------------------------
MODEL_PATH = "model_pipeline.pkl"  # Place the .pkl in the same folder
pipe = joblib.load(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection App")
st.write("Predict whether news is Fake (0) or Real (1) using the trained model.")

# Input choice
input_type = st.sidebar.radio("Select input type:", ("Single News Text", "Upload CSV"))

# Threshold slider
st.sidebar.subheader("Decision Threshold")
threshold = st.sidebar.slider(
    "Set probability threshold for classifying as Real news",
    0.0, 1.0, 0.5, 0.01
)

# ----------------------------
# Single text input
# ----------------------------
if input_type == "Single News Text":
    news_text = st.text_area("Enter the news text here:")
    if st.button("Predict"):
        if news_text.strip() == "":
            st.error("Please enter some text to predict.")
        else:
            prob = pipe.predict_proba([news_text])[0, 1]
            pred = 1 if prob >= threshold else 0
            label = "Real 🟢" if pred == 1 else "Fake 🔴"
            st.subheader(f"Prediction: {label}")
            st.write(f"Probability of being Real news: {prob:.3f}")

# ----------------------------
# CSV input
# ----------------------------
else:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Predict CSV"):
                probabilities = pipe.predict_proba(df["text"])[:, 1]
                predictions = [1 if p >= threshold else 0 for p in probabilities]
                df["Prediction"] = ["Real 🟢" if p == 1 else "Fake 🔴" for p in predictions]
                df["Probability_Real"] = probabilities

                st.subheader("Prediction Results:")
                st.dataframe(df)

                st.write("Summary Counts:")
                st.write(df["Prediction"].value_counts())

# Footer
st.markdown("---")
st.markdown(
    "⚡ This app uses a Kaggle-trained model. "
    "Set the threshold to adjust sensitivity for Real vs Fake news."
)
