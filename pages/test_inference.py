import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Inference", page_icon="🎯", layout="wide")

ART = "./trained_models"

st.title("🎯 Model Inference (Upload Test Data)")

uploaded_file = st.file_uploader("📂 Upload Test CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Input shape:", df.shape)

    # ================= LOAD ARTIFACTS =================
    scaler = joblib.load(f"{ART}/scaler.pkl")
    median = joblib.load(f"{ART}/median.pkl")
    features = joblib.load(f"{ART}/features.pkl")

    # ================= FEATURE ENGINEERING =================
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    df["release_year"] = df["ReleaseDate"].dt.year.fillna(0)
    df["game_age_days"] = (pd.Timestamp.today() - df["ReleaseDate"]).dt.days.fillna(0)
    df.drop(columns=["ReleaseDate"], inplace=True)

    df["discount_ratio"] = (df["PriceInitial"] - df["PriceFinal"]) / (df["PriceInitial"] + 1e-9)

    # ================= ALIGN FEATURES =================
    df = df.fillna(median)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]

    # ================= SCALE =================
    X_scaled = scaler.transform(X)

    st.success("✔ Preprocessing applied successfully")

    st.write("Sample output matrix shape:", X_scaled.shape)

    # (optional placeholder for model later)
    st.info("👉 Connect trained model here (LightGBM / CatBoost / etc.)")