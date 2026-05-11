import streamlit as st
import pandas as pd
import numpy as np
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Train Pipeline", page_icon="⚙️", layout="wide")

OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

st.title("⚙️ Training Pipeline (Upload + Preprocess + Save)")

uploaded_file = st.file_uploader("📂 Upload Training CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded: {df.shape}")

    # ================= TARGET =================
    df["target_log"] = np.log1p(df["GamePopularity_enc"])

    # ================= DATE FEATURES =================
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    df["release_year"] = df["ReleaseDate"].dt.year.fillna(df["ReleaseDate"].dt.year.median())
    df["game_age_days"] = (pd.Timestamp.today() - df["ReleaseDate"]).dt.days.fillna(0)
    df.drop(columns=["ReleaseDate"], inplace=True)

    # ================= SIMPLE FEATURES =================
    df["discount_ratio"] = (df["PriceInitial"] - df["PriceFinal"]) / (df["PriceInitial"] + 1e-9)

    # ================= DROP OBJECT =================
    obj_cols = df.select_dtypes(include="object").columns
    df.drop(columns=obj_cols, inplace=True)

    X = df.drop(columns=["GamePopularity_enc", "target_log"], errors="ignore")
    y = df["target_log"]

    # ================= SPLIT =================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # ================= FILL =================
    median = X_train.median()
    X_train = X_train.fillna(median)
    X_test = X_test.fillna(median)

    # ================= OUTLIERS =================
    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X_train) == 1
    X_train, y_train = X_train[mask], y_train[mask]

    # ================= SCALING =================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ================= SAVE ARTIFACTS =================
    joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
    joblib.dump(iso, f"{OUT_DIR}/iso.pkl")
    joblib.dump(median, f"{OUT_DIR}/median.pkl")
    joblib.dump(list(X.columns), f"{OUT_DIR}/features.pkl")

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)

    st.success("✔ Training Completed + Saved in /artifacts")