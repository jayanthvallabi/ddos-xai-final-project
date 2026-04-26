import json
import pathlib

import joblib
import pandas as pd
import streamlit as st
import requests

BASE = pathlib.Path(__file__).resolve().parent

DATA_PATH = BASE / "data" / "processed" / "nsl_clean_full.csv"
RF_MODEL_PATH = BASE / "artifacts" / "models" / "rf_nsl.joblib"
IF_MODEL_PATH = BASE / "artifacts" / "models" / "if_nsl.joblib"
REPORT_PATH = BASE / "artifacts" / "metrics" / "report.json"
PLOTS_DIR = BASE / "artifacts" / "figures" / "shap_rf_nsl_synth"

st.set_page_config(page_title="DDoS XAI Dashboard", layout="wide")

st.title("Explainable AI-Based DDoS Detection Dashboard")
st.write("Random Forest, Isolation Forest, and SHAP-based explainability for NSL-KDD network traffic.")

FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login"
]

# Load data
st.header("1. Dataset Preview")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success(f"Loaded dataset: {DATA_PATH}")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    rows = st.slider("Rows to preview", 5, 100, 10)
    st.dataframe(df.head(rows))
else:
    st.error("Dataset not found. Expected: data/processed/nsl_clean_full.csv")
    df = None

# Load models
st.header("2. Model Status")

rf_model = None
if_model = None

col1, col2 = st.columns(2)

with col1:
    if RF_MODEL_PATH.exists():
        rf_model = joblib.load(RF_MODEL_PATH)
        st.success("Random Forest model loaded")
    else:
        st.error("Random Forest model not found. Train it first.")

with col2:
    if IF_MODEL_PATH.exists():
        if_model = joblib.load(IF_MODEL_PATH)
        st.success("Isolation Forest model loaded")
    else:
        st.warning("Isolation Forest model not found. Train it first.")

# Metrics
st.header("3. Evaluation Metrics")

def parse_metrics_report(text):
    results = {}
    current_model = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("==") and line.endswith("=="):
            current_model = line.replace("=", "").strip()
            results[current_model] = {}

        elif ":" in line and current_model:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                results[current_model][key] = float(value)
            except:
                results[current_model][key] = value

    return results


if REPORT_PATH.exists() and REPORT_PATH.stat().st_size > 0:
    report_text = REPORT_PATH.read_text()
    metrics = parse_metrics_report(report_text)

    if metrics:
        for model_name, vals in metrics.items():
            st.subheader(model_name)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", round(vals.get("Accuracy", 0), 4))
            c2.metric("Precision", round(vals.get("Precision", 0), 4))
            c3.metric("Recall", round(vals.get("Recall", 0), 4))
            c4.metric("F1 Score", round(vals.get("F1", 0), 4))

            chart_data = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1"],
                "Score": [
                    vals.get("Accuracy", 0),
                    vals.get("Precision", 0),
                    vals.get("Recall", 0),
                    vals.get("F1", 0),
                ]
            })

            st.bar_chart(chart_data.set_index("Metric"))

            if "ROC-AUC" in vals:
                st.metric("ROC-AUC", round(vals.get("ROC-AUC", 0), 4))

            st.divider()
    else:
        st.warning("Could not parse evaluation metrics. Showing raw report:")
        st.text(report_text)

else:
    st.warning("Evaluation report not found or empty. Run evaluation first.")

# Manual Prediction
st.header("4. Manual Traffic Prediction")

st.write("Enter one traffic sample manually. This does not change the dataset; it only tests a new input.")

manual_values = {}

cols = st.columns(3)
for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        manual_values[feature] = st.number_input(feature, value=0.0)

sample_df = pd.DataFrame([manual_values])

if st.button("Predict Manual Sample"):
    if rf_model is not None:
        rf_pred = rf_model.predict(sample_df)[0]
        st.subheader("Random Forest Prediction")
        st.write("Prediction:", "Attack" if rf_pred == 1 else "Normal")

    if if_model is not None:
        iso_pred = if_model.predict(sample_df)[0]
        iso_label = 1 if iso_pred == -1 else 0
        st.subheader("Isolation Forest Prediction")
        st.write("Prediction:", "Attack / Anomaly" if iso_label == 1 else "Normal")

# Bulk prediction
st.header("5. Bulk Prediction from Dataset")

if df is not None and rf_model is not None:
    n = st.slider("Number of rows to test", 5, 200, 20)

    if st.button("Run Bulk Prediction"):
        X = df[FEATURES].head(n)

        rf_preds = rf_model.predict(X)

        result_df = X.copy()
        result_df["RandomForest_Prediction"] = rf_preds
        result_df["RandomForest_Label"] = result_df["RandomForest_Prediction"].map({
            0: "Normal",
            1: "Attack"
        })

        if if_model is not None:
            iso_preds = if_model.predict(X)
            iso_labels = [1 if p == -1 else 0 for p in iso_preds]
            result_df["IsolationForest_Prediction"] = iso_labels
            result_df["IsolationForest_Label"] = result_df["IsolationForest_Prediction"].map({
                0: "Normal",
                1: "Attack / Anomaly"
            })

        st.dataframe(result_df)

# Plots
st.header("6. Visual Results and SHAP Explainability")

if PLOTS_DIR.exists():
    image_files = list(PLOTS_DIR.glob("*.png"))

    if image_files:
        for img in image_files:
            st.subheader(img.name)
            st.image(str(img), use_container_width=True)
    else:
        st.warning("No plot images found in artifacts/plots.")
else:
    st.warning("Plots folder not found.")
st.header("7. Live API Prediction Test")

api_url = st.text_input("API URL", "http://127.0.0.1:8000/predict")

if st.button("Send Sample to Live API"):
    sample = {
        "duration": 1,
        "src_bytes": 5000,
        "dst_bytes": 200,
        "count": 300,
        "srv_count": 250,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0
    }

    try:
        response = requests.post(api_url, json=sample)
        st.json(response.json())
    except Exception as e:
        st.error(f"API connection failed: {e}")