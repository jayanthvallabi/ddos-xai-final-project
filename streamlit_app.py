import json
import pathlib
import shap
import matplotlib.pyplot as plt
import numpy as np

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
# ---------------------------------------------------------
# Bulk Prediction from Dataset
# ---------------------------------------------------------

st.header("5. Bulk Prediction from Dataset")

st.write(
    "This section runs predictions on multiple traffic records from the dataset. "
    "It shows how both models classify each row as Normal or Attack."
)

if df is not None and rf_model is not None:

    n = st.slider("Number of rows to test", 5, 200, 20)

    if st.button("Run Bulk Prediction"):

        X = df[FEATURES].head(n).copy()

        result_df = X.copy()

        # Random Forest prediction
        rf_preds = rf_model.predict(X)
        result_df["RF Prediction"] = [
            "Attack" if pred == 1 else "Normal" for pred in rf_preds
        ]

        # Random Forest probability
        if hasattr(rf_model, "predict_proba"):
            rf_probs = rf_model.predict_proba(X)
            result_df["RF Attack Probability"] = [
                f"{prob[1] * 100:.2f}%" for prob in rf_probs
            ]

        # Isolation Forest prediction
        if if_model is not None:
            iso_preds = if_model.predict(X)

            result_df["IF Prediction"] = [
                "Attack / Anomaly" if pred == -1 else "Normal"
                for pred in iso_preds
            ]

        st.subheader("Bulk Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        # Summary cards
        st.subheader("Prediction Summary")

        rf_attack_count = result_df["RF Prediction"].value_counts().get("Attack", 0)
        rf_normal_count = result_df["RF Prediction"].value_counts().get("Normal", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Tested", n)
        c2.metric("RF Attacks Detected", rf_attack_count)
        c3.metric("RF Normal Traffic", rf_normal_count)

        if if_model is not None:
            iso_attack_count = result_df["IF Prediction"].value_counts().get("Attack / Anomaly", 0)
            iso_normal_count = result_df["IF Prediction"].value_counts().get("Normal", 0)

            c4, c5 = st.columns(2)
            c4.metric("IF Anomalies Detected", iso_attack_count)
            c5.metric("IF Normal Traffic", iso_normal_count)

        st.info(
            "Random Forest is supervised and usually performs better on labeled NSL-KDD data. "
            "Isolation Forest is unsupervised and is useful for detecting unusual traffic patterns."
        )

else:
    st.warning("Dataset or Random Forest model is not loaded.")

# Plots
# ---------------------------------------------------------
# Global SHAP Summary — Optional
# ---------------------------------------------------------

# st.header("6. Global SHAP Summary (Optional)")

# st.write(
#     "This section shows overall feature importance across the dataset. "
#     "It is a global explanation of how the model behaves."
# )

# summary_found = False

# if PLOTS_DIR.exists():
#     summary_path = PLOTS_DIR / "summary.png"

#     if summary_path.exists():
#         st.image(str(summary_path), use_container_width=True)
#         st.info(
#             "This plot shows global feature importance. "
#             "Use the Advanced XAI section below for per-sample explanation."
#         )
#         summary_found = True

# if not summary_found:
#     st.warning("Global SHAP summary not found. Advanced XAI section will still work.")

# ---------------------------------------------------------
# Advanced XAI Explanation
# ---------------------------------------------------------
api_url = st.text_input(
    "API URL",
    value="https://ddos-xai-final-project.onrender.com/predict",
    key="live_cloud_api_url"
)
st.header("8. Advanced XAI Explanation")

st.write(
    "This section explains why a selected traffic record is classified as "
    "Attack or Normal using Random Forest probability and SHAP feature contributions."
)

if df is not None and rf_model is not None:

    X_all = df[FEATURES].copy()

    selected_row = st.number_input(
        "Select a traffic row to explain",
        min_value=0,
        max_value=len(X_all) - 1,
        value=0,
        step=1
    )

    if st.button("Generate Advanced XAI Explanation"):

        try:
            sample = X_all.iloc[[selected_row]]

            # Prediction
            prediction = int(rf_model.predict(sample)[0])

            if hasattr(rf_model, "predict_proba"):
                proba = rf_model.predict_proba(sample)[0]
                normal_prob = float(proba[0])
                attack_prob = float(proba[1])
            else:
                normal_prob = 0.0
                attack_prob = 1.0 if prediction == 1 else 0.0

            prediction_label = "Attack" if prediction == 1 else "Normal"

            st.subheader("Prediction Result")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", prediction_label)
            c2.metric("Attack Probability", f"{attack_prob * 100:.2f}%")
            c3.metric("Normal Probability", f"{normal_prob * 100:.2f}%")

            # SHAP explanation
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(sample)

            if isinstance(shap_values, list):
                shap_for_attack = shap_values[1][0]
            else:
                arr = np.array(shap_values)

                # Handles newer SHAP output shapes
                if arr.ndim == 3:
                    shap_for_attack = arr[0, :, 1]
                elif arr.ndim == 2:
                    shap_for_attack = arr[0]
                else:
                    shap_for_attack = arr

            contribution_df = pd.DataFrame({
                "Feature": FEATURES,
                "Feature Value": sample.iloc[0].values,
                "SHAP Value": shap_for_attack
            })

            contribution_df["Absolute Impact"] = contribution_df["SHAP Value"].abs()
            contribution_df = contribution_df.sort_values(
                by="Absolute Impact",
                ascending=False
            )

            st.subheader("Top Features Influencing This Prediction")
            st.dataframe(contribution_df.head(10), use_container_width=True)

            # Positive and negative contributors
            positive_df = contribution_df[contribution_df["SHAP Value"] > 0].head(5)
            negative_df = contribution_df[contribution_df["SHAP Value"] < 0].head(5)

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Features Pushing Toward Attack")
                if len(positive_df) > 0:
                    st.dataframe(
                        positive_df[["Feature", "Feature Value", "SHAP Value"]],
                        use_container_width=True
                    )
                else:
                    st.info("No strong positive contributors found.")

            with col_b:
                st.subheader("Features Pushing Toward Normal")
                if len(negative_df) > 0:
                    st.dataframe(
                        negative_df[["Feature", "Feature Value", "SHAP Value"]],
                        use_container_width=True
                    )
                else:
                    st.info("No strong negative contributors found.")

            # Visual bar chart
            st.subheader("SHAP Contribution Chart")

            chart_df = contribution_df.head(10)[["Feature", "SHAP Value"]].set_index("Feature")
            st.bar_chart(chart_df)

            # Natural language explanation
            st.subheader("Natural Language Explanation")

            top_features = contribution_df.head(3)["Feature"].tolist()

            if prediction == 1:
                explanation = (
                    f"The model predicted this traffic record as an Attack mainly because "
                    f"features such as {', '.join(top_features)} had strong influence on the prediction."
                )
                action = (
                    "Recommended Action: Monitor this traffic source, check repeated connection attempts, "
                    "apply rate limiting if needed, and investigate abnormal traffic behavior."
                )
                st.error(explanation)
                st.warning(action)
            else:
                explanation = (
                    f"The model predicted this traffic record as Normal because the most influential features "
                    f"({', '.join(top_features)}) did not strongly indicate attack behavior."
                )
                action = (
                    "Recommended Action: Continue monitoring. No immediate blocking action is required."
                )
                st.success(explanation)
                st.info(action)

            # Optional waterfall plot
            st.subheader("SHAP Waterfall Plot")

            try:
                expected_value = explainer.expected_value

                if isinstance(expected_value, (list, np.ndarray)):
                    base_value = expected_value[1]
                else:
                    base_value = expected_value

                shap_exp = shap.Explanation(
                    values=shap_for_attack,
                    base_values=base_value,
                    data=sample.iloc[0].values,
                    feature_names=FEATURES
                )

                fig = plt.figure(figsize=(10, 5))
                shap.plots.waterfall(shap_exp, show=False)
                st.pyplot(fig)
                plt.close(fig)

            except Exception as waterfall_error:
                st.warning(f"Waterfall plot could not be generated: {waterfall_error}")

        except Exception as e:
            st.error(f"Advanced XAI explanation failed: {e}")

else:
    st.warning("Dataset or Random Forest model is not loaded.")

# ---------------------------------------------------------
# Live Cloud API Prediction Test
# ---------------------------------------------------------

st.header("7. Live Cloud API Prediction Test")

st.write(
    "This section sends a sample traffic record to the deployed FastAPI backend on Render "
    "and returns live predictions from the cloud model."
)

api_url = st.text_input(
    "API URL",
    "https://ddos-xai-final-project.onrender.com/predict"
)

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
        response = requests.post(api_url, json=sample, timeout=30)

        if response.status_code == 200:
            st.success("Cloud API response received successfully.")
            st.json(response.json())
        else:
            st.error(f"API returned error code: {response.status_code}")
            st.text(response.text)

    except Exception as e:
        st.error(f"API connection failed: {e}")