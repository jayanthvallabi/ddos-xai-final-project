from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[2]

RF_MODEL_PATH = BASE / "artifacts" / "models" / "rf_nsl.joblib"
IF_MODEL_PATH = BASE / "artifacts" / "models" / "if_nsl.joblib"

FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login"
]

app = FastAPI(title="DDoS Live Detection API")

rf_model = joblib.load(RF_MODEL_PATH)
if_model = joblib.load(IF_MODEL_PATH)

class TrafficSample(BaseModel):
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    count: float = 0
    srv_count: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 0
    num_compromised: float = 0
    root_shell: float = 0
    su_attempted: float = 0
    num_root: float = 0
    num_file_creations: float = 0
    num_access_files: float = 0
    num_outbound_cmds: float = 0
    is_host_login: float = 0
    is_guest_login: float = 0

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(sample: TrafficSample):
    X = pd.DataFrame([sample.dict()], columns=FEATURES)

    rf_pred = int(rf_model.predict(X)[0])
    iso_raw = int(if_model.predict(X)[0])
    iso_pred = 1 if iso_raw == -1 else 0

    return {
        "RandomForest": "attack" if rf_pred == 1 else "normal",
        "IsolationForest": "attack" if iso_pred == 1 else "normal"
    }