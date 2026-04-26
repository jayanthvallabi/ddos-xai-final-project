import pandas as pd
import requests
import time
import argparse

FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login"
]

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/processed/nsl_clean_full.csv")
parser.add_argument("--api", default="http://127.0.0.1:8000/predict")
parser.add_argument("--delay", type=float, default=1.0)
parser.add_argument("--rows", type=int, default=20)
args = parser.parse_args()

df = pd.read_csv(args.csv)

print("Starting simulated live traffic stream...")

for i, row in df.head(args.rows).iterrows():
    payload = {feature: float(row[feature]) for feature in FEATURES}

    try:
        response = requests.post(args.api, json=payload)
        print(f"\nRow {i}")
        print(response.json())
    except Exception as e:
        print("Error:", e)

    time.sleep(args.delay)

print("\nLive simulation completed.")