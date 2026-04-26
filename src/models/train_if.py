import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import IsolationForest
from pathlib import Path
import numpy as np

def main(data: str, model_path: str, contamination: float = 0.1, random_state: int = 42):
    df = pd.read_csv(data)
    y = df["label"].values  # 0 normal, 1 attack
    X = df.drop(columns=["label"])

    # Train on normal traffic only for IF
    X_train = X[y == 0]
    if X_train.empty:
        raise ValueError("No normal samples found for IsolationForest training.")
    clf = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train)

    # Decision: anomaly -> attack (1), inlier -> normal (0)
    preds = (clf.predict(X) == -1).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    print("IF metrics (threshold at -1):")
    print("Precision:", p, "Recall:", r, "F1:", f)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print("Saved model ->", model_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--contamination", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.model, args.contamination, args.random_state)
