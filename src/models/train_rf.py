import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def main(data: str, model_path: str, test_size: float = 0.3, random_state: int = 42):
    df = pd.read_csv(data)
    y = df["label"].values
    X = df.drop(columns=["label"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]
    print("RF metrics:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print("Saved model ->", model_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.model, args.test_size, args.random_state)
