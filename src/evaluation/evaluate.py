# src/evaluation/evaluate.py
import argparse, joblib, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

def eval_rf_like(df, model):
    """For supervised classifiers (e.g., RandomForest)."""
    y = df["label"].values
    X = df.drop(columns=["label"])
    preds = model.predict(X)
    out = []
    out.append(f"Accuracy: {accuracy_score(y, preds):.4f}")
    out.append(f"Precision: {precision_score(y, preds):.4f}")
    out.append(f"Recall: {recall_score(y, preds):.4f}")
    out.append(f"F1: {f1_score(y, preds):.4f}")
    try:
        proba = model.predict_proba(X)[:, 1]
        out.append(f"ROC-AUC: {roc_auc_score(y, proba):.4f}")
    except Exception:
        pass
    out.append("Confusion Matrix:")
    out.append(str(confusion_matrix(y, preds)))
    return "\n".join(out)

def eval_isolation_forest(df, model):
    """For IsolationForest (unsupervised). Map predict() => -1 anomaly -> attack=1."""
    y = df["label"].values  # 0 normal, 1 attack
    X = df.drop(columns=["label"])
    preds = (model.predict(X) == -1).astype(int)
    out = []
    out.append("IF metrics (threshold at -1):")
    out.append(f"Accuracy: {accuracy_score(y, preds):.4f}")
    out.append(f"Precision: {precision_score(y, preds):.4f}")
    out.append(f"Recall: {recall_score(y, preds):.4f}")
    out.append(f"F1: {f1_score(y, preds):.4f}")
    out.append("Confusion Matrix:")
    out.append(str(confusion_matrix(y, preds)))
    return "\n".join(out)

def main(data, models, report):
    df = pd.read_csv(data)
    lines = []
    for mpath in models:
        model = joblib.load(mpath)
        name = type(model).__name__
        lines.append(f"== {mpath} ==")
        try:
            if name == "IsolationForest":
                lines.append(eval_isolation_forest(df, model))
            else:
                lines.append(eval_rf_like(df, model))
        except Exception as e:
            lines.append(f"(Evaluation failed: {e})")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    # ✅ ensure parent dir exists
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        f.write(text)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    main(args.data, args.models, args.report)
