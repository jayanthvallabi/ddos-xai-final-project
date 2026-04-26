import argparse, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

def main(in_dir: str, out: str):
    in_dir = Path(in_dir)
    # Expect preprocessed NSL-KDD CSVs (or convert from original .txt elsewhere)
    # Minimal example expects a single CSV with 'label' column (normal vs attack)
    # TODO: implement actual NSL-KDD parsing if needed.
    # For now, try to find a CSV in the folder.
    csvs = list(in_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {in_dir}.")
    df = pd.read_csv(csvs[0])
    # Basic cleaning
    df = df.dropna().reset_index(drop=True)
    # Encode label to binary {0,1}: normal=0, attack=1
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column with normal/attack values.")
    df["label"] = df["label"].str.lower().map(lambda x: 0 if "normal" in x else 1)
    # Separate features/labels
    y = df["label"].values
    X = df.drop(columns=["label"])
    # One-hot for any object/categorical columns
    X = pd.get_dummies(X, drop_first=True)
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X["label"] = y
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out, index=False)
    print(f"Wrote {out} with shape {X.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.in_dir, args.out)
