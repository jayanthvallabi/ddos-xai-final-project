import argparse, pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main(in_dir: str, out: str):
    in_dir = Path(in_dir)
    csvs = list(in_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {in_dir}. Place CIC-DDoS2019 CSVs here.")
    # Concatenate all CSVs with shared columns
    dfs = [pd.read_csv(p) for p in csvs]
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    dfs = [d[list(common_cols)].copy() for d in dfs]
    df = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)
    # Expect a label column naming normal vs attack
    label_col = None
    for c in df.columns:
        if c.lower() in ("label", "class"):
            label_col = c
            break
    if label_col is None:
        raise ValueError("Expected a label/class column in CIC-DDoS2019 CSV(s).")
    y = (df[label_col].astype(str).str.lower().str.contains("attack|ddos|flood")).astype(int).values
    X = df.drop(columns=[label_col])
    X = pd.get_dummies(X, drop_first=True)
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
