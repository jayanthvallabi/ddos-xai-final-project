import argparse, joblib, pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt

def main(data: str, model_path: str, out_dir: str):
    df = pd.read_csv(data)
    y = df["label"].values
    X = df.drop(columns=["label"])
    model = joblib.load(model_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Use TreeExplainer when possible (e.g., RandomForest)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    summary_path = f"{out_dir}/summary.png"
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # Force plot for first 1 sample
    try:
        # Save as image by using matplotlib rendering
        sample_idx = 0
        shap.plots.force(shap_values[sample_idx], matplotlib=True, show=False)
        plt.tight_layout()
        force_path = f"{out_dir}/force_sample0.png"
        plt.savefig(force_path, dpi=200)
        plt.close()
    except Exception as e:
        with open(f"{out_dir}/force_plot_error.txt", "w") as f:
            f.write(str(e))

    print("Wrote:", summary_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.data, args.model, args.out_dir)
