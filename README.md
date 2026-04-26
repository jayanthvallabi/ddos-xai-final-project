# DDoS Mitigation Using Explainable AI (SHAP)

This repo scaffolds the project work: build Random Forest / Isolation Forest detectors, explain predictions with SHAP, and evaluate on NSL-KDD and CIC-DDoS2019.

## Quickstart (Local)

1. **Create env & install deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Place datasets**
   - Put NSL-KDD CSVs in `data/raw/nsl_kdd/` (e.g., `KDDTrain+.txt`, `KDDTest+.txt` preprocessed to CSV).
   - Put CIC-DDoS2019 CSV(s) in `data/raw/cicddos2019/` (convert PCAPs via CICFlowMeter or equivalent).

3. **Preprocess**
   ```bash
   python -m src.data.prepare_nsl_kdd --in_dir data/raw/nsl_kdd --out data/processed/nsl_kdd.csv
   python -m src.data.prepare_cicddos2019 --in_dir data/raw/cicddos2019 --out data/processed/cicddos2019.csv
   ```

4. **Train models**
   ```bash
   python -m src.models.train_rf --data data/processed/nsl_kdd.csv --model artifacts/models/rf_nsl.joblib
   python -m src.models.train_if --data data/processed/nsl_kdd.csv --model artifacts/models/if_nsl.joblib
   ```

5. **Evaluate & compare**
   ```bash
   python -m src.evaluation.evaluate --data data/processed/nsl_kdd.csv --models artifacts/models/rf_nsl.joblib artifacts/models/if_nsl.joblib --report artifacts/figures/nsl_eval.txt
   ```

6. **Explain with SHAP**
   ```bash
   python -m src.explain.shap_plots --data data/processed/nsl_kdd.csv --model artifacts/models/rf_nsl.joblib --out_dir artifacts/figures/shap_rf_nsl
   ```

7. **Repeat for CIC-DDoS2019** by swapping dataset paths and model filenames.

## Project flow (what each script does)

- `src/data/prepare_nsl_kdd.py`: loads raw NSL-KDD, cleans, encodes labels (binary: normal vs attack), normalizes numeric features, writes one CSV.
- `src/data/prepare_cicddos2019.py`: loads CIC-DDoS2019 CSV(s), ensures consistent columns, cleans, encodes labels, normalizes, writes one CSV.
- `src/models/train_rf.py`: trains RandomForestClassifier with a train/val split (stratified), saves model via joblib.
- `src/models/train_if.py`: trains IsolationForest (unsupervised), aligns decision function to attack/normal thresholding, saves model.
- `src/evaluation/evaluate.py`: prints accuracy/precision/recall/F1, confusion matrix, ROC-AUC for supervised models; for IF, reports precision/recall vs threshold.
- `src/explain/shap_plots.py`: makes SHAP summary and force plots for the trained model (tree-based SHAP for RF).

## Week-by-week guide

- **Weeks 1–2**: Finalize scope; gather datasets; run preprocessing; basic EDA in `notebooks/01_eda.ipynb`.
- **Weeks 3–4**: Implement baseline RF/IF; initial evaluation; start SHAP integration (summary/force plots).
- **Weeks 5–6**: Hyperparameter tuning; cross-dataset validation (train on NSL-KDD, test on CIC, and vice versa).
- **Weeks 7–8**: Refine visuals, write report sections; finalize demo.

> Go throgh the course proposal for official objectives, tools, datasets, and a weekly timeline.
