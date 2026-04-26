# DDoS Mitigation Using Explainable AI with SHAP

This is the cleaned submission package for the Explainable AI-based DDoS attack detection project.

## Project Goal
Build a machine-learning-based DDoS detection system and explain model decisions using SHAP.

## Main Components
- **Dataset:** NSL-KDD cleaned dataset (`data/processed/nsl_clean_full.csv`)
- **Models:** Random Forest and Isolation Forest
- **Explainability:** SHAP summary and force plots
- **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC/PR-AUC where applicable

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Example
```powershell
python -m src.models.train_rf --data data/processed/nsl_clean_full.csv --model artifacts/models/rf_nsl.joblib
python -m src.models.train_if --data data/processed/nsl_clean_full.csv --model artifacts/models/if_nsl.joblib
python -m src.evaluation.evaluate --data data/processed/nsl_clean_full.csv --models artifacts/models/rf_nsl.joblib artifacts/models/if_nsl.joblib --report artifacts/figures/nsl_eval.txt
python -m src.explain.shap_plots --data data/processed/nsl_clean_full.csv --model artifacts/models/rf_nsl.joblib --out_dir artifacts/figures/shap_rf_nsl
```

## Important Note
The `.venv` folder is intentionally not included. It should be recreated using `requirements.txt`.
