#!/usr/bin/env bash
set -e
source .venv/bin/activate || true
python -m src.data.prepare_nsl_kdd --in_dir data/raw/nsl_kdd --out data/processed/nsl_kdd.csv
python -m src.models.train_rf --data data/processed/nsl_kdd.csv --model artifacts/models/rf_nsl.joblib
python -m src.explain.shap_plots --data data/processed/nsl_kdd.csv --model artifacts/models/rf_nsl.joblib --out_dir artifacts/figures/shap_rf_nsl
python -m src.evaluation.evaluate --data data/processed/nsl_kdd.csv --models artifacts/models/rf_nsl.joblib --report artifacts/figures/nsl_eval.txt
