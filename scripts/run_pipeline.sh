
#!/usr/bin/env bash
set -e

DATA_DIR="data/raw"
INTERMEDIATE="data/intermediate/synced_epochs.pkl"
FEATURES_CSV="data/processed/features.csv"
REPORTS_DIR="artifacts/reports"
MODELS_DIR="artifacts/models"

python src/sync_epochs.py --data_dir "$DATA_DIR" --out_pickle "$INTERMEDIATE" --config config.yaml
python src/extract_features.py --in_pickle "$INTERMEDIATE" --out_csv "$FEATURES_CSV" --config config.yaml
python src/train_models.py --features "$FEATURES_CSV" --reports_dir "$REPORTS_DIR" --models_dir "$MODELS_DIR" --config config.yaml
