
# Sleep Staging (R&K / AASM) — End-to-End Pipeline

This repository provides an end-to-end pipeline to classify sleep stages from EEG (and optionally EOG/EMG) using the Sleep-EDF format:

1. Ingest `.rec`/`.edf` polysomnography signals and `.hyp` hypnograms
2. Synchronize contiguous 30-second epochs with labels
3. Extract robust spectral features (Welch bandpowers + entropy)
4. Train multiple ML models with subject-wise cross-validation and evaluate

## Highlights
- **Subject-wise GroupKFold** to avoid train-test leakage across files/subjects
- **Class balancing** and **hyperparameter tuning**
- **Welch PSD** features for speed and robustness; optional STFT/PWVD hooks
- **Metrics**: Accuracy, Macro F1, Cohen’s kappa, per-class report, confusion matrices

> Note: Accuracy depends on dataset quality, channels, and label balance. This repo is tuned for generalization.

## Quickstart

```bash
# 1) Create and activate virtual env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Put Sleep-EDF files in data/raw/
#    Example: sc4002e0.rec / sc4002e0.hyp (or .edf)

# 4) Run full pipeline
bash scripts/run_pipeline.sh
# Or: run step-by-step
python src/sync_epochs.py --data_dir data/raw --out_pickle data/intermediate/synced_epochs.pkl
python src/extract_features.py --in_pickle data/intermediate/synced_epochs.pkl --out_csv data/processed/features.csv
python src/train_models.py --features data/processed/features.csv --reports_dir artifacts/reports --models_dir artifacts/models
```

## Config
Adjust `config.yaml` to control epoch length, channels, band definitions, and label mapping (R&K vs AASM).

## Data
Place your Sleep-EDF `.rec/.edf` and `.hyp` files in `data/raw/`. The pipeline pairs files by stem (`sc4002e0.rec` ↔ `sc4002e0.hyp`).

## Results
- Best model saved to `artifacts/models/`
- Reports (metrics, confusion matrices) in `artifacts/reports/`

## License
MIT
