
import os
import argparse
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame, drop_classes=("Movement", "Undefined", "Unknown")):
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError("Features CSV must include 'label' column.")

    # Drop unwanted classes
    if drop_classes:
        drop_set = set([c for c in drop_classes if c is not None])
        if drop_set:
            df = df[~df["label"].isin(drop_set)]

    # Groups by file (for subject-wise CV)
    groups = df["file"].values if "file" in df.columns else np.arange(len(df))

    # Features
    feature_cols = [c for c in df.columns if c not in ["file", "epoch_idx", "label", "sfreq", "duration_sec"]]
    X = df[feature_cols].values
    y = df["label"].values

    return X, y, groups, feature_cols


def evaluate_and_plot(y_true, y_pred, labels_order, out_dir_prefix):
    os.makedirs(os.path.dirname(out_dir_prefix), exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    # Save metrics
    with open(out_dir_prefix + "_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1_macro, "kappa": kappa}, f, indent=2)
    with open(out_dir_prefix + "_report.txt", "w") as f:
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir_prefix + "_confusion_matrix.png")
    plt.close()

    print(f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | kappa={kappa:.4f}")
    return acc, f1_macro, kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, help="CSV with extracted features")
    parser.add_argument("--reports_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.features)
    X, y, groups, feature_cols = prepare_data(
        df, drop_classes=("Movement" if cfg.get("drop_movement", True) else None,
                          "Undefined" if cfg.get("drop_undefined", True) else None,
                          "Unknown")
    )

    labels_order = sorted(pd.Series(y).unique().tolist())

    # Pipelines
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", multi_class="auto", random_state=RANDOM_STATE))
    ])
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE))
    ])
    xgb_pipe = Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE
        ))
    ])

    # Hyperparameter grids
    lr_grid = {
        "clf__C": np.logspace(-2, 2, 10)
    }
    rf_grid = {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10]
    }
    xgb_grid = {
        "clf__n_estimators": [400, 800, 1200],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
        "clf__reg_lambda": [0.0, 0.5, 1.0]
    }

    # GroupKFold CV for fairness across subjects/files
    gkf = GroupKFold(n_splits=cfg.get("n_splits_group_kfold", 5))
    scoring = "f1_macro"

    searches = [
        ("logreg", lr_pipe, lr_grid),
        ("random_forest", rf_pipe, rf_grid),
        ("xgboost", xgb_pipe, xgb_grid)
    ]

    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    best_overall = {"name": None, "score": -np.inf, "model": None}

    for name, pipe, grid in searches:
        print(f"
=== Tuning {name} ===")
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=min(20, int(np.prod([len(v) for v in grid.values()]))),
            cv=gkf.split(X, y, groups),
            scoring=scoring,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
            refit=True
        )
        rs.fit(X, y)
        best_model = rs.best_estimator_
        print(f"Best params for {name}: {rs.best_params_}")

        # CV predictions via StratifiedKFold for quick evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        y_true_all, y_pred_all = [], []
        for tr, te in skf.split(X, y):
            best_model.fit(X[tr], y[tr])
            y_pred = best_model.predict(X[te])
            y_true_all.extend(y[te])
            y_pred_all.extend(y_pred)
        out_prefix = os.path.join(args.reports_dir, f"{name}")
        acc, f1m, kappa = evaluate_and_plot(y_true_all, y_pred_all, labels_order, out_prefix)

        # Save model
        model_path = os.path.join(args.models_dir, f"{name}_best.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved model -> {model_path}")

        if f1m > best_overall["score"]:
            best_overall = {"name": name, "score": f1m, "model": best_model}

    # Save summary
    summary_path = os.path.join(args.reports_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"best_model": best_overall["name"], "best_f1_macro": best_overall["score"]}, f, indent=2)
    print(f"
Best model: {best_overall['name']} with F1-macro={best_overall['score']:.4f}")


if __name__ == "__main__":
    main()
