import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/ton_iot/network_v1.csv"
MODEL_PATH = ROOT / "models/network_v2_isolation_forest.joblib"
REPORT_PATH = ROOT / "audit/network_v2_evaluation.json"
PREDICTIONS_PATH = ROOT / "audit/network_v2_holdout_predictions.csv"

RANDOM_STATE = 42
CALIBRATION_PERCENTILE = 95


def main():
    print("=" * 80)
    print("NETWORK IDS — CHRONOLOGICAL EVALUATION V2")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)

    required = {"window_start_ts", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["window_start_ts"] = pd.to_numeric(
        df["window_start_ts"], errors="raise"
    )

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")

    df = (
        df.sort_values("window_start_ts")
        .reset_index(drop=True)
    )

    if df["window_start_ts"].duplicated().any():
        raise ValueError("Duplicate windows detected.")

    # Chronological 70 / 15 / 15 split.
    n = len(df)
    train_end = int(n * 0.70)
    calibration_end = int(n * 0.85)

    train_period = df.iloc[:train_end].copy()
    calibration_period = df.iloc[train_end:calibration_end].copy()
    holdout = df.iloc[calibration_end:].copy()

    # Only normal observations may fit or calibrate the detector.
    train = train_period[train_period["label"] == 0].copy()
    calibration = calibration_period[
        calibration_period["label"] == 0
    ].copy()

    if train.empty or calibration.empty:
        raise ValueError("Training or calibration has no normal windows.")

    if holdout["label"].nunique() != 2:
        raise ValueError(
            "The final holdout must contain both normal and attack windows."
        )

    # Remove labels, timestamps, and descriptive metadata.
    excluded = {
        "label",
        "attack_type",
        "type",
        "window_start_ts",
        "window_end_ts",
    }

    feature_columns = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    X_calibration = calibration[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    X_holdout = holdout[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    if X_train.isna().all().any():
        bad = X_train.columns[X_train.isna().all()].tolist()
        raise ValueError(f"Entirely missing training features: {bad}")

    print("\nDATA SPLIT")
    print("-" * 80)

    for name, part in [
        ("Training period", train_period),
        ("Calibration period", calibration_period),
        ("Holdout period", holdout),
    ]:
        print(
            f"{name:20s} "
            f"rows={len(part):6d} "
            f"normal={(part['label'] == 0).sum():6d} "
            f"attack={(part['label'] == 1).sum():6d} "
            f"range={part['window_start_ts'].min():.0f}"
            f"–{part['window_start_ts'].max():.0f}"
        )

    print(f"\nNormal training windows:     {len(train)}")
    print(f"Normal calibration windows:  {len(calibration)}")
    print(f"Holdout windows:             {len(holdout)}")
    print(f"Numeric features:            {len(feature_columns)}")

    # Train Isolation Forest on normal observations only.
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        (
            "detector",
            IsolationForest(
                n_estimators=300,
                contamination="auto",
                max_samples="auto",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTRAINING")
    print("-" * 80)

    model.fit(X_train)

    # Higher scores indicate greater anomaly likelihood.
    calibration_scores = -model.score_samples(X_calibration)
    holdout_scores = -model.score_samples(X_holdout)

    threshold = float(
        np.percentile(
            calibration_scores,
            CALIBRATION_PERCENTILE,
        )
    )

    y_true = holdout["label"].to_numpy()
    y_pred = (holdout_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, holdout_scores)),
        "pr_auc": float(
            average_precision_score(y_true, holdout_scores)
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0
        ),
        "true_negative_rate": float(
            tn / (tn + fp) if (tn + fp) else 0
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    report = {
        "experiment": "network_v2_chronological",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "model_type": "IsolationForest",
        "capability": "anomaly_detection",
        "feature_profile": "network_v1",
        "feature_names": feature_columns,
        "split": {
            "strategy": "chronological_70_15_15",
            "total_windows": int(n),
            "training_period": {
                "start_ts": float(train_period["window_start_ts"].min()),
                "end_ts": float(train_period["window_start_ts"].max()),
                "normal_windows_used": int(len(train)),
            },
            "calibration_period": {
                "start_ts": float(calibration_period["window_start_ts"].min()),
                "end_ts": float(calibration_period["window_start_ts"].max()),
                "normal_windows_used": int(len(calibration)),
            },
            "holdout_period": {
                "start_ts": float(holdout["window_start_ts"].min()),
                "end_ts": float(holdout["window_start_ts"].max()),
                "normal_windows": int((y_true == 0).sum()),
                "attack_windows": int((y_true == 1).sum()),
            },
        },
        "calibration": {
            "method": "normal_score_percentile",
            "percentile": CALIBRATION_PERCENTILE,
            "threshold": threshold,
            "normal_calibration_windows": int(len(calibration)),
            "observed_calibration_flag_rate": float(
                np.mean(calibration_scores > threshold)
            ),
        },
        "random_state": RANDOM_STATE,
        "metrics": metrics,
    }

    bundle = {
        "profile": "network_v1",
        "feature_version": "backend-features-v1",
        "feature_names": feature_columns,
        "model_type": "IsolationForest",
        "capability": "anomaly_detection",
        "model": model,
        "threshold": threshold,
        "score_direction": "higher_is_more_anomalous",
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "random_state": RANDOM_STATE,
        "evaluation": report,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, MODEL_PATH)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    predictions = holdout[
        ["window_start_ts", "label"]
    ].copy()

    predictions["anomaly_score"] = holdout_scores
    predictions["predicted_label"] = y_pred
    predictions["correct"] = (
        predictions["label"] == predictions["predicted_label"]
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nCALIBRATION")
    print("-" * 80)
    print(f"Percentile:              {CALIBRATION_PERCENTILE}")
    print(f"Threshold:               {threshold:.6f}")
    print(
        "Calibration flag rate:  "
        f"{metrics['false_positive_rate'] * 0:.4f}"
    )
    print(
        "Observed calibration rate: "
        f"{report['calibration']['observed_calibration_flag_rate']:.4f}"
    )

    print("\nHOLDOUT RESULTS")
    print("-" * 80)

    for key, value in metrics.items():
        print(f"{key:24s}: {value}")

    print("\nARTIFACTS")
    print("-" * 80)
    print(f"Model:       {MODEL_PATH.relative_to(ROOT)}")
    print(f"Report:      {REPORT_PATH.relative_to(ROOT)}")
    print(f"Predictions: {PREDICTIONS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
