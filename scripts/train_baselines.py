
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "ton_iot"
MODEL_DIR = ROOT / "models"
AUDIT_DIR = ROOT / "audit"

RANDOM_STATE = 42
N_ESTIMATORS = 300
CALIBRATION_PERCENTILE = 95.0


# ---------------------------------------------------------------------
# Data loading and feature selection
# ---------------------------------------------------------------------

def load_profile(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"window_start_ts", "label", "attack_type"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    df = df.sort_values("window_start_ts").reset_index(drop=True)

    if df["label"].isna().any():
        raise ValueError(f"{name}: missing labels detected")

    return df


def get_features(df: pd.DataFrame) -> list[str]:
    excluded = {
        "window_start_ts",
        "label",
        "attack_type",
        "block",
    }

    features = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not features:
        raise ValueError("No numeric model features found")

    return features


def chronological_blocks(
    df: pd.DataFrame,
    n_blocks: int = 10,
) -> list[pd.DataFrame]:
    indices = np.array_split(np.arange(len(df)), n_blocks)

    return [
        df.iloc[index].copy()
        for index in indices
    ]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "detector",
                IsolationForest(
                    n_estimators=N_ESTIMATORS,
                    contamination="auto",
                    max_samples="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_anomaly_scores(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    # Isolation Forest score_samples:
    # lower values indicate more anomalous observations.
    # Negating makes higher values indicate greater anomaly.
    return -model.score_samples(X)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\n{name}")
    print("-" * 60)

    print("Windows:", len(y_true))
    print("Normal:", int((y_true == 0).sum()))
    print("Attack:", int((y_true == 1).sum()))
    print("Threshold:", round(float(threshold), 6))

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["normal", "attack"],
            zero_division=0,
        )
    )

    metrics = {
        "windows": int(len(y_true)),
        "normal_windows": int((y_true == 0).sum()),
        "attack_windows": int((y_true == 1).sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": (
            float(fp / (fp + tn))
            if fp + tn > 0 else None
        ),
        "attack_recall": (
            float(tp / (tp + fn))
            if tp + fn > 0 else None
        ),
        "attack_precision": (
            float(tp / (tp + fp))
            if tp + fp > 0 else None
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )
        metrics["pr_auc"] = float(
            average_precision_score(y_true, scores)
        )

        print("ROC AUC:", round(metrics["roc_auc"], 6))
        print("PR AUC:", round(metrics["pr_auc"], 6))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        print("ROC AUC / PR AUC: unavailable (one class only)")

    return metrics


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_profile(
    name: str,
    train_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    test_data: pd.DataFrame,
    future_normal_data: pd.DataFrame | None = None,
) -> dict:

    print("\n" + "=" * 65)
    print("PROFILE:", name)
    print("=" * 65)

    # Feature schema is determined from the complete profile.
    # Labels, timestamps and raw attack types are excluded.
    features = get_features(train_data)

    missing_features = set(features) - set(calibration_data.columns)

    if missing_features:
        raise ValueError(
            f"Calibration data missing features: {sorted(missing_features)}"
        )

    missing_features = set(features) - set(test_data.columns)

    if missing_features:
        raise ValueError(
            f"Test data missing features: {sorted(missing_features)}"
        )

    # Train only on normal observations.
    train_normal = train_data[
        train_data["label"] == 0
    ].copy()

    if train_normal.empty:
        raise ValueError(f"{name}: no normal training observations")

    # Calibration also uses normal observations only.
    calibration_normal = calibration_data[
        calibration_data["label"] == 0
    ].copy()

    if calibration_normal.empty:
        raise ValueError(f"{name}: no normal calibration observations")

    X_train = train_normal[features]
    X_calibration = calibration_normal[features]
    X_test = test_data[features]

    y_test = test_data["label"].astype(int).to_numpy()

    if X_train.isna().all(axis=0).any():
        bad = X_train.columns[X_train.isna().all(axis=0)].tolist()
        raise ValueError(
            f"{name}: entirely missing training features: {bad}"
        )

    print("Features:", len(features))
    print("Training normal windows:", len(train_normal))
    print("Calibration normal windows:", len(calibration_normal))
    print("Test windows:", len(test_data))
    print("Test attacks:", int((y_test == 1).sum()))

    # Fit model.
    model = build_model()
    model.fit(X_train)

    # Calibrate threshold using normal calibration scores.
    calibration_scores = calculate_anomaly_scores(
        model,
        X_calibration,
    )

    threshold = float(
        np.percentile(
            calibration_scores,
            CALIBRATION_PERCENTILE,
        )
    )

    print(
        "\nCalibration threshold:",
        round(threshold, 6),
    )

    # Evaluate attack-containing or normal-only test period.
    test_scores = calculate_anomaly_scores(
        model,
        X_test,
    )

    test_metrics = evaluate(
        name + " / primary evaluation",
        y_test,
        test_scores,
        threshold,
    )

    # Optional future normal-only evaluation.
    future_metrics = None

    if future_normal_data is not None and not future_normal_data.empty:

        future = future_normal_data[
            future_normal_data["label"] == 0
        ].copy()

        if not future.empty:
            future_scores = calculate_anomaly_scores(
                model,
                future[features],
            )

            future_predictions = (
                future_scores >= threshold
            ).astype(int)

            false_positives = int(future_predictions.sum())

            future_metrics = {
                "windows": int(len(future)),
                "false_positives": false_positives,
                "false_positive_rate": float(
                    false_positives / len(future)
                ),
            }

            print("\nFuture normal-only evaluation")
            print("-" * 60)
            print("Windows:", len(future))
            print("False positives:", false_positives)
            print(
                "False-positive rate:",
                round(future_metrics["false_positive_rate"], 6),
            )

    # Save reproducible model bundle.
    bundle = {
        "profile": name,
        "feature_version": "backend-features-v1",
        "feature_names": features,
        "model_type": "isolation_forest",
        "capability": ["anomaly_detection"],
        "model": model,
        "threshold": threshold,
        "score_direction": "higher_is_more_anomalous",
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "random_state": RANDOM_STATE,
        "training": {
            "normal_windows": int(len(train_normal)),
            "attack_windows": 0,
            "training_source": "chronological_normal_only",
        },
        "calibration": {
            "normal_windows": int(len(calibration_normal)),
            "attack_windows_excluded": int(
                (calibration_data["label"] == 1).sum()
            ),
        },
        "evaluation": {
            **test_metrics,
            "future_normal_only": future_metrics,
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_path = MODEL_DIR / f"{name}_isolation_forest.joblib"

    joblib.dump(bundle, output_path)

    print("\nModel saved:", output_path.relative_to(ROOT))

    return bundle


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # Network profile
    # -------------------------------------------------------------

    network = load_profile("network_v1")

    first_attack_ts = network.loc[
        network["label"] == 1,
        "window_start_ts",
    ].min()

    if pd.isna(first_attack_ts):
        raise ValueError("network_v1 contains no attack observations")

    network_before_attack = network[
        network["window_start_ts"] < first_attack_ts
    ].copy()

    network_attack_period = network[
        network["window_start_ts"] >= first_attack_ts
    ].copy()

    if network_attack_period["label"].sum() == 0:
        raise ValueError("Network attack period contains no attacks")

    # Use the first 80% of the pre-attack period for training.
    # Remaining 20% is used for normal-only calibration.
    split_index = int(len(network_before_attack) * 0.80)

    network_train = network_before_attack.iloc[:split_index]
    network_calibration = network_before_attack.iloc[split_index:]

    train_profile(
        name="network_v1",
        train_data=network_train,
        calibration_data=network_calibration,
        test_data=network_attack_period,
    )

    # -------------------------------------------------------------
    # Linux host profile
    # -------------------------------------------------------------

    linux = load_profile("linux_host_v1")
    linux_blocks = chronological_blocks(linux, n_blocks=10)

    linux_train = pd.concat(
        linux_blocks[:5],
        ignore_index=True,
    )

    linux_calibration = linux_blocks[5]
    linux_test = linux_blocks[6]

    linux_future_normal = pd.concat(
        linux_blocks[7:10],
        ignore_index=True,
    )

    train_profile(
        name="linux_host_v1",
        train_data=linux_train,
        calibration_data=linux_calibration,
        test_data=linux_test,
        future_normal_data=linux_future_normal,
    )

    print("\n" + "=" * 65)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 65)

    print("\nGenerated model files:")

    for path in sorted(MODEL_DIR.glob("*_isolation_forest.joblib")):
        print(" -", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
