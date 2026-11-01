
import json
import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_FILE = PROCESSED_DIR / "training_dataset.csv"

MODEL_DIR = PROCESSED_DIR / "model_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "failure_risk_model.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"

TARGET = "failure_within_24h"
RANDOM_SEED = 20261022

EMBARGO_HOURS = 24


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    data = pd.read_csv(DATASET_FILE)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", TARGET]
    ).copy()

    data[TARGET] = data[TARGET].astype(int)

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "The dataset must contain both positive and negative labels."
        )

    return data


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(data):
    excluded = {
        "machine_id",
        "timestamp",
        TARGET,
        "failure_time",
        "failure_type",
        "severity",
        "ground_truth_label",
        "issue_type",
        "operating_state",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric features found.")

    X = data[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    usable = [
        column
        for column in X.columns
        if X[column].notna().any()
    ]

    X = X[usable]

    return X, data[TARGET], usable


# --------------------------------------------------
# CHRONOLOGICAL SPLIT WITH EMBARGO
# --------------------------------------------------

def chronological_split(data, X, y):
    timestamps = np.sort(
        data["timestamp"].unique()
    )

    n = len(timestamps)

    train_position = int(n * 0.60)
    validation_position = int(n * 0.80)

    train_end = pd.Timestamp(
        timestamps[train_position - 1]
    )

    validation_start = (
        pd.Timestamp(timestamps[train_position])
        + pd.Timedelta(hours=EMBARGO_HOURS)
    )

    validation_end = pd.Timestamp(
        timestamps[validation_position - 1]
    )

    test_start = (
        pd.Timestamp(timestamps[validation_position])
        + pd.Timedelta(hours=EMBARGO_HOURS)
    )

    train_mask = data["timestamp"] <= train_end

    validation_mask = (
        (data["timestamp"] >= validation_start)
        & (data["timestamp"] <= validation_end)
    )

    test_mask = data["timestamp"] >= test_start

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    for name, labels in [
        ("Training", y_train),
        ("Validation", y_validation),
        ("Testing", y_test),
    ]:
        if labels.empty or labels.nunique() != 2:
            raise ValueError(
                f"{name} period does not contain both classes. "
                "Review the failure distribution."
            )

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("--------------------------------")
    print(f"Training end: {train_end}")
    print(f"Validation start: {validation_start}")
    print(f"Validation end: {validation_end}")
    print(f"Test start: {test_start}")

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows: {len(X_test):,}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

def train_model(X_train, y_train):
    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining improved Random Forest...")
    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# SELECT THRESHOLD ON VALIDATION DATA ONLY
# --------------------------------------------------

def choose_threshold(model, X_validation, y_validation):
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50
    best_f2 = -1

    for threshold in np.arange(0.05, 0.81, 0.01):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_validation,
            predictions,
            beta=2,
            zero_division=0,
        )

        if score > best_f2:
            best_f2 = score
            best_threshold = float(threshold)

    print("\nVALIDATION THRESHOLD")
    print("--------------------------------")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation F2-score: {best_f2:.4f}")

    return best_threshold


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, threshold):
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_test, predictions, beta=2, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("\nFINAL TEST RESULTS")
    print("--------------------------------")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2-score: {f2:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f2_score": float(f2),
        "pr_auc": float(pr_auc),
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(y_test)),
        "positive_test_rows": int(y_test.sum()),
        "negative_test_rows": int((y_test == 0).sum()),
    }


# --------------------------------------------------
# SAVE FEATURE IMPORTANCE
# --------------------------------------------------

def save_feature_importance(model, feature_columns):
    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": classifier.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    print("\nTOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("\nIMPROVED INDUSTRIAL FAILURE MODEL")
    print("=" * 45)

    data = load_data()

    print(f"Dataset rows: {len(data):,}")
    print(f"Positive labels: {int(data[TARGET].sum()):,}")
    print(f"Negative labels: {int((data[TARGET] == 0).sum()):,}")

    X, y, feature_columns = prepare_features(data)

    print(f"Numeric features: {len(feature_columns)}")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = chronological_split(data, X, y)

    model = train_model(X_train, y_train)

    threshold = choose_threshold(
        model,
        X_validation,
        y_validation,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "prediction_horizon_hours": 24,
        },
        MODEL_FILE,
    )

    save_feature_importance(
        model,
        feature_columns,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL TRAINING COMPLETE")
    print("=" * 45)
    print(f"Model: {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Feature importance: {IMPORTANCE_FILE}")


if __name__ == "__main__":
    main()
