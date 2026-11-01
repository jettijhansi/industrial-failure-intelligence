
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import PROCESSED_DIR


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_FILE = (
    PROCESSED_DIR
    / "model_artifacts"
    / "failure_risk_model.joblib"
)

app = FastAPI(
    title="Industrial Failure Risk API",
    description=(
        "Predicts whether an industrial machine may experience "
        "a failure within the next 24 hours."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model_bundle = None
model = None
feature_columns = []
prediction_threshold = 0.5
prediction_horizon = 24


def load_model():
    global model_bundle
    global model
    global feature_columns
    global prediction_threshold
    global prediction_horizon

    if not MODEL_FILE.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_FILE}. "
            "Run the model training script first."
        )

    model_bundle = joblib.load(MODEL_FILE)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    prediction_threshold = float(
        model_bundle.get("threshold", 0.5)
    )

    prediction_horizon = int(
        model_bundle.get("prediction_horizon_hours", 24)
    )


@app.on_event("startup")
def startup_event():
    load_model()


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class PredictionRequest(BaseModel):
    machine_id: str = Field(
        min_length=1,
        max_length=100
    )

    features: dict[str, Optional[float]]


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    return {
        "model_type": "RandomForestClassifier",
        "prediction_horizon_hours": prediction_horizon,
        "classification_threshold": prediction_threshold,
        "number_of_features": len(feature_columns),
        "required_features": feature_columns,
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_failure(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable."
        )

    supplied_features = request.features

    unknown_features = [
        name for name in supplied_features
        if name not in feature_columns
    ]

    if unknown_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unknown feature names supplied.",
                "unknown_features": unknown_features,
            },
        )

    # Missing features are passed as NaN.
    # The trained pipeline's median imputer handles them.
    row = {}

    for feature in feature_columns:
        value = supplied_features.get(feature)

        if value is not None:
            value = float(value)

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{feature}' must be finite."
                )

        row[feature] = np.nan if value is None else value

    import pandas as pd

    input_frame = pd.DataFrame(
        [row],
        columns=feature_columns
    )

    try:
        probability = float(
            model.predict_proba(input_frame)[0][1]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    predicted_failure = (
        probability >= prediction_threshold
    )

    if probability >= max(prediction_threshold, 0.70):
        risk_level = "HIGH"
    elif predicted_failure:
        risk_level = "ELEVATED"
    else:
        risk_level = "LOW"

    return {
        "machine_id": request.machine_id,
        "failure_probability": round(probability, 4),
        "failure_probability_percent": round(
            probability * 100, 2
        ),
        "prediction_horizon_hours": prediction_horizon,
        "classification_threshold": round(
            prediction_threshold, 4
        ),
        "predicted_failure": bool(predicted_failure),
        "risk_level": risk_level,
        "recommendation": (
            "Prioritize inspection and maintenance assessment."
            if predicted_failure
            else "Continue routine monitoring."
        ),
        "model_type": "RandomForestClassifier",
        "warning": (
            "This is a model estimate based on synthetic training data, "
            "not a confirmed equipment diagnosis."
        ),
    }
