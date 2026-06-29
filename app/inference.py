# app/inference.py

from typing import Any

import mlflow.sklearn
import pandas as pd


MODEL_URI = "models:/credit_risk_random_forest/latest"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def load_model() -> Any:
    """
    Loads the latest registered credit risk model from the MLflow Model Registry.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)

    return model


def predict_credit_risk(
    model: Any,
    input_data: pd.DataFrame,
) -> dict:
    """
    Predicts the credit risk class and class probabilities for one input row.
    """

    prediction = model.predict(input_data)[0]

    prediction_label = (
        "bad"
        if prediction == 1
        else "good"
    )

    result = {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        result["probability_good"] = float(probabilities[0])
        result["probability_bad"] = float(probabilities[1])

    return result