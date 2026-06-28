from typing import Annotated, Dict

import pandas as pd
import mlflow
import numpy as np
from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


@step
def test_evaluation(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
)  -> tuple[
    Annotated[Dict[str, float], "test_metrics"],
    Annotated[np.ndarray, "predicted_labels"],
    Annotated[np.ndarray, "predicted_probabilities"],
]:
    """
    Evaluates a trained model pipeline on the test dataset.

    Calculates:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - MCC

    The input model_pipeline already contains the fitted preprocessor
    and the fitted model.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test_evaluation")

    X_test = X_test.copy()
    y_test = y_test.copy()

    y_pred = model_pipeline.predict(X_test)

    if hasattr(model_pipeline, "predict_proba"):
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    test_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc),
    }

    with mlflow.start_run(run_name=f"{model_name}_test_evaluation"):
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                metric_value,
            )

    print(f"Test Evaluation abgeschlossen für: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")

    return test_metrics, y_pred, y_proba
