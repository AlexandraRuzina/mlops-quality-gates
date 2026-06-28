from typing import Any

import mlflow
import mlflow.sklearn
from zenml import step


@step
def register_model(
    model_pipeline: Any,
    model_name: str = "credit_risk_random_forest",
) -> dict:
    """
    Registers the validated sklearn pipeline in the MLflow Model Registry.

    The pipeline includes preprocessing, encoders and the trained model.
    This step should only run after all quality gates have passed.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit_risk_model_registry")

    with mlflow.start_run(run_name="register_validated_credit_risk_model") as run:
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            registered_model_name=model_name,
        )

        model_uri = f"runs:/{run.info.run_id}/model"

    return {
        "registered_model_name": model_name,
        "model_uri": model_uri,
        "run_id": run.info.run_id,
    }