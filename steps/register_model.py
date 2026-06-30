from typing import Annotated
import mlflow
from zenml import step


@step
def register_model(
    checked: bool,
    mlflow_run_id: str,
    model_name: str = "credit_risk_random_forest",
) -> Annotated[dict, "registered_model_data"]:
    """
    Registers the already logged sklearn pipeline in the MLflow Model Registry.

    This step should only run after all quality gates have passed.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if not checked:
        raise ValueError(
            "Model registration failed: production readiness checks did not pass."
        )

    model_uri = f"runs:/{mlflow_run_id}/final_random_forest_model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )


    return {
        "registered_model_name": model_name,
        "model_uri": model_uri,
        "run_id": mlflow_run_id,
        "model_version": registered_model.version,
    }