from typing import Dict, Any, Annotated

import pandas as pd
import mlflow
import mlflow.sklearn

from scripts.encoder import create_random_forest_preprocessor
from zenml import get_step_context
from zenml import step

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


@step
def train_final_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
) -> tuple[
    Annotated[Pipeline, "rf_pipeline"],
    Annotated[str, "mlflow_run_id"],
]:
    """
    Trains the final Random Forest model using the best hyperparameters
    found during hyperparameter tuning.

    The model is trained on the full training dataset and logged to MLflow.
    The MLflow run ID is returned so that following pipeline steps can log
    their metrics and artifacts to the same MLflow run.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("final_training")

    context = get_step_context()
    preprocessor = create_random_forest_preprocessor(X_train)

    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )

    final_model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run(
        run_name="final_random_forest_training",
    ) as run:
        mlflow_run_id = run.info.run_id

        mlflow.set_tag(
            "zenml_run_id",
            str(context.pipeline_run.id),
        )

        mlflow.set_tag(
            "zenml_pipeline_run_name",
            context.pipeline_run.name,
        )

        final_model_pipeline.fit(X_train, y_train)

        mlflow.log_params(best_params)

        mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            artifact_path="final_random_forest_model",
        )

    return final_model_pipeline, mlflow_run_id