from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from zenml import step

from scripts.encoder import create_logistic_regression_preprocessor


@step
def train_final_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
) -> Pipeline:
    """
    Trains the final Logistic Regression model using the best hyperparameters
    found during hyperparameter tuning.

    The model is trained on the full training dataset and logged to MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("final_training")
    preprocessor = create_logistic_regression_preprocessor(X_train)

    model = LogisticRegression(
        **best_params,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )

    final_model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run(
        run_name="final_logistic_regression_training",
        nested=True,
    ):
        final_model_pipeline.fit(X_train, y_train)

        mlflow.log_params(best_params)

        mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            artifact_path="final_logistic_regression_model",
        )

    return final_model_pipeline