from typing import Dict, Any
import pandas as pd
import mlflow
import mlflow.sklearn
from scripts.encoder import create_random_forest_preprocessor

from zenml import step

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

@step
def train_final_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
) -> Pipeline:
    """
    Trains the final Random Forest model using the best hyperparameters
    found during hyperparameter tuning.

    The model is trained on the full training dataset and logged to MLflow.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("final_training")
    preprocessor = create_random_forest_preprocessor(X_train)

    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    final_model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run(run_name="final_random_forest_training", nested=True):
        final_model_pipeline.fit(X_train, y_train)

        mlflow.log_params(best_params)

        mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            artifact_path="final_random_forest_model",
        )

    return final_model_pipeline