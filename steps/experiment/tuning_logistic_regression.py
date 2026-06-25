from typing import Annotated

import pandas as pd
import mlflow
import mlflow.sklearn

from zenml import step
from scripts.encoder import create_logistic_regression_preprocessor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef


@step
def tuning_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[dict, "best_logistic_regression_params"]:
    """
    Trainiert ein Logistic-Regression-Baseline-Modell mit Encoding,
    Skalierung, Hyperparameter-Tuning, 5-Fold Stratified Cross Validation
    und MLflow Tracking.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tuning")

    X_train = X_train.copy()
    y_train = y_train.copy()

    preprocessor = create_logistic_regression_preprocessor(X_train)

    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    param_distributions = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "mcc": make_scorer(matthews_corrcoef),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring=scoring,
        refit="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )

    with mlflow.start_run(run_name="logistic_regression_training"):

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("refit_metric", "f1")
        mlflow.log_param("n_iter", 10)
        mlflow.log_param("scaling", "StandardScaler")

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params_with_prefix = search.best_params_
        best_index = search.best_index_

        mlflow.log_metric(
            "best_cv_accuracy",
            search.cv_results_["mean_test_accuracy"][best_index],
        )
        mlflow.log_metric(
            "best_cv_precision",
            search.cv_results_["mean_test_precision"][best_index],
        )
        mlflow.log_metric(
            "best_cv_recall",
            search.cv_results_["mean_test_recall"][best_index],
        )
        mlflow.log_metric(
            "best_cv_f1",
            search.cv_results_["mean_test_f1"][best_index],
        )
        mlflow.log_metric(
            "best_cv_roc_auc",
            search.cv_results_["mean_test_roc_auc"][best_index],
        )
        mlflow.log_metric(
            "best_cv_mcc",
            search.cv_results_["mean_test_mcc"][best_index],
        )

        for param_name, param_value in best_params_with_prefix.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_logistic_regression_model",
        )

        best_params = {
            key.replace("model__", ""): value
            for key, value in best_params_with_prefix.items()
        }

        print("Logistic Regression Training abgeschlossen.")
        print(f"Beste Parameter: {best_params}")
        print(f"Best CV Accuracy: {search.cv_results_['mean_test_accuracy'][best_index]:.4f}")
        print(f"Best CV Precision: {search.cv_results_['mean_test_precision'][best_index]:.4f}")
        print(f"Best CV Recall: {search.cv_results_['mean_test_recall'][best_index]:.4f}")
        print(f"Best CV F1: {search.cv_results_['mean_test_f1'][best_index]:.4f}")
        print(f"Best CV ROC-AUC: {search.cv_results_['mean_test_roc_auc'][best_index]:.4f}")
        print(f"Best CV MCC: {search.cv_results_['mean_test_mcc'][best_index]:.4f}")

    return best_params