from typing import Annotated

import pandas as pd
import mlflow
import mlflow.sklearn

from zenml import step

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@step
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    Annotated[Pipeline, "best_logistic_regression_model"],
    Annotated[dict, "best_logistic_regression_params"],
    Annotated[float, "best_logistic_regression_cv_score"],
]:
    """
    Trainiert ein Logistic-Regression-Baseline-Modell mit Encoding,
    Skalierung, Hyperparameter-Tuning, 5-Fold Stratified Cross Validation
    und MLflow Tracking.
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit_risk_logistic_regression")

    X_train = X_train.copy()
    y_train = y_train.copy()

    ordinal_features = [
        "employment",
        "job",
    ]

    nominal_features = [
        "purpose",
        "credit_history",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "age_group",
        "checking_status",
        "savings_status",
    ]

    binary_features = [
        "is_negative_checking",
        "has_additional_security",
    ]

    numeric_features = X_train.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    numeric_features = [
        col for col in numeric_features
        if col not in binary_features
    ]

    employment_categories = [
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
    ]

    job_categories = [
        [
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt",
        ]
    ]

    ordinal_categories = employment_categories + job_categories

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                Pipeline(
                    steps=[
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=ordinal_categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ordinal_features,
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                nominal_features,
            ),
            (
                "binary",
                "passthrough",
                binary_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

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

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )

    #mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="logistic_regression_training"):

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("n_iter", 10)
        mlflow.log_param("scaling", "StandardScaler")

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        mlflow.log_metric("best_cv_f1_score", best_cv_score)

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_logistic_regression_model",
        )

        print("Logistic Regression Training abgeschlossen.")
        print(f"Beste Parameter: {best_params}")
        print(f"Bester CV-F1-Score: {best_cv_score:.4f}")

    return best_model, best_params, best_cv_score