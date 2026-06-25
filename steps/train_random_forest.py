from typing import Annotated

import pandas as pd
import mlflow
import mlflow.sklearn

from zenml import step

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


@step
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    Annotated[Pipeline, "best_random_forest_model"],
    Annotated[dict, "best_random_forest_params"],
    Annotated[float, "best_random_forest_cv_score"],
]:
    # Rückgabe bestes trainiertes Modell, besten Hyperparameter, besten Cross-Validation-F1-Score
    """
    Trainiert ein Random-Forest-Modell mit Encoding, Hyperparameter-Tuning,
    5-Fold Stratified Cross Validation und MLflow Tracking.
    """

    #MLflow konfigurieren, Ergebnisse an lokalen MLflow-Server geschickt
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit_risk_random_forest")

    X_train = X_train.copy()
    y_train = y_train.copy()

    ordinal_features = [
        "employment",
        "job",
    ]

    binary_features = [
        "is_negative_checking",
        "has_additional_security",
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

    # Alle übrigen numerischen Spalten automatisch durchreichen
    numeric_features = X_train.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    #Binäre Features aus numerischen Features entfernen
    numeric_features = [
        col for col in numeric_features
        if col not in binary_features
    ]

    # Reihenfolge für ordinales Encoding
    employment_categories = [
        ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
    ]

    job_categories = [
        ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"]
    ]

    #Zusammenführung der Listen
    ordinal_categories = employment_categories + job_categories

    #Welche Spalten sollen wie verarbeitet werden
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
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
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",  #Alle Spalten, die nicht in einer der Gruppen enthalten sind, werden entfernt.
    )

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    )

    # Encoder fit nur auf Fold-Trainingsdaten
    # Encoder transformiert Fold-Validierungsdaten
    # Modell trainiert auf Fold-Trainingsdaten
    # Modell bewertet auf Fold-Validierungsdaten
    # Vermeiden von Data Leakage
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    #Hyperparameter-Suchraum
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500], #Anzahl der Bäume
        "model__max_depth": [None, 5, 10, 20, 30], #Maximale Tiefe der Bäume
        "model__min_samples_split": [2, 5, 10], #Mindestanzahl Samples, damit ein Knoten weiter geteilt wird.
        "model__min_samples_leaf": [1, 2, 4], #Mindestanzahl Samples in einem Blatt.
        "model__max_features": ["sqrt", "log2"], #Wie viele Features pro Split berücksichtigt werden.
        "model__bootstrap": [True, False], #Ob Bäume mit Bootstrap-Samples trainiert werden.
    }

    #5-Fold Stratified Cross Validation.
    #Trainingsdatensatz wird in 5 Teile geteilt --> Dann wird 5-mal trainiert
    # Stratified --> Klassenverteilung bleibt in jedem Fold ungefähr gleich
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    #Hyperparameter-Tuning
    #wählt zufällig 20 Kombinationen aus dem Suchraum
    #Für jede Kombination macht es 5-Fold Cross Validation
    # 20 Kombinationen × 5 Folds = 100 Trainingsläufe
    # Aber in cv_results_ wird pro Kombination eine zusammengefasste Zeile gespeichert.
    #Bestes Modell wird nach F1-Score ausgewählt
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )

    #mlflow.sklearn.autolog(log_models=False)

    #Alles innerhalb dieses Blocks wird in MLflow als ein Run gespeichert.
    with mlflow.start_run(run_name="random_forest_training"):

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("n_iter", 20)

        #Training starten
        search.fit(X_train, y_train)
        #Beste Ergebnisse auswählen
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
        #Beste Ergebnisse in mlflow speichern
        mlflow.log_metric("best_cv_f1_score", best_cv_score)

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        #speichert ganze Pipeline
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_random_forest_model",
        )

        print("Random Forest Training abgeschlossen.")
        print(f"Beste Parameter: {best_params}")
        print(f"Bester CV-F1-Score: {best_cv_score:.4f}")

    return best_model, best_params, best_cv_score