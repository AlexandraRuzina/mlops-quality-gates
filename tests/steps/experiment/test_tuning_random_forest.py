import sys
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TUNING_RANDOM_FOREST_PATH = (
    PROJECT_ROOT / "steps" / "experiment" / "tuning_random_forest.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "tuning_random_forest",
    TUNING_RANDOM_FOREST_PATH,
)
tuning_random_forest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning_random_forest)


class DummyPipelineRun:
    id = "dummy-zenml-run-id"
    name = "dummy-pipeline-run"


class DummyContext:
    pipeline_run = DummyPipelineRun()


class DummyMLflowRunInfo:
    run_id = "dummy-mlflow-run-id"


class DummyMLflowRun:
    info = DummyMLflowRunInfo()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyRandomizedSearchCV:
    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter,
        scoring,
        refit,
        cv,
        random_state,
        n_jobs,
        verbose,
        return_train_score,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.refit = refit
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.return_train_score = return_train_score

        self.best_estimator_ = estimator
        self.best_params_ = {
            "model__n_estimators": 300,
            "model__max_depth": 5,
            "model__min_samples_split": 2,
            "model__min_samples_leaf": 2,
            "model__max_features": "sqrt",
            "model__bootstrap": True,
        }
        self.best_index_ = 0
        self.cv_results_ = {
            "mean_test_accuracy": [0.75],
            "mean_test_precision": [0.56],
            "mean_test_recall": [0.73],
            "mean_test_f1": [0.64],
            "mean_test_roc_auc": [0.80],
            "mean_test_mcc": [0.46],
        }

        DummyRandomizedSearchCV.last_instance = self

    def fit(self, X_train, y_train):
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.fit_called_ = True
        return self


def create_sample_training_data():
    X_train = pd.DataFrame(
        [
            {
                "employment": "1<=X<4",
                "job": "skilled",
                "has_additional_security": 0,
                "high_risk_financial_status": 1,
                "purpose": "new car",
                "credit_history": "existing paid",
                "other_parties": "none",
                "property_magnitude": "car",
                "other_payment_plans": "none",
                "housing": "own",
                "age_group": "25-40",
                "checking_status": "<0",
                "savings_status": "<100",
                "duration": 24,
                "credit_amount": 3000,
                "installment_commitment": 2,
                "residence_since": 2,
                "existing_credits": 1,
                "num_dependents": 1,
                "monthly_payment": 125.0,
            },
            {
                "employment": ">=7",
                "job": "high qualif/self emp/mgmt",
                "has_additional_security": 1,
                "high_risk_financial_status": 0,
                "purpose": "business",
                "credit_history": "critical/other existing credit",
                "other_parties": "guarantor",
                "property_magnitude": "real estate",
                "other_payment_plans": "bank",
                "housing": "own",
                "age_group": "40-60",
                "checking_status": ">=200",
                "savings_status": ">=1000",
                "duration": 36,
                "credit_amount": 6000,
                "installment_commitment": 3,
                "residence_since": 4,
                "existing_credits": 2,
                "num_dependents": 2,
                "monthly_payment": 166.67,
            },
        ]
    )

    y_train = pd.Series([0, 1])

    return X_train, y_train


def patch_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "experiment": None,
        "tags": {},
        "params": {},
        "metrics": {},
        "logged_model": None,
    }

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update({"tracking_uri": uri}),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "set_experiment",
        lambda experiment: calls.update({"experiment": experiment}),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "start_run",
        lambda run_name: DummyMLflowRun(),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "set_tag",
        lambda key, value: calls["tags"].update({key: value}),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "log_param",
        lambda key, value: calls["params"].update({key: value}),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow,
        "log_metric",
        lambda key, value: calls["metrics"].update({key: value}),
    )

    monkeypatch.setattr(
        tuning_random_forest.mlflow.sklearn,
        "log_model",
        lambda sk_model, artifact_path: calls.update(
            {
                "logged_model": {
                    "sk_model": sk_model,
                    "artifact_path": artifact_path,
                }
            }
        ),
    )

    return calls


def run_tuning_random_forest(X_train, y_train):
    return tuning_random_forest.tuning_random_forest.entrypoint(
        X_train,
        y_train,
    )


def test_tuning_random_forest_returns_best_params_without_model_prefix(monkeypatch):
    X_train, y_train = create_sample_training_data()

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )

    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    result = run_tuning_random_forest(X_train, y_train)

    assert result == {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    }


def test_tuning_random_forest_configures_mlflow(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["tracking_uri"] == "http://127.0.0.1:5000"
    assert calls["experiment"] == "tuning"


def test_tuning_random_forest_logs_zenml_tags(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["tags"]["zenml_run_id"] == "dummy-zenml-run-id"
    assert calls["tags"]["zenml_pipeline_run_name"] == "dummy-pipeline-run"


def test_tuning_random_forest_logs_static_parameters(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["params"]["model_type"] == "RandomForestClassifier"
    assert calls["params"]["cv_strategy"] == "StratifiedKFold"
    assert calls["params"]["cv_folds"] == 5
    assert calls["params"]["refit_metric"] == "f1"
    assert calls["params"]["n_iter"] == 20


def test_tuning_random_forest_logs_best_cv_metrics(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["metrics"]["best_cv_accuracy"] == 0.75
    assert calls["metrics"]["best_cv_precision"] == 0.56
    assert calls["metrics"]["best_cv_recall"] == 0.73
    assert calls["metrics"]["best_cv_f1"] == 0.64
    assert calls["metrics"]["best_cv_roc_auc"] == 0.80
    assert calls["metrics"]["best_cv_mcc"] == 0.46


def test_tuning_random_forest_logs_best_params_with_prefix(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["params"]["best_model__n_estimators"] == 300
    assert calls["params"]["best_model__max_depth"] == 5
    assert calls["params"]["best_model__min_samples_split"] == 2
    assert calls["params"]["best_model__min_samples_leaf"] == 2
    assert calls["params"]["best_model__max_features"] == "sqrt"
    assert calls["params"]["best_model__bootstrap"] is True


def test_tuning_random_forest_logs_best_model(monkeypatch):
    X_train, y_train = create_sample_training_data()

    calls = patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    assert calls["logged_model"]["artifact_path"] == "best_random_forest_model"


def test_tuning_random_forest_uses_randomized_search_configuration(monkeypatch):
    X_train, y_train = create_sample_training_data()

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    search = DummyRandomizedSearchCV.last_instance

    assert search.n_iter == 20
    assert search.refit == "f1"
    assert search.random_state == 42
    assert search.n_jobs == 1
    assert search.verbose == 2
    assert search.return_train_score is True


def test_tuning_random_forest_calls_fit(monkeypatch):
    X_train, y_train = create_sample_training_data()

    patch_mlflow(monkeypatch)

    monkeypatch.setattr(
        tuning_random_forest,
        "get_step_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        tuning_random_forest,
        "RandomizedSearchCV",
        DummyRandomizedSearchCV,
    )

    _ = run_tuning_random_forest(X_train, y_train)

    search = DummyRandomizedSearchCV.last_instance

    assert search.fit_called_ is True
    pd.testing.assert_frame_equal(search.X_train_, X_train)
    pd.testing.assert_series_equal(search.y_train_, y_train)